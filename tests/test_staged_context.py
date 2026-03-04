"""Tests for staged context-length training support.

Covers:
1. compute_piece_weights() — all three modes
2. Prefix detection (_detect_prefix_length) — with and without tokenizer
3. BachDataset prefix-preserving crops
4. BachDataset piece_ids filtering synchronization
5. load_checkpoint override_max_seq_len
6. CLI --piece-balance option
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from bach_gen.data.dataset import (
    BachDataset,
    compute_piece_weights,
    create_dataset,
    _detect_prefix_length,
)
from bach_gen.model.config import ModelConfig
from bach_gen.model.architecture import BachTransformer
from bach_gen.model.trainer import Trainer


# ---------------------------------------------------------------------------
# Mock tokenizer helper
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(key_token_id: int = 59) -> MagicMock:
    """Create a mock tokenizer with a token_to_name dict that maps key_token_id to KEY_C_major."""
    tok = MagicMock()
    tok.token_to_name = {
        0: "PAD", 1: "BOS", 2: "EOS",
        20: "STYLE_BACH", 25: "FORM_INVENTION", 16: "MODE_2PART",
        37: "METER_4_4", key_token_id: "KEY_C_major",
        83: "OCT_2", 89: "DEG_1", 98: "Dur_480",
    }
    tok.name_to_token = {v: k for k, v in tok.token_to_name.items()}
    tok.vocab_size = 125
    return tok


def _make_seq_with_prefix(prefix_tokens: list[int], body_len: int, body_start: int = 100) -> list[int]:
    """Build a sequence = prefix + body of length body_len."""
    body = list(range(body_start, body_start + body_len))
    return prefix_tokens + body


# ===========================================================================
# 1. compute_piece_weights
# ===========================================================================

class TestComputePieceWeights:
    def test_none_mode(self):
        ids = ["a", "a", "b"]
        w = compute_piece_weights(ids, mode="none")
        assert w == [1.0, 1.0, 1.0]

    def test_empty_ids(self):
        w = compute_piece_weights([], mode="sqrt")
        assert w == []

    def test_sqrt_mode(self):
        ids = ["a", "a", "a", "a", "b"]
        w = compute_piece_weights(ids, mode="sqrt")
        # "a" has 4 chunks → 1/sqrt(4) = 0.5; "b" has 1 → 1/sqrt(1) = 1.0
        assert len(w) == 5
        assert w[0] == pytest.approx(0.5)
        assert w[3] == pytest.approx(0.5)
        assert w[4] == pytest.approx(1.0)

    def test_inverse_mode(self):
        ids = ["a", "a", "b"]
        w = compute_piece_weights(ids, mode="inverse")
        assert w[0] == pytest.approx(0.5)
        assert w[1] == pytest.approx(0.5)
        assert w[2] == pytest.approx(1.0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown piece_balance mode"):
            compute_piece_weights(["a"], mode="cubic")


# ===========================================================================
# 2. Prefix detection
# ===========================================================================

class TestPrefixDetection:
    def test_with_key_token(self):
        tok = _make_mock_tokenizer(key_token_id=59)
        seq = [1, 20, 25, 16, 37, 59, 83, 89, 98]
        #       BOS STYLE FORM MODE METER KEY body...
        prefix_len = _detect_prefix_length(seq, tok)
        # KEY_ is at index 5, so prefix_len should be 6 (include the KEY token)
        assert prefix_len == 6

    def test_no_key_token(self):
        tok = _make_mock_tokenizer(key_token_id=59)
        seq = [1, 20, 25, 83, 89, 98]  # no KEY_ token
        prefix_len = _detect_prefix_length(seq, tok)
        assert prefix_len == 0

    def test_no_tokenizer(self):
        assert _detect_prefix_length([1, 2, 3], None) == 0

    def test_tokenizer_without_token_to_name(self):
        tok = MagicMock(spec=[])  # no attributes
        assert _detect_prefix_length([1, 2, 3], tok) == 0


# ===========================================================================
# 3. BachDataset prefix-preserving crops
# ===========================================================================

class TestPrefixPreservingCrops:
    def test_prefix_always_present_in_crop(self):
        tok = _make_mock_tokenizer(key_token_id=59)
        prefix = [1, 20, 25, 16, 37, 59]  # 6 tokens
        seq = _make_seq_with_prefix(prefix, body_len=200)
        # seq is 206 tokens total, seq_len=50 forces a crop
        ds = BachDataset([seq], seq_len=50, tokenizer=tok)

        # Sample multiple times to exercise the random crop
        for _ in range(20):
            item = ds[0]
            # input_ids is seq[:-1], so it's length seq_len-1=49
            ids = item["input_ids"].tolist()
            # First 6 tokens of seq are the prefix; since input_ids = seq[:-1],
            # we only check the first 5 are the prefix minus last token of prefix
            # Actually: seq = prefix + body → crop → input_ids = crop[:-1]
            # The crop starts with the full prefix, so ids[:6] == prefix
            assert ids[:6] == prefix

    def test_no_tokenizer_uses_random_crop(self):
        prefix = [1, 20, 25, 16, 37, 59]
        seq = _make_seq_with_prefix(prefix, body_len=200)
        ds = BachDataset([seq], seq_len=50, tokenizer=None)

        # Without tokenizer, prefix_len is 0, so standard random crop
        first_tokens = set()
        for _ in range(50):
            item = ds[0]
            first_tokens.add(item["input_ids"][0].item())

        # Should vary since random crop starts at different offsets
        assert len(first_tokens) > 1

    def test_short_sequence_gets_padded(self):
        tok = _make_mock_tokenizer(key_token_id=59)
        prefix = [1, 20, 59]
        seq = _make_seq_with_prefix(prefix, body_len=10)  # 13 total < 20, filtered out
        # Need at least 20 tokens
        seq_long = _make_seq_with_prefix(prefix, body_len=20)  # 23 total
        ds = BachDataset([seq_long], seq_len=50, tokenizer=tok)

        item = ds[0]
        # Should be padded to seq_len - 1 = 49
        assert item["input_ids"].shape[0] == 49
        # Prefix should be intact
        assert item["input_ids"][:3].tolist() == prefix[:3]

    def test_prefix_fills_most_of_seq_len(self):
        """When prefix is almost as long as seq_len, body window is small."""
        tok = _make_mock_tokenizer(key_token_id=59)
        # Make a long prefix (45 tokens before KEY, KEY at position 45)
        long_prefix_ids = list(range(1, 45)) + [59]  # 45 tokens, KEY at index 44
        # We need the mock tokenizer to recognize token 59 as KEY
        seq = long_prefix_ids + list(range(200, 220))  # 65 total
        ds = BachDataset([seq], seq_len=50, tokenizer=tok)

        item = ds[0]
        ids = item["input_ids"].tolist()
        # Prefix should be preserved (45 tokens), body window is 5
        assert ids[:45] == long_prefix_ids

    def test_exact_seq_len_no_crop(self):
        tok = _make_mock_tokenizer(key_token_id=59)
        prefix = [1, 20, 59]
        seq = _make_seq_with_prefix(prefix, body_len=47)  # exactly 50
        ds = BachDataset([seq], seq_len=50, tokenizer=tok)

        item = ds[0]
        assert item["input_ids"].shape[0] == 49
        assert item["input_ids"][:3].tolist() == prefix[:3]


# ===========================================================================
# 4. BachDataset piece_ids synchronization
# ===========================================================================

class TestPieceIdsSynchronization:
    def test_piece_ids_filtered_with_sequences(self):
        """Short sequences are removed; piece_ids must stay in sync."""
        seqs = [
            list(range(50)),   # kept (50 >= 20)
            list(range(5)),    # filtered out (5 < 20)
            list(range(30)),   # kept
        ]
        pids = ["piece_a", "piece_b", "piece_c"]
        ds = BachDataset(seqs, seq_len=100, piece_ids=pids)
        assert len(ds.sequences) == 2
        assert ds.piece_ids == ["piece_a", "piece_c"]

    def test_no_piece_ids_gives_empty_list(self):
        seqs = [list(range(50))]
        ds = BachDataset(seqs, seq_len=100)
        assert ds.piece_ids == []

    def test_create_dataset_passes_piece_ids_through(self):
        seqs = [list(range(20, 70)) for _ in range(20)]
        pids = [f"piece_{i % 5}" for i in range(20)]
        train_ds, val_ds = create_dataset(seqs, seq_len=100, piece_ids=pids)
        # All pieces should appear in either train or val
        all_pids = set(train_ds.piece_ids) | set(val_ds.piece_ids)
        assert all_pids == set(pids)


# ===========================================================================
# 5. load_checkpoint override_max_seq_len
# ===========================================================================

class TestLoadCheckpointOverrideSeqLen:
    def test_override_changes_config_max_seq_len(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=128)
        model = BachTransformer(config)
        ckpt_path = tmp_path / "test.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": config,
            "best_val_loss": 1.0,
        }, ckpt_path)

        loaded_model, loaded_config = Trainer.load_checkpoint(
            ckpt_path,
            device=torch.device("cpu"),
            override_max_seq_len=256,
        )
        assert loaded_config.max_seq_len == 256

    def test_no_override_preserves_original(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=128)
        model = BachTransformer(config)
        ckpt_path = tmp_path / "test.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": config,
            "best_val_loss": 1.0,
        }, ckpt_path)

        loaded_model, loaded_config = Trainer.load_checkpoint(
            ckpt_path,
            device=torch.device("cpu"),
        )
        assert loaded_config.max_seq_len == 128


# ===========================================================================
# 6. CLI --piece-balance option
# ===========================================================================

class TestCLIPieceBalance:
    def test_train_has_piece_balance_option(self):
        from click.testing import CliRunner
        from bach_gen.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--piece-balance" in result.output
        assert "sqrt" in result.output
        assert "inverse" in result.output
        assert "none" in result.output


# ===========================================================================
# 7. Trainer _make_train_loader
# ===========================================================================

class TestMakeTrainLoader:
    def test_no_balance_uses_shuffle(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=64)
        model = BachTransformer(config)
        seqs = [list(range(20, 60)) for _ in range(10)]
        ds = BachDataset(seqs, seq_len=64)
        trainer = Trainer(
            model=model, train_dataset=ds,
            batch_size=2, checkpoint_dir=tmp_path / "m",
            device=torch.device("cpu"), piece_balance="none",
        )
        loader = trainer._make_train_loader()
        # No sampler → shuffle mode
        assert loader.sampler is not None  # PyTorch adds a RandomSampler internally

    def test_sqrt_balance_with_piece_ids(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=64)
        model = BachTransformer(config)
        seqs = [list(range(20, 60)) for _ in range(10)]
        pids = ["a"] * 8 + ["b"] * 2
        ds = BachDataset(seqs, seq_len=64, piece_ids=pids)
        trainer = Trainer(
            model=model, train_dataset=ds,
            batch_size=2, checkpoint_dir=tmp_path / "m",
            device=torch.device("cpu"), piece_balance="sqrt",
        )
        loader = trainer._make_train_loader()
        # Should use WeightedRandomSampler
        from torch.utils.data import WeightedRandomSampler
        assert isinstance(loader.sampler, WeightedRandomSampler)

    def test_no_piece_ids_falls_back_to_shuffle(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=64)
        model = BachTransformer(config)
        seqs = [list(range(20, 60)) for _ in range(10)]
        ds = BachDataset(seqs, seq_len=64)  # no piece_ids
        trainer = Trainer(
            model=model, train_dataset=ds,
            batch_size=2, checkpoint_dir=tmp_path / "m",
            device=torch.device("cpu"), piece_balance="sqrt",
        )
        loader = trainer._make_train_loader()
        from torch.utils.data import WeightedRandomSampler
        assert not isinstance(loader.sampler, WeightedRandomSampler)


# ===========================================================================
# 8. Trainer.transition_seq_len
# ===========================================================================

class TestTransitionSeqLen:
    def test_updates_dataset_and_config(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=64)
        model = BachTransformer(config)
        seqs = [list(range(20, 60)) for _ in range(10)]
        train_ds = BachDataset(seqs, seq_len=64)
        val_ds = BachDataset(seqs[:2], seq_len=64)
        trainer = Trainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds,
            batch_size=2, checkpoint_dir=tmp_path / "m",
            device=torch.device("cpu"),
        )

        trainer.transition_seq_len(128)

        assert train_ds.seq_len == 128
        assert val_ds.seq_len == 128
        assert config.max_seq_len == 128
        # Check attention layers
        for layer in model.layers:
            assert layer.attn.max_seq_len == 128

    def test_transition_without_val_dataset(self, tmp_path):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=64)
        model = BachTransformer(config)
        seqs = [list(range(20, 60)) for _ in range(10)]
        train_ds = BachDataset(seqs, seq_len=64)
        trainer = Trainer(
            model=model, train_dataset=train_ds, val_dataset=None,
            batch_size=2, checkpoint_dir=tmp_path / "m",
            device=torch.device("cpu"),
        )
        # Should not raise
        trainer.transition_seq_len(256)
        assert train_ds.seq_len == 256


# ===========================================================================
# 9. Staged training (seq_len_stages in train())
# ===========================================================================

class TestStagedTraining:
    def _make_staged_trainer(self, tmp_path, seq_len=64):
        config = ModelConfig(vocab_size=100, embed_dim=32, num_heads=2,
                             num_layers=1, max_seq_len=seq_len)
        model = BachTransformer(config)
        seqs = [list(range(20, 20 + seq_len)) for _ in range(12)]
        train_ds = BachDataset(seqs[:10], seq_len=seq_len)
        val_ds = BachDataset(seqs[10:], seq_len=seq_len)
        return Trainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds,
            lr=1e-4, batch_size=2, checkpoint_dir=tmp_path / "m",
            device=torch.device("cpu"),
        )

    def test_stages_run_correct_epoch_count(self, tmp_path):
        trainer = self._make_staged_trainer(tmp_path)
        stages = [(32, 3), (64, 2)]  # 5 total epochs
        history = trainer.train(
            epochs=999,  # should be ignored
            log_interval=1, val_interval=1,
            seq_len_stages=stages,
        )
        assert history["epochs_ran"] == 5

    def test_stages_transition_seq_len(self, tmp_path):
        trainer = self._make_staged_trainer(tmp_path)
        stages = [(32, 2), (64, 2)]
        seq_lens_seen = []

        original_train_epoch = trainer._train_epoch

        def spy_train_epoch(loader, use_rope=True):
            seq_lens_seen.append(trainer.train_dataset.seq_len)
            return original_train_epoch(loader, use_rope=use_rope)

        with patch.object(trainer, "_train_epoch", side_effect=spy_train_epoch):
            trainer.train(
                log_interval=1, val_interval=999,
                seq_len_stages=stages,
            )

        # First 2 epochs at 32, next 2 at 64
        assert seq_lens_seen == [32, 32, 64, 64]

    def test_stage_checkpoints_saved(self, tmp_path):
        trainer = self._make_staged_trainer(tmp_path)
        stages = [(32, 1), (64, 1)]
        trainer.train(
            log_interval=1, val_interval=1,
            seq_len_stages=stages,
        )
        models_dir = tmp_path / "m"
        assert (models_dir / "stage1.pt").exists()
        assert (models_dir / "stage2.pt").exists()
        assert (models_dir / "final.pt").exists()

    def test_no_stages_behaves_like_before(self, tmp_path):
        trainer = self._make_staged_trainer(tmp_path)
        history = trainer.train(
            epochs=3,
            log_interval=1, val_interval=1,
            seq_len_stages=None,
        )
        assert history["epochs_ran"] == 3

    def test_final_config_has_last_stage_seq_len(self, tmp_path):
        trainer = self._make_staged_trainer(tmp_path)
        stages = [(32, 2), (128, 1)]
        trainer.train(
            log_interval=1, val_interval=999,
            seq_len_stages=stages,
        )
        assert trainer.model.config.max_seq_len == 128


# ===========================================================================
# 10. CLI --seq-len-stages option
# ===========================================================================

class TestCLISeqLenStages:
    def test_train_has_seq_len_stages_option(self):
        from click.testing import CliRunner
        from bach_gen.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--seq-len-stages" in result.output
