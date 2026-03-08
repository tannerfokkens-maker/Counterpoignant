"""Tests for LoopLM (weight-tied recurrence) implementation."""

from __future__ import annotations

import torch
import pytest

from bach_gen.model.config import ModelConfig
from bach_gen.model.architecture import (
    BachTransformer,
    LoopLMOutput,
    compute_exit_distribution,
)


def _tiny_config(**overrides) -> ModelConfig:
    """Create a minimal model config for fast tests."""
    defaults = dict(
        vocab_size=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
        dropout=0.0,
        pos_encoding="rope",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestComputeExitDistribution:
    """Unit tests for the exit distribution computation."""

    def test_single_step_returns_ones(self):
        dist = compute_exit_distribution([], num_steps=1)
        assert dist.shape == (1, 1, 1)
        torch.testing.assert_close(dist, torch.ones_like(dist))

    def test_two_steps_sums_to_one(self):
        lam = [
            torch.tensor([[0.4, 0.6]]),
        ]
        dist = compute_exit_distribution(lam)
        assert dist.shape == (2, 1, 2)
        sums = dist.sum(dim=0)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_three_steps_sums_to_one(self):
        lam = [
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.4, 0.5, 0.6]]),
        ]
        dist = compute_exit_distribution(lam)
        assert dist.shape == (3, 1, 3)
        sums = dist.sum(dim=0)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_last_step_gets_remaining_mass(self):
        # If lambda_0 = 0, all mass should go to the last step
        lam = [
            torch.tensor([[0.0]]),
            torch.tensor([[0.5]]),
        ]
        dist = compute_exit_distribution(lam)
        # Step 0: lambda_0 * S_0 = 0.0 * 1.0 = 0.0
        # Step 1: lambda_1 * S_1 = 0.5 * 1.0 = 0.5
        # Step 2 (last): remaining survival mass = 0.5
        assert dist[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert dist[1, 0, 0].item() == pytest.approx(0.5, abs=1e-4)
        assert dist[2, 0, 0].item() == pytest.approx(0.5, abs=1e-4)


class TestLoopLMForward:
    """Test LoopLM forward pass variants."""

    def test_standard_path_unchanged(self):
        """T_max=1 should produce identical output to a standard transformer."""
        config = _tiny_config(num_recurrent_steps=1)
        model = BachTransformer(config)
        model.eval()

        ids = torch.randint(0, 32, (2, 8))
        out = model(ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, 32)

    def test_looped_forward_returns_logits(self):
        """T_max>1 without return_all_steps returns plain logits."""
        config = _tiny_config(num_recurrent_steps=3)
        model = BachTransformer(config)
        model.eval()

        ids = torch.randint(0, 32, (2, 8))
        out = model(ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, 32)

    def test_looped_forward_return_all_steps(self):
        """return_all_steps=True returns LoopLMOutput."""
        config = _tiny_config(num_recurrent_steps=3)
        model = BachTransformer(config)
        model.eval()

        ids = torch.randint(0, 32, (2, 8))
        out = model(ids, return_all_steps=True)
        assert isinstance(out, LoopLMOutput)
        assert out.logits.shape == (2, 8, 32)
        assert len(out.all_logits) == 3
        for logits in out.all_logits:
            assert logits.shape == (2, 8, 32)
        # No exit gate → no lambdas
        assert len(out.exit_lambdas) == 0

    def test_exit_gate_produces_lambdas(self):
        """With exit gate enabled, return_all_steps includes exit_lambdas."""
        config = _tiny_config(
            num_recurrent_steps=3,
            looplm_exit_gate=True,
        )
        model = BachTransformer(config)
        model.eval()
        assert model.exit_gate is not None

        ids = torch.randint(0, 32, (2, 8))
        out = model(ids, return_all_steps=True)
        assert isinstance(out, LoopLMOutput)
        assert len(out.exit_lambdas) == 2
        for lam in out.exit_lambdas:
            assert lam.shape == (2, 8)
            assert (lam >= 0).all() and (lam <= 1).all()

    def test_full_depth_equivalence_when_threshold_is_one(self):
        """q=1.0 should match the ordinary full-depth recurrent output."""
        config = _tiny_config(
            num_recurrent_steps=4,
            looplm_exit_gate=True,
            looplm_exit_threshold=1.0,
        )
        model = BachTransformer(config)
        model.eval()

        ids = torch.randint(0, 32, (2, 8))
        gated_logits = model(ids)
        all_steps = model(ids, return_all_steps=True)

        torch.testing.assert_close(gated_logits, all_steps.logits)

    def test_low_threshold_exits_in_fewer_steps_than_full_depth(self):
        """A low exit threshold should reduce recurrent steps during eval."""
        config = _tiny_config(
            num_recurrent_steps=4,
            looplm_exit_gate=True,
            looplm_exit_threshold=0.3,
        )
        model = BachTransformer(config)
        model.eval()

        with torch.no_grad():
            model.exit_gate.weight.zero_()
            model.exit_gate.bias.fill_(10.0)

        ids = torch.randint(0, 32, (2, 8))
        first_layer = model.layers[0]
        original_forward = first_layer.forward

        def count_steps() -> int:
            calls = {"count": 0}

            def wrapped_forward(*args, **kwargs):
                calls["count"] += 1
                return original_forward(*args, **kwargs)

            first_layer.forward = wrapped_forward
            try:
                _ = model(ids)
            finally:
                first_layer.forward = original_forward
            return calls["count"]

        early_steps = count_steps()
        model.config.looplm_exit_threshold = 1.0
        full_steps = count_steps()

        assert early_steps < full_steps
        assert full_steps == config.num_recurrent_steps

    def test_sandwich_norm_creates_extra_norms(self):
        """Sandwich norm adds ln1_post and ln2_post to each layer."""
        config = _tiny_config(
            num_recurrent_steps=2,
            looplm_sandwich_norm=True,
        )
        model = BachTransformer(config)
        for layer in model.layers:
            assert hasattr(layer, "ln1_post")
            assert hasattr(layer, "ln2_post")

    def test_no_sandwich_norm_by_default(self):
        """Without sandwich norm flag, no post-norms exist."""
        config = _tiny_config(num_recurrent_steps=2)
        model = BachTransformer(config)
        for layer in model.layers:
            assert not layer.sandwich_norm

    def test_looped_kv_cache_incremental(self):
        """LoopLM with KV cache should work for incremental decoding."""
        config = _tiny_config(num_recurrent_steps=2)
        model = BachTransformer(config)
        model.eval()

        # Prefill
        ids = torch.randint(0, 32, (1, 8))
        out, caches = model(ids, use_cache=True)
        assert out.shape == (1, 8, 32)
        assert len(caches) == 2  # num_layers

        # Incremental step
        next_id = torch.randint(0, 32, (1, 1))
        out2, caches2 = model(next_id, use_cache=True, kv_cache=caches)
        assert out2.shape == (1, 1, 32)


class TestLoopLMTrainingLoss:
    """Test LoopLM loss computation in the trainer."""

    def test_looplm_loss_without_gate(self):
        """LoopLM loss without exit gate uses uniform weighting."""
        from bach_gen.model.trainer import Trainer

        config = _tiny_config(num_recurrent_steps=3)
        model = BachTransformer(config)

        # Minimal dataset
        from bach_gen.data.dataset import BachDataset
        seqs = [[1] + [10] * 30 + [2]] * 4  # BOS + tokens + EOS
        ds = BachDataset(seqs, seq_len=16)

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            val_dataset=ds,
            lr=1e-3,
            batch_size=2,
            device=torch.device("cpu"),
        )

        assert trainer._use_looplm_loss is True

        # Run one training step
        loader = trainer._make_train_loader()
        loss, _ = trainer._train_epoch(loader)
        assert loss > 0

    def test_looplm_loss_with_gate(self):
        """LoopLM loss with exit gate uses learned weighting + entropy reg."""
        from bach_gen.model.trainer import Trainer
        from bach_gen.data.dataset import BachDataset

        config = _tiny_config(
            num_recurrent_steps=3,
            looplm_exit_gate=True,
            looplm_kl_beta=0.1,
        )
        model = BachTransformer(config)

        seqs = [[1] + [10] * 30 + [2]] * 4
        ds = BachDataset(seqs, seq_len=16)

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            val_dataset=ds,
            lr=1e-3,
            batch_size=2,
            device=torch.device("cpu"),
        )

        loader = trainer._make_train_loader()
        loss, _ = trainer._train_epoch(loader)
        assert loss > 0

    def test_looplm_validation(self):
        """Validation also uses LoopLM loss when enabled."""
        from bach_gen.model.trainer import Trainer
        from bach_gen.data.dataset import BachDataset
        from torch.utils.data import DataLoader

        config = _tiny_config(num_recurrent_steps=2)
        model = BachTransformer(config)

        seqs = [[1] + [10] * 30 + [2]] * 4
        ds = BachDataset(seqs, seq_len=16)

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            val_dataset=ds,
            lr=1e-3,
            batch_size=2,
            device=torch.device("cpu"),
        )

        val_loader = DataLoader(ds, batch_size=2, shuffle=False)
        val_loss, _ = trainer._validate(val_loader)
        assert val_loss > 0

    def test_gate_stage_updates_only_exit_gate(self, tmp_path):
        """Stage II gate training should not modify frozen backbone weights."""
        from bach_gen.model.trainer import Trainer
        from bach_gen.data.dataset import BachDataset

        config = _tiny_config(
            num_recurrent_steps=3,
            looplm_exit_gate=True,
        )
        model = BachTransformer(config)

        seqs = [[1] + [10] * 30 + [2]] * 4
        ds = BachDataset(seqs, seq_len=16)

        trainer = Trainer(
            model=model,
            train_dataset=ds,
            val_dataset=ds,
            lr=1e-3,
            batch_size=2,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path,
        )

        before = {
            name: param.detach().clone()
            for name, param in trainer.model.named_parameters()
        }

        history = trainer.train_exit_gate(epochs=1, lr=1e-2)
        assert history["epochs_ran"] == 1

        after = {
            name: param.detach().clone()
            for name, param in trainer.model.named_parameters()
        }

        exit_gate_changed = False
        for name in before:
            if "exit_gate" in name:
                if not torch.equal(before[name], after[name]):
                    exit_gate_changed = True
            else:
                torch.testing.assert_close(before[name], after[name])

        assert exit_gate_changed is True


class TestLoopLMCheckpointCompat:
    """Test that LoopLM checkpoints reconcile correctly."""

    def test_load_standard_into_looplm(self, tmp_path):
        """Loading a standard checkpoint into a LoopLM model fills missing keys."""
        from bach_gen.model.trainer import Trainer

        # Save a standard model checkpoint
        standard_config = _tiny_config(num_recurrent_steps=1)
        standard_model = BachTransformer(standard_config)
        ckpt_path = tmp_path / "standard.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": standard_model.state_dict(),
            "optimizer_state_dict": {},
            "config": standard_config,
            "best_val_loss": 1.0,
        }, ckpt_path)

        # Load into a LoopLM model (with sandwich norm + exit gate)
        looplm_config = _tiny_config(
            num_recurrent_steps=3,
            looplm_sandwich_norm=True,
            looplm_exit_gate=True,
        )
        looplm_model = BachTransformer(looplm_config)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        state, reconciled = Trainer._reconcile_looplm_state_dict(
            state, looplm_model.state_dict(),
        )
        assert reconciled is True
        # Should be loadable now
        looplm_model.load_state_dict(state)

    def test_load_looplm_into_standard(self, tmp_path):
        """Loading a LoopLM checkpoint into a standard model drops extra keys."""
        from bach_gen.model.trainer import Trainer

        looplm_config = _tiny_config(
            num_recurrent_steps=3,
            looplm_sandwich_norm=True,
            looplm_exit_gate=True,
        )
        looplm_model = BachTransformer(looplm_config)
        ckpt_path = tmp_path / "looplm.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": looplm_model.state_dict(),
            "optimizer_state_dict": {},
            "config": looplm_config,
            "best_val_loss": 1.0,
        }, ckpt_path)

        standard_config = _tiny_config(num_recurrent_steps=1)
        standard_model = BachTransformer(standard_config)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        state, reconciled = Trainer._reconcile_looplm_state_dict(
            state, standard_model.state_dict(),
        )
        assert reconciled is True
        standard_model.load_state_dict(state)


class TestModelConfigLoopLM:
    """Test ModelConfig LoopLM-related properties."""

    def test_default_no_recurrence(self):
        config = _tiny_config()
        assert config.num_recurrent_steps == 1
        assert config.looplm_sandwich_norm is False
        assert config.looplm_exit_gate is False

    def test_num_params_accounts_for_sandwich_norm(self):
        base = _tiny_config(num_recurrent_steps=2)
        sandwich = _tiny_config(num_recurrent_steps=2, looplm_sandwich_norm=True)
        # Sandwich norm adds 2 extra RMSNorm per layer (each = embed_dim params)
        diff = sandwich.num_params - base.num_params
        expected = 2 * base.num_layers * base.embed_dim  # 2 norms * N layers * D
        assert diff == expected

    def test_num_params_accounts_for_exit_gate(self):
        base = _tiny_config(num_recurrent_steps=2)
        gated = _tiny_config(num_recurrent_steps=2, looplm_exit_gate=True)
        diff = gated.num_params - base.num_params
        expected = base.embed_dim + 1  # Linear(embed_dim, 1) = weight + bias
        assert diff == expected
