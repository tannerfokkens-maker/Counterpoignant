"""Checkpoint resume tests for vocabulary-size migrations."""

from __future__ import annotations

import torch

from bach_gen.data.dataset import BachDataset
from bach_gen.data.tokenizer import BachTokenizer
from bach_gen.model.architecture import BachTransformer
from bach_gen.model.config import ModelConfig
from bach_gen.model.trainer import Trainer


def _tiny_dataset() -> BachDataset:
    seq = [1, 3, 4, 5, 6, 7, 8, 9, 2] + [0] * 24
    return BachDataset([seq], seq_len=16)


def _tiny_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=16,
        dropout=0.0,
    )


def _tiny_config_with_rel_bias(vocab_size: int, enabled: bool) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=16,
        dropout=0.0,
        rel_attn_bias=enabled,
        rel_attn_max_distance=16,
    )


def _tiny_looplm_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        ffn_dim=64,
        max_seq_len=16,
        dropout=0.0,
        pos_encoding="pope",
        num_recurrent_steps=3,
        looplm_exit_gate=True,
    )


def test_tokenizer_uses_subject_boundary_slots_without_subj_aliases():
    tokenizer = BachTokenizer()

    assert tokenizer.name_to_token["SUBJECT_START"] == 7
    assert tokenizer.name_to_token["SUBJECT_END"] == 8


def test_resume_from_checkpoint_resizes_token_embedding_rows(tmp_path):
    device = torch.device("cpu")
    train_ds = _tiny_dataset()
    current_vocab_size = BachTokenizer().vocab_size
    legacy_vocab_size = current_vocab_size - 4

    small_model = BachTransformer(_tiny_config(legacy_vocab_size))
    trainer_small = Trainer(
        model=small_model,
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=1,
        checkpoint_dir=tmp_path,
        device=device,
    )
    trainer_small.save_checkpoint("small.pt")

    big_model = BachTransformer(_tiny_config(current_vocab_size))
    trainer_big = Trainer(
        model=big_model,
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=1,
        checkpoint_dir=tmp_path,
        device=device,
    )

    next_epoch = trainer_big.resume_from_checkpoint(tmp_path / "small.pt")

    assert next_epoch == 1
    assert trainer_big.model.token_embed.weight.shape[0] == current_vocab_size
    assert trainer_big.model.head.weight.shape[0] == current_vocab_size


def test_resume_from_checkpoint_backfills_missing_relative_bias(tmp_path):
    device = torch.device("cpu")
    train_ds = _tiny_dataset()
    vocab_size = BachTokenizer().vocab_size

    trainer_small = Trainer(
        model=BachTransformer(_tiny_config_with_rel_bias(vocab_size, enabled=False)),
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=1,
        checkpoint_dir=tmp_path,
        device=device,
    )
    trainer_small.save_checkpoint("no_bias.pt")

    model_with_bias = BachTransformer(_tiny_config_with_rel_bias(vocab_size, enabled=True))
    initial_bias = model_with_bias.layers[0].attn.rel_attn_bias.detach().clone()
    trainer_big = Trainer(
        model=model_with_bias,
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=1,
        checkpoint_dir=tmp_path,
        device=device,
    )

    next_epoch = trainer_big.resume_from_checkpoint(tmp_path / "no_bias.pt")

    assert next_epoch == 1
    torch.testing.assert_close(
        trainer_big.model.layers[0].attn.rel_attn_bias,
        initial_bias,
    )


def test_resume_from_checkpoint_drops_unexpected_relative_bias(tmp_path):
    device = torch.device("cpu")
    train_ds = _tiny_dataset()
    vocab_size = BachTokenizer().vocab_size

    trainer_small = Trainer(
        model=BachTransformer(_tiny_config_with_rel_bias(vocab_size, enabled=True)),
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=1,
        checkpoint_dir=tmp_path,
        device=device,
    )
    trainer_small.save_checkpoint("with_bias.pt")

    trainer_big = Trainer(
        model=BachTransformer(_tiny_config_with_rel_bias(vocab_size, enabled=False)),
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=1,
        checkpoint_dir=tmp_path,
        device=device,
    )

    next_epoch = trainer_big.resume_from_checkpoint(tmp_path / "with_bias.pt")

    assert next_epoch == 1


def test_load_checkpoint_normalizes_legacy_flat_layer_stack(tmp_path):
    vocab_size = BachTokenizer().vocab_size
    model = BachTransformer(_tiny_looplm_config(vocab_size))
    state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith("loop_layers."):
            flat_key = key.replace("loop_layers.", "layers.", 1)
            state_dict[flat_key] = value.detach().clone()
        else:
            state_dict[key] = value.detach().clone()

    legacy_config = _tiny_looplm_config(vocab_size)
    legacy_config.num_front_layers = 0
    legacy_config.num_loop_layers = 0
    legacy_config.num_back_layers = 0

    checkpoint_path = tmp_path / "legacy_flat_layers.pt"
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": legacy_config,
            "epoch": 0,
        },
        checkpoint_path,
    )

    loaded_model, loaded_config = Trainer.load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert loaded_config.num_front_layers == 0
    assert loaded_config.num_loop_layers == loaded_config.num_layers == 2
    assert loaded_config.num_back_layers == 0
    assert len(loaded_model.front_layers) == 0
    assert len(loaded_model.loop_layers) == 2
    assert len(loaded_model.back_layers) == 0
    torch.testing.assert_close(
        loaded_model.state_dict()["loop_layers.0.ln1.weight"],
        model.state_dict()["loop_layers.0.ln1.weight"],
    )
