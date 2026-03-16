from __future__ import annotations

import torch
import torch.nn as nn

from bach_gen.cli import _maybe_cast_generation_model_for_mps


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()


def test_mps_bf16_cast_enabled_on_mps():
    model = _DummyModel()

    cast_model, state = _maybe_cast_generation_model_for_mps(
        model,
        mps_bf16=True,
        device_type_override="mps",
    )

    assert cast_model.linear.weight.dtype == torch.bfloat16
    assert cast_model.eval_called is True
    assert state == "enabled"


def test_mps_bf16_cast_ignored_on_non_mps():
    model = _DummyModel()

    cast_model, state = _maybe_cast_generation_model_for_mps(
        model,
        mps_bf16=True,
        device_type_override="cpu",
    )

    assert cast_model.linear.weight.dtype == torch.float32
    assert state == "ignored_non_mps"


def test_mps_bf16_cast_noop_when_disabled():
    model = _DummyModel()

    cast_model, state = _maybe_cast_generation_model_for_mps(
        model,
        mps_bf16=False,
        device_type_override="mps",
    )

    assert cast_model.linear.weight.dtype == torch.float32
    assert state is None
