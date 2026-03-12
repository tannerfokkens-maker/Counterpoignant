"""Single-file autoresearch experiment harness for the bach-gen pipeline."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from prepare import (
    EVAL_SEED,
    TIME_BUDGET_SECONDS,
    VAL_MAX_BATCHES,
    build_trainer,
    clear_device_cache,
    current_memory_mb,
    is_oom_error,
    load_data_bundle,
    load_model_weights_only,
    make_val_loader,
    set_seed,
    synchronize_device,
)

from bach_gen.model.config import ModelConfig


DEFAULT_MPS_HIGH_WATERMARK_RATIO = 0.9
DEFAULT_MPS_LOW_WATERMARK_RATIO = 0.5
_MPS_ENV_CONFIGURED = False


@dataclass
class ExperimentConfig:
    """Editable experiment configuration.

    Treat this file as the single search surface, mirroring upstream
    `autoresearch/train.py`.
    """

    # Current default is the best-feasible stage-1 long-context run on this machine.
    description: str = "stage4096_lr4e4_eb16_mb1x16_looplm_t2_l4_gate_b005"
    seq_len: int = 4096
    init_checkpoint: str | None = None
    batch_size: int = 1
    accumulation_steps: int = 16
    lr: float = 4e-4
    embed_dim: int = 384
    num_heads: int = 8
    num_layers: int = 4
    num_front_layers: int = 0
    num_loop_layers: int = 0
    num_back_layers: int = 0
    pos_encoding: str = "pope"
    piece_balance: str = "sqrt"
    rel_attn_bias: bool = False
    rel_attn_max_distance: int = 2048
    num_recurrent_steps: int = 2
    loop_step_embedding: bool = True
    looplm_sandwich_norm: bool = False
    looplm_exit_gate: bool = True
    looplm_kl_beta: float = 0.05
    # Try 0.05 later if the broader LoopLM search plateaus near the current gate range.
    looplm_exit_threshold: float = 1.0
    fp16: bool = False
    seed: int = 1337
    time_budget_seconds: float = TIME_BUDGET_SECONDS
    val_max_batches: int = VAL_MAX_BATCHES
    use_rope: bool = True

    # Adaptive runtime controls for local experimentation on the Mac Studio.
    auto_adjust: bool = True
    min_batch_size: int = 1
    preflight_warmup_steps: int = 0
    preflight_measure_steps: int = 1
    min_projected_steps: int = 1
    memory_soft_limit_gb: float = 32.0
    retry_on_runtime_oom: bool = True
    stability_probe_interval_seconds: float = 60.0
    stability_probe_batches: int = 1

    # Optional staged curriculum inside the fixed autoresearch budget.
    # Format per stage: "seq_len@time_weight@recurrent_steps@lr_scale@batch_scale@kl_beta"
    use_stage_curriculum: bool = False
    curriculum_stages: str = (
        "1024@1@3@1.0@1.0@0.05,"
        "2048@1@3@0.5@0.5@0.05,"
        "4096@1@3@0.25@0.25@0.05"
    )

    # Optional recurrent-specific optimizer settings.
    optimizer_weight_decay: float = 0.01
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.98
    optimizer_eps: float = 1e-9
    use_recurrent_optimizer_overrides: bool = False
    recurrent_optimizer_weight_decay: float = 0.1
    recurrent_optimizer_beta1: float = 0.9
    recurrent_optimizer_beta2: float = 0.95
    recurrent_optimizer_eps: float = 1e-9

    # Optional Stage II gate-only pass after the final curriculum stage.
    gate_stage_fraction: float = 0.0
    gate_stage_lr: float = 1e-3
    gate_target_margin: float = 0.005
    gate_target_sharpness: float = 50.0


@dataclass(frozen=True)
class CurriculumStage:
    seq_len: int
    time_weight: float
    recurrent_steps: int
    lr_scale: float
    batch_scale: float
    kl_beta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke test setup with a tiny budget")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Override an ExperimentConfig field for this run.",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="Override training time budget for this invocation",
    )
    parser.add_argument(
        "--val-max-batches",
        type=int,
        default=None,
        help="Override number of validation batches",
    )
    return parser.parse_args()


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    updates: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected FIELD=VALUE")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field {key!r}")

        current = getattr(cfg, key)
        if isinstance(current, bool):
            lowered = raw_value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                parsed: Any = True
            elif lowered in {"0", "false", "no", "off"}:
                parsed = False
            else:
                raise ValueError(f"Invalid boolean override for {key!r}: {raw_value!r}")
        elif isinstance(current, int) and not isinstance(current, bool):
            parsed = int(raw_value)
        elif isinstance(current, float):
            parsed = float(raw_value)
        elif current is None:
            parsed = None if raw_value.lower() == "none" else raw_value
        else:
            parsed = raw_value

        updates[key] = parsed

    return replace(cfg, **updates) if updates else cfg


def parse_curriculum_stages(spec: str) -> list[CurriculumStage]:
    stages: list[CurriculumStage] = []
    for raw_stage in spec.split(","):
        stage_text = raw_stage.strip()
        if not stage_text:
            continue

        parts = [part.strip() for part in stage_text.split("@")]
        if len(parts) != 6:
            raise ValueError(
                f"Invalid curriculum stage {stage_text!r}; expected "
                "'seq_len@time_weight@recurrent_steps@lr_scale@batch_scale@kl_beta'"
            )

        seq_len = int(parts[0])
        time_weight = float(parts[1])
        recurrent_steps = int(parts[2])
        lr_scale = float(parts[3])
        batch_scale = float(parts[4])
        kl_beta = float(parts[5])

        if seq_len < 1:
            raise ValueError(f"Invalid seq_len in stage {stage_text!r}")
        if time_weight <= 0.0:
            raise ValueError(f"Invalid time_weight in stage {stage_text!r}")
        if recurrent_steps < 1:
            raise ValueError(f"Invalid recurrent_steps in stage {stage_text!r}")
        if lr_scale <= 0.0:
            raise ValueError(f"Invalid lr_scale in stage {stage_text!r}")
        if batch_scale <= 0.0:
            raise ValueError(f"Invalid batch_scale in stage {stage_text!r}")
        if kl_beta < 0.0:
            raise ValueError(f"Invalid kl_beta in stage {stage_text!r}")

        stages.append(
            CurriculumStage(
                seq_len=seq_len,
                time_weight=time_weight,
                recurrent_steps=recurrent_steps,
                lr_scale=lr_scale,
                batch_scale=batch_scale,
                kl_beta=kl_beta,
            )
        )

    if not stages:
        raise ValueError("Curriculum stage list is empty")
    return stages


def cycle_batches(loader):
    while True:
        for batch in loader:
            yield batch


def build_model_config(cfg: ExperimentConfig, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        num_front_layers=cfg.num_front_layers,
        num_loop_layers=cfg.num_loop_layers,
        num_back_layers=cfg.num_back_layers,
        max_seq_len=cfg.seq_len,
        pos_encoding=cfg.pos_encoding,
        rel_attn_bias=cfg.rel_attn_bias,
        rel_attn_max_distance=cfg.rel_attn_max_distance,
        num_recurrent_steps=cfg.num_recurrent_steps,
        loop_step_embedding=cfg.loop_step_embedding,
        looplm_sandwich_norm=cfg.looplm_sandwich_norm,
        looplm_exit_gate=cfg.looplm_exit_gate,
        looplm_kl_beta=cfg.looplm_kl_beta,
        looplm_exit_threshold=cfg.looplm_exit_threshold,
    )


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_model_state_into_trainer(
    trainer,
    state_dict: dict[str, torch.Tensor],
) -> None:
    state = dict(state_dict)
    state, _ = trainer._maybe_resize_vocab_state_dict(
        state,
        target_vocab_size=trainer.model.config.vocab_size,
    )
    state, _ = trainer._reconcile_optional_attention_state_dict(
        state,
        trainer.model.state_dict(),
    )
    state, _ = trainer._reconcile_looplm_state_dict(
        state,
        trainer.model.state_dict(),
    )
    trainer.model.load_state_dict(state, strict=True)
    trainer.best_val_loss = float("inf")
    trainer.epoch = 0


def optimizer_kwargs_for_cfg(cfg: ExperimentConfig) -> dict[str, Any]:
    if cfg.use_recurrent_optimizer_overrides and cfg.num_recurrent_steps > 1:
        return {
            "lr": cfg.lr,
            "weight_decay": cfg.recurrent_optimizer_weight_decay,
            "betas": (cfg.recurrent_optimizer_beta1, cfg.recurrent_optimizer_beta2),
            "eps": cfg.recurrent_optimizer_eps,
        }

    return {
        "lr": cfg.lr,
        "weight_decay": cfg.optimizer_weight_decay,
        "betas": (cfg.optimizer_beta1, cfg.optimizer_beta2),
        "eps": cfg.optimizer_eps,
    }


def configure_trainer_optimizer(trainer, cfg: ExperimentConfig) -> None:
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        **optimizer_kwargs_for_cfg(cfg),
    )


def build_pitch_token_ids(tokenizer) -> tuple[int, ...]:
    token_to_name = getattr(tokenizer, "token_to_name", {})
    if not isinstance(token_to_name, dict):
        return ()

    pitch_ids: list[int] = []
    for tok, name in token_to_name.items():
        try:
            tok_id = int(tok)
        except (TypeError, ValueError):
            continue
        if not isinstance(name, str):
            continue
        if name.startswith("DEG_") or name in ("SHARP", "FLAT") or name.startswith("Pitch_"):
            pitch_ids.append(tok_id)
    return tuple(sorted(set(pitch_ids)))


def pitch_token_mask(flat_labels: torch.Tensor, pitch_token_ids: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(flat_labels, dtype=torch.bool)
    for tok_id in pitch_token_ids:
        mask |= flat_labels == tok_id
    return mask


def summarize_series(prefix: str, values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_start": float("nan"),
            f"{prefix}_end": float("nan"),
            f"{prefix}_best": float("nan"),
            f"{prefix}_range": float("nan"),
            f"{prefix}_delta": float("nan"),
            f"{prefix}_reversals": 0,
        }

    reversals = 0
    prev_sign = 0
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        if abs(delta) <= 1e-6:
            continue
        sign = 1 if delta > 0.0 else -1
        if prev_sign and sign != prev_sign:
            reversals += 1
        prev_sign = sign

    return {
        f"{prefix}_count": len(values),
        f"{prefix}_start": values[0],
        f"{prefix}_end": values[-1],
        f"{prefix}_best": min(values),
        f"{prefix}_range": max(values) - min(values),
        f"{prefix}_delta": values[-1] - values[0],
        f"{prefix}_reversals": reversals,
    }


def configure_mps_environment() -> None:
    """Apply safe MPS watermark defaults and repair invalid env combinations."""
    global _MPS_ENV_CONFIGURED

    if _MPS_ENV_CONFIGURED or not torch.backends.mps.is_built():
        return

    def _read_ratio(name: str) -> float | None:
        value = os.environ.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    high = _read_ratio("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if high is None or high <= 0.0:
        high = DEFAULT_MPS_HIGH_WATERMARK_RATIO
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high)

    low = _read_ratio("PYTORCH_MPS_LOW_WATERMARK_RATIO")
    if low is None or low < 0.0 or low > high:
        low = min(DEFAULT_MPS_LOW_WATERMARK_RATIO, max(high - 0.1, 0.0))
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(low)

    _MPS_ENV_CONFIGURED = True


def default_device() -> torch.device:
    configure_mps_environment()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_validation_metrics(
    trainer,
    val_loader,
    *,
    use_rope: bool,
    pitch_token_ids: tuple[int, ...],
    max_batches: int = VAL_MAX_BATCHES,
) -> dict[str, float]:
    """Deterministic validation pass with comparable and objective losses."""
    model = trainer.model
    model.eval()
    total_loss = 0.0
    total_objective_loss = 0.0
    n_batches = 0
    pitch_loss_sum = 0.0
    n_pitch_tokens = 0
    loop_val_loss_sums: list[float] | None = None
    loop_pitch_loss_sums: list[float] | None = None
    use_looplm = trainer._use_looplm_loss

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            attention_mask = (input_ids != 0).long()

            with torch.amp.autocast(trainer.device.type, enabled=trainer.fp16):
                if use_looplm:
                    output = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_rope=use_rope,
                        return_all_steps=True,
                    )
                    objective_loss, final_logits = trainer._compute_looplm_loss(output, labels)
                    loss = trainer.criterion(
                        final_logits.reshape(-1, final_logits.size(-1)),
                        labels.reshape(-1),
                    )
                    if loop_val_loss_sums is None:
                        loop_val_loss_sums = [0.0 for _ in output.all_logits]
                        loop_pitch_loss_sums = [0.0 for _ in output.all_logits]
                else:
                    final_logits = model(input_ids, attention_mask=attention_mask, use_rope=use_rope)
                    loss = trainer.criterion(
                        final_logits.reshape(-1, final_logits.size(-1)),
                        labels.reshape(-1),
                    )
                    objective_loss = loss

            total_loss += float(loss.item())
            total_objective_loss += float(objective_loss.item())
            n_batches += 1

            flat_labels = labels.reshape(-1)
            valid_mask = flat_labels != trainer.criterion.ignore_index
            if pitch_token_ids:
                valid_pitch_mask = valid_mask & pitch_token_mask(flat_labels, pitch_token_ids)
            else:
                valid_pitch_mask = torch.zeros_like(flat_labels, dtype=torch.bool)

            if torch.any(valid_pitch_mask):
                per_token_losses = F.cross_entropy(
                    final_logits.reshape(-1, final_logits.size(-1)),
                    flat_labels,
                    ignore_index=trainer.criterion.ignore_index,
                    label_smoothing=trainer.criterion.label_smoothing,
                    reduction="none",
                )
                pitch_loss_sum += float(per_token_losses[valid_pitch_mask].sum().item())
                n_pitch_tokens += int(valid_pitch_mask.sum().item())

            if use_looplm and loop_val_loss_sums is not None and loop_pitch_loss_sums is not None:
                for step_idx, step_logits in enumerate(output.all_logits):
                    step_loss = trainer.criterion(
                        step_logits.reshape(-1, step_logits.size(-1)),
                        labels.reshape(-1),
                    )
                    loop_val_loss_sums[step_idx] += float(step_loss.item())

                    if torch.any(valid_pitch_mask):
                        step_per_token_losses = F.cross_entropy(
                            step_logits.reshape(-1, step_logits.size(-1)),
                            flat_labels,
                            ignore_index=trainer.criterion.ignore_index,
                            label_smoothing=trainer.criterion.label_smoothing,
                            reduction="none",
                        )
                        loop_pitch_loss_sums[step_idx] += float(
                            step_per_token_losses[valid_pitch_mask].sum().item()
                        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    if n_batches == 0:
        raise RuntimeError("Validation loader produced zero batches")

    loop_val_losses = (
        [loss_sum / n_batches for loss_sum in loop_val_loss_sums]
        if loop_val_loss_sums is not None
        else []
    )
    loop_pitch_losses = (
        [loss_sum / n_pitch_tokens for loss_sum in loop_pitch_loss_sums]
        if loop_pitch_loss_sums is not None and n_pitch_tokens > 0
        else []
    )

    return {
        "val_loss": total_loss / n_batches,
        "objective_val_loss": total_objective_loss / n_batches,
        "pitch_val_loss": (pitch_loss_sum / n_pitch_tokens) if n_pitch_tokens > 0 else float("nan"),
        "loop_val_losses": loop_val_losses,
        "loop_pitch_losses": loop_pitch_losses,
    }


def run_optimizer_step(
    trainer,
    cfg: ExperimentConfig,
    train_iter,
) -> tuple[float, int, float]:
    """Run one optimizer step, including accumulation, and return metrics."""
    model = trainer.model
    optimizer = trainer.optimizer
    use_looplm = trainer._use_looplm_loss

    optimizer.zero_grad(set_to_none=True)
    total_tokens = 0
    loss_sum = 0.0
    peak_memory_mb = current_memory_mb(trainer.device)

    synchronize_device(trainer.device)
    step_start = time.perf_counter()

    for _ in range(cfg.accumulation_steps):
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        attention_mask = (input_ids != 0).long()

        with torch.amp.autocast(trainer.device.type, enabled=trainer.fp16):
            if use_looplm:
                output = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_rope=cfg.use_rope,
                    return_all_steps=True,
                )
                loss, _ = trainer._compute_looplm_loss(output, labels)
            else:
                logits = model(input_ids, attention_mask=attention_mask, use_rope=cfg.use_rope)
                loss = trainer.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

        trainer.scaler.scale(loss / cfg.accumulation_steps).backward()
        loss_sum += float(loss.item())
        total_tokens += int(input_ids.numel())
        peak_memory_mb = max(peak_memory_mb, current_memory_mb(trainer.device))

    trainer.scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    trainer.scaler.step(optimizer)
    trainer.scaler.update()
    optimizer.zero_grad(set_to_none=True)

    synchronize_device(trainer.device)
    step_seconds = time.perf_counter() - step_start
    peak_memory_mb = max(peak_memory_mb, current_memory_mb(trainer.device))
    return loss_sum / max(cfg.accumulation_steps, 1), total_tokens, step_seconds, peak_memory_mb


def build_trainer_for_cfg(
    cfg: ExperimentConfig,
    data,
    *,
    initial_model_state: dict[str, torch.Tensor] | None = None,
) -> tuple[Any, int]:
    set_seed(cfg.seed)
    config = build_model_config(cfg, data.tokenizer.vocab_size)
    trainer = build_trainer(
        config,
        data,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        accumulation_steps=cfg.accumulation_steps,
        fp16=cfg.fp16,
        piece_balance=cfg.piece_balance,
    )
    configure_trainer_optimizer(trainer, cfg)
    checkpoint_loaded_epoch = 0
    if initial_model_state is not None:
        load_model_state_into_trainer(trainer, initial_model_state)
    elif cfg.init_checkpoint:
        checkpoint_loaded_epoch = load_model_weights_only(trainer, cfg.init_checkpoint)
    return trainer, checkpoint_loaded_epoch


def probe_cfg(
    cfg: ExperimentConfig,
    data,
    *,
    initial_model_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run a short preflight to estimate memory and step time."""
    trainer = None
    train_iter = None
    device = default_device()

    try:
        trainer, checkpoint_loaded_epoch = build_trainer_for_cfg(
            cfg,
            data,
            initial_model_state=initial_model_state,
        )
        device = trainer.device
        train_iter = cycle_batches(trainer._make_train_loader())

        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

        trainer.model.train()
        clear_device_cache(trainer.device)

        warmup_steps = max(0, cfg.preflight_warmup_steps)
        measure_steps = max(1, cfg.preflight_measure_steps)
        durations: list[float] = []
        peak_memory_mb = current_memory_mb(trainer.device)
        total_tokens = 0

        for _ in range(warmup_steps):
            _, tokens, _, step_peak_mb = run_optimizer_step(trainer, cfg, train_iter)
            total_tokens += tokens
            peak_memory_mb = max(peak_memory_mb, step_peak_mb)

        for _ in range(measure_steps):
            _, tokens, step_seconds, step_peak_mb = run_optimizer_step(trainer, cfg, train_iter)
            total_tokens += tokens
            durations.append(step_seconds)
            peak_memory_mb = max(peak_memory_mb, step_peak_mb)

        avg_step_seconds = sum(durations) / max(len(durations), 1)
        projected_steps = int(cfg.time_budget_seconds / max(avg_step_seconds, 1e-6))
        projected_tokens_m = (projected_steps * (total_tokens / max(warmup_steps + measure_steps, 1))) / 1e6

        return {
            "checkpoint_loaded_epoch": checkpoint_loaded_epoch,
            "avg_step_seconds": avg_step_seconds,
            "peak_memory_mb": peak_memory_mb,
            "projected_steps": projected_steps,
            "projected_tokens_m": projected_tokens_m,
        }
    finally:
        train_iter = None
        trainer = None
        clear_device_cache(device)


def choose_runtime_cfg(
    cfg: ExperimentConfig,
    data,
    *,
    initial_model_state: dict[str, torch.Tensor] | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Adaptively reduce batch size until the run looks safe on this machine."""
    candidate = cfg
    attempts: list[dict[str, Any]] = []
    memory_soft_limit_mb = cfg.memory_soft_limit_gb * 1024.0

    while True:
        attempt_info: dict[str, Any] = {"batch_size": candidate.batch_size}
        try:
            probe = probe_cfg(candidate, data, initial_model_state=initial_model_state)
            attempt_info.update(probe)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            attempt_info.update({"status": "oom", "reason": str(exc)})
            attempts.append(attempt_info)
            if not cfg.auto_adjust or candidate.batch_size <= cfg.min_batch_size:
                raise
            next_batch = max(cfg.min_batch_size, candidate.batch_size // 2)
            if next_batch == candidate.batch_size:
                raise
            candidate = replace(candidate, batch_size=next_batch)
            continue

        over_memory = probe["peak_memory_mb"] > memory_soft_limit_mb
        too_slow = probe["projected_steps"] < candidate.min_projected_steps
        if not cfg.auto_adjust or candidate.batch_size <= cfg.min_batch_size or (not over_memory and not too_slow):
            attempt_info["status"] = "selected"
            attempt_info["adjustment_reason"] = (
                "memory"
                if over_memory
                else "speed"
                if too_slow
                else "ok"
            )
            attempts.append(attempt_info)
            return candidate, {"attempts": attempts, "selected": attempt_info}

        attempt_info["status"] = "backoff"
        attempt_info["adjustment_reason"] = "memory" if over_memory else "speed"
        attempts.append(attempt_info)
        next_batch = max(cfg.min_batch_size, candidate.batch_size // 2)
        if next_batch == candidate.batch_size:
            return candidate, {"attempts": attempts, "selected": attempt_info}
        candidate = replace(candidate, batch_size=next_batch)


def run_timed_training(
    cfg: ExperimentConfig,
    data,
    *,
    initial_model_state: dict[str, torch.Tensor] | None = None,
    return_model_state: bool = False,
) -> dict[str, Any]:
    trainer = None
    train_iter = None
    val_loader = None
    device = default_device()
    summary: dict[str, Any] | None = None

    try:
        trainer, checkpoint_loaded_epoch = build_trainer_for_cfg(
            cfg,
            data,
            initial_model_state=initial_model_state,
        )
        device = trainer.device
        train_iter = cycle_batches(trainer._make_train_loader())
        val_loader = make_val_loader(data.val_dataset, cfg.batch_size)
        pitch_token_ids = build_pitch_token_ids(data.tokenizer)

        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

        model = trainer.model
        model.train()
        clear_device_cache(trainer.device)

        running_loss = 0.0
        step_count = 0
        total_tokens = 0
        peak_memory_mb = current_memory_mb(trainer.device)
        next_stability_probe_seconds = cfg.stability_probe_interval_seconds
        stability_pitch_history: list[float] = []
        stability_val_history: list[float] = []

        synchronize_device(trainer.device)
        training_start = time.perf_counter()

        while True:
            if step_count > 0 and (time.perf_counter() - training_start) >= cfg.time_budget_seconds:
                break

            loss, step_tokens, _, step_peak_mb = run_optimizer_step(trainer, cfg, train_iter)
            running_loss += loss
            step_count += 1
            total_tokens += step_tokens
            peak_memory_mb = max(peak_memory_mb, step_peak_mb)

            elapsed_seconds = time.perf_counter() - training_start
            if (
                cfg.stability_probe_batches > 0
                and cfg.stability_probe_interval_seconds > 0.0
                and elapsed_seconds >= next_stability_probe_seconds
            ):
                probe_metrics = evaluate_validation_metrics(
                    trainer,
                    val_loader,
                    use_rope=cfg.use_rope,
                    pitch_token_ids=pitch_token_ids,
                    max_batches=cfg.stability_probe_batches,
                )
                stability_pitch_history.append(probe_metrics["pitch_val_loss"])
                stability_val_history.append(probe_metrics["val_loss"])
                model.train()
                next_stability_probe_seconds += cfg.stability_probe_interval_seconds

        synchronize_device(trainer.device)
        training_seconds = time.perf_counter() - training_start

        metrics = evaluate_validation_metrics(
            trainer,
            val_loader,
            use_rope=cfg.use_rope,
            pitch_token_ids=pitch_token_ids,
            max_batches=cfg.val_max_batches,
        )

        summary = {
            "checkpoint_loaded_epoch": checkpoint_loaded_epoch,
            "val_loss": metrics["val_loss"],
            "objective_val_loss": metrics["objective_val_loss"],
            "pitch_val_loss": metrics["pitch_val_loss"],
            "loop_val_losses": metrics["loop_val_losses"],
            "loop_pitch_losses": metrics["loop_pitch_losses"],
            "train_loss": running_loss / max(step_count, 1),
            "training_seconds": training_seconds,
            "peak_memory_mb": peak_memory_mb,
            "total_tokens_M": total_tokens / 1e6,
            "num_steps": step_count,
            "num_params_M": model.count_parameters() / 1e6,
            "device": trainer.device.type,
            "seq_len": cfg.seq_len,
            "selected_batch_size": cfg.batch_size,
            "effective_batch_size": cfg.batch_size * cfg.accumulation_steps,
        }
        summary.update(summarize_series("stability_pitch", stability_pitch_history))
        summary.update(summarize_series("stability_val", stability_val_history))
        summary.update(summarize_series("loop_val", metrics["loop_val_losses"]))
        summary.update(summarize_series("loop_pitch", metrics["loop_pitch_losses"]))
        if return_model_state:
            summary["final_model_state"] = clone_model_state(model)
    finally:
        val_loader = None
        train_iter = None
        trainer = None
        clear_device_cache(device)

    if summary is None:
        raise RuntimeError("Training did not produce a summary")
    return summary


def run_timed_gate_training(
    cfg: ExperimentConfig,
    data,
    *,
    initial_model_state: dict[str, torch.Tensor],
    time_budget_seconds: float,
) -> dict[str, Any]:
    trainer = None
    train_iter = None
    val_loader = None
    device = default_device()
    summary: dict[str, Any] | None = None

    try:
        trainer, checkpoint_loaded_epoch = build_trainer_for_cfg(
            cfg,
            data,
            initial_model_state=initial_model_state,
        )
        device = trainer.device
        if trainer.model.exit_gate is None:
            raise RuntimeError("Gate-only stage requested but the model has no exit gate")

        train_iter = cycle_batches(trainer._make_train_loader())
        val_loader = make_val_loader(data.val_dataset, cfg.batch_size)
        pitch_token_ids = build_pitch_token_ids(data.tokenizer)

        requires_grad_state = {
            name: param.requires_grad for name, param in trainer.model.named_parameters()
        }
        for name, param in trainer.model.named_parameters():
            param.requires_grad = "exit_gate" in name

        gate_optimizer = torch.optim.AdamW(
            trainer.model.exit_gate.parameters(),
            lr=cfg.gate_stage_lr,
            weight_decay=0.0,
        )
        trainer.optimizer = gate_optimizer
        trainer.model.eval()
        clear_device_cache(trainer.device)

        peak_memory_mb = current_memory_mb(trainer.device)
        running_loss = 0.0
        step_count = 0

        synchronize_device(trainer.device)
        training_start = time.perf_counter()

        while True:
            if step_count > 0 and (time.perf_counter() - training_start) >= time_budget_seconds:
                break

            gate_optimizer.zero_grad(set_to_none=True)
            step_peak_mb = current_memory_mb(trainer.device)
            loss_sum = 0.0

            for _ in range(cfg.accumulation_steps):
                batch = next(train_iter)
                input_ids = batch["input_ids"].to(trainer.device)
                labels = batch["labels"].to(trainer.device)
                attention_mask = (input_ids != 0).long()

                with torch.amp.autocast(trainer.device.type, enabled=trainer.fp16):
                    output = trainer.model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_rope=cfg.use_rope,
                        return_all_steps=True,
                    )
                    losses_stack, valid, n_valid = trainer._compute_looplm_per_step_losses(output, labels)

                    if not output.exit_lambdas:
                        loss = losses_stack.new_zeros(())
                    else:
                        gate_losses = []
                        for step_idx, lam in enumerate(output.exit_lambdas):
                            curr_loss = losses_stack[step_idx]
                            next_loss = losses_stack[step_idx + 1]
                            improvement = torch.clamp(curr_loss - next_loss, min=0.0)
                            continue_target = torch.sigmoid(
                                cfg.gate_target_sharpness * (improvement - cfg.gate_target_margin)
                            )
                            exit_target = 1.0 - continue_target
                            bce = F.binary_cross_entropy(
                                lam.reshape(-1),
                                exit_target,
                                reduction="none",
                            )
                            gate_losses.append(bce)

                        gate_stack = torch.stack(gate_losses, dim=0)
                        loss = (gate_stack * valid.unsqueeze(0)).sum() / (
                            n_valid * gate_stack.size(0)
                        )

                trainer.scaler.scale(loss / cfg.accumulation_steps).backward()
                loss_sum += float(loss.item())
                step_peak_mb = max(step_peak_mb, current_memory_mb(trainer.device))

            trainer.scaler.unscale_(gate_optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.exit_gate.parameters(), 1.0)
            trainer.scaler.step(gate_optimizer)
            trainer.scaler.update()
            gate_optimizer.zero_grad(set_to_none=True)

            running_loss += loss_sum / max(cfg.accumulation_steps, 1)
            step_count += 1
            peak_memory_mb = max(peak_memory_mb, step_peak_mb)

        synchronize_device(trainer.device)
        training_seconds = time.perf_counter() - training_start
        metrics = evaluate_validation_metrics(
            trainer,
            val_loader,
            use_rope=cfg.use_rope,
            pitch_token_ids=pitch_token_ids,
            max_batches=cfg.val_max_batches,
        )

        summary = {
            "checkpoint_loaded_epoch": checkpoint_loaded_epoch,
            "val_loss": metrics["val_loss"],
            "objective_val_loss": metrics["objective_val_loss"],
            "pitch_val_loss": metrics["pitch_val_loss"],
            "loop_val_losses": metrics["loop_val_losses"],
            "loop_pitch_losses": metrics["loop_pitch_losses"],
            "gate_train_loss": running_loss / max(step_count, 1),
            "gate_training_seconds": training_seconds,
            "peak_memory_mb": peak_memory_mb,
            "num_steps": step_count,
            "num_params_M": trainer.model.count_parameters() / 1e6,
            "device": trainer.device.type,
            "seq_len": cfg.seq_len,
            "selected_batch_size": cfg.batch_size,
            "effective_batch_size": cfg.batch_size * cfg.accumulation_steps,
            "final_model_state": clone_model_state(trainer.model),
        }
        summary.update(summarize_series("loop_val", metrics["loop_val_losses"]))
        summary.update(summarize_series("loop_pitch", metrics["loop_pitch_losses"]))
    finally:
        if trainer is not None:
            for name, param in trainer.model.named_parameters():
                if "requires_grad_state" in locals():
                    param.requires_grad = requires_grad_state[name]
        val_loader = None
        train_iter = None
        trainer = None
        clear_device_cache(device)

    if summary is None:
        raise RuntimeError("Gate-only training did not produce a summary")
    return summary


def run_stage_curriculum(cfg: ExperimentConfig) -> dict[str, Any]:
    stages = parse_curriculum_stages(cfg.curriculum_stages)
    stage_weight_total = sum(stage.time_weight for stage in stages)
    gate_weight = max(cfg.gate_stage_fraction, 0.0)
    total_weight = stage_weight_total + gate_weight
    if total_weight <= 0.0:
        raise ValueError("Curriculum time weights must sum to > 0")

    initial_model_state: dict[str, torch.Tensor] | None = None
    total_start = time.perf_counter()
    final_summary: dict[str, Any] | None = None
    combined_summary: dict[str, Any] = {
        "curriculum_stage_count": len(stages),
        "curriculum_schedule": cfg.curriculum_stages,
    }

    for stage_idx, stage in enumerate(stages, start=1):
        stage_budget = cfg.time_budget_seconds * (stage.time_weight / total_weight)
        stage_cfg = replace(
            cfg,
            description=f"{cfg.description}_stage{stage_idx}",
            seq_len=stage.seq_len,
            num_recurrent_steps=stage.recurrent_steps,
            looplm_kl_beta=stage.kl_beta,
            lr=cfg.lr * stage.lr_scale,
            batch_size=max(1, int(math.floor(cfg.batch_size * stage.batch_scale))),
            time_budget_seconds=stage_budget,
            use_stage_curriculum=False,
        )

        data = load_data_bundle(stage_cfg.seq_len)
        runtime_cfg, adaptive_info = choose_runtime_cfg(
            stage_cfg,
            data,
            initial_model_state=initial_model_state,
        )
        if runtime_cfg.batch_size != stage_cfg.batch_size:
            print(
                f"curriculum_stage{stage_idx}_auto_adjust: "
                f"batch_size {stage_cfg.batch_size} -> {runtime_cfg.batch_size} "
                f"({adaptive_info['selected']['adjustment_reason']})"
            )

        stage_summary = run_timed_training(
            runtime_cfg,
            data,
            initial_model_state=initial_model_state,
            return_model_state=True,
        )
        initial_model_state = stage_summary.pop("final_model_state")
        final_summary = stage_summary

        prefix = f"stage{stage_idx}"
        combined_summary[f"{prefix}_seq_len"] = runtime_cfg.seq_len
        combined_summary[f"{prefix}_recurrent_steps"] = runtime_cfg.num_recurrent_steps
        combined_summary[f"{prefix}_lr"] = runtime_cfg.lr
        combined_summary[f"{prefix}_requested_batch_size"] = stage_cfg.batch_size
        combined_summary[f"{prefix}_selected_batch_size"] = runtime_cfg.batch_size
        combined_summary[f"{prefix}_effective_batch_size"] = runtime_cfg.batch_size * runtime_cfg.accumulation_steps
        combined_summary[f"{prefix}_time_budget_seconds"] = stage_budget
        combined_summary[f"{prefix}_preflight_attempts"] = len(adaptive_info["attempts"])
        combined_summary[f"{prefix}_preflight_peak_memory_mb"] = adaptive_info["selected"].get("peak_memory_mb", 0.0)
        combined_summary[f"{prefix}_val_loss"] = stage_summary["val_loss"]
        combined_summary[f"{prefix}_objective_val_loss"] = stage_summary["objective_val_loss"]
        combined_summary[f"{prefix}_pitch_val_loss"] = stage_summary["pitch_val_loss"]

    if final_summary is None:
        raise RuntimeError("Curriculum run produced no stage summaries")

    if gate_weight > 0.0:
        if initial_model_state is None:
            raise RuntimeError("Gate-only stage requires a preceding model state")
        gate_budget = cfg.time_budget_seconds * (gate_weight / total_weight)
        gate_stage = stages[-1]
        gate_cfg = replace(
            cfg,
            description=f"{cfg.description}_gate",
            seq_len=gate_stage.seq_len,
            num_recurrent_steps=gate_stage.recurrent_steps,
            looplm_kl_beta=gate_stage.kl_beta,
            lr=cfg.lr * gate_stage.lr_scale,
            batch_size=max(1, int(math.floor(cfg.batch_size * gate_stage.batch_scale))),
            time_budget_seconds=gate_budget,
            use_stage_curriculum=False,
        )
        gate_data = load_data_bundle(gate_cfg.seq_len)
        gate_runtime_cfg, gate_adaptive = choose_runtime_cfg(
            gate_cfg,
            gate_data,
            initial_model_state=initial_model_state,
        )
        gate_summary = run_timed_gate_training(
            gate_runtime_cfg,
            gate_data,
            initial_model_state=initial_model_state,
            time_budget_seconds=gate_budget,
        )
        initial_model_state = gate_summary.pop("final_model_state")
        final_summary = gate_summary

        combined_summary["gate_stage_time_budget_seconds"] = gate_budget
        combined_summary["gate_stage_selected_batch_size"] = gate_runtime_cfg.batch_size
        combined_summary["gate_stage_effective_batch_size"] = gate_runtime_cfg.batch_size * gate_runtime_cfg.accumulation_steps
        combined_summary["gate_stage_preflight_attempts"] = len(gate_adaptive["attempts"])
        combined_summary["gate_stage_val_loss"] = gate_summary["val_loss"]
        combined_summary["gate_stage_objective_val_loss"] = gate_summary["objective_val_loss"]
        combined_summary["gate_stage_pitch_val_loss"] = gate_summary["pitch_val_loss"]
        combined_summary["gate_stage_train_loss"] = gate_summary["gate_train_loss"]

    final_summary["description"] = cfg.description
    final_summary["total_seconds"] = time.perf_counter() - total_start
    final_summary.update(combined_summary)
    return final_summary


def main() -> None:
    args = parse_args()
    configure_mps_environment()
    cfg = apply_overrides(ExperimentConfig(), args.set)

    if args.time_budget_seconds is not None:
        cfg.time_budget_seconds = args.time_budget_seconds
    if args.val_max_batches is not None:
        cfg.val_max_batches = args.val_max_batches
    if args.dry_run:
        cfg.time_budget_seconds = min(cfg.time_budget_seconds, 2.0)
        cfg.val_max_batches = min(cfg.val_max_batches, 1)
        cfg.preflight_measure_steps = 1
        cfg.min_projected_steps = 1

    if cfg.use_stage_curriculum:
        summary = run_stage_curriculum(cfg)
    else:
        total_start = time.perf_counter()
        data = load_data_bundle(cfg.seq_len)

        runtime_cfg, adaptive_info = choose_runtime_cfg(cfg, data)
        if runtime_cfg.batch_size != cfg.batch_size:
            print(
                f"auto_adjust: batch_size {cfg.batch_size} -> {runtime_cfg.batch_size} "
                f"({adaptive_info['selected']['adjustment_reason']})"
            )

        while True:
            try:
                summary = run_timed_training(runtime_cfg, data)
                break
            except RuntimeError as exc:
                if not (runtime_cfg.retry_on_runtime_oom and is_oom_error(exc) and runtime_cfg.batch_size > runtime_cfg.min_batch_size):
                    raise
                next_batch = max(runtime_cfg.min_batch_size, runtime_cfg.batch_size // 2)
                if next_batch == runtime_cfg.batch_size:
                    raise
                runtime_cfg = replace(runtime_cfg, batch_size=next_batch)

        synchronize_device(torch.device(summary["device"]))
        summary["description"] = cfg.description
        summary["total_seconds"] = time.perf_counter() - total_start
        summary["preflight_attempts"] = len(adaptive_info["attempts"])
        summary["preflight_avg_step_seconds"] = adaptive_info["selected"].get("avg_step_seconds", 0.0)
        summary["preflight_peak_memory_mb"] = adaptive_info["selected"].get("peak_memory_mb", 0.0)
        summary["preflight_projected_steps"] = adaptive_info["selected"].get("projected_steps", 0)
        summary["preflight_projected_tokens_M"] = adaptive_info["selected"].get("projected_tokens_m", 0.0)

    print("---")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
