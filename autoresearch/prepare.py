"""Fixed helpers for autoresearch experiments on the bach-gen pipeline."""

from __future__ import annotations

import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bach_gen.data.dataset import BachDataset, create_dataset
from bach_gen.data.tokenizer import load_tokenizer
from bach_gen.model.architecture import BachTransformer
from bach_gen.model.config import ModelConfig
from bach_gen.model.trainer import Trainer, get_device

DATA_DIR = ROOT / "datamidiall"
CHECKPOINT_DIR = ROOT / "models" / "autoresearch"
TIME_BUDGET_SECONDS = 300.0
SPLIT_SEED = 1337
EVAL_SEED = 4242
VAL_MAX_BATCHES = 8


@dataclass
class DataBundle:
    tokenizer: object
    train_dataset: BachDataset
    val_dataset: BachDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_bundle(seq_len: int) -> DataBundle:
    """Load tokenizer + deterministic train/val datasets for a target seq_len."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    seq_path = DATA_DIR / "sequences.json"
    tok_path = DATA_DIR / "tokenizer.json"
    piece_ids_path = DATA_DIR / "piece_ids.json"

    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequences file: {seq_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer file: {tok_path}")

    with open(seq_path) as f:
        sequences = json.load(f)

    piece_ids = None
    if piece_ids_path.exists():
        with open(piece_ids_path) as f:
            piece_ids = json.load(f)

    tokenizer = load_tokenizer(tok_path)

    rng_state = random.getstate()
    try:
        random.seed(SPLIT_SEED)
        train_dataset, val_dataset = create_dataset(
            sequences,
            seq_len=seq_len,
            piece_ids=piece_ids,
            tokenizer=tokenizer,
        )
    finally:
        random.setstate(rng_state)

    return DataBundle(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def build_trainer(
    config: ModelConfig,
    data: DataBundle,
    *,
    lr: float,
    batch_size: int,
    accumulation_steps: int,
    fp16: bool,
    piece_balance: str,
) -> Trainer:
    """Instantiate a Trainer with a fresh experiment checkpoint directory."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model = BachTransformer(config)
    return Trainer(
        model=model,
        train_dataset=data.train_dataset,
        val_dataset=data.val_dataset,
        lr=lr,
        batch_size=batch_size,
        checkpoint_dir=CHECKPOINT_DIR,
        device=get_device(),
        accumulation_steps=accumulation_steps,
        fp16=fp16,
        piece_balance=piece_balance,
    )


def load_model_weights_only(trainer: Trainer, checkpoint_path: str | Path) -> int:
    """Load model weights without restoring optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    state = checkpoint["model_state_dict"]
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
    return checkpoint.get("epoch", 0)


def make_val_loader(dataset: BachDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    synchronize_device(device)


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "not enough memory" in message
        or "mps backend out of memory" in message
    )


def current_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.memory_allocated(device)) / (1024**2)
    if device.type == "mps":
        try:
            return float(torch.mps.current_allocated_memory()) / (1024**2)
        except RuntimeError:
            return 0.0
    return 0.0


@torch.no_grad()
def evaluate_validation_loss(
    trainer: Trainer,
    val_loader: DataLoader,
    *,
    use_rope: bool,
    max_batches: int = VAL_MAX_BATCHES,
) -> float:
    """Run a deterministic limited validation pass for experiment comparison."""
    model = trainer.model
    model.eval()
    total_loss = 0.0
    n_batches = 0
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
                    loss, _ = trainer._compute_looplm_loss(output, labels)
                else:
                    logits = model(input_ids, attention_mask=attention_mask, use_rope=use_rope)
                    loss = trainer.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    )

            total_loss += float(loss.item())
            n_batches += 1
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    if n_batches == 0:
        raise RuntimeError("Validation loader produced zero batches")
    return total_loss / n_batches
