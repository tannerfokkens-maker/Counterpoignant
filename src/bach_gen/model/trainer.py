"""Training loop with MPS/CUDA/CPU support and checkpointing."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from bach_gen.model.config import ModelConfig
from bach_gen.model.architecture import BachTransformer
from bach_gen.data.dataset import BachDataset
from bach_gen.utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_LABEL_SMOOTHING,
)

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Training manager for the Bach Transformer."""

    def __init__(
        self,
        model: BachTransformer,
        train_dataset: BachDataset,
        val_dataset: BachDataset | None = None,
        lr: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
        checkpoint_dir: str | Path = "models",
        device: torch.device | None = None,
        accumulation_steps: int = 1,
        fp16: bool = False,
        token_category_map: list[int] | None = None,
        token_category_names: list[str] | None = None,
        piece_balance: str = "none",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.piece_balance = piece_balance

        # Mixed precision — only on CUDA
        self.fp16 = fp16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.fp16)
        if self.fp16:
            logger.info("Mixed precision (fp16) enabled")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # ignore PAD
            label_smoothing=label_smoothing,
        )

        self.best_val_loss = float("inf")
        self.epoch = 0

        # Optional token-category monitoring (pure logging, not used for gradients).
        self.token_category_names = token_category_names or []
        self._token_category_map = None
        if token_category_map is not None and self.token_category_names:
            if len(token_category_map) != self.model.config.vocab_size:
                raise ValueError(
                    "token_category_map length must match vocab size: "
                    f"{len(token_category_map)} != {self.model.config.vocab_size}"
                )
            self._token_category_map = torch.tensor(
                token_category_map, dtype=torch.long, device=self.device
            )

    def reset_for_finetuning(
        self,
        train_dataset: BachDataset,
        val_dataset: BachDataset | None,
        lr: float,
        save_checkpoint_name: str | None = "pretrain_final.pt",
    ) -> None:
        """Swap datasets and reset optimizer for fine-tuning phase.

        Optionally saves the current model checkpoint, then replaces the
        training/validation datasets, creates a fresh optimizer with the
        given learning rate, and resets ``best_val_loss``.
        """
        if save_checkpoint_name:
            self._save_checkpoint(save_checkpoint_name)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.optimizer.defaults.get("weight_decay", 0.01),
            betas=self.optimizer.defaults.get("betas", (0.9, 0.98)),
            eps=self.optimizer.defaults.get("eps", 1e-9),
        )

        self.best_val_loss = float("inf")
        logger.info(
            f"Reset for fine-tuning: lr={lr}, "
            f"train={len(train_dataset)}, "
            f"val={len(val_dataset) if val_dataset else 0}"
        )

    @staticmethod
    def _resize_vocab_rows(
        weight: torch.Tensor,
        target_rows: int,
    ) -> torch.Tensor:
        """Resize embedding/head matrix rows while preserving shared prefix."""
        if weight.size(0) == target_rows:
            return weight

        resized = weight.new_empty((target_rows, weight.size(1)))
        keep_rows = min(weight.size(0), target_rows)
        if keep_rows > 0:
            resized[:keep_rows].copy_(weight[:keep_rows])

        if target_rows > keep_rows:
            std = float(weight.std().item()) if weight.numel() else 0.02
            if not math.isfinite(std) or std <= 0.0:
                std = 0.02
            nn.init.normal_(resized[keep_rows:], mean=0.0, std=std)

        return resized

    @classmethod
    def _maybe_resize_vocab_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        target_vocab_size: int,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Resize token embedding/head rows when checkpoint vocab differs."""
        embed_key = "token_embed.weight"
        if embed_key not in state_dict:
            return state_dict, False

        embed_weight = state_dict[embed_key]
        if embed_weight.size(0) == target_vocab_size:
            return state_dict, False

        resized_embed = cls._resize_vocab_rows(embed_weight, target_vocab_size)
        state_dict[embed_key] = resized_embed

        head_key = "head.weight"
        if head_key in state_dict:
            state_dict[head_key] = cls._resize_vocab_rows(state_dict[head_key], target_vocab_size)

        return state_dict, True

    def save_checkpoint(self, filename: str) -> None:
        """Public checkpoint save helper for phase transitions."""
        self._save_checkpoint(filename)

    def resume_from_checkpoint(self, path: str | Path) -> int:
        """Load model, optimizer, and training state from checkpoint.

        Returns:
            The epoch to resume from (next epoch after the saved one).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint["model_state_dict"]
        state, resized_vocab = self._maybe_resize_vocab_state_dict(
            state,
            target_vocab_size=self.model.config.vocab_size,
        )
        self.model.load_state_dict(state, strict=True)
        if resized_vocab:
            logger.info(
                "Checkpoint vocab resized to %d rows; optimizer state reset.",
                self.model.config.vocab_size,
            )
        else:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state from {path}: {e}")
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        saved_epoch = checkpoint.get("epoch", 0)
        self.epoch = saved_epoch
        logger.info(f"Resumed from {path} (epoch {saved_epoch}, best_val={self.best_val_loss:.4f})")
        return saved_epoch + 1

    def _make_train_loader(self, dataset: BachDataset | None = None) -> DataLoader:
        """Build a training DataLoader, optionally with piece-balanced sampling.

        When ``self.piece_balance`` is ``"sqrt"`` or ``"inverse"`` and the
        dataset has ``piece_ids``, a ``WeightedRandomSampler`` is used so that
        heavily-chunked pieces are down-weighted.
        """
        from bach_gen.data.dataset import compute_piece_weights

        ds = dataset or self.train_dataset
        use_sampler = (
            self.piece_balance != "none"
            and hasattr(ds, "piece_ids")
            and ds.piece_ids
        )

        if use_sampler:
            weights = compute_piece_weights(ds.piece_ids, mode=self.piece_balance)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(ds),
                replacement=True,
            )
            # Log diagnostics
            from collections import Counter
            piece_counts = Counter(ds.piece_ids)
            top = piece_counts.most_common(10)
            logger.info(
                f"Piece-balance sampler ({self.piece_balance}): "
                f"{len(piece_counts)} unique pieces, "
                f"top-10 chunks: {[(pid, cnt) for pid, cnt in top]}"
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                sampler=sampler,
                drop_last=True,
                num_workers=0,
            )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    def transition_seq_len(self, new_seq_len: int) -> None:
        """Update the effective context length for training.

        Adjusts the dataset ``seq_len`` (which controls cropping/padding in
        ``__getitem__``), the model config, and the ``max_seq_len`` cached on
        each attention layer (used only for KV-cache allocation during
        generation, but kept consistent for correctness).  Positional-
        embedding caches auto-extend on the next forward pass so no explicit
        rebuild is needed.
        """
        old = self.train_dataset.seq_len
        self.train_dataset.seq_len = new_seq_len
        if self.val_dataset is not None:
            self.val_dataset.seq_len = new_seq_len
        self.model.config.max_seq_len = new_seq_len

        # Update per-layer KV cache pre-allocation size.
        for layer in self.model.layers:
            if hasattr(layer, "attn") and hasattr(layer.attn, "max_seq_len"):
                layer.attn.max_seq_len = new_seq_len

        logger.info(f"Transitioned seq_len: {old} → {new_seq_len}")

    def train(
        self,
        epochs: int = DEFAULT_EPOCHS,
        start_epoch: int = 1,
        log_interval: int = 10,
        val_interval: int = 5,
        progress_callback=None,
        early_stop: bool = False,
        patience: int = 20,
        min_delta: float = 1e-4,
        min_epochs: int = 10,
        phase_name: str | None = None,
        checkpoint_prefix: str = "",
        use_rope: bool | None = None,
        seq_len_stages: list[tuple[int, int]] | None = None,
    ) -> dict:
        """Run training loop.

        Args:
            epochs: Total number of epochs (target epoch count).  When
                ``seq_len_stages`` is provided this is ignored and the total
                is computed from the stage definitions.
            start_epoch: Epoch to start from (1 for fresh, >1 when resuming).
            log_interval: Log every N epochs.
            val_interval: Validate every N epochs.
            progress_callback: Optional callback(epoch, train_loss, val_loss).
            early_stop: Whether to stop before ``epochs`` on val loss plateau.
            patience: Allowed consecutive non-improving validation checks.
            min_delta: Minimum val loss improvement to reset patience.
            min_epochs: Minimum epochs before early stop can trigger.
            phase_name: Optional phase label for log messages.
            checkpoint_prefix: Optional checkpoint name prefix per phase.
            use_rope: Whether to use positional embeddings. Defaults to
                ``not model.config.drope_trained``.
            seq_len_stages: Optional list of ``(seq_len, num_epochs)`` for
                staged context-length training.  When provided, the training
                loop is divided into consecutive stages, each with its own
                context length, cosine-annealed LR schedule, and early-
                stopping counter.

        Returns:
            Dict with training history.
        """
        if use_rope is None:
            use_rope = not getattr(self.model.config, "drope_trained", False)
        phase_tag = f"[{phase_name}] " if phase_name else ""
        ckpt_prefix = checkpoint_prefix or ""

        # --- Build stage schedule ---
        if seq_len_stages:
            stages = list(seq_len_stages)
        else:
            stages = [(self.train_dataset.seq_len, epochs)]

        total_epochs = sum(ep for _, ep in stages)

        history: dict = {"train_loss": [], "val_loss": [], "lr": []}
        if self._token_category_map is not None:
            history["train_category_loss"] = []
            history["val_category_loss"] = []

        effective_batch = self.batch_size * self.accumulation_steps
        logger.info(
            f"{phase_tag}Training on {self.device} for {total_epochs} epochs "
            f"({len(stages)} stage{'s' if len(stages) > 1 else ''}, "
            f"starting at epoch {start_epoch})"
        )
        if len(stages) > 1:
            stage_desc = ", ".join(f"{sl}@{ep}ep" for sl, ep in stages)
            logger.info(f"{phase_tag}Seq-len stages: {stage_desc}")
        logger.info(f"{phase_tag}Model params: {self.model.count_parameters():,}")
        logger.info(
            f"{phase_tag}Batch size: {self.batch_size} x {self.accumulation_steps} accumulation"
            f" = {effective_batch} effective"
        )

        if start_epoch > total_epochs:
            logger.warning(
                f"Start epoch {start_epoch} exceeds target epochs {total_epochs}. Nothing to train."
            )
            return history

        if early_stop:
            logger.info(
                f"{phase_tag}Early stopping enabled: patience={patience}, "
                f"min_delta={min_delta}, min_epochs={min_epochs}"
            )

        global_epoch = 0  # 0-based counter across all stages
        stop_reason = "max_epochs_reached"

        for stage_idx, (stage_seq_len, stage_epochs) in enumerate(stages):
            # Transition context length
            self.transition_seq_len(stage_seq_len)

            train_loader = self._make_train_loader()
            val_loader = None
            if self.val_dataset and len(self.val_dataset) > 0:
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

            # Per-stage cosine annealing from current LR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=stage_epochs, eta_min=1e-6,
            )

            # Fast-forward scheduler if resuming into this stage
            stage_start_global = sum(ep for _, ep in stages[:stage_idx])
            epochs_to_skip = max(0, (start_epoch - 1) - stage_start_global)
            if epochs_to_skip >= stage_epochs:
                global_epoch += stage_epochs
                continue  # This entire stage was already completed
            for _ in range(epochs_to_skip):
                scheduler.step()

            if len(stages) > 1:
                logger.info(
                    f"{phase_tag}Stage {stage_idx + 1}/{len(stages)}: "
                    f"seq_len={stage_seq_len}, {stage_epochs} epochs"
                )

            bad_epochs = 0
            stage_stopped_early = False

            for stage_ep in range(1, stage_epochs + 1):
                global_epoch = stage_start_global + stage_ep
                if global_epoch < start_epoch:
                    continue  # Skip already-completed epochs

                self.epoch = global_epoch
                train_loss, train_cat_losses = self._train_epoch(train_loader, use_rope=use_rope)
                history["train_loss"].append(train_loss)
                history["lr"].append(scheduler.get_last_lr()[0])
                if "train_category_loss" in history:
                    history["train_category_loss"].append(train_cat_losses)

                scheduler.step()

                val_loss = None
                val_cat_losses: dict[str, float | None] = {}
                if val_loader and global_epoch % val_interval == 0:
                    val_loss, val_cat_losses = self._validate(val_loader, use_rope=use_rope)
                    history["val_loss"].append(val_loss)
                    if "val_category_loss" in history:
                        history["val_category_loss"].append(val_cat_losses)

                    if val_loss < (self.best_val_loss - min_delta):
                        self.best_val_loss = val_loss
                        bad_epochs = 0
                        self._save_checkpoint(f"{ckpt_prefix}best.pt")
                    elif val_loss < self.best_val_loss:
                        # Improved but below min_delta threshold
                        self.best_val_loss = val_loss
                        bad_epochs += 1
                        self._save_checkpoint(f"{ckpt_prefix}best.pt")
                    else:
                        bad_epochs += 1

                if global_epoch % log_interval == 0:
                    msg = f"{phase_tag}Epoch {global_epoch}/{total_epochs} | train_loss={train_loss:.4f}"
                    if len(stages) > 1:
                        msg += f" | seq_len={stage_seq_len}"
                    if train_cat_losses:
                        msg += self._format_category_losses(train_cat_losses, label="train_cat")
                    if val_loss is not None:
                        msg += f" | val_loss={val_loss:.4f}"
                        if val_cat_losses:
                            msg += self._format_category_losses(val_cat_losses, label="val_cat")
                    msg += f" | lr={scheduler.get_last_lr()[0]:.6f}"
                    logger.info(msg)

                if progress_callback:
                    progress_callback(global_epoch, train_loss, val_loss)

                # Save after every epoch so training can be stopped at any time
                self._save_checkpoint(f"{ckpt_prefix}latest.pt")

                if early_stop and stage_ep >= min_epochs and bad_epochs >= patience:
                    stop_reason = (
                        f"early_stop(stage={stage_idx+1}, patience={patience}, min_delta={min_delta})"
                    )
                    logger.info(f"{phase_tag}Early stop at epoch {global_epoch}: {stop_reason}")
                    stage_stopped_early = True
                    break

            # Save stage checkpoint
            if len(stages) > 1:
                self._save_checkpoint(f"{ckpt_prefix}stage{stage_idx+1}.pt")

        # Save final checkpoint
        self._save_checkpoint(f"{ckpt_prefix}final.pt")

        history["epochs_ran"] = len(history["train_loss"])
        history["stop_reason"] = stop_reason

        return history

    def _train_epoch(self, loader: DataLoader, use_rope: bool = True) -> tuple[float, dict[str, float | None]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        category_sums, category_counts = self._init_category_accumulators()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Create attention mask (non-PAD tokens)
            attention_mask = (input_ids != 0).long()

            with torch.amp.autocast(self.device.type, enabled=self.fp16):
                logits = self.model(input_ids, attention_mask=attention_mask, use_rope=use_rope)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            self._accumulate_category_losses(logits, labels, category_sums, category_counts)

            # Scale loss for accumulation
            scaled_loss = loss / self.accumulation_steps
            self.scaler.scale(scaled_loss).backward()

            total_loss += loss.item()
            n_batches += 1

            # Step when we've accumulated enough, or at the last batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                # Gradient clipping (unscale first for fp16)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        return total_loss / max(n_batches, 1), self._finalize_category_losses(category_sums, category_counts)

    @torch.no_grad()
    def _validate(self, loader: DataLoader, use_rope: bool = True) -> tuple[float, dict[str, float | None]]:
        """Validate on held-out data."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        category_sums, category_counts = self._init_category_accumulators()

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = (input_ids != 0).long()

            with torch.amp.autocast(self.device.type, enabled=self.fp16):
                logits = self.model(input_ids, attention_mask=attention_mask, use_rope=use_rope)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            self._accumulate_category_losses(logits, labels, category_sums, category_counts)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1), self._finalize_category_losses(category_sums, category_counts)

    def recalibrate_drope(
        self,
        epochs: int,
        lr: float,
        early_stop: bool = True,
        patience: int = 5,
        min_delta: float = 1e-4,
        min_epochs: int = 4,
        warmup_epochs: int = 1,
    ) -> dict:
        """Run DroPE recalibration: continue training without RoPE.

        Per Gelberg et al. 2025: after normal RoPE training, drop all
        positional embeddings and train for a short recalibration phase
        at the original context length.  The model learns to recover
        positional information from causal masking and BEAT tokens.

        Args:
            epochs: Maximum number of recalibration epochs.
            lr: Learning rate for recalibration (typically higher, e.g. 1e-3).
            early_stop: Whether to stop before ``epochs`` on plateau.
            patience: Allowed consecutive non-improving epochs.
            min_delta: Minimum metric improvement to reset patience.
            min_epochs: Minimum epochs before early stop is allowed.
            warmup_epochs: Warmup epochs before cosine decay.

        Returns:
            Dict with training history for the recalibration phase.
        """
        # Save pre-DroPE checkpoint
        self._save_checkpoint("pre_drope.pt")

        # Mark model as DroPE-trained before saving DroPE checkpoints so the
        # flag persists when loading drope_best.pt for downstream phases.
        self.model.config.drope_trained = True

        # Record the training sequence length before DroPE
        self.model.config.drope_train_seq_len = self.model.config.max_seq_len

        # Create fresh optimizer with the recalibration LR
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.optimizer.defaults.get("weight_decay", 0.01),
            betas=self.optimizer.defaults.get("betas", (0.9, 0.98)),
            eps=self.optimizer.defaults.get("eps", 1e-9),
        )

        self.best_val_loss = float("inf")

        train_loader = self._make_train_loader()

        val_loader = None
        if self.val_dataset and len(self.val_dataset) > 0:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        total_epochs = max(1, epochs)
        warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
        if warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.2,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=max(1, total_epochs - warmup_epochs),
                        eta_min=1e-6,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=1e-6,
            )

        history = {"train_loss": [], "val_loss": [], "lr": []}
        if self._token_category_map is not None:
            history["train_category_loss"] = []
            history["val_category_loss"] = []

        logger.info(
            f"DroPE recalibration: {epochs} epochs, lr={lr}, "
            f"dropping {self.model.config.pos_encoding} positional encoding, "
            f"warmup_epochs={warmup_epochs}, "
            f"early_stop={early_stop}, patience={patience}, min_delta={min_delta}, min_epochs={min_epochs}"
        )

        best_metric = float("inf")
        bad_epochs = 0
        stop_reason = "max_epochs_reached"

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            train_loss, train_cat_losses = self._train_epoch(train_loader, use_rope=False)
            history["train_loss"].append(train_loss)
            history["lr"].append(scheduler.get_last_lr()[0])
            if "train_category_loss" in history:
                history["train_category_loss"].append(train_cat_losses)

            scheduler.step()

            val_loss = None
            val_cat_losses: dict[str, float | None] = {}
            if val_loader:
                val_loss, val_cat_losses = self._validate(val_loader, use_rope=False)
                history["val_loss"].append(val_loss)
                if "val_category_loss" in history:
                    history["val_category_loss"].append(val_cat_losses)
            else:
                history["val_loss"].append(None)
                if "val_category_loss" in history:
                    history["val_category_loss"].append({})

            metric = val_loss if val_loss is not None else train_loss
            if metric < (best_metric - min_delta):
                best_metric = metric
                bad_epochs = 0
                if val_loss is not None:
                    self.best_val_loss = val_loss
                self._save_checkpoint("drope_best.pt")
            else:
                bad_epochs += 1

            msg = f"[DROPE] Epoch {epoch}/{epochs} | train_loss={train_loss:.4f}"
            if train_cat_losses:
                msg += self._format_category_losses(train_cat_losses, label="train_cat")
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.4f}"
                if val_cat_losses:
                    msg += self._format_category_losses(val_cat_losses, label="val_cat")
            msg += f" | lr={scheduler.get_last_lr()[0]:.6f}"
            logger.info(msg)

            self._save_checkpoint("drope_latest.pt")

            if early_stop and epoch >= min_epochs and bad_epochs >= patience:
                stop_reason = (
                    f"early_stop(patience={patience}, min_delta={min_delta})"
                )
                logger.info(f"DroPE early stop at epoch {epoch}: {stop_reason}")
                break

        # Save final DroPE checkpoint
        self._save_checkpoint("drope_final.pt")

        history["epochs_ran"] = len(history["train_loss"])
        history["stop_reason"] = stop_reason
        history["best_metric"] = best_metric

        logger.info(
            f"DroPE recalibration complete (epochs_ran={history['epochs_ran']}, "
            f"stop_reason={stop_reason})"
        )
        return history

    def _init_category_accumulators(self):
        if self._token_category_map is None or not self.token_category_names:
            return None, None
        n = len(self.token_category_names)
        return torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.long)

    def _accumulate_category_losses(self, logits, labels, sums, counts) -> None:
        if sums is None or counts is None or self._token_category_map is None:
            return

        with torch.no_grad():
            flat_labels = labels.reshape(-1)
            valid_mask = flat_labels != self.criterion.ignore_index
            if not torch.any(valid_mask):
                return

            flat_logits = logits.detach().reshape(-1, logits.size(-1)).float()
            per_token_losses = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self.criterion.ignore_index,
                label_smoothing=self.criterion.label_smoothing,
                reduction="none",
            )

            valid_labels = flat_labels[valid_mask]
            valid_losses = per_token_losses[valid_mask]
            category_idx = self._token_category_map[valid_labels]

            # Use CPU for bincount compatibility across devices (e.g., MPS).
            category_idx = category_idx.to("cpu")
            valid_losses = valid_losses.to("cpu", dtype=torch.float64)

            batch_sums = torch.bincount(
                category_idx,
                weights=valid_losses,
                minlength=len(self.token_category_names),
            )
            batch_counts = torch.bincount(
                category_idx,
                minlength=len(self.token_category_names),
            ).to(torch.long)

            sums += batch_sums
            counts += batch_counts

    def _finalize_category_losses(self, sums, counts) -> dict[str, float | None]:
        if sums is None or counts is None:
            return {}

        result: dict[str, float | None] = {}
        for i, name in enumerate(self.token_category_names):
            count = int(counts[i].item())
            result[name] = (float(sums[i].item()) / count) if count > 0 else None
        return result

    def _format_category_losses(self, losses: dict[str, float | None], label: str) -> str:
        parts = []
        for name in self.token_category_names:
            value = losses.get(name)
            if value is None:
                parts.append(f"{name}=n/a")
            else:
                parts.append(f"{name}={value:.4f}")
        return f" | {label}[{', '.join(parts)}]"

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        device: torch.device | None = None,
        override_max_seq_len: int | None = None,
    ) -> tuple[BachTransformer, ModelConfig]:
        """Load model from checkpoint.

        Args:
            path: Path to the checkpoint file.
            device: Device to load the model onto.
            override_max_seq_len: When provided, override
                ``config.max_seq_len`` before constructing the model so
                positional embedding caches are built at the new size.
                Useful for staged context-length training (e.g. 8k → 16k).

        Returns:
            (model, config)
        """
        device = device or get_device()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        # Backward compat: old checkpoints lack pos_encoding
        if not hasattr(config, "pos_encoding"):
            config.pos_encoding = "rope"
        # Backward compat: old checkpoints lack num_kv_heads
        if not hasattr(config, "num_kv_heads"):
            config.num_kv_heads = None

        # Override max_seq_len for context-length extension.  Positional
        # embedding caches are registered as non-persistent buffers so they
        # won't conflict with the state dict — they are simply rebuilt at the
        # new size during model construction.
        if override_max_seq_len is not None:
            old_len = config.max_seq_len
            config.max_seq_len = override_max_seq_len
            logger.info(
                "Overriding max_seq_len: %d → %d",
                old_len,
                override_max_seq_len,
            )

        model = BachTransformer(config)

        # Migrate old combined QKV weights to separate Q/K/V projections
        state = checkpoint["model_state_dict"]
        qkv_keys = [k for k in state if ".attn.qkv." in k]
        if qkv_keys:
            for key in list(state.keys()):
                if ".attn.qkv.weight" in key:
                    w = state.pop(key)  # (3*embed_dim, embed_dim)
                    q_w, k_w, v_w = w.chunk(3, dim=0)
                    prefix = key.replace(".qkv.weight", "")
                    state[f"{prefix}.q_proj.weight"] = q_w
                    state[f"{prefix}.k_proj.weight"] = k_w
                    state[f"{prefix}.v_proj.weight"] = v_w
                elif ".attn.qkv.bias" in key:
                    b = state.pop(key)  # (3*embed_dim,)
                    q_b, k_b, v_b = b.chunk(3, dim=0)
                    prefix = key.replace(".qkv.bias", "")
                    state[f"{prefix}.q_proj.bias"] = q_b
                    state[f"{prefix}.k_proj.bias"] = k_b
                    state[f"{prefix}.v_proj.bias"] = v_b
            logger.info("Migrated old QKV weights to separate Q/K/V projections")

        state, resized_vocab = Trainer._maybe_resize_vocab_state_dict(
            state,
            target_vocab_size=config.vocab_size,
        )
        if resized_vocab:
            logger.info(
                "Resized checkpoint token embeddings/head to vocab_size=%d",
                config.vocab_size,
            )

        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        if device.type == "cuda":
            model = model.half()
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return model, config
