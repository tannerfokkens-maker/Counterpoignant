"""PyTorch Dataset for training the Bach model."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bach_gen.utils.constants import DEFAULT_SEQ_LEN

logger = logging.getLogger(__name__)


class BachDataset(Dataset):
    """Dataset of tokenized Bach voice pairs."""

    def __init__(
        self,
        sequences: list[list[int]],
        seq_len: int = DEFAULT_SEQ_LEN,
        pad_token: int = 0,
    ):
        self.seq_len = seq_len
        self.pad_token = pad_token

        # Filter sequences: keep those with at least a few tokens
        self.sequences = [s for s in sequences if len(s) >= 20]
        logger.info(f"Dataset: {len(self.sequences)} sequences, seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        # Truncate or pad
        if len(seq) > self.seq_len:
            # Random offset crop
            max_start = len(seq) - self.seq_len
            start = random.randint(0, max_start)
            seq = seq[start:start + self.seq_len]
        elif len(seq) < self.seq_len:
            seq = seq + [self.pad_token] * (self.seq_len - len(seq))

        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}

    def save(self, path: str | Path) -> None:
        """Save sequences to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.sequences, f)
        logger.info(f"Saved {len(self.sequences)} sequences to {path}")

    @classmethod
    def load(cls, path: str | Path, seq_len: int = DEFAULT_SEQ_LEN) -> "BachDataset":
        """Load sequences from a JSON file."""
        with open(path) as f:
            sequences = json.load(f)
        return cls(sequences, seq_len=seq_len)


def create_dataset(
    sequences: list[list[int]],
    seq_len: int = DEFAULT_SEQ_LEN,
    val_split: float = 0.1,
    piece_ids: list[str] | None = None,
) -> tuple[BachDataset, BachDataset]:
    """Create train/val datasets from tokenized sequences.

    Args:
        sequences: List of token sequences.
        seq_len: Maximum sequence length.
        val_split: Fraction of data for validation.
        piece_ids: Optional list parallel to sequences identifying the source
            piece. When provided, the split is done by piece so that all
            chunks from the same piece end up on the same side.

    Returns:
        (train_dataset, val_dataset)
    """
    if piece_ids is not None and len(piece_ids) == len(sequences):
        # Split by piece to avoid data leakage from overlapping chunks
        unique_ids = list(set(piece_ids))
        random.shuffle(unique_ids)
        split_idx = max(1, int(len(unique_ids) * (1 - val_split)))
        train_ids = set(unique_ids[:split_idx])
        val_ids = set(unique_ids[split_idx:])

        train_seqs = [s for s, pid in zip(sequences, piece_ids) if pid in train_ids]
        val_seqs = [s for s, pid in zip(sequences, piece_ids) if pid in val_ids]

        logger.info(
            f"Piece-level split: {len(train_ids)} train pieces, "
            f"{len(val_ids)} val pieces, no overlap: True"
        )
    else:
        if piece_ids is not None:
            logger.warning("piece_ids length mismatch, falling back to random split")
        random.shuffle(sequences)
        split_idx = max(1, int(len(sequences) * (1 - val_split)))
        train_seqs = sequences[:split_idx]
        val_seqs = sequences[split_idx:]

    train_ds = BachDataset(train_seqs, seq_len=seq_len)
    val_ds = BachDataset(val_seqs, seq_len=seq_len)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


def compute_corpus_stats(sequences: list[list[int]], vocab_size: int, tokenizer=None) -> dict:
    """Compute statistics from the corpus for evaluation.

    Args:
        sequences: Tokenized sequences.
        vocab_size: Size of the token vocabulary.
        tokenizer: Optional tokenizer instance. Falls back to BachTokenizer()
                   when None (backward compat).

    Returns dict with pitch_class_dist (and/or scale_degree_dist), interval_dist,
    duration_dist.
    """
    if tokenizer is None:
        from bach_gen.data.tokenizer import BachTokenizer
        tokenizer = BachTokenizer()

    from bach_gen.utils.constants import DURATION_BINS
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer

    is_scale_degree = isinstance(tokenizer, ScaleDegreeTokenizer)

    pitch_counts = np.zeros(12)
    scale_degree_counts = np.zeros(7)
    interval_counts = np.zeros(25)  # -12 to +12
    duration_bins = getattr(getattr(tokenizer, "config", None), "duration_bins", DURATION_BINS)
    duration_counts = np.zeros(len(duration_bins))

    for seq in sequences:
        comp = tokenizer.decode(seq)

        for voice_notes in comp.voices:
            prev_pitch = None
            for _, dur, pitch in voice_notes:
                pc = pitch % 12
                pitch_counts[pc] += 1

                # Scale degree distribution (key-agnostic)
                if is_scale_degree:
                    from bach_gen.utils.music_theory import midi_to_scale_degree
                    _, degree, _ = midi_to_scale_degree(pitch, comp.key_root, comp.key_mode)
                    scale_degree_counts[degree - 1] += 1

                if prev_pitch is not None:
                    interval = pitch - prev_pitch
                    interval = max(-12, min(12, interval))
                    interval_counts[interval + 12] += 1

                prev_pitch = pitch

                dur_idx = _nearest_bin_idx(dur, duration_bins)
                duration_counts[dur_idx] += 1

    pitch_dist = pitch_counts / (pitch_counts.sum() + 1e-10)
    interval_dist = interval_counts / (interval_counts.sum() + 1e-10)
    duration_dist = duration_counts / (duration_counts.sum() + 1e-10)

    result = {
        "pitch_class_dist": pitch_dist.tolist(),
        "interval_dist": interval_dist.tolist(),
        "duration_dist": duration_dist.tolist(),
    }

    if is_scale_degree:
        sd_dist = scale_degree_counts / (scale_degree_counts.sum() + 1e-10)
        result["scale_degree_dist"] = sd_dist.tolist()

    return result


def _nearest_bin_idx(value: int, bins: list[int]) -> int:
    """Find index of the nearest bin."""
    return min(range(len(bins)), key=lambda i: abs(bins[i] - value))
