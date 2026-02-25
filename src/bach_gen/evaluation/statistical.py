"""Statistical similarity evaluation.

Measures Jensen-Shannon divergence of interval/rhythm/pitch-class distributions
between generated music and the Bach training corpus.

When scale-degree stats are available (from a scale-degree tokenizer), pitch-class
comparison is replaced with a key-agnostic scale-degree comparison (7 bins).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.utils.constants import DURATION_BINS

logger = logging.getLogger(__name__)

# Default corpus stats (will be replaced by actual computed stats)
_corpus_stats: dict | None = None
_STATS_PATH = Path("data/corpus_stats.json")


def load_corpus_stats(path: str | Path | None = None) -> dict:
    """Load precomputed corpus statistics."""
    global _corpus_stats
    p = Path(path) if path else _STATS_PATH
    if p.exists():
        with open(p) as f:
            _corpus_stats = json.load(f)
    else:
        # Return uniform distributions as fallback
        _corpus_stats = {
            "pitch_class_dist": [1 / 12] * 12,
            "interval_dist": [1 / 25] * 25,
            "duration_dist": [1 / 11] * 11,
        }
        logger.warning("No corpus stats found, using uniform distributions")
    return _corpus_stats


def _get_all_voices(item: VoicePair | VoiceComposition) -> list[list[tuple[int, int, int]]]:
    """Extract list of voice note-lists from either type."""
    if isinstance(item, VoiceComposition):
        return [v for v in item.voices if v]
    return [v for v in [item.upper, item.lower] if v]


def score_statistical(item: VoicePair | VoiceComposition) -> tuple[float, dict]:
    """Score statistical similarity to Bach corpus.

    Accepts VoicePair or VoiceComposition.

    When the corpus stats contain scale_degree_dist (computed from a scale-degree
    tokenizer), pitch comparison is done in scale-degree space — making it
    key-agnostic. Otherwise falls back to absolute pitch-class comparison.

    Returns:
        (score 0-1, details dict)
    """
    global _corpus_stats
    if _corpus_stats is None:
        load_corpus_stats()

    voices = _get_all_voices(item)
    corpus = _corpus_stats

    # Determine key info for scale-degree conversion
    if isinstance(item, VoiceComposition):
        key_root, key_mode = item.key_root, item.key_mode
    else:
        key_root, key_mode = item.key_root, item.key_mode

    # Pitch comparison: use scale-degree dist if available, else pitch class
    use_scale_degree = "scale_degree_dist" in corpus
    if use_scale_degree:
        pitch_dist = _compute_scale_degree_dist(voices, key_root, key_mode)
        jsd_pitch = _safe_jsd(pitch_dist, np.array(corpus["scale_degree_dist"]))
    else:
        pitch_dist = _compute_pitch_class_dist(voices)
        jsd_pitch = _safe_jsd(pitch_dist, np.array(corpus["pitch_class_dist"]))

    interval_dist = _compute_interval_dist(voices)
    duration_dist = _compute_duration_dist(voices)

    jsd_interval = _safe_jsd(interval_dist, np.array(corpus["interval_dist"]))
    jsd_duration = _safe_jsd(duration_dist, np.array(corpus["duration_dist"]))

    details = {
        "jsd_pitch_class": float(jsd_pitch),
        "jsd_interval": float(jsd_interval),
        "jsd_duration": float(jsd_duration),
        "pitch_mode": "scale_degree" if use_scale_degree else "absolute",
    }

    # Convert JSD to similarity score (0 = very different, 1 = identical)
    # JSD ranges from 0 to ~0.83 (ln(2) for completely disjoint distributions).
    # Bach pieces typically have JSD ~0.1-0.3 against the corpus average.
    # Use a softer scaling so real music scores in the 0.5-0.9 range.
    avg_jsd = (jsd_pitch + jsd_interval + jsd_duration) / 3
    score = max(0.0, 1.0 - avg_jsd * 2)

    return score, details


def _compute_pitch_class_dist(voices: list[list[tuple[int, int, int]]]) -> np.ndarray:
    """Compute 12-element pitch class distribution."""
    counts = np.zeros(12)
    for voice in voices:
        for _, _, pitch in voice:
            counts[pitch % 12] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(12) / 12
    return counts


def _compute_scale_degree_dist(
    voices: list[list[tuple[int, int, int]]],
    key_root: int,
    key_mode: str,
) -> np.ndarray:
    """Compute 7-element scale degree distribution (key-agnostic).

    Maps each pitch to its nearest scale degree (1-7) relative to the key,
    ignoring accidentals. This produces the same distribution regardless of
    which key the piece is in.
    """
    from bach_gen.utils.music_theory import midi_to_scale_degree

    counts = np.zeros(7)
    for voice in voices:
        for _, _, pitch in voice:
            _, degree, _ = midi_to_scale_degree(pitch, key_root, key_mode)
            counts[degree - 1] += 1  # degree is 1-indexed
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(7) / 7
    return counts


def _compute_interval_dist(voices: list[list[tuple[int, int, int]]]) -> np.ndarray:
    """Compute melodic interval distribution (-12 to +12 semitones)."""
    counts = np.zeros(25)
    for voice in voices:
        for i in range(1, len(voice)):
            interval = voice[i][2] - voice[i - 1][2]
            interval = max(-12, min(12, interval))
            counts[interval + 12] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(25) / 25
    return counts


def _compute_duration_dist(voices: list[list[tuple[int, int, int]]]) -> np.ndarray:
    """Compute duration distribution aligned to tokenizer bins."""
    bins = DURATION_BINS
    counts = np.zeros(len(bins))

    for voice in voices:
        for _, dur, _ in voice:
            idx = min(range(len(bins)), key=lambda i: abs(bins[i] - dur))
            counts[idx] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(len(bins)) / len(bins)
    return counts


def _safe_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence safely."""
    # Ensure same length
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]

    # Add small epsilon to avoid zeros
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(jensenshannon(p, q))
