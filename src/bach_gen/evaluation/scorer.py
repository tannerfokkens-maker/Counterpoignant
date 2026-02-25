"""Composite score orchestrator.

Combines all evaluation dimensions into a single score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.evaluation.voice_leading import score_voice_leading
from bach_gen.evaluation.statistical import score_statistical
from bach_gen.evaluation.structural import score_structural, score_thematic_recall
from bach_gen.evaluation.contrapuntal import score_contrapuntal
from bach_gen.evaluation.information import score_information

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Breakdown of evaluation scores."""

    voice_leading: float = 0.0
    statistical: float = 0.0
    structural: float = 0.0
    information: float = 0.0
    contrapuntal: float = 0.0
    completeness: float = 0.0
    thematic_recall: float = 0.0
    composite: float = 0.0
    details: dict | None = None


# Default weights
DEFAULT_WEIGHTS = {
    "voice_leading": 0.25,
    "statistical": 0.15,
    "structural": 0.15,
    "information": 0.15,
    "contrapuntal": 0.10,
    "completeness": 0.10,
    "thematic_recall": 0.10,
}


def score_composition(
    comp: VoiceComposition,
    token_sequence: list[int] | None = None,
    model: torch.nn.Module | None = None,
    weights: dict[str, float] | None = None,
    vocab_size: int | None = None,
    tokenizer=None,
) -> ScoreBreakdown:
    """Score a VoiceComposition on all evaluation dimensions.

    Args:
        comp: The composition to evaluate.
        token_sequence: Tokenized version (for information scoring).
        model: Trained model (for information scoring).
        weights: Custom weights (default: DEFAULT_WEIGHTS).
        vocab_size: Vocabulary size for information scoring.
                    Falls back to BachTokenizer().vocab_size when None.
        tokenizer: Tokenizer instance (for thematic recall subject extraction).

    Returns:
        ScoreBreakdown with individual and composite scores.
    """
    w = weights or DEFAULT_WEIGHTS
    all_details: dict = {}

    # Voice leading — evaluates all voice pairs
    vl_score, vl_details = score_voice_leading(comp)
    all_details["voice_leading"] = vl_details

    # Statistical similarity
    stat_score, stat_details = score_statistical(comp)
    all_details["statistical"] = stat_details

    # Structural coherence
    struct_score, struct_details = score_structural(comp)
    all_details["structural"] = struct_details

    # Contrapuntal quality — evaluates all voice pairs
    cp_score, cp_details = score_contrapuntal(comp)
    all_details["contrapuntal"] = cp_details

    # Information-theoretic
    if token_sequence is not None:
        if vocab_size is None:
            from bach_gen.data.tokenizer import BachTokenizer
            vocab_size = BachTokenizer().vocab_size
        info_score, info_details = score_information(
            token_sequence, model=model, vocab_size=vocab_size
        )
    else:
        info_score = 0.5  # neutral if no tokens
        info_details = {"note": "no token sequence provided"}
    all_details["information"] = info_details

    # Completeness
    comp_score = _score_completeness(comp)
    all_details["completeness"] = {"score": comp_score}

    # Thematic recall
    tr_score = score_thematic_recall(comp, token_sequence=token_sequence, tokenizer=tokenizer)
    all_details["thematic_recall"] = {"score": tr_score}

    # Composite
    composite = (
        vl_score * w.get("voice_leading", 0.25)
        + stat_score * w.get("statistical", 0.15)
        + struct_score * w.get("structural", 0.15)
        + info_score * w.get("information", 0.15)
        + cp_score * w.get("contrapuntal", 0.10)
        + comp_score * w.get("completeness", 0.10)
        + tr_score * w.get("thematic_recall", 0.10)
    )

    return ScoreBreakdown(
        voice_leading=vl_score,
        statistical=stat_score,
        structural=struct_score,
        information=info_score,
        contrapuntal=cp_score,
        completeness=comp_score,
        thematic_recall=tr_score,
        composite=composite,
        details=all_details,
    )


def score_voice_pair(
    pair: VoicePair,
    token_sequence: list[int] | None = None,
    model: torch.nn.Module | None = None,
    weights: dict[str, float] | None = None,
) -> ScoreBreakdown:
    """Score a voice pair (backward-compatible wrapper).

    Converts to VoiceComposition and delegates to score_composition.
    """
    comp = VoiceComposition.from_voice_pair(pair)
    return score_composition(comp, token_sequence=token_sequence, model=model, weights=weights)


def _score_completeness(comp: VoiceComposition) -> float:
    """Score completeness: proper opening, development, and ending."""
    from bach_gen.utils.constants import TICKS_PER_QUARTER

    all_notes = [n for voice in comp.voices for n in voice]
    if not all_notes:
        return 0.0

    score = 0.0

    # 1. Has content in all voices
    non_empty = sum(1 for v in comp.voices if v)
    if non_empty >= 2:
        score += 0.3

    # 2. Sufficient length (at least 8 bars)
    total_ticks = max(n[0] + n[1] for n in all_notes) - min(n[0] for n in all_notes)
    bars = total_ticks / (TICKS_PER_QUARTER * 4)
    if bars >= 8:
        score += 0.2
    elif bars >= 4:
        score += 0.1

    # 3. All voices present at beginning and end
    min_time = min(n[0] for n in all_notes)
    max_time = max(n[0] + n[1] for n in all_notes)
    first_quarter = min_time + TICKS_PER_QUARTER * 4
    last_quarter = max_time - TICKS_PER_QUARTER * 4

    voices_at_start = sum(1 for v in comp.voices if v and any(n[0] < first_quarter for n in v))
    voices_at_end = sum(1 for v in comp.voices if v and any(n[0] + n[1] > last_quarter for n in v))

    if voices_at_start >= non_empty:
        score += 0.1
    if voices_at_end >= non_empty:
        score += 0.1

    # 4. Ends on a long note (final cadence feel) — check lowest voice
    lowest_voice = comp.voices[-1] if comp.voices else []
    highest_voice = comp.voices[0] if comp.voices else []
    if lowest_voice and highest_voice:
        last_low_dur = lowest_voice[-1][1]
        last_high_dur = highest_voice[-1][1]
        if last_low_dur >= TICKS_PER_QUARTER and last_high_dur >= TICKS_PER_QUARTER:
            score += 0.15

    # 5. Tonic ending — check lowest voice
    tonic = comp.key_root
    if lowest_voice:
        last_lower_pc = lowest_voice[-1][2] % 12
        if last_lower_pc == tonic:
            score += 0.15

    return min(1.0, score)
