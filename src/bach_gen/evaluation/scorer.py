"""Composite score orchestrator.

Combines all evaluation dimensions into a single score.

When per-form calibration weights have been computed (via ``bach-gen
calibrate-forms``), they are loaded automatically and used whenever a
``form`` is supplied to :func:`score_composition`.

Dimensions:
  - voice_leading: Local note-to-note rule violations
  - statistical: Distribution similarity to Bach corpus
  - structural: Key consistency, cadences, phrase structure, modulation
  - contrapuntal: Texture & technique (sequences, register, stretto, etc.)
  - completeness: Voice count, length, proper endings
  - thematic_recall: Subject recurrence across voices and time

Information-theoretic scoring has been removed — calibration showed it
has zero discrimination power (identical scores on real Bach, shuffled
notes, and random pitches when no model is passed during generation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.evaluation.voice_leading import score_voice_leading
from bach_gen.evaluation.statistical import score_statistical
from bach_gen.evaluation.structural import score_structural, score_thematic_recall
from bach_gen.evaluation.contrapuntal import score_contrapuntal

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Breakdown of evaluation scores."""

    voice_leading: float = 0.0
    statistical: float = 0.0
    structural: float = 0.0
    information: float = 0.0  # kept for backward compat, always 0.0
    contrapuntal: float = 0.0
    completeness: float = 0.0
    thematic_recall: float = 0.0
    composite: float = 0.0
    details: dict | None = None


# Default weights (used when no calibration data is available)
# Favor perceptual structure and flow over single-axis "cleanliness."
DEFAULT_WEIGHTS = {
    "voice_leading": 0.20,
    "statistical": 0.09,
    "structural": 0.22,
    "contrapuntal": 0.20,
    "completeness": 0.05,
    "thematic_recall": 0.24,
}

# Calibrated per-form weights (populated by load_form_weights)
_form_weights: dict[str, dict[str, float]] | None = None


def load_form_weights(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-form calibrated weights from calibration_forms.json.

    Called automatically on first use if the file exists. Can also be
    called explicitly to reload after re-calibration.
    """
    global _form_weights
    if path is None:
        path = Path("data/calibration_forms.json")
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        _form_weights = data.get("weights_by_form", {})
        logger.info(f"Loaded calibrated weights for forms: {list(_form_weights.keys())}")
    else:
        _form_weights = {}
    return _form_weights


def get_weights_for_form(form: str | None = None) -> dict[str, float]:
    """Return the best available weights for the given form.

    Priority:
      1. Calibrated per-form weights (if calibrate-forms has been run)
      2. DEFAULT_WEIGHTS fallback
    """
    global _form_weights

    # Auto-load on first call
    if _form_weights is None:
        load_form_weights()

    if form and _form_weights and form in _form_weights:
        return _form_weights[form]

    # Try normalised form names
    form_aliases = {
        "2-part": "invention",
        "3-part": "sinfonia",
        "4-part": "chorale",
    }
    if form and _form_weights:
        alias = form_aliases.get(form)
        if alias and alias in _form_weights:
            return _form_weights[alias]

    return DEFAULT_WEIGHTS


def score_composition(
    comp: VoiceComposition,
    token_sequence: list[int] | None = None,
    model: torch.nn.Module | None = None,
    weights: dict[str, float] | None = None,
    vocab_size: int | None = None,
    tokenizer=None,
    form: str | None = None,
) -> ScoreBreakdown:
    """Score a VoiceComposition on all evaluation dimensions.

    Args:
        comp: The composition to evaluate.
        token_sequence: Tokenized version (unused, kept for API compat).
        model: Trained model (unused, kept for API compat).
        weights: Custom weights override. If None, uses calibrated per-form
                 weights when available, else DEFAULT_WEIGHTS.
        vocab_size: Unused, kept for API compat.
        tokenizer: Tokenizer instance (for thematic recall subject extraction).
        form: Composition form (e.g. "fugue", "chorale", "invention").
              Used to select calibrated weights when ``weights`` is None.

    Returns:
        ScoreBreakdown with individual and composite scores.
    """
    w = weights or get_weights_for_form(form)
    all_details: dict = {}

    # Voice leading — local rule violations
    vl_score, vl_details = score_voice_leading(comp)
    all_details["voice_leading"] = vl_details

    # Statistical similarity
    stat_score, stat_details = score_statistical(comp)
    all_details["statistical"] = stat_details

    # Structural coherence
    struct_score, struct_details = score_structural(comp)
    all_details["structural"] = struct_details

    # Contrapuntal quality — texture and technique
    cp_score, cp_details = score_contrapuntal(comp)
    all_details["contrapuntal"] = cp_details

    # Completeness
    comp_score = _score_completeness(comp)
    all_details["completeness"] = {"score": comp_score}

    # Thematic recall
    tr_score = score_thematic_recall(comp, token_sequence=token_sequence, tokenizer=tokenizer)
    all_details["thematic_recall"] = {"score": tr_score}

    # Composite (information dimension removed — zero discrimination power)
    raw_composite = (
        vl_score * w.get("voice_leading", 0.22)
        + stat_score * w.get("statistical", 0.10)
        + struct_score * w.get("structural", 0.15)
        + cp_score * w.get("contrapuntal", 0.18)
        + comp_score * w.get("completeness", 0.05)
        + tr_score * w.get("thematic_recall", 0.30)
    )
    guardrail_mult, guardrail_flags = _guardrail_multiplier(
        form=form,
        comp=comp,
        voice_leading=vl_score,
        structural_details=struct_details,
        contrapuntal_details=cp_details,
        completeness=comp_score,
    )
    guarded_composite = raw_composite * guardrail_mult
    interaction_delta, interaction_flags = _interaction_adjustment(
        form=form,
        voice_leading=vl_score,
        structural_details=struct_details,
        contrapuntal_details=cp_details,
    )
    composite = max(0.0, min(1.0, guarded_composite + interaction_delta))
    if guardrail_flags:
        all_details["guardrails"] = {
            "multiplier": guardrail_mult,
            "flags": guardrail_flags,
            "raw_composite": raw_composite,
        }
    if interaction_flags:
        all_details["interactions"] = {
            "delta": interaction_delta,
            "flags": interaction_flags,
            "pre_interaction_composite": guarded_composite,
        }

    return ScoreBreakdown(
        voice_leading=vl_score,
        statistical=stat_score,
        structural=struct_score,
        information=0.0,  # removed — kept for backward compat
        contrapuntal=cp_score,
        completeness=comp_score,
        thematic_recall=tr_score,
        composite=composite,
        details=all_details,
    )


def _guardrail_multiplier(
    *,
    form: str | None,
    comp: VoiceComposition,
    voice_leading: float,
    structural_details: dict,
    contrapuntal_details: dict,
    completeness: float,
) -> tuple[float, list[str]]:
    """Compute post-weighting guardrail multiplier and trigger flags."""
    mult = 1.0
    flags: list[str] = []

    # Global quality floors.
    if voice_leading < 0.65:
        mult *= 0.80
        flags.append("low_voice_leading")
    if completeness < 0.50:
        mult *= 0.80
        flags.append("low_completeness")

    if form == "fugue":
        total_notes = sum(len(v) for v in comp.voices if v)
        # Avoid over-penalising tiny toy snippets used in tests/debug.
        if total_notes < 32:
            return mult, flags

        non_empty = sum(1 for v in comp.voices if v)
        cadence = float(structural_details.get("cadence", 0.0))
        onset_stagger = float(contrapuntal_details.get("onset_staggering", 0.0))
        voice_indep = float(contrapuntal_details.get("voice_independence", 0.0))
        voice_balance = float(contrapuntal_details.get("voice_balance", 1.0))

        if non_empty < 4:
            mult *= 0.70
            flags.append("fugue_missing_voice")
        if cadence < 0.45:
            mult *= 0.85
            flags.append("fugue_weak_cadence")
        if voice_balance < 0.35:
            mult *= 0.75
            flags.append("fugue_unbalanced_voices")
        if onset_stagger > 0.90 and voice_indep < 0.45:
            mult *= 0.80
            flags.append("fugue_fragmented_not_dialogic")
        if onset_stagger < 0.10:
            mult *= 0.85
            flags.append("fugue_over_lockstep")
        if completeness < 0.65:
            mult *= 0.80
            flags.append("fugue_incomplete_form")

    return max(0.0, min(1.0, mult)), flags


def _interaction_adjustment(
    *,
    form: str | None,
    voice_leading: float,
    structural_details: dict,
    contrapuntal_details: dict,
) -> tuple[float, list[str]]:
    """Compute small additive interaction adjustments for form-specific quality."""
    if form is None:
        return 0.0, []

    profiles: dict[str, dict[str, object]] = {
        "fugue": {
            "tag": "fugue",
            "rhetoric": (0.62, 0.74, 0.74, 0.008, "fugue_rhetorical_shape"),
            "rhetoric_strong": (0.70, 0.84, 0.80, 0.010, "fugue_strong_rhetorical_shape"),
            "flow": (0.78, 0.40, 0.88, 0.006, "fugue_flowing_texture"),
            "clean_flat": (0.90, 0.66, 0.40, -0.012, "fugue_clean_but_flat"),
            "safe_homog": (0.92, 0.78, 0.66, 0.44, -0.018, "fugue_safe_homogenized_texture"),
            "bound": 0.04,
        },
        "sinfonia": {
            "tag": "sinfonia",
            "rhetoric": (0.58, 0.70, 0.72, 0.006, "sinfonia_rhetorical_shape"),
            "rhetoric_strong": (0.66, 0.80, 0.78, 0.008, "sinfonia_strong_rhetorical_shape"),
            "flow": (0.74, 0.38, 0.86, 0.005, "sinfonia_flowing_texture"),
            "clean_flat": (0.89, 0.64, 0.38, -0.010, "sinfonia_clean_but_flat"),
            "safe_homog": (0.91, 0.76, 0.64, 0.42, -0.014, "sinfonia_safe_homogenized_texture"),
            "bound": 0.035,
        },
        "invention": {
            "tag": "invention",
            "rhetoric": (0.55, 0.68, 0.70, 0.006, "invention_rhetorical_shape"),
            "rhetoric_strong": (0.63, 0.78, 0.76, 0.008, "invention_strong_rhetorical_shape"),
            "flow": (0.70, 0.36, 0.85, 0.004, "invention_flowing_texture"),
            "clean_flat": (0.88, 0.62, 0.36, -0.010, "invention_clean_but_flat"),
            "safe_homog": (0.90, 0.74, 0.62, 0.40, -0.014, "invention_safe_homogenized_texture"),
            "bound": 0.035,
        },
        # Apply invention profile to legacy/alias 2-voice form.
        "2-part": {
            "tag": "invention",
            "rhetoric": (0.55, 0.68, 0.70, 0.006, "invention_rhetorical_shape"),
            "rhetoric_strong": (0.63, 0.78, 0.76, 0.008, "invention_strong_rhetorical_shape"),
            "flow": (0.70, 0.36, 0.85, 0.004, "invention_flowing_texture"),
            "clean_flat": (0.88, 0.62, 0.36, -0.010, "invention_clean_but_flat"),
            "safe_homog": (0.90, 0.74, 0.62, 0.40, -0.014, "invention_safe_homogenized_texture"),
            "bound": 0.035,
        },
    }
    profile = profiles.get(form)
    if profile is None:
        return 0.0, []

    delta = 0.0
    flags: list[str] = []

    cadence = float(structural_details.get("cadence", 0.0))
    phrase = float(structural_details.get("phrase_structure", 0.0))
    key_consistency = float(structural_details.get("key_consistency", 0.0))
    onset_stagger = float(contrapuntal_details.get("onset_staggering", 0.0))
    contrary = float(contrapuntal_details.get("contrary_at_cadences", 0.0))
    melodic = float(contrapuntal_details.get("melodic_coherence", 0.0))
    voice_indep = float(contrapuntal_details.get("voice_independence", 0.0))

    cad_min, phr_min, key_min, boost, boost_flag = profile["rhetoric"]  # type: ignore[index]
    if cadence >= cad_min and phrase >= phr_min and key_consistency >= key_min:
        delta += boost
        flags.append(str(boost_flag))

    cad_s, phr_s, key_s, boost_s, boost_s_flag = profile["rhetoric_strong"]  # type: ignore[index]
    if cadence >= cad_s and phrase >= phr_s and key_consistency >= key_s:
        delta += boost_s
        flags.append(str(boost_s_flag))

    # Reward flowing independent texture near cadential rhetoric.
    onset_min, contrary_min, melodic_min, flow_boost, flow_flag = profile["flow"]  # type: ignore[index]
    if onset_stagger >= onset_min and contrary >= contrary_min and melodic >= melodic_min:
        delta += flow_boost
        flags.append(str(flow_flag))

    # Penalize "clean but flat" outputs.
    vl_floor, phrase_floor, contrary_floor, clean_penalty, clean_flag = profile["clean_flat"]  # type: ignore[index]
    if voice_leading >= vl_floor and (phrase < phrase_floor or contrary < contrary_floor):
        delta += clean_penalty
        flags.append(str(clean_flag))

    # Penalize safe-but-homogenized texture dressed up with clean voice-leading.
    (
        vl_safe_floor,
        indep_floor,
        onset_floor,
        contrary_safe_floor,
        safe_penalty,
        safe_flag,
    ) = profile["safe_homog"]  # type: ignore[index]
    if (
        voice_leading >= vl_safe_floor
        and voice_indep < indep_floor
        and onset_stagger < onset_floor
        and contrary < contrary_safe_floor
    ):
        delta += safe_penalty
        flags.append(str(safe_flag))

    # Keep interaction effects bounded and interpretable.
    bound = float(profile.get("bound", 0.04))
    delta = max(-bound, min(bound, delta))
    return delta, flags


def score_voice_pair(
    pair: VoicePair,
    token_sequence: list[int] | None = None,
    model: torch.nn.Module | None = None,
    weights: dict[str, float] | None = None,
) -> ScoreBreakdown:
    """Score a voice pair (backward-compatible wrapper)."""
    comp = VoiceComposition.from_voice_pair(pair)
    return score_composition(comp, token_sequence=token_sequence, model=model, weights=weights)


def _score_completeness(comp: VoiceComposition) -> float:
    """Score completeness: proper opening, development, and ending."""
    from bach_gen.utils.constants import TICKS_PER_QUARTER

    all_notes = [n for voice in comp.voices for n in voice]
    if not all_notes:
        return 0.0

    score = 0.0

    # 1. Has content in multiple voices
    non_empty = sum(1 for v in comp.voices if v)
    if non_empty >= 2:
        score += 0.25

    # 2. Sufficient length (at least 8 bars)
    total_ticks = max(n[0] + n[1] for n in all_notes) - min(n[0] for n in all_notes)
    bars = total_ticks / (TICKS_PER_QUARTER * 4)
    if bars >= 8:
        score += 0.15
    elif bars >= 4:
        score += 0.08

    # 3. All voices present at beginning and end
    min_time = min(n[0] for n in all_notes)
    max_time = max(n[0] + n[1] for n in all_notes)
    first_quarter = min_time + TICKS_PER_QUARTER * 4
    last_quarter = max_time - TICKS_PER_QUARTER * 4

    voices_at_start = sum(1 for v in comp.voices if v and any(n[0] < first_quarter for n in v))
    voices_at_end = sum(1 for v in comp.voices if v and any(n[0] + n[1] > last_quarter for n in v))

    if voices_at_start >= non_empty:
        score += 0.10
    if voices_at_end >= non_empty:
        score += 0.10

    # 4. Ends on a long note (final cadence feel)
    lowest_voice = comp.voices[-1] if comp.voices else []
    highest_voice = comp.voices[0] if comp.voices else []
    if lowest_voice and highest_voice:
        last_low_dur = lowest_voice[-1][1]
        last_high_dur = highest_voice[-1][1]
        if last_low_dur >= TICKS_PER_QUARTER and last_high_dur >= TICKS_PER_QUARTER:
            score += 0.15

    # 5. Tonic ending
    tonic = comp.key_root
    if lowest_voice:
        last_lower_pc = lowest_voice[-1][2] % 12
        if last_lower_pc == tonic:
            score += 0.15

    # 6. Piece reached natural end (EOS) vs truncated at max_length
    # This is checked in the token_sequence but since we may not have it,
    # approximate: if last notes are very short or cut off, likely truncated
    if lowest_voice and highest_voice:
        last_bass_end = lowest_voice[-1][0] + lowest_voice[-1][1]
        last_sop_end = highest_voice[-1][0] + highest_voice[-1][1]
        # Voices end near each other = proper ending
        if abs(last_bass_end - last_sop_end) < TICKS_PER_QUARTER * 2:
            score += 0.10

    return min(1.0, score)
