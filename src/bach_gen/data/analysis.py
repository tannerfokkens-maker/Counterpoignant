"""Algorithmic analysis of VoiceComposition for conditioning labels.

The labeler is intentionally Bach-first and form-aware. It computes numeric
features first, then buckets them into the existing token vocabulary so the
rest of the pipeline can stay stable while the detectors improve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from bach_gen.data.conditioning import detect_subject_entries
from bach_gen.data.conditioning_config import (
    get_form_thresholds,
    load_conditioning_thresholds,
    normalize_conditioning_form,
)
from bach_gen.data.extraction import VoiceComposition
from bach_gen.utils.constants import (
    TICKS_PER_QUARTER,
    beat_tick_positions,
    ticks_per_measure,
)
from bach_gen.utils.music_theory import detect_mode_family, get_modal_scale, get_scale
from bach_gen.utils.voice_index import VoiceIndex


@dataclass(frozen=True)
class ConditioningFeatures:
    """Continuous features used to derive conditioning labels."""

    texture_sync_ratio: float
    texture_active_overlap: float
    texture_onset_staggering: float
    texture_shared_onset_ratio: float
    texture_polyphony_score: float
    imitation_match_density: float
    harmonic_rhythm_changes_per_measure: float
    harmonic_rhythm_outer_voice_changes_per_measure: float
    harmonic_rhythm_score: float
    harmonic_tension_attack_dissonance_ratio: float
    harmonic_tension_strongbeat_dissonance_ratio: float
    harmonic_tension_outer_voice_dissonance_ratio: float
    harmonic_tension_score: float
    chromaticism_ratio: float


def _sorted_voices(comp: VoiceComposition) -> list[list[tuple[int, int, int]]]:
    return [sorted(voice, key=lambda n: n[0]) for voice in comp.voices if voice]


def _time_signature(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if time_sig is not None:
        return time_sig
    return comp.time_signature if hasattr(comp, "time_signature") else (4, 4)


def _max_tick(voices: list[list[tuple[int, int, int]]]) -> int:
    if not voices:
        return 0
    return max(start + dur for voice in voices for start, dur, _ in voice)


def _measure_count(max_tick: int, measure_ticks: int) -> int:
    if measure_ticks <= 0:
        return 1
    return max(1, math.ceil(max_tick / measure_ticks))


def _active_pitches(indexes: list[VoiceIndex], tick: int) -> list[int]:
    pitches: list[int] = []
    for idx in indexes:
        pitch = idx.pitch_at(tick)
        if pitch is not None:
            pitches.append(pitch)
    return pitches


def _strong_beat_offsets(time_sig: tuple[int, int]) -> list[int]:
    offsets = beat_tick_positions(time_sig)
    if not offsets:
        return [0]
    strong = [offsets[0]]
    if len(offsets) == 2:
        strong.append(offsets[1])
    elif len(offsets) >= 4:
        strong.append(offsets[len(offsets) // 2])
    return sorted(set(strong))


def _texture_feature_components(
    comp: VoiceComposition,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> dict[str, float]:
    voices = _sorted_voices(comp)
    if not voices:
        return {
            "sync_ratio": 0.5,
            "active_overlap_ratio": 0.5,
            "stagger_ratio": 0.5,
            "shared_onset_ratio": 0.5,
            "polyphony_score": 0.5,
        }

    form_cfg = get_form_thresholds(form, thresholds)
    all_thresholds = thresholds or load_conditioning_thresholds()
    weights = all_thresholds["texture_weights"]

    grid_quantum = TICKS_PER_QUARTER // 2
    onset_sets: list[set[int]] = []
    all_slots: set[int] = set()
    for voice in voices:
        onset_set = {start // grid_quantum for start, _, _ in voice}
        onset_sets.append(onset_set)
        all_slots |= onset_set

    if not all_slots:
        return {
            "sync_ratio": 0.5,
            "active_overlap_ratio": 0.5,
            "stagger_ratio": 0.5,
            "shared_onset_ratio": 0.5,
            "polyphony_score": 0.5,
        }

    slot_list = sorted(all_slots)
    n_voices = len(onset_sets)
    shared_fraction = float(form_cfg["texture"].get("shared_voice_fraction", 0.75))
    shared_threshold = max(2, math.ceil(n_voices * shared_fraction)) if n_voices > 1 else 1

    onset_counts = [sum(1 for onset_set in onset_sets if slot in onset_set) for slot in slot_list]
    sync_ratio = sum(count / n_voices for count in onset_counts) / len(onset_counts)
    stagger_ratio = sum(1 for count in onset_counts if 1 <= count < shared_threshold) / len(onset_counts)
    shared_onset_ratio = sum(1 for count in onset_counts if count >= shared_threshold) / len(onset_counts)

    indexes = [VoiceIndex(voice) for voice in voices]
    max_tick = _max_tick(voices)
    active_any = 0
    active_overlap = 0
    for tick in range(0, max_tick + grid_quantum, grid_quantum):
        active_count = sum(1 for idx in indexes if idx.is_active(tick))
        if active_count == 0:
            continue
        active_any += 1
        if active_count >= min(2, len(indexes)):
            active_overlap += 1
    active_overlap_ratio = active_overlap / active_any if active_any else 0.5

    polyphony_score = (
        weights["sync_inverse"] * (1.0 - sync_ratio)
        + weights["stagger"] * stagger_ratio
        + weights["active_overlap"] * active_overlap_ratio
        + weights["shared_inverse"] * (1.0 - shared_onset_ratio)
    )

    return {
        "sync_ratio": max(0.0, min(1.0, sync_ratio)),
        "active_overlap_ratio": max(0.0, min(1.0, active_overlap_ratio)),
        "stagger_ratio": max(0.0, min(1.0, stagger_ratio)),
        "shared_onset_ratio": max(0.0, min(1.0, shared_onset_ratio)),
        "polyphony_score": max(0.0, min(1.0, polyphony_score)),
    }


def _imitation_match_density(comp: VoiceComposition) -> float:
    voices = _sorted_voices(comp)
    if not voices:
        return 0.0

    ngram_len = 6
    min_offset = TICKS_PER_QUARTER

    voice_ngram_times: list[dict[tuple[int, ...], list[int]]] = []
    for voice in voices:
        ngram_time_map: dict[tuple[int, ...], list[int]] = {}
        for i in range(len(voice) - ngram_len):
            ngram = tuple(voice[k + 1][2] - voice[k][2] for k in range(i, i + ngram_len))
            ngram_time_map.setdefault(ngram, []).append(voice[i][0])
        voice_ngram_times.append(ngram_time_map)

    total_notes = sum(len(v) for v in voices)
    if total_notes == 0:
        return 0.0

    matches = 0
    for vi in range(len(voice_ngram_times)):
        for vj in range(vi + 1, len(voice_ngram_times)):
            map_i = voice_ngram_times[vi]
            map_j = voice_ngram_times[vj]
            for ngram_i, times_i in map_i.items():
                candidates = {ngram_i}
                for pos in range(len(ngram_i)):
                    for delta in (-1, 1):
                        variant = list(ngram_i)
                        variant[pos] += delta
                        candidates.add(tuple(variant))
                for candidate in candidates:
                    if candidate not in map_j:
                        continue
                    for t_i in times_i:
                        for t_j in map_j[candidate]:
                            if abs(t_j - t_i) >= min_offset:
                                matches += 1
    return matches / total_notes


def _harmonic_rhythm_components(
    comp: VoiceComposition,
    *,
    time_sig: tuple[int, int],
) -> dict[str, float]:
    voices = _sorted_voices(comp)
    if not voices:
        return {
            "changes_per_measure": 0.0,
            "outer_changes_per_measure": 0.0,
            "score": 0.0,
        }

    measure_ticks = ticks_per_measure(time_sig)
    if measure_ticks <= 0:
        return {
            "changes_per_measure": 0.0,
            "outer_changes_per_measure": 0.0,
            "score": 0.0,
        }

    max_tick = _max_tick(voices)
    n_measures = _measure_count(max_tick, measure_ticks)
    beat_offsets = beat_tick_positions(time_sig)
    beat_ticks: list[int] = []
    for measure_idx in range(n_measures + 1):
        for offset in beat_offsets:
            tick = measure_idx * measure_ticks + offset
            if tick <= max_tick:
                beat_ticks.append(tick)
    beat_ticks = sorted(set(beat_ticks))
    if len(beat_ticks) < 2:
        return {
            "changes_per_measure": 0.0,
            "outer_changes_per_measure": 0.0,
            "score": 0.0,
        }

    indexes = [VoiceIndex(voice) for voice in voices]

    def pcs_at(tick: int) -> frozenset[int]:
        return frozenset(pitch % 12 for pitch in _active_pitches(indexes, tick))

    def outer_pcs_at(tick: int) -> tuple[int | None, int | None]:
        soprano = indexes[0].pitch_at(tick) if indexes else None
        bass = indexes[-1].pitch_at(tick) if indexes else None
        return (
            soprano % 12 if soprano is not None else None,
            bass % 12 if bass is not None else None,
        )

    prev_pcs = pcs_at(beat_ticks[0])
    prev_outer = outer_pcs_at(beat_ticks[0])
    pcs_changes = 0
    outer_changes = 0
    for tick in beat_ticks[1:]:
        current_pcs = pcs_at(tick)
        current_outer = outer_pcs_at(tick)
        if current_pcs and current_pcs != prev_pcs:
            pcs_changes += 1
        if current_pcs and current_outer != prev_outer:
            outer_changes += 1
        prev_pcs = current_pcs
        prev_outer = current_outer

    changes_per_measure = pcs_changes / n_measures
    outer_changes_per_measure = outer_changes / n_measures
    score = 0.6 * changes_per_measure + 0.4 * outer_changes_per_measure
    return {
        "changes_per_measure": changes_per_measure,
        "outer_changes_per_measure": outer_changes_per_measure,
        "score": score,
    }


def _dissonance_ratio(
    comp: VoiceComposition,
    sample_ticks: list[int],
    voice_pairs: list[tuple[int, int]] | None = None,
) -> float:
    voices = _sorted_voices(comp)
    if len(voices) < 2 or not sample_ticks:
        return 0.0

    indexes = [VoiceIndex(voice) for voice in voices]
    if voice_pairs is None:
        pairs = [(i, j) for i in range(len(indexes)) for j in range(i + 1, len(indexes))]
    else:
        pairs = [(i, j) for i, j in voice_pairs if i < len(indexes) and j < len(indexes)]
    if not pairs:
        return 0.0

    dissonant_set = {1, 2, 6, 10, 11}
    dissonant_intervals = 0
    interval_count = 0

    for tick in sample_ticks:
        for i, j in pairs:
            pitch_i = indexes[i].pitch_at(tick)
            pitch_j = indexes[j].pitch_at(tick)
            if pitch_i is None or pitch_j is None:
                continue
            interval_count += 1
            if abs(pitch_i - pitch_j) % 12 in dissonant_set:
                dissonant_intervals += 1

    if interval_count == 0:
        return 0.0
    return dissonant_intervals / interval_count


def _harmonic_tension_components(
    comp: VoiceComposition,
    *,
    time_sig: tuple[int, int],
) -> dict[str, float]:
    voices = _sorted_voices(comp)
    if len(voices) < 2:
        return {
            "attack_dissonance_ratio": 0.0,
            "strongbeat_dissonance_ratio": 0.0,
            "outer_voice_dissonance_ratio": 0.0,
            "score": 0.0,
        }

    max_tick = _max_tick(voices)
    attack_ticks = sorted({start for voice in voices for start, _, _ in voice})
    measure_ticks = ticks_per_measure(time_sig)
    strong_offsets = _strong_beat_offsets(time_sig)
    strongbeat_ticks: list[int] = []
    if measure_ticks > 0:
        n_measures = _measure_count(max_tick, measure_ticks)
        for measure_idx in range(n_measures + 1):
            base = measure_idx * measure_ticks
            for offset in strong_offsets:
                tick = base + offset
                if tick <= max_tick:
                    strongbeat_ticks.append(tick)
    strongbeat_ticks = sorted(set(strongbeat_ticks))

    outer_pairs = {(0, len(voices) - 1)}
    if len(voices) >= 3:
        outer_pairs.add((0, len(voices) - 2))
        outer_pairs.add((1, len(voices) - 1))

    attack_ratio = _dissonance_ratio(comp, attack_ticks)
    strongbeat_ratio = _dissonance_ratio(comp, strongbeat_ticks)
    outer_ratio = _dissonance_ratio(comp, attack_ticks, sorted(outer_pairs))
    score = 0.45 * strongbeat_ratio + 0.35 * attack_ratio + 0.20 * outer_ratio
    return {
        "attack_dissonance_ratio": attack_ratio,
        "strongbeat_dissonance_ratio": strongbeat_ratio,
        "outer_voice_dissonance_ratio": outer_ratio,
        "score": score,
    }


def _chromaticism_components(
    comp: VoiceComposition,
    *,
    form: str | None = None,
) -> dict[str, float | str]:
    piece_time_sig = _time_signature(comp)
    normalized_form = normalize_conditioning_form(form)
    scale_pcs = set(get_scale(comp.key_root, comp.key_mode))
    reference_system = "tonal"

    if normalized_form in {"motet", "vocal_polyphony"} or getattr(comp, "style", "") in {"renaissance", "medieval"}:
        mode_diag = detect_mode_family(comp.voices, time_signature=piece_time_sig)
        if mode_diag.system == "modal" and mode_diag.confidence >= 0.58:
            scale_pcs = set(get_modal_scale(mode_diag.root_pc, mode_diag.mode_family))
            reference_system = "modal"

    total_notes = 0
    chromatic_notes = 0
    for voice in comp.voices:
        for _, _, pitch in voice:
            total_notes += 1
            if (pitch % 12) not in scale_pcs:
                chromatic_notes += 1
    if total_notes == 0:
        return {"ratio": 0.0, "reference_system": reference_system}
    return {
        "ratio": chromatic_notes / total_notes,
        "reference_system": reference_system,
    }


def compute_conditioning_features(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> ConditioningFeatures:
    """Return continuous analysis features for one composition."""
    piece_time_sig = _time_signature(comp, time_sig)
    normalized_form = normalize_conditioning_form(form)

    texture = _texture_feature_components(comp, form=normalized_form, thresholds=thresholds)
    imitation_density = _imitation_match_density(comp)
    harmonic_rhythm = _harmonic_rhythm_components(comp, time_sig=piece_time_sig)
    harmonic_tension = _harmonic_tension_components(comp, time_sig=piece_time_sig)
    chromaticism = _chromaticism_components(comp, form=normalized_form)

    return ConditioningFeatures(
        texture_sync_ratio=texture["sync_ratio"],
        texture_active_overlap=texture["active_overlap_ratio"],
        texture_onset_staggering=texture["stagger_ratio"],
        texture_shared_onset_ratio=texture["shared_onset_ratio"],
        texture_polyphony_score=texture["polyphony_score"],
        imitation_match_density=imitation_density,
        harmonic_rhythm_changes_per_measure=harmonic_rhythm["changes_per_measure"],
        harmonic_rhythm_outer_voice_changes_per_measure=harmonic_rhythm["outer_changes_per_measure"],
        harmonic_rhythm_score=harmonic_rhythm["score"],
        harmonic_tension_attack_dissonance_ratio=harmonic_tension["attack_dissonance_ratio"],
        harmonic_tension_strongbeat_dissonance_ratio=harmonic_tension["strongbeat_dissonance_ratio"],
        harmonic_tension_outer_voice_dissonance_ratio=harmonic_tension["outer_voice_dissonance_ratio"],
        harmonic_tension_score=harmonic_tension["score"],
        chromaticism_ratio=float(chromaticism["ratio"]),
    )


def compute_texture(
    comp: VoiceComposition,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> str:
    """Classify texture as homophonic, mixed, or polyphonic."""
    form_cfg = get_form_thresholds(form, thresholds)
    score = _texture_feature_components(
        comp,
        form=form,
        thresholds=thresholds,
    )["polyphony_score"]
    if score >= float(form_cfg["texture"]["polyphonic_min"]):
        return "polyphonic"
    if score >= float(form_cfg["texture"]["mixed_min"]):
        return "mixed"
    return "homophonic"


def compute_imitation(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> str:
    """Classify imitation level as none, low, or high."""
    piece_time_sig = _time_signature(comp, time_sig)
    form_cfg = get_form_thresholds(form, thresholds)
    imitation_cfg = form_cfg.get("imitation", {})

    normalised = _imitation_match_density(comp)
    effective_density = normalised
    length_norm_notes = float(imitation_cfg.get("length_normalization_notes", 0.0))
    if length_norm_notes > 0:
        total_notes = sum(len(voice) for voice in _sorted_voices(comp))
        effective_density = normalised / max(1.0, total_notes / length_norm_notes)
    high_min = float(imitation_cfg.get("high_min", 0.30))
    low_min = float(imitation_cfg.get("low_min", 0.10))

    if effective_density >= high_min:
        return "high"
    if effective_density >= low_min:
        return "low"

    floor_count_min = int(imitation_cfg.get("subject_floor_count_min", 99))
    floor_density_min = float(imitation_cfg.get("subject_floor_density_min", 99.0))
    normalized_form = normalize_conditioning_form(form)
    if normalized_form in {"fugue", "invention", "sinfonia"}:
        entries = detect_subject_entries(comp, form=form, thresholds=thresholds)
        if entries:
            measure_ticks = ticks_per_measure(piece_time_sig)
            measures = _measure_count(_max_tick(_sorted_voices(comp)), measure_ticks)
            subject_density = len(entries) * 10.0 / max(1, measures)
            if len(entries) >= floor_count_min and subject_density >= floor_density_min:
                return "low"

    return "none"


def compute_harmonic_rhythm(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> str:
    """Classify harmonic rhythm as slow, moderate, or fast."""
    piece_time_sig = _time_signature(comp, time_sig)
    form_cfg = get_form_thresholds(form, thresholds)
    score = _harmonic_rhythm_components(comp, time_sig=piece_time_sig)["score"]
    slow_max = float(form_cfg["harmonic_rhythm"]["slow_max"])
    moderate_max = float(form_cfg["harmonic_rhythm"]["moderate_max"])
    if score <= slow_max:
        return "slow"
    if score <= moderate_max:
        return "moderate"
    return "fast"


def compute_harmonic_tension(
    comp: VoiceComposition,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> str:
    """Classify harmonic tension as low, moderate, or high."""
    piece_time_sig = _time_signature(comp)
    form_cfg = get_form_thresholds(form, thresholds)
    score = _harmonic_tension_components(comp, time_sig=piece_time_sig)["score"]
    low_max = float(form_cfg["harmonic_tension"]["low_max"])
    moderate_max = float(form_cfg["harmonic_tension"]["moderate_max"])
    if score <= low_max:
        return "low"
    if score <= moderate_max:
        return "moderate"
    return "high"


def compute_chromaticism(
    comp: VoiceComposition,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> str:
    """Classify chromaticism as low, moderate, or high."""
    form_cfg = get_form_thresholds(form, thresholds)
    ratio = float(_chromaticism_components(comp, form=form)["ratio"])
    low_max = float(form_cfg["chromaticism"]["low_max"])
    moderate_max = float(form_cfg["chromaticism"]["moderate_max"])
    if ratio <= low_max:
        return "low"
    if ratio <= moderate_max:
        return "moderate"
    return "high"


def conditioning_feature_dict(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
    *,
    form: str | None = None,
    thresholds: dict | None = None,
) -> dict[str, float | str]:
    """Return conditioning features as a plain dict for audits and reports."""
    features = asdict(
        compute_conditioning_features(
            comp,
            time_sig,
            form=form,
            thresholds=thresholds,
        )
    )
    chromaticism = _chromaticism_components(comp, form=form)
    features["chromaticism_reference_system"] = str(chromaticism["reference_system"])
    return features


def analyze_composition(
    comp: VoiceComposition,
    time_sig: tuple[int, int] | None = None,
    form: str | None = None,
    thresholds: dict | None = None,
) -> dict[str, str]:
    """Return token-ready labels for one composition."""
    return {
        "texture": compute_texture(comp, form=form, thresholds=thresholds),
        "imitation": compute_imitation(
            comp,
            time_sig,
            form=form,
            thresholds=thresholds,
        ),
        "harmonic_rhythm": compute_harmonic_rhythm(
            comp,
            time_sig,
            form=form,
            thresholds=thresholds,
        ),
        "harmonic_tension": compute_harmonic_tension(
            comp,
            form=form,
            thresholds=thresholds,
        ),
        "chromaticism": compute_chromaticism(
            comp,
            form=form,
            thresholds=thresholds,
        ),
    }
