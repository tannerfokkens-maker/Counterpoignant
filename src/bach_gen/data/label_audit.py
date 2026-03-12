"""Audit utilities for conditioning-label quality on Bach gold references."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import mido
import music21
import numpy as np

from bach_gen.data.analysis import analyze_composition, conditioning_feature_dict
from bach_gen.data.conditioning import detect_cadence_events, detect_subject_entries
from bach_gen.data.conditioning_config import load_conditioning_thresholds
from bach_gen.data.extraction import (
    VoiceComposition,
    _detect_key as _detect_score_key,
    _detect_time_signature as _detect_score_time_signature,
    detect_form,
)
from bach_gen.utils.constants import (
    DIR_TO_STYLE,
    METER_MAP,
    bwv_to_form,
    compute_measure_count,
    length_bucket,
)
from bach_gen.utils.midi_io import load_midi, midi_to_note_events
from bach_gen.utils.midi_io import midi_key_signature, midi_time_signature
from bach_gen.utils.music_theory import (
    detect_composition_key,
    detect_mode_family,
    parse_midi_key_signature,
    parse_explicit_key_from_text,
    pc_to_note_name,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_MIDI_ROOT = REPO_ROOT / "data" / "midi"
DATA_MIDI_ALL_ROOT = DATA_MIDI_ROOT / "all"
DEFAULT_MANIFEST = REPO_ROOT / "data/benchmarks/bach_gold.json"

_BACH_INVENTION_SINFONIA_KEY_CYCLE: tuple[tuple[int, str], ...] = (
    (0, "major"),
    (0, "minor"),
    (2, "major"),
    (2, "minor"),
    (3, "major"),
    (4, "major"),
    (4, "minor"),
    (5, "major"),
    (5, "minor"),
    (7, "major"),
    (7, "minor"),
    (9, "major"),
    (9, "minor"),
    (10, "major"),
    (11, "minor"),
)


EXPECTED_PRIORS: dict[str, dict[str, float]] = {
    "fugue": {
        "non_homophonic_texture_min": 0.60,
        "high_imitation_min": 0.80,
        "harmonic_rhythm_collapsed_max": 0.85,
        "harmonic_tension_collapsed_max": 0.85,
        "cadence_mean_min": 3.0,
        "cadence_mean_max": 10.0,
        "subject_coverage_min": 0.85,
        "form_path_match_min": 0.95,
    },
    "invention": {
        "non_homophonic_texture_min": 0.60,
        "harmonic_rhythm_collapsed_max": 0.85,
        "harmonic_tension_collapsed_max": 0.85,
        "subject_coverage_min": 0.80,
        "form_path_match_min": 0.95,
    },
    "sinfonia": {
        "non_homophonic_texture_min": 0.60,
        "harmonic_rhythm_collapsed_max": 0.85,
        "harmonic_tension_collapsed_max": 0.85,
        "subject_coverage_min": 0.80,
        "form_path_match_min": 0.95,
    },
    "chorale": {
        "homophonic_texture_min": 0.90,
        "none_imitation_min": 0.95,
        "harmonic_rhythm_collapsed_max": 0.85,
        "harmonic_tension_collapsed_max": 0.85,
        "cadence_median_min": 1.0,
        "cadence_mean_min": 1.5,
        "form_path_match_min": 0.95,
    },
}


@dataclass(frozen=True)
class ReferenceMetadata:
    """Reference metadata recovered from a companion source file."""

    kind: str
    path: str
    style: str
    path_form: str | None = None
    score_form: str | None = None
    key_root: int | None = None
    key_mode: str | None = None
    key_name: str | None = None
    time_signature: tuple[int, int] | None = None
    meter: str | None = None


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _expand_group_files(group: dict[str, Any]) -> list[Path]:
    files: set[Path] = set()
    for pattern in group.get("patterns", []):
        files.update(REPO_ROOT.glob(pattern))
    for explicit in group.get("files", []):
        files.add(REPO_ROOT / explicit)
    resolved = sorted((p.resolve() for p in files if p.exists()), key=_natural_key)
    include_regex = group.get("include_regex")
    if include_regex:
        matcher = re.compile(str(include_regex), re.IGNORECASE)
        resolved = [
            p for p in resolved
            if matcher.search(p.as_posix())
        ]
    exclude_regex = group.get("exclude_regex")
    if exclude_regex:
        matcher = re.compile(str(exclude_regex), re.IGNORECASE)
        resolved = [
            p for p in resolved
            if not matcher.search(p.as_posix())
        ]
    expected = group.get("expected_count")
    if expected is not None and len(resolved) != expected:
        raise ValueError(
            f"group {group['name']} expected {expected} files, found {len(resolved)}"
        )
    return resolved


def _infer_style_from_path(path: Path) -> str:
    try:
        rel_parts = path.resolve().relative_to(DATA_MIDI_ROOT.resolve()).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts[:-1]:
        style = DIR_TO_STYLE.get(part.lower())
        if style:
            return style
    return "other"


def _infer_form_from_source(source: str, num_tracks: int) -> str | None:
    source_lower = source.lower().replace("\\", "/")

    def _has_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    bwv_match = re.search(r"bwv[-_\s]?(\d+)", source_lower, re.IGNORECASE)
    if bwv_match:
        bwv_num = int(bwv_match.group(1))
        form = bwv_to_form(bwv_num)
        if form is not None:
            return form

    keyword_map = [
        ("artfugue", "fugue"),
        ("wtc1f", "fugue"),
        ("wtc2f", "fugue"),
        ("fugue", "fugue"),
        ("three-part_inventions", "sinfonia"),
        ("three-part-inventions", "sinfonia"),
        ("three_part_inventions", "sinfonia"),
        ("three part inventions", "sinfonia"),
        ("sinfonias_bwv", "sinfonia"),
        ("sinfonias bwv", "sinfonia"),
        ("sinfonias-bwv", "sinfonia"),
        ("two-part_inventions", "invention"),
        ("two-part-inventions", "invention"),
        ("two_part_inventions", "invention"),
        ("invention", "invention"),
    ]
    for keyword, form in keyword_map:
        if keyword in source_lower:
            return form

    filename = source_lower.rsplit("/", 1)[-1]
    strict_chorale = bool(
        re.search(r"(^|[_\-.])chor\d{3,}", filename)
        or re.search(r"chorales?[_\-\s]\d{3,}", source_lower)
        or _has_any_keyword(
            source_lower,
            (
                "four-part chorales",
                "four part chorales",
                "chorale harmonization",
                "chorale harmonizations",
                "harmonized chorales",
            ),
        )
    )
    if "chorale" in source_lower:
        if strict_chorale:
            return "chorale"
        if _infer_style_from_path(Path(source)) in {"renaissance", "medieval"}:
            return "vocal_polyphony"
        return "keyboard_piece"

    if "sinfonia" in source_lower:
        if _has_any_keyword(
            source_lower,
            (
                "symphony",
                "orchestra",
                "orchestral",
                "concerto",
                "overture",
                "opera",
                "ballet",
                "cantata",
                "requiem",
                "mass",
                "missa",
                "for orchestra",
                "for_orchestra",
            ),
        ) or _infer_style_from_path(Path(source)) == "classical":
            return "orchestral_reduction"
        if _has_any_keyword(
            source_lower,
            (
                "piano",
                "keyboard",
                "klavier",
                "clavier",
                "harpsichord",
                "etude",
                "prelude",
                "nocturne",
                "mazurka",
                "waltz",
                "valse",
                "polonaise",
                "impromptu",
                "ballade",
                "bagatelle",
                "rhapsody",
                "romance",
                "liebestraume",
            ),
        ):
            return "keyboard_piece"
        return "chamber_piece"

    filename = Path(source_lower).name
    if re.search(r"(^|[_\-.])chor\d+", filename):
        return "chorale"
    if _has_any_keyword(
        source_lower,
        (
            "symphony",
            "orchestra",
            "orchestral",
            "concerto",
            "overture",
            "opera",
            "ballet",
            "cantata",
            "requiem",
            "mass",
            "missa",
            "for orchestra",
            "for_orchestra",
        ),
    ):
        return "orchestral_reduction"
    if _has_any_keyword(
        source_lower,
        (
            "piano",
            "keyboard",
            "klavier",
            "clavier",
            "harpsichord",
            "etude",
            "prelude",
            "nocturne",
            "mazurka",
            "waltz",
            "valse",
            "polonaise",
            "impromptu",
            "ballade",
            "bagatelle",
            "rhapsody",
            "romance",
            "liebestraume",
        ),
    ):
        return "keyboard_piece"
    if num_tracks <= 2:
        return "keyboard_piece"
    if num_tracks == 3:
        return "chamber_piece"
    if num_tracks >= 4:
        return "chamber_piece"
    return None


def _midi_time_signature(mid: mido.MidiFile) -> tuple[int, int]:
    return midi_time_signature(mid)


def _infer_bach_catalog_bwv(path: Path, form: str | None) -> int | None:
    source = path.as_posix().lower()
    bwv_match = re.search(r"bwv[-_\s]?(\d+)", source)
    if bwv_match:
        return int(bwv_match.group(1))

    if form == "invention":
        match = re.search(r"two[-_ ]part[-_ ]inventions[-_ ](\d+)", source)
        if match:
            index = int(match.group(1))
            if 1 <= index <= 15:
                return 771 + index

    if form == "sinfonia":
        match = re.search(r"three[-_ ]part[-_ ]inventions[-_ ](\d+)", source)
        if match:
            index = int(match.group(1))
            if 1 <= index <= 15:
                return 786 + index

    return None


def _infer_bach_catalog_key(path: Path, form: str | None) -> tuple[int, str] | None:
    bwv = _infer_bach_catalog_bwv(path, form)
    if bwv is None:
        return None
    if 772 <= bwv <= 801:
        return _BACH_INVENTION_SINFONIA_KEY_CYCLE[(bwv - 772) % 15]
    return None


def _resolve_companion_source(path: Path) -> Path | None:
    try:
        rel = path.resolve().relative_to(DATA_MIDI_ALL_ROOT.resolve())
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    composer_dir = rel.parts[0]
    filename = rel.name
    if "__" not in filename:
        return None
    source_family, remainder = filename.split("__", 1)
    family_root = DATA_MIDI_ROOT / source_family / composer_dir
    candidates: list[Path] = []
    if remainder.endswith(".fromscore.mid"):
        candidates.append(family_root / remainder[:-len(".fromscore.mid")])
    if remainder.endswith(".fromscore.midi"):
        candidates.append(family_root / remainder[:-len(".fromscore.midi")])
    candidates.append(family_root / remainder)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=2048)
def _load_reference_metadata_cached(companion_path_str: str, expected_style: str) -> dict[str, Any]:
    companion_path = Path(companion_path_str)
    style = _infer_style_from_path(companion_path) or expected_style
    path_form = _infer_form_from_source(str(companion_path), 0)
    catalog_key = _infer_bach_catalog_key(companion_path, path_form) if style == "bach" else None

    if companion_path.suffix.lower() in {".krn", ".mxl", ".xml", ".musicxml"}:
        score = music21.converter.parse(str(companion_path))
        key_root, key_mode = _detect_score_key(score)
        time_signature = _detect_score_time_signature(score)
        score_form, _ = detect_form(score, str(companion_path), style)
        return asdict(
            ReferenceMetadata(
                kind="score",
                path=str(companion_path),
                style=style,
                path_form=path_form,
                score_form=score_form,
                key_root=key_root,
                key_mode=key_mode,
                key_name=f"{pc_to_note_name(key_root)} {key_mode}",
                time_signature=time_signature,
                meter=METER_MAP.get(time_signature),
            )
        )

    if companion_path.suffix.lower() == ".mid":
        mid = load_midi(companion_path)
        time_signature = _midi_time_signature(mid)
        path_form = path_form or _infer_form_from_source(str(companion_path), len(mid.tracks))
        if catalog_key is not None:
            return asdict(
                ReferenceMetadata(
                    kind="catalog",
                    path=str(companion_path),
                    style=style,
                    path_form=path_form,
                    key_root=catalog_key[0],
                    key_mode=catalog_key[1],
                    key_name=f"{pc_to_note_name(catalog_key[0])} {catalog_key[1]}",
                    time_signature=time_signature,
                    meter=METER_MAP.get(time_signature),
                )
            )
        key_sig = midi_key_signature(mid)
        parsed_key = parse_midi_key_signature(key_sig or "")
        return asdict(
            ReferenceMetadata(
                kind="midi",
                path=str(companion_path),
                style=style,
                path_form=path_form,
                key_root=parsed_key[0] if parsed_key else None,
                key_mode=parsed_key[1] if parsed_key else None,
                key_name=(
                    f"{pc_to_note_name(parsed_key[0])} {parsed_key[1]}"
                    if parsed_key
                    else None
                ),
                time_signature=time_signature,
                meter=METER_MAP.get(time_signature),
            )
        )

    return asdict(
        ReferenceMetadata(
            kind="unknown",
            path=str(companion_path),
            style=style,
            path_form=path_form,
        )
    )


def _load_reference_metadata(path: Path, expected_style: str) -> ReferenceMetadata | None:
    companion = _resolve_companion_source(path)
    hinted_key = parse_explicit_key_from_text(path.stem)
    if companion is not None:
        try:
            reference = ReferenceMetadata(**_load_reference_metadata_cached(str(companion), expected_style))
            if (
                reference.kind == "midi"
                and hinted_key is not None
            ):
                path_form = reference.path_form or _infer_form_from_source(str(path), 0)
                return ReferenceMetadata(
                    kind="filename",
                    path=str(path),
                    style=expected_style,
                    path_form=path_form,
                    key_root=hinted_key[0],
                    key_mode=hinted_key[1],
                    key_name=f"{pc_to_note_name(hinted_key[0])} {hinted_key[1]}",
                    time_signature=reference.time_signature,
                    meter=reference.meter,
                )
            if reference.key_root is not None and reference.key_mode is not None:
                return reference
        except Exception:
            pass

    if hinted_key is None:
        return None

    path_form = _infer_form_from_source(str(path), 0)
    return ReferenceMetadata(
        kind="filename",
        path=str(path),
        style=expected_style,
        path_form=path_form,
        key_root=hinted_key[0],
        key_mode=hinted_key[1],
        key_name=f"{pc_to_note_name(hinted_key[0])} {hinted_key[1]}",
    )


def _load_comp(path: Path) -> tuple[VoiceComposition, str, tuple[int, int], float]:
    mid = load_midi(path)
    tracks = midi_to_note_events(mid)
    voices = [voice for voice in tracks if voice]
    if len(voices) < 2 and len(tracks) == 1 and tracks[0]:
        all_notes = tracks[0]
        median_pitch = np.median([note[2] for note in all_notes])
        upper = [(s, d, p) for s, d, p in all_notes if p >= median_pitch]
        lower = [(s, d, p) for s, d, p in all_notes if p < median_pitch]
        voices = [upper, lower]
    if len(voices) < 2:
        raise ValueError(f"{path} does not contain at least two non-empty voices")

    time_signature = _midi_time_signature(mid)
    key_root, key_mode, key_corr, _key_source = detect_composition_key(
        voices,
        time_signature=time_signature,
        midi_key_signature=midi_key_signature(mid),
        style=_infer_style_from_path(path),
        source_key_hint=parse_explicit_key_from_text(path.stem),
    )
    key_name = f"{pc_to_note_name(key_root)} {key_mode}"
    return (
        VoiceComposition(
            voices=voices,
            key_root=key_root,
            key_mode=key_mode,
            source=str(path),
            style=_infer_style_from_path(path),
            time_signature=time_signature,
        ),
        key_name,
        time_signature,
        float(key_corr),
    )


def _audit_desc_for_path(path: Path) -> str:
    """Build the same source description used by ``prepare-data`` local ingestion."""
    resolved = path.resolve()
    for root in (DATA_MIDI_ALL_ROOT.resolve(), DATA_MIDI_ROOT.resolve()):
        try:
            return str(resolved.relative_to(root)).rsplit(".", 1)[0]
        except ValueError:
            continue
    return path.stem


def _load_comp_prepare_data(path: Path) -> tuple[VoiceComposition, str, tuple[int, int], float] | None:
    """Load a composition as the local ``prepare-data`` pipeline would."""
    from bach_gen.data.corpus import _parse_and_extract_file_entry

    style = _infer_style_from_path(path)
    desc = _audit_desc_for_path(path)
    results = _parse_and_extract_file_entry((str(path), desc, style, 4, 1, None))
    if not results:
        return None
    comp, _form = results[0]
    key_name = f"{pc_to_note_name(comp.key_root)} {comp.key_mode}"
    return comp, key_name, comp.time_signature, 1.0


def build_conditioning_audit_rows(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    forms: set[str] | None = None,
    groups: set[str] | None = None,
    max_per_group: int | None = None,
    thresholds_path: Path | None = None,
    pipeline: str = "midi_eval",
) -> list[dict[str, Any]]:
    """Return per-piece conditioning-label audit rows."""
    if pipeline not in {"midi_eval", "prepare_data"}:
        raise ValueError(f"unknown audit pipeline: {pipeline}")

    manifest = _load_manifest(manifest_path)
    selected_groups = manifest.get("groups", [])
    if forms:
        selected_groups = [group for group in selected_groups if group["form"] in forms]
    if groups:
        selected_groups = [group for group in selected_groups if group["name"] in groups]

    thresholds = load_conditioning_thresholds(thresholds_path)
    rows: list[dict[str, Any]] = []
    for group in selected_groups:
        files = _expand_group_files(group)
        if max_per_group is not None:
            files = files[:max_per_group]
        for path in files:
            if pipeline == "prepare_data":
                loaded = _load_comp_prepare_data(path)
                if loaded is None:
                    continue
                comp, key_name, time_signature, key_corr = loaded
            else:
                comp, key_name, time_signature, key_corr = _load_comp(path)
            labels = analyze_composition(comp, form=group["form"], thresholds=thresholds)
            features = conditioning_feature_dict(comp, form=group["form"], thresholds=thresholds)
            cadence_events = detect_cadence_events(comp, form=group["form"])
            mode_diag = detect_mode_family(comp.voices, time_signature=time_signature)
            subject_entries = (
                detect_subject_entries(comp, form=group["form"], thresholds=thresholds)
                if group["form"] in {"fugue", "invention", "sinfonia"}
                else []
            )

            expected_style = _infer_style_from_path(path)
            path_form = _infer_form_from_source(str(path), comp.num_voices)
            reference = _load_reference_metadata(path, expected_style)
            measure_count = compute_measure_count(comp.voices, time_signature)
            subject_density = (len(subject_entries) * 10.0 / max(1, measure_count))
            exposition_count = sum(1 for entry in subject_entries if entry.is_exposition)

            row: dict[str, Any] = {
                "group": group["name"],
                "form": group["form"],
                "file": str(path),
                "analysis_pipeline": pipeline,
                "style": expected_style,
                "key": key_name,
                "key_correlation": key_corr,
                "time_signature": f"{time_signature[0]}/{time_signature[1]}",
                "meter": METER_MAP.get(time_signature, "unknown"),
                "length_bars": measure_count,
                "length_bucket": length_bucket(measure_count),
                "num_voices": comp.num_voices,
                "path_form": path_form or "",
                "path_form_match": int(path_form == group["form"]) if path_form else "",
                "texture": labels["texture"],
                "imitation": labels["imitation"],
                "harmonic_rhythm": labels["harmonic_rhythm"],
                "harmonic_tension": labels["harmonic_tension"],
                "chromaticism": labels["chromaticism"],
                "cadence_count": len(cadence_events),
                "cadence_density_per_100_bars": len(cadence_events) * 100.0 / max(1, measure_count),
                "cadence_types": ",".join(event.token_name for event in cadence_events),
                "subject_entry_count": len(subject_entries),
                "subject_entry_density_per_10_bars": subject_density,
                "subject_exposition_count": exposition_count,
                "has_subject_entries": int(bool(subject_entries)),
                "mode_family": mode_diag.mode_family,
                "mode_system": mode_diag.system,
                "mode_confidence": mode_diag.confidence,
                "reference_kind": reference.kind if reference else "",
                "reference_path": reference.path if reference else "",
                "reference_style": reference.style if reference else "",
                "reference_key": reference.key_name or "" if reference else "",
                "reference_time_signature": (
                    f"{reference.time_signature[0]}/{reference.time_signature[1]}"
                    if reference and reference.time_signature is not None
                    else ""
                ),
                "reference_meter": reference.meter or "" if reference else "",
                "reference_path_form": reference.path_form or "" if reference else "",
                "reference_score_form": reference.score_form or "" if reference else "",
                "key_reference_available": int(bool(reference and reference.key_root is not None)),
                "meter_reference_available": int(bool(reference and reference.time_signature is not None)),
                "score_form_reference_available": int(bool(reference and reference.score_form)),
                "key_match_strict": (
                    int(reference.key_root == comp.key_root and reference.key_mode == comp.key_mode)
                    if reference and reference.key_root is not None and reference.key_mode is not None
                    else ""
                ),
                "key_match_mode": (
                    int(reference.key_mode == comp.key_mode)
                    if reference and reference.key_mode is not None
                    else ""
                ),
                "meter_match": (
                    int(reference.time_signature == time_signature)
                    if reference and reference.time_signature is not None
                    else ""
                ),
                "reference_path_form_match": (
                    int(reference.path_form == group["form"])
                    if reference and reference.path_form
                    else ""
                ),
                "reference_score_form_match": (
                    int(reference.score_form == group["form"])
                    if reference and reference.score_form
                    else ""
                ),
                "reference_style_match": (
                    int(reference.style == expected_style)
                    if reference and reference.style
                    else ""
                ),
            }
            row.update(features)
            rows.append(row)
    return rows


def _match_rate(form_rows: list[dict[str, Any]], field: str) -> float | None:
    available = [int(row[field]) for row in form_rows if str(row.get(field, "")) != ""]
    if not available:
        return None
    return float(sum(available) / len(available))


def _summarize_bucket_rows(bucket_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(bucket_rows)
    texture_counts = Counter(str(row["texture"]) for row in bucket_rows)
    imitation_counts = Counter(str(row["imitation"]) for row in bucket_rows)
    harmonic_rhythm_counts = Counter(str(row["harmonic_rhythm"]) for row in bucket_rows)
    harmonic_tension_counts = Counter(str(row["harmonic_tension"]) for row in bucket_rows)
    chromaticism_counts = Counter(str(row["chromaticism"]) for row in bucket_rows)
    mode_system_counts = Counter(str(row.get("mode_system", "")) for row in bucket_rows if str(row.get("mode_system", "")))
    chromaticism_ref_counts = Counter(
        str(row.get("chromaticism_reference_system", ""))
        for row in bucket_rows
        if str(row.get("chromaticism_reference_system", ""))
    )
    meter_counts = Counter(str(row["meter"]) for row in bucket_rows)
    length_bucket_counts = Counter(str(row["length_bucket"]) for row in bucket_rows)
    style_counts = Counter(str(row["style"]) for row in bucket_rows)
    cadence_counts = [int(row["cadence_count"]) for row in bucket_rows]
    cadence_density = [float(row["cadence_density_per_100_bars"]) for row in bucket_rows]
    subject_counts = [int(row["subject_entry_count"]) for row in bucket_rows]
    subject_density = [float(row["subject_entry_density_per_10_bars"]) for row in bucket_rows]
    feature_means = {
        key: float(mean(float(row[key]) for row in bucket_rows))
        for key in [
            "texture_polyphony_score",
            "texture_sync_ratio",
            "texture_active_overlap",
            "imitation_match_density",
            "harmonic_rhythm_score",
            "harmonic_tension_score",
            "chromaticism_ratio",
            "mode_confidence",
        ]
        if all(str(row.get(key, "")) != "" for row in bucket_rows)
    }
    coverage = float(sum(1 for count in subject_counts if count > 0) / n) if n else 0.0
    path_form_match_rate = _match_rate(bucket_rows, "path_form_match")
    score_form_match_rate = _match_rate(bucket_rows, "reference_score_form_match")
    meter_match_rate = _match_rate(bucket_rows, "meter_match")
    key_strict_match_rate = _match_rate(bucket_rows, "key_match_strict")
    key_mode_match_rate = _match_rate(bucket_rows, "key_match_mode")
    style_match_rate = _match_rate(bucket_rows, "reference_style_match")

    return {
        "n": n,
        "style": dict(style_counts),
        "meter": dict(meter_counts),
        "length_bucket": dict(length_bucket_counts),
        "texture": dict(texture_counts),
        "imitation": dict(imitation_counts),
        "harmonic_rhythm": dict(harmonic_rhythm_counts),
        "harmonic_tension": dict(harmonic_tension_counts),
        "chromaticism": dict(chromaticism_counts),
        "mode_system": dict(mode_system_counts),
        "chromaticism_reference_system": dict(chromaticism_ref_counts),
        "form_path_match_rate": path_form_match_rate,
        "form_score_match_rate": score_form_match_rate,
        "style_match_rate": style_match_rate,
        "key_strict_match_rate": key_strict_match_rate,
        "key_mode_match_rate": key_mode_match_rate,
        "meter_match_rate": meter_match_rate,
        "cadence_count": {
            "mean": float(mean(cadence_counts)) if cadence_counts else 0.0,
            "median": float(median(cadence_counts)) if cadence_counts else 0.0,
            "max": max(cadence_counts) if cadence_counts else 0,
        },
        "cadence_density_per_100_bars": {
            "mean": float(mean(cadence_density)) if cadence_density else 0.0,
            "median": float(median(cadence_density)) if cadence_density else 0.0,
        },
        "subject_entries": {
            "mean": float(mean(subject_counts)) if subject_counts else 0.0,
            "median": float(median(subject_counts)) if subject_counts else 0.0,
            "coverage": coverage,
            "density_per_10_bars_mean": float(mean(subject_density)) if subject_density else 0.0,
        },
        "feature_means": feature_means,
    }


def _evaluate_expectations(bucket_summary: dict[str, Any], expectations: dict[str, Any]) -> list[str]:
    n = max(1, int(bucket_summary["n"]))
    texture = bucket_summary["texture"]
    imitation = bucket_summary["imitation"]
    harmonic_rhythm = bucket_summary["harmonic_rhythm"]
    harmonic_tension = bucket_summary["harmonic_tension"]
    chromaticism = bucket_summary["chromaticism"]
    mode_system = bucket_summary.get("mode_system", {})

    non_homophonic = (texture.get("polyphonic", 0) + texture.get("mixed", 0)) / n
    homophonic = texture.get("homophonic", 0) / n
    high_imitation = imitation.get("high", 0) / n
    none_imitation = imitation.get("none", 0) / n
    high_chromaticism = chromaticism.get("high", 0) / n
    modal_fraction = mode_system.get("modal", 0) / n
    collapsed_hr = max(harmonic_rhythm.values(), default=0) / n
    collapsed_ht = max(harmonic_tension.values(), default=0) / n

    violations: list[str] = []
    if "non_homophonic_texture_min" in expectations and non_homophonic < expectations["non_homophonic_texture_min"]:
        violations.append(
            f"texture non-homophonic fraction {non_homophonic:.2f} < {expectations['non_homophonic_texture_min']:.2f}"
        )
    if "texture_non_homophonic_min" in expectations and non_homophonic < expectations["texture_non_homophonic_min"]:
        violations.append(
            f"texture non-homophonic fraction {non_homophonic:.2f} < {expectations['texture_non_homophonic_min']:.2f}"
        )
    if "homophonic_texture_min" in expectations and homophonic < expectations["homophonic_texture_min"]:
        violations.append(
            f"texture homophonic fraction {homophonic:.2f} < {expectations['homophonic_texture_min']:.2f}"
        )
    if "high_imitation_min" in expectations and high_imitation < expectations["high_imitation_min"]:
        violations.append(
            f"high imitation fraction {high_imitation:.2f} < {expectations['high_imitation_min']:.2f}"
        )
    if "none_imitation_min" in expectations and none_imitation < expectations["none_imitation_min"]:
        violations.append(
            f"none imitation fraction {none_imitation:.2f} < {expectations['none_imitation_min']:.2f}"
        )
    if "imitation_none_max" in expectations and none_imitation > expectations["imitation_none_max"]:
        violations.append(
            f"none imitation fraction {none_imitation:.2f} > {expectations['imitation_none_max']:.2f}"
        )
    if "imitation_high_max" in expectations and high_imitation > expectations["imitation_high_max"]:
        violations.append(
            f"high imitation fraction {high_imitation:.2f} > {expectations['imitation_high_max']:.2f}"
        )
    if (
        "harmonic_rhythm_collapsed_max" in expectations
        and collapsed_hr > expectations["harmonic_rhythm_collapsed_max"]
    ):
        violations.append(
            "harmonic rhythm collapsed into one bucket "
            f"({collapsed_hr:.2f} > {expectations['harmonic_rhythm_collapsed_max']:.2f})"
        )
    if (
        "harmonic_tension_collapsed_max" in expectations
        and collapsed_ht > expectations["harmonic_tension_collapsed_max"]
    ):
        violations.append(
            "harmonic tension collapsed into one bucket "
            f"({collapsed_ht:.2f} > {expectations['harmonic_tension_collapsed_max']:.2f})"
        )
    if "cadence_mean_min" in expectations and bucket_summary["cadence_count"]["mean"] < expectations["cadence_mean_min"]:
        violations.append(
            f"cadence mean {bucket_summary['cadence_count']['mean']:.2f} < {expectations['cadence_mean_min']:.2f}"
        )
    if "cadence_mean_max" in expectations and bucket_summary["cadence_count"]["mean"] > expectations["cadence_mean_max"]:
        violations.append(
            f"cadence mean {bucket_summary['cadence_count']['mean']:.2f} > {expectations['cadence_mean_max']:.2f}"
        )
    if "cadence_median_min" in expectations and bucket_summary["cadence_count"]["median"] < expectations["cadence_median_min"]:
        violations.append(
            f"cadence median {bucket_summary['cadence_count']['median']:.2f} < {expectations['cadence_median_min']:.2f}"
        )
    if "subject_coverage_min" in expectations and bucket_summary["subject_entries"]["coverage"] < expectations["subject_coverage_min"]:
        violations.append(
            f"subject coverage {bucket_summary['subject_entries']['coverage']:.2f} < {expectations['subject_coverage_min']:.2f}"
        )
    if (
        "form_path_match_min" in expectations
        and bucket_summary["form_path_match_rate"] is not None
        and bucket_summary["form_path_match_rate"] < expectations["form_path_match_min"]
    ):
        violations.append(
            f"path-form match rate {bucket_summary['form_path_match_rate']:.2f} < {expectations['form_path_match_min']:.2f}"
        )
    if (
        "key_strict_match_min" in expectations
        and bucket_summary["key_strict_match_rate"] is not None
        and bucket_summary["key_strict_match_rate"] < expectations["key_strict_match_min"]
    ):
        violations.append(
            f"key strict match rate {bucket_summary['key_strict_match_rate']:.2f} < {expectations['key_strict_match_min']:.2f}"
        )
    if "mode_system_modal_min" in expectations and modal_fraction < expectations["mode_system_modal_min"]:
        violations.append(
            f"modal system fraction {modal_fraction:.2f} < {expectations['mode_system_modal_min']:.2f}"
        )
    if "chromaticism_high_max" in expectations and high_chromaticism > expectations["chromaticism_high_max"]:
        violations.append(
            f"high chromaticism fraction {high_chromaticism:.2f} > {expectations['chromaticism_high_max']:.2f}"
        )
    return violations


def summarize_conditioning_audit(
    rows: list[dict[str, Any]],
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Summarize label distributions and audit violations by form and group."""
    summary: dict[str, Any] = {"forms": {}, "groups": {}, "violations": []}
    manifest = _load_manifest(manifest_path) if manifest_path is not None else None
    expectations_by_group: dict[str, dict[str, Any]] = {
        str(group["name"]): dict(group.get("audit_expectations", {}))
        for group in manifest.get("groups", [])
    } if manifest else {}

    rows_by_form: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_form[str(row["form"])].append(row)
        rows_by_group[str(row["group"])].append(row)

    for form, form_rows in sorted(rows_by_form.items()):
        form_summary = _summarize_bucket_rows(form_rows)
        priors = EXPECTED_PRIORS.get(form, {})
        violations = _evaluate_expectations(form_summary, priors)
        form_summary["violations"] = violations
        summary["forms"][form] = form_summary
        for violation in violations:
            summary["violations"].append({"scope": "form", "name": form, "message": violation})

    for group, group_rows in sorted(rows_by_group.items()):
        group_summary = _summarize_bucket_rows(group_rows)
        group_summary["form"] = str(group_rows[0]["form"]) if group_rows else ""
        expectations = expectations_by_group.get(group, {})
        violations = _evaluate_expectations(group_summary, expectations)
        group_summary["violations"] = violations
        summary["groups"][group] = group_summary
        for violation in violations:
            summary["violations"].append({"scope": "group", "name": group, "message": violation})

    return summary


def write_conditioning_audit(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    out_csv: Path | None = None,
    out_json: Path | None = None,
) -> None:
    """Write audit results to disk."""
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w") as f:
            json.dump(summary, f, indent=2)


def render_conditioning_audit_summary(summary: dict[str, Any]) -> str:
    """Return a concise human-readable audit summary."""
    lines = ["Conditioning label audit"]
    for form, form_summary in summary.get("forms", {}).items():
        texture = form_summary["texture"]
        imitation = form_summary["imitation"]
        cadence = form_summary["cadence_count"]
        subject = form_summary["subject_entries"]
        lines.append(
            (
                f"{form}: n={form_summary['n']} "
                f"texture={texture} "
                f"imitation={imitation} "
                f"cadence(mean={cadence['mean']:.2f}, median={cadence['median']:.2f}) "
                f"subject_coverage={subject['coverage']:.2f}"
            )
        )
        extra = []
        if form_summary.get("form_path_match_rate") is not None:
            extra.append(f"path_form={form_summary['form_path_match_rate']:.2f}")
        if form_summary.get("form_score_match_rate") is not None:
            extra.append(f"score_form={form_summary['form_score_match_rate']:.2f}")
        if form_summary.get("meter_match_rate") is not None:
            extra.append(f"meter_ref={form_summary['meter_match_rate']:.2f}")
        if form_summary.get("key_strict_match_rate") is not None:
            extra.append(f"key_ref={form_summary['key_strict_match_rate']:.2f}")
        if extra:
            lines.append("  " + " ".join(extra))
        for violation in form_summary.get("violations", []):
            lines.append(f"  violation: {violation}")
    if summary.get("groups"):
        lines.append("Groups")
        for group, group_summary in summary.get("groups", {}).items():
            cadence = group_summary["cadence_count"]
            mode_system = group_summary.get("mode_system", {})
            extra = [
                f"form={group_summary.get('form', '')}",
                f"n={group_summary['n']}",
                f"cadence_mean={cadence['mean']:.2f}",
            ]
            if mode_system:
                extra.append(f"mode_system={mode_system}")
            lines.append(f"{group}: " + " ".join(extra))
            for violation in group_summary.get("violations", []):
                lines.append(f"  violation: {violation}")
    if not summary.get("violations"):
        lines.append("No prior violations.")
    return "\n".join(lines)
