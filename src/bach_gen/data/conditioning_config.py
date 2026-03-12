"""Shared configuration helpers for conditioning-label analysis."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "data/conditioning_thresholds.json"


DEFAULT_CONDITIONING_THRESHOLDS: dict[str, Any] = {
    "texture_weights": {
        "sync_inverse": 0.35,
        "stagger": 0.30,
        "active_overlap": 0.20,
        "shared_inverse": 0.15,
    },
    "forms": {
        "default": {
            "texture": {
                "polyphonic_min": 0.50,
                "mixed_min": 0.33,
                "shared_voice_fraction": 0.75,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 2.60, "moderate_max": 3.40},
            "harmonic_tension": {"low_max": 0.12, "moderate_max": 0.19},
            "chromaticism": {"low_max": 0.05, "moderate_max": 0.15},
            "cadence": {
                "min_confidence": 2.0,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.5,
                "min_spacing_measures": 1.0,
                "max_events_per_32_measures": 32,
                "max_events_per_100_bars": 100.0,
                "require_phrase_support": False,
            },
            "subject": {
                "min_match_ratio": 0.70,
                "min_quality": 0.80,
                "min_notes": 4,
                "min_same_voice_spacing_ratio": 0.50,
                "min_span_ratio": 0.50,
                "max_span_ratio": 2.50,
                "default_bars": 2,
                "max_bars": 4,
                "min_notes_before_gap_break": 5,
                "min_span_quarters_before_gap_break": 4.0,
            },
        },
        "chorale": {
            "texture": {
                "polyphonic_min": 0.60,
                "mixed_min": 0.52,
                "shared_voice_fraction": 0.75,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 2.9, "moderate_max": 3.45},
            "harmonic_tension": {"low_max": 0.12, "moderate_max": 0.18},
            "chromaticism": {"low_max": 0.04, "moderate_max": 0.10},
            "cadence": {
                "min_confidence": 1.5,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.75,
                "min_spacing_measures": 1.0,
                "max_events_per_32_measures": 24,
                "max_events_per_100_bars": 100.0,
                "require_phrase_support": False,
            },
        },
        "fugue": {
            "texture": {
                "polyphonic_min": 0.52,
                "mixed_min": 0.34,
                "shared_voice_fraction": 0.75,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 2,
                "subject_floor_density_min": 0.75,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 3.25, "moderate_max": 3.65},
            "harmonic_tension": {"low_max": 0.13, "moderate_max": 0.19},
            "chromaticism": {"low_max": 0.05, "moderate_max": 0.14},
            "cadence": {
                "min_confidence": 2.0,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.5,
                "min_spacing_measures": 1.0,
                "max_events_per_32_measures": 32,
                "max_events_per_100_bars": 100.0,
                "require_phrase_support": False,
            },
            "subject": {
                "min_match_ratio": 0.70,
                "min_quality": 0.80,
                "min_same_voice_spacing_ratio": 0.55,
            },
        },
        "invention": {
            "texture": {
                "polyphonic_min": 0.36,
                "mixed_min": 0.24,
                "shared_voice_fraction": 1.0,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 2,
                "subject_floor_density_min": 0.75,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 3.35, "moderate_max": 3.70},
            "harmonic_tension": {"low_max": 0.12, "moderate_max": 0.18},
            "chromaticism": {"low_max": 0.04, "moderate_max": 0.12},
            "cadence": {
                "min_confidence": 1.8,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.5,
                "min_spacing_measures": 1.0,
                "max_events_per_32_measures": 32,
                "max_events_per_100_bars": 100.0,
                "require_phrase_support": False,
            },
            "subject": {
                "min_match_ratio": 0.72,
                "min_quality": 0.82,
                "min_same_voice_spacing_ratio": 0.60,
                "min_span_ratio": 0.55,
                "max_span_ratio": 2.25,
            },
        },
        "sinfonia": {
            "texture": {
                "polyphonic_min": 0.56,
                "mixed_min": 0.36,
                "shared_voice_fraction": 0.67,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 2,
                "subject_floor_density_min": 0.75,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 3.45, "moderate_max": 3.74},
            "harmonic_tension": {"low_max": 0.13, "moderate_max": 0.20},
            "chromaticism": {"low_max": 0.05, "moderate_max": 0.13},
            "cadence": {
                "min_confidence": 1.8,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.5,
                "min_spacing_measures": 1.0,
                "max_events_per_32_measures": 32,
                "max_events_per_100_bars": 100.0,
                "require_phrase_support": False,
            },
            "subject": {
                "min_match_ratio": 0.75,
                "min_quality": 0.85,
                "min_same_voice_spacing_ratio": 0.80,
                "min_span_ratio": 0.65,
                "max_span_ratio": 1.80,
                "min_notes_before_gap_break": 6,
            },
        },
        "2-part": {
            "texture": {
                "polyphonic_min": 0.48,
                "mixed_min": 0.30,
                "shared_voice_fraction": 1.0,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 2,
                "subject_floor_density_min": 0.75,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 2.5, "moderate_max": 3.6},
            "harmonic_tension": {"low_max": 0.12, "moderate_max": 0.18},
            "chromaticism": {"low_max": 0.04, "moderate_max": 0.12},
            "cadence": {
                "min_confidence": 1.8,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.5,
                "min_spacing_measures": 1.0,
                "max_events_per_32_measures": 32,
                "max_events_per_100_bars": 100.0,
                "require_phrase_support": False,
            },
        },
        "motet": {
            "texture": {
                "polyphonic_min": 0.44,
                "mixed_min": 0.28,
                "shared_voice_fraction": 0.60,
            },
            "imitation": {
                "low_min": 0.07,
                "high_min": 0.22,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 2.40, "moderate_max": 3.20},
            "harmonic_tension": {"low_max": 0.10, "moderate_max": 0.16},
            "chromaticism": {"low_max": 0.08, "moderate_max": 0.18},
            "cadence": {
                "min_confidence": 2.1,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.75,
                "min_spacing_measures": 1.5,
                "max_events_per_32_measures": 8,
                "max_events_per_100_bars": 8.0,
                "require_phrase_support": True,
            },
        },
        "trio_sonata": {
            "texture": {
                "polyphonic_min": 0.42,
                "mixed_min": 0.28,
                "shared_voice_fraction": 0.67,
            },
            "imitation": {
                "low_min": 0.09,
                "high_min": 0.26,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 2.50, "moderate_max": 3.50},
            "harmonic_tension": {"low_max": 0.11, "moderate_max": 0.18},
            "chromaticism": {"low_max": 0.05, "moderate_max": 0.14},
            "cadence": {
                "min_confidence": 2.0,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.5,
                "min_spacing_measures": 1.5,
                "max_events_per_32_measures": 10,
                "max_events_per_100_bars": 10.0,
                "require_phrase_support": False,
            },
        },
        "quartet": {
            "texture": {
                "polyphonic_min": 0.48,
                "mixed_min": 0.30,
                "shared_voice_fraction": 0.75,
            },
            "imitation": {
                "low_min": 0.12,
                "high_min": 0.34,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 24.0,
            },
            "harmonic_rhythm": {"slow_max": 2.30, "moderate_max": 3.10},
            "harmonic_tension": {"low_max": 0.10, "moderate_max": 0.17},
            "chromaticism": {"low_max": 0.06, "moderate_max": 0.16},
            "cadence": {
                "min_confidence": 2.6,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 1.0,
                "min_spacing_measures": 2.0,
                "max_events_per_32_measures": 3,
                "max_events_per_100_bars": 1.2,
                "require_phrase_support": True,
            },
        },
        "keyboard_piece": {
            "texture": {
                "polyphonic_min": 0.42,
                "mixed_min": 0.28,
                "shared_voice_fraction": 0.75,
            },
            "imitation": {
                "low_min": 0.10,
                "high_min": 0.30,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 8.0,
            },
            "harmonic_rhythm": {"slow_max": 2.40, "moderate_max": 3.40},
            "harmonic_tension": {"low_max": 0.11, "moderate_max": 0.19},
            "chromaticism": {"low_max": 0.05, "moderate_max": 0.18},
            "cadence": {
                "min_confidence": 2.2,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.75,
                "min_spacing_measures": 1.5,
                "max_events_per_32_measures": 10,
                "max_events_per_100_bars": 8.0,
                "require_phrase_support": False,
            },
        },
        "chamber_piece": {
            "texture": {
                "polyphonic_min": 0.46,
                "mixed_min": 0.30,
                "shared_voice_fraction": 0.72,
            },
            "imitation": {
                "low_min": 0.11,
                "high_min": 0.28,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 16.0,
            },
            "harmonic_rhythm": {"slow_max": 2.35, "moderate_max": 3.20},
            "harmonic_tension": {"low_max": 0.10, "moderate_max": 0.17},
            "chromaticism": {"low_max": 0.05, "moderate_max": 0.16},
            "cadence": {
                "min_confidence": 2.4,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.75,
                "min_spacing_measures": 1.5,
                "max_events_per_32_measures": 8,
                "max_events_per_100_bars": 4.0,
                "require_phrase_support": True,
            },
        },
        "orchestral_reduction": {
            "texture": {
                "polyphonic_min": 0.44,
                "mixed_min": 0.28,
                "shared_voice_fraction": 0.78,
            },
            "imitation": {
                "low_min": 0.12,
                "high_min": 0.30,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 24.0,
            },
            "harmonic_rhythm": {"slow_max": 2.20, "moderate_max": 3.05},
            "harmonic_tension": {"low_max": 0.10, "moderate_max": 0.17},
            "chromaticism": {"low_max": 0.06, "moderate_max": 0.18},
            "cadence": {
                "min_confidence": 2.6,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 1.0,
                "min_spacing_measures": 2.0,
                "max_events_per_32_measures": 6,
                "max_events_per_100_bars": 2.5,
                "require_phrase_support": True,
            },
        },
        "vocal_polyphony": {
            "texture": {
                "polyphonic_min": 0.44,
                "mixed_min": 0.28,
                "shared_voice_fraction": 0.60,
            },
            "imitation": {
                "low_min": 0.07,
                "high_min": 0.22,
                "subject_floor_count_min": 99,
                "subject_floor_density_min": 99.0,
                "length_normalization_notes": 0.0,
            },
            "harmonic_rhythm": {"slow_max": 2.30, "moderate_max": 3.10},
            "harmonic_tension": {"low_max": 0.10, "moderate_max": 0.16},
            "chromaticism": {"low_max": 0.08, "moderate_max": 0.18},
            "cadence": {
                "min_confidence": 2.1,
                "boundary_window_quarters": 1.0,
                "phrase_support_quarters": 0.75,
                "min_spacing_measures": 1.5,
                "max_events_per_32_measures": 8,
                "max_events_per_100_bars": 8.0,
                "require_phrase_support": True,
            },
        },
    },
}


def normalize_conditioning_form(form: str | None) -> str:
    """Normalize form names used by the labeling pipeline."""
    if not form:
        return "default"
    normalized = form.strip().lower()
    if normalized == "2-part":
        return "invention"
    return normalized


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=4)
def _load_conditioning_thresholds_cached(path_str: str | None) -> dict[str, Any]:
    thresholds = deepcopy(DEFAULT_CONDITIONING_THRESHOLDS)
    path = Path(path_str) if path_str else DEFAULT_THRESHOLDS_PATH
    if path.exists():
        with path.open() as f:
            loaded = json.load(f)
        thresholds = _deep_update(thresholds, loaded)
    return thresholds


def load_conditioning_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    """Load conditioning-threshold config from disk with defaults applied."""
    path_str = str(Path(path).resolve()) if path is not None else None
    return deepcopy(_load_conditioning_thresholds_cached(path_str))


def get_form_thresholds(
    form: str | None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fully merged thresholds for one form."""
    all_thresholds = thresholds or load_conditioning_thresholds()
    forms = all_thresholds.get("forms", {})
    base = deepcopy(forms.get("default", {}))
    form_name = normalize_conditioning_form(form)
    if form_name in forms:
        base = _deep_update(base, forms[form_name])
    return base
