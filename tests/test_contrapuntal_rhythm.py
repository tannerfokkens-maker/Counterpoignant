"""Tests for rhythmic complementarity behavior in contrapuntal scoring."""

from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.evaluation.contrapuntal import _score_rhythmic_complementarity, score_contrapuntal


def _voice(onsets: list[int], dur: int = 480, pitch: int = 60) -> list[tuple[int, int, int]]:
    return [(t, dur, pitch) for t in onsets]


def test_rhythmic_complementarity_penalizes_lockstep():
    """Perfectly synchronized onsets should score lower than staggered counterpoint."""
    a = _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=60)
    lockstep_b = _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=67)
    staggered_b = _voice([240, 720, 1200, 1680, 2160, 2640, 3120, 3600], pitch=67)

    lockstep_score = _score_rhythmic_complementarity(a, lockstep_b)
    staggered_score = _score_rhythmic_complementarity(a, staggered_b)

    assert lockstep_score < staggered_score
    assert lockstep_score < 0.65
    assert staggered_score > 0.70


def test_rhythmic_complementarity_penalizes_disconnected_call_response():
    """Voices that barely overlap should not score as strong complementarity."""
    first_half = _voice([0, 480, 960, 1440], pitch=60)
    second_half = _voice([1920, 2400, 2880, 3360], pitch=67)
    staggered = _voice([240, 720, 1200, 1680, 2160, 2640, 3120, 3600], pitch=67)

    disconnected_score = _score_rhythmic_complementarity(first_half, second_half)
    staggered_score = _score_rhythmic_complementarity(first_half + _voice([1920, 2400, 2880, 3360], pitch=60), staggered)

    assert disconnected_score < 0.40
    assert disconnected_score < staggered_score


def test_contrapuntal_onset_staggering_prefers_moderate_staggering():
    """Bell-shaped staggering should prefer moderate overlap over extremes."""
    lockstep_voices = [
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=72),
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=67),
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=64),
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=55),
    ]
    fully_staggered_voices = [
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=72),
        _voice([120, 600, 1080, 1560, 2040, 2520, 3000, 3480], pitch=67),
        _voice([240, 720, 1200, 1680, 2160, 2640, 3120, 3600], pitch=64),
        _voice([360, 840, 1320, 1800, 2280, 2760, 3240, 3720], pitch=55),
    ]
    moderate_voices = [
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=72),
        _voice([0, 480, 960, 1440, 1920, 2400, 2880, 3360], pitch=67),
        _voice([240, 720, 1200, 1680, 2160, 2640, 3120, 3600], pitch=64),
        _voice([240, 720, 1200, 1680, 2160, 2640, 3120, 3600], pitch=55),
    ]

    lockstep_comp = VoiceComposition(voices=lockstep_voices, key_root=0, key_mode="minor", source="lock")
    full_stag_comp = VoiceComposition(voices=fully_staggered_voices, key_root=0, key_mode="minor", source="full-stag")
    moderate_comp = VoiceComposition(voices=moderate_voices, key_root=0, key_mode="minor", source="moderate")

    lock_score, lock_details = score_contrapuntal(lockstep_comp)
    full_stag_score, full_stag_details = score_contrapuntal(full_stag_comp)
    moderate_score, moderate_details = score_contrapuntal(moderate_comp)

    assert lock_details["onset_staggering"] < moderate_details["onset_staggering"]
    assert full_stag_details["onset_staggering"] < moderate_details["onset_staggering"]
    assert lock_score < moderate_score
    assert full_stag_score < moderate_score
