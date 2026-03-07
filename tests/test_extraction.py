from __future__ import annotations

from music21 import key, meter, note, stream

from bach_gen.data.extraction import _detect_key, _part_to_notes
from bach_gen.utils.constants import TICKS_PER_QUARTER


def test_detect_key_prefers_explicit_key_metadata() -> None:
    score = stream.Score()
    part = stream.Part()
    part.insert(0, key.Key("E", "major"))
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, note.Note("E4", quarterLength=1.0))
    part.insert(1, note.Note("G#4", quarterLength=1.0))
    score.insert(0, part)

    # If explicit key metadata is ignored, this test should fail loudly.
    score.analyze = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected analyze()"))  # type: ignore[attr-defined]

    assert _detect_key(score) == (4, "major")


def test_part_to_notes_uses_absolute_offsets_after_flatten() -> None:
    part = stream.Part()
    measure_1 = stream.Measure(number=1)
    measure_1.insert(0, note.Note("C4", quarterLength=1.0))
    measure_1.insert(2, note.Note("D4", quarterLength=1.0))
    measure_2 = stream.Measure(number=2)
    measure_2.insert(0, note.Note("E4", quarterLength=1.0))
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, measure_1)
    part.insert(4, measure_2)

    notes = _part_to_notes(part)

    assert notes == [
        (0 * TICKS_PER_QUARTER, 1 * TICKS_PER_QUARTER, 60),
        (2 * TICKS_PER_QUARTER, 1 * TICKS_PER_QUARTER, 62),
        (4 * TICKS_PER_QUARTER, 1 * TICKS_PER_QUARTER, 64),
    ]
