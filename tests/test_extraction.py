from __future__ import annotations

from music21 import key, meter, note, stream

from bach_gen.data.extraction import _detect_key, _part_to_notes, detect_form
from bach_gen.utils.constants import TICKS_PER_QUARTER


def _score_with_n_parts(n_parts: int) -> stream.Score:
    score = stream.Score()
    for idx in range(n_parts):
        part = stream.Part()
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, note.Note(60 + idx, quarterLength=1.0))
        score.insert(0, part)
    return score


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


def test_detect_form_uses_keyboard_piece_for_romantic_etude_fallback() -> None:
    form, voices = detect_form(
        _score_with_n_parts(2),
        "data/midi/all/liszt/kernscores__etude10-02.krn.fromscore.mid",
        "romantic",
    )
    assert (form, voices) == ("keyboard_piece", 2)


def test_detect_form_uses_orchestral_reduction_for_reduced_symphony() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/liszt/kunstderfuge__beethoven-liszt_symphony_9_2_(c)pepperdine.mid",
        "romantic",
    )
    assert (form, voices) == ("orchestral_reduction", 4)


def test_detect_form_does_not_treat_classical_sinfonia_as_bach_sinfonia() -> None:
    form, voices = detect_form(
        _score_with_n_parts(3),
        "data/midi/all/mozart/kunstderfuge__sinfonia_40_550_1_hisamori.mid",
        "classical",
    )
    assert (form, voices) == ("orchestral_reduction", 3)


def test_detect_form_keeps_bach_three_part_sinfonia_family() -> None:
    form, voices = detect_form(
        _score_with_n_parts(3),
        "data/midi/all/bach/kunstderfuge__sinfonias_bwv-788_(c)sankey.mid",
        "bach",
    )
    assert (form, voices) == ("sinfonia", 3)


def test_detect_form_keeps_numbered_bach_chorales_as_strict_chorales() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/bach/kunstderfuge__chorales_013701b_(c)greentree.mid",
        "bach",
    )
    assert (form, voices) == ("chorale", 4)


def test_detect_form_moves_chorale_preludes_to_keyboard_piece() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/buxtehude/kunstderfuge__chorale-preludes_buxwv-188_(mccloskey).mid",
        "baroque",
    )
    assert (form, voices) == ("keyboard_piece", 4)


def test_detect_form_moves_schubler_chorales_to_keyboard_piece() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/bach/kunstderfuge__organ_schubler_chorales_bwv-650_(c)fenske.mid",
        "bach",
    )
    assert (form, voices) == ("keyboard_piece", 4)


def test_detect_form_uses_vocal_polyphony_for_renaissance_generic_fallback() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/byrd/kernscores__aveverum.krn.fromscore.mid",
        "renaissance",
    )
    assert (form, voices) == ("vocal_polyphony", 4)


def test_detect_form_uses_vocal_polyphony_for_renaissance_missa() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/josquin/jrp__Jos0301a-Missa_Ave_maris_stella-Kyrie.krn.fromscore.mid",
        "renaissance",
    )
    assert (form, voices) == ("vocal_polyphony", 4)


def test_detect_form_does_not_force_brahms_dir_to_quartet() -> None:
    form, voices = detect_form(
        _score_with_n_parts(4),
        "data/midi/all/brahms/kunstderfuge__hungarian_dances_book_1_no_5_(c)yogore.mid",
        "romantic",
    )
    assert (form, voices) == ("chamber_piece", 4)
