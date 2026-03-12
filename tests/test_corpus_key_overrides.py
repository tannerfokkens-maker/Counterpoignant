from __future__ import annotations

from pathlib import Path

import pytest

from bach_gen.data.corpus import (
    _load_companion_overrides,
    _parse_and_extract_file_entry,
    _resolve_companion_source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("midi_relpath", "companion_relpath"),
    [
        (
            "data/midi/all/chopin/kernscores__ballade52.krn.fromscore.mid",
            "data/midi/kernscores/chopin/ballade52.krn",
        ),
        (
            "data/midi/all/liszt/kernscores__ballade2.krn.fromscore.mid",
            "data/midi/kernscores/liszt/ballade2.krn",
        ),
        (
            "data/midi/all/josquin/jrp__Jos0301a-Missa_Ave_maris_stella-Kyrie.krn.fromscore.mid",
            "data/midi/jrp/josquin/Jos0301a-Missa_Ave_maris_stella-Kyrie.krn",
        ),
    ],
)
def test_resolve_companion_source_finds_original_score(
    midi_relpath: str,
    companion_relpath: str,
) -> None:
    midi_path = REPO_ROOT / midi_relpath
    expected = (REPO_ROOT / companion_relpath).resolve()
    assert _resolve_companion_source(midi_path) == expected


@pytest.mark.parametrize(
    ("midi_relpath", "style", "expected_key"),
    [
        ("data/midi/all/chopin/kernscores__ballade52.krn.fromscore.mid", "romantic", (5, "minor")),
        ("data/midi/all/liszt/kernscores__ballade2.krn.fromscore.mid", "romantic", (11, "minor")),
        ("data/midi/all/brahms/kernscores__brahms51-1-3.krn.fromscore.mid", "romantic", (5, "major")),
    ],
)
def test_load_companion_overrides_recovers_score_backed_key(
    midi_relpath: str,
    style: str,
    expected_key: tuple[int, str],
) -> None:
    overrides = _load_companion_overrides(REPO_ROOT / midi_relpath, style)
    assert overrides is not None
    key_override, time_sig_override, _form, _voices = overrides
    assert key_override == expected_key
    assert time_sig_override[0] > 0
    assert time_sig_override[1] > 0


@pytest.mark.parametrize(
    ("midi_relpath", "style", "expected_key"),
    [
        ("data/midi/all/chopin/kernscores__ballade52.krn.fromscore.mid", "romantic", (5, "minor")),
        ("data/midi/all/liszt/kernscores__ballade2.krn.fromscore.mid", "romantic", (11, "minor")),
    ],
)
def test_parse_and_extract_file_entry_uses_companion_key_override(
    midi_relpath: str,
    style: str,
    expected_key: tuple[int, str],
) -> None:
    midi_path = REPO_ROOT / midi_relpath
    desc = str(midi_path.relative_to(REPO_ROOT / "data/midi/all")).rsplit(".", 1)[0]
    results = _parse_and_extract_file_entry((str(midi_path), desc, style, 4, 1, None))
    assert results
    comp, _form = results[0]
    assert (comp.key_root, comp.key_mode) == expected_key
