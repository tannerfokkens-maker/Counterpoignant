"""Extract Bach works from music21 corpus.

Loads Bach works from multiple sources:
1. Broad music21 corpus search (composer, title, catalog number)
2. Targeted BWV loading from constants.py lists
3. User-supplied MIDI files in data/midi/

All sources are deduplicated by sourcePath before returning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import music21
from music21 import corpus

from bach_gen.utils.constants import ALL_TARGETED_BWV, DIR_TO_STYLE

logger = logging.getLogger(__name__)


def _search_bach_broad() -> list[tuple[str, music21.stream.Score, str]]:
    """Search music21 corpus broadly for Bach works.

    Searches by composer, title keywords, and catalog number to cast a
    wide net.  Deduplicates by sourcePath so the same file is never
    loaded twice.

    Returns:
        List of (sourcePath, Score, style) tuples.
    """
    seen_paths: set[str] = set()
    results: list[tuple[str, music21.stream.Score, str]] = []

    search_queries = [
        ("bach", "composer"),
        ("bach", "title"),
    ]

    for query, field in search_queries:
        try:
            refs = corpus.search(query, field)
        except Exception as e:
            logger.debug(f"Corpus search ({query!r}, {field!r}) failed: {e}")
            continue

        for ref in refs:
            path = str(ref.sourcePath)
            if path in seen_paths:
                continue
            seen_paths.add(path)

            try:
                score = ref.parse()
                parts = list(score.parts)
                if len(parts) >= 2:
                    results.append((path, score, "bach"))
            except Exception as e:
                logger.debug(f"Could not parse {path}: {e}")

    logger.info(f"Broad search loaded {len(results)} works ({len(seen_paths)} paths seen)")
    return results


def _load_by_bwv(
    bwv_numbers: list[int],
    already_loaded: set[str],
) -> list[tuple[str, music21.stream.Score, str]]:
    """Load specific BWV numbers from music21, skipping already-loaded paths.

    Args:
        bwv_numbers: List of BWV catalog numbers to search for.
        already_loaded: Set of sourcePath strings already loaded.

    Returns:
        List of (sourcePath, Score, style) tuples for newly loaded works.
    """
    results: list[tuple[str, music21.stream.Score, str]] = []

    for bwv in bwv_numbers:
        try:
            refs = corpus.search(str(bwv), "number")
        except Exception as e:
            logger.debug(f"BWV {bwv} search failed: {e}")
            continue

        for ref in refs:
            path = str(ref.sourcePath)
            if path in already_loaded:
                continue
            already_loaded.add(path)

            try:
                score = ref.parse()
                parts = list(score.parts)
                if len(parts) >= 2:
                    results.append((path, score, "bach"))
            except Exception as e:
                logger.debug(f"Could not parse BWV {bwv} ({path}): {e}")

    logger.info(f"BWV-targeted loading added {len(results)} works")
    return results


def get_midi_files(midi_dir: str | Path = "data/midi") -> list[tuple[str, music21.stream.Score, str]]:
    """Load additional score files from a directory.

    Supports all formats music21 can parse: MIDI, MusicXML, Humdrum kern,
    MuseScore, ABC, LilyPond, etc.  Scans recursively so you can organize
    files into subdirectories (e.g. data/midi/bach/, data/midi/handel/).

    Style is inferred from the first subdirectory name under midi_dir using
    DIR_TO_STYLE.  Falls back to empty string if not recognised.

    Returns:
        List of (description, Score, style) tuples.
    """
    midi_path = Path(midi_dir)
    if not midi_path.exists():
        return []

    extensions = [
        "*.mid", "*.midi",           # MIDI
        "*.mxl", "*.musicxml", "*.xml",  # MusicXML (compressed & uncompressed)
        "*.krn",                      # Humdrum kern
        "*.mscz", "*.mscx",          # MuseScore
        "*.abc",                      # ABC notation
        "*.ly",                       # LilyPond
        "*.mei",                      # MEI (Music Encoding Initiative)
        "*.cap", "*.capx",           # Capella
    ]

    files = []
    for ext in extensions:
        files.extend(midi_path.rglob(ext))
    files = sorted(set(files))

    results = []
    for f in files:
        try:
            score = music21.converter.parse(str(f))
            parts = list(score.parts)
            if len(parts) >= 2:
                # Use relative path from midi_dir as description
                desc = str(f.relative_to(midi_path)).rsplit(".", 1)[0]
                # Infer style from directory components using DIR_TO_STYLE
                rel_parts = f.relative_to(midi_path).parts
                style = ""
                for part in rel_parts[:-1]:  # exclude filename
                    dir_name = part.lower()
                    if dir_name in DIR_TO_STYLE:
                        style = DIR_TO_STYLE[dir_name]
                        break
                results.append((desc, score, style))
            else:
                logger.debug(f"Skipping {f.name}: fewer than 2 parts")
        except Exception as e:
            logger.debug(f"Could not parse {f}: {e}")

    if results:
        logger.info(f"Loaded {len(results)} files from {midi_dir} ({len(files)} found, {len(files) - len(results)} skipped)")
    return results


def get_all_works(
    composer_filter: list[str] | None = None,
) -> list[tuple[str, music21.stream.Score, str]]:
    """Load works for training, optionally filtered by composer/style.

    1. Broad corpus search (composer + title)
    2. Targeted BWV loading from ALL_TARGETED_BWV
    3. User-supplied MIDI files from data/midi/

    All sources deduplicated by sourcePath.

    Args:
        composer_filter: If set, only include works whose style matches one
            of these keys (from DIR_TO_STYLE values or keys, e.g.
            ``["bach"]``, ``["baroque", "renaissance"]``).  Matching is
            case-insensitive.  When ``None``, all works are returned.

    Returns:
        List of (description, Score, style) tuples.
    """
    # Normalise filter to lowercase set of accepted styles/composer names
    accepted: set[str] | None = None
    if composer_filter is not None:
        accepted = {c.strip().lower() for c in composer_filter}
        # Expand composer dir names to their style values so that e.g.
        # "buxtehude" matches style "baroque"
        expanded = set(accepted)
        for dir_name, style in DIR_TO_STYLE.items():
            if dir_name in accepted or style in accepted:
                expanded.add(style)
                expanded.add(dir_name)
        accepted = expanded
        logger.info(f"Composer filter active — accepting: {sorted(accepted)}")

    def _style_accepted(style: str) -> bool:
        if accepted is None:
            return True
        return style.lower() in accepted

    # Phase 1: broad search
    works = _search_bach_broad()
    loaded_paths = {path for path, _, _ in works}

    # Phase 2: targeted BWV loading
    bwv_works = _load_by_bwv(ALL_TARGETED_BWV, loaded_paths)
    works.extend(bwv_works)

    # Phase 3: user MIDI files (always additive, no dedup needed)
    works.extend(get_midi_files())

    # Apply composer filter
    if accepted is not None:
        before = len(works)
        works = [(d, s, st) for d, s, st in works if _style_accepted(st)]
        logger.info(f"Composer filter kept {len(works)}/{before} works")

    logger.info(f"Total works loaded: {len(works)}")
    return works


def get_all_bach_works() -> list[tuple[str, music21.stream.Score, str]]:
    """Load all available works for training (unfiltered).

    Convenience wrapper around :func:`get_all_works` for backwards
    compatibility.
    """
    return get_all_works(composer_filter=None)
