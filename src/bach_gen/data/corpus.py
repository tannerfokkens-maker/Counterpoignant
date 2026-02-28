"""Extract works from music21 corpus and user-supplied files.

Loads works from multiple sources:
1. Broad music21 corpus search (Bach, Palestrina, Monteverdi, Mozart, etc.)
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


def _search_corpus_broad() -> list[tuple[str, music21.stream.Score, str]]:
    """Search music21 corpus broadly for all relevant composers.

    Searches by composer name to cast a wide net across the built-in
    corpus.  Deduplicates by sourcePath so the same file is never
    loaded twice.

    Returns:
        List of (sourcePath, Score, style) tuples.
    """
    seen_paths: set[str] = set()
    results: list[tuple[str, music21.stream.Score, str]] = []

    # (search_query, field, style)
    CORPUS_SEARCHES = [
        # Bach — composer and title (catches BWV references)
        ("bach", "composer", "bach"),
        ("bach", "title", "bach"),
        # Renaissance
        ("palestrina", "composer", "renaissance"),
        ("monteverdi", "composer", "renaissance"),
        ("josquin", "composer", "renaissance"),
        ("ciconia", "composer", "renaissance"),
        ("luca", "composer", "renaissance"),
        ("trecento", "composer", "renaissance"),
        # Baroque
        ("handel", "composer", "baroque"),
        ("corelli", "composer", "baroque"),
        ("cpebach", "composer", "baroque"),
        # Classical
        ("haydn", "composer", "classical"),
        ("mozart", "composer", "classical"),
        ("beethoven", "composer", "classical"),
        # Romantic
        ("schubert", "composer", "classical"),
        ("schumann_robert", "composer", "classical"),
        ("schumann_clara", "composer", "classical"),
    ]

    for query, field, style in CORPUS_SEARCHES:
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
                    results.append((path, score, style))
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


def _infer_style_from_rel_parts(rel_parts: tuple[str, ...]) -> str:
    """Infer style from directory components of a path relative to `data/midi`."""
    for part in rel_parts[:-1]:  # exclude filename
        style = DIR_TO_STYLE.get(part.lower())
        if style:
            return style
    return ""


def _work_matches_filter(desc: str, style: str, accepted: set[str] | None) -> bool:
    """Return True when a work matches the accepted style/composer filter."""
    if accepted is None:
        return True

    if style and style.lower() in accepted:
        return True

    # Also match explicit directory/name tokens in description path.
    for part in Path(desc).parts:
        if part.lower() in accepted:
            return True
    return False


def get_midi_files(
    midi_dir: str | Path = "data/midi",
    accepted: set[str] | None = None,
) -> list[tuple[str, music21.stream.Score, str]]:
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

    files: dict[Path, str] = {}
    kdf_root = midi_path / "kunstderfuge"
    kdf_bucket = kdf_root / "_voice_buckets" / "dataset_2to4"
    use_kdf_bucket = kdf_bucket.exists()
    if use_kdf_bucket:
        logger.info(
            "Using curated Kunstderfuge bucket (2-4 voices only): %s",
            kdf_bucket,
        )

    kdf_triage: dict[str, dict] = {}
    if kdf_root.exists() and not use_kdf_bucket:
        triage_path = kdf_root / "_triage_report.json"
        if triage_path.exists():
            try:
                import json
                kdf_triage = json.loads(triage_path.read_text())
            except Exception as e:
                logger.warning(f"Could not read {triage_path}: {e}")
        else:
            logger.warning(
                "No Kunstderfuge triage report found at %s; raw Kunstderfuge files will be skipped.",
                triage_path,
            )

    for ext in extensions:
        for f in midi_path.rglob(ext):
            rel = f.relative_to(midi_path)
            rel_parts = rel.parts

            # Ignore generated helper directories by default.
            if any(part.startswith("_") for part in rel_parts[:-1]):
                if not (use_kdf_bucket and f.is_relative_to(kdf_bucket)):
                    continue

            # Kunstderfuge must be restricted to 2-4 voices:
            # prefer curated bucket when present, otherwise consult triage.
            if kdf_root.exists() and f.is_relative_to(kdf_root):
                if use_kdf_bucket:
                    if not f.is_relative_to(kdf_bucket):
                        continue
                else:
                    triage_key = str(f.relative_to(kdf_root))
                    info = kdf_triage.get(triage_key)
                    if not isinstance(info, dict):
                        continue
                    num_tracks = int(info.get("num_tracks", 0))
                    if not (2 <= num_tracks <= 4):
                        continue

            style_guess = _infer_style_from_rel_parts(rel_parts)
            desc_guess = str(rel).rsplit(".", 1)[0]
            if accepted is not None and not _work_matches_filter(desc_guess, style_guess, accepted):
                continue
            files[f] = style_guess

    sorted_files = sorted(files)

    results = []
    for f in sorted_files:
        style = files[f]
        try:
            score = music21.converter.parse(str(f))
            parts = list(score.parts)
            if len(parts) >= 2:
                # Use relative path from midi_dir as description
                desc = str(f.relative_to(midi_path)).rsplit(".", 1)[0]
                results.append((desc, score, style))
            else:
                logger.debug(f"Skipping {f.name}: fewer than 2 parts")
        except Exception as e:
            logger.debug(f"Could not parse {f}: {e}")

    if results:
        logger.info(
            f"Loaded {len(results)} files from {midi_dir} "
            f"({len(sorted_files)} candidates, {len(sorted_files) - len(results)} skipped)"
        )
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

    # Phase 1: broad search
    works = _search_corpus_broad()
    loaded_paths = {path for path, _, _ in works}

    # Phase 2: targeted BWV loading
    bwv_works = _load_by_bwv(ALL_TARGETED_BWV, loaded_paths)
    works.extend(bwv_works)

    # Phase 3: user MIDI files (always additive, no dedup needed)
    works.extend(get_midi_files(accepted=accepted))

    # Apply composer filter
    if accepted is not None:
        before = len(works)
        works = [(d, s, st) for d, s, st in works if _work_matches_filter(d, st, accepted)]
        logger.info(f"Composer filter kept {len(works)}/{before} works")

    logger.info(f"Total works loaded: {len(works)}")
    return works


def get_all_bach_works() -> list[tuple[str, music21.stream.Score, str]]:
    """Load all available works for training (unfiltered).

    Convenience wrapper around :func:`get_all_works` for backwards
    compatibility.
    """
    return get_all_works(composer_filter=None)
