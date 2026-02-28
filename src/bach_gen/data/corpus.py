"""Extract works from music21 corpus and user-supplied files.

Loads works from multiple sources:
1. Broad music21 corpus search (Bach, Palestrina, Monteverdi, Mozart, etc.)
2. Targeted BWV loading from constants.py lists
3. User-supplied MIDI files in data/midi/

All sources are deduplicated by sourcePath before returning.

File discovery is sequential (fast); file parsing **and voice extraction**
are parallelized across CPU cores using ``concurrent.futures.ProcessPoolExecutor``.
Workers return picklable ``VoiceComposition`` objects (lists of tuples) instead
of ``music21.stream.Score`` objects, which corrupt silently during pickling.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import music21
from music21 import corpus

from bach_gen.utils.constants import ALL_TARGETED_BWV, DIR_TO_STYLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for pickling)
# ---------------------------------------------------------------------------


def _parse_and_extract_corpus_entry(args: tuple) -> list[tuple]:
    """Worker: parse a music21 corpus entry and extract voice groups.

    Args:
        args: (source_path, style, max_source_voices, max_groups, voices_override)

    Returns:
        List of (VoiceComposition, form_str) tuples, or [] on failure.
    """
    source_path, style, max_source_voices, max_groups, voices_override = args
    try:
        from bach_gen.data.extraction import detect_form, extract_voice_groups

        score = corpus.parse(source_path)
        parts = list(score.parts)
        if len(parts) < 2 or len(parts) > max_source_voices:
            return []
        form, num_voices = detect_form(score, source_path, style)
        if voices_override is not None:
            num_voices = voices_override
        groups = extract_voice_groups(
            score, num_voices, source=source_path, form=form,
        )
        if max_groups > 0:
            groups = groups[:max_groups]
        for comp in groups:
            comp.style = style
        return [(comp, form) for comp in groups]
    except Exception:
        return []


def _parse_and_extract_file_entry(args: tuple) -> list[tuple]:
    """Worker: parse a file path and extract voice groups.

    Args:
        args: (file_path_str, description, style, max_source_voices,
               max_groups, voices_override)

    Returns:
        List of (VoiceComposition, form_str) tuples, or [] on failure.
    """
    file_path_str, desc, style, max_source_voices, max_groups, voices_override = args
    try:
        from bach_gen.data.extraction import detect_form, extract_voice_groups

        score = music21.converter.parse(file_path_str)
        parts = list(score.parts)
        if len(parts) < 2 or len(parts) > max_source_voices:
            return []
        form, num_voices = detect_form(score, desc, style)
        if voices_override is not None:
            num_voices = voices_override
        groups = extract_voice_groups(
            score, num_voices, source=desc, form=form,
        )
        if max_groups > 0:
            groups = groups[:max_groups]
        for comp in groups:
            comp.style = style
        return [(comp, form) for comp in groups]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Parallel dispatch helper
# ---------------------------------------------------------------------------


def _parallel_process(
    tasks: list,
    worker_fn,
    max_workers: int,
    label: str = "Processing",
) -> list:
    """Process *tasks* in parallel using ProcessPoolExecutor.

    Workers return lists of results; this function flattens them.
    Falls back to sequential processing if multiprocessing raises a
    pickling error on the first batch.

    Args:
        tasks: List of argument tuples to pass to *worker_fn*.
        worker_fn: Top-level function that accepts a single tuple arg
            and returns a list of results.
        max_workers: Number of processes.
        label: Label for log messages.

    Returns:
        Flattened list of results from all workers.
    """
    if not tasks:
        return []

    # If only 1 worker requested, skip multiprocessing overhead entirely
    if max_workers <= 1:
        results = []
        for t in tasks:
            r = worker_fn(t)
            if r:
                results.extend(r)
        logger.info(f"{label}: {len(results)} items from {len(tasks)} tasks (sequential)")
        return results

    results: list = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_fn, t): i for i, t in enumerate(tasks)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                if done_count % 50 == 0 or done_count == len(tasks):
                    logger.info(f"{label}: {done_count}/{len(tasks)} files processed...")
                try:
                    r = future.result()
                    if r:
                        results.extend(r)
                except Exception as e:
                    logger.debug(f"{label} worker error: {e}")
    except Exception as e:
        # Pickling or other multiprocessing failure — fall back to sequential
        logger.warning(
            f"{label}: parallel processing failed ({e}); falling back to sequential"
        )
        results = []
        for t in tasks:
            try:
                r = worker_fn(t)
                if r:
                    results.extend(r)
            except Exception as inner_e:
                logger.debug(f"{label} sequential fallback error: {inner_e}")

    logger.info(f"{label}: {len(results)} items from {len(tasks)} tasks")
    return results


# ---------------------------------------------------------------------------
# Corpus search (discovery + parallel parse+extract)
# ---------------------------------------------------------------------------


def _search_corpus_broad(
    max_workers: int = 1,
    max_source_voices: int = 4,
    max_groups: int = 1,
    voices_override: int | None = None,
) -> list[tuple]:
    """Search music21 corpus broadly for all relevant composers.

    Phase 1 (fast, sequential): discover all source paths via corpus.search.
    Phase 2 (slow, parallel): parse and extract voice groups across workers.

    Returns:
        List of (VoiceComposition, form_str) tuples.
    """
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

    # Phase 1: discover all (sourcePath, style) pairs, deduplicating
    seen_paths: set[str] = set()
    discovered: list[tuple[str, str]] = []

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
            discovered.append((path, style))

    logger.info(
        f"Broad search discovered {len(discovered)} unique paths "
        f"({len(seen_paths)} total seen)"
    )

    # Phase 2: parse + extract in parallel
    parse_tasks = [
        (path, style, max_source_voices, max_groups, voices_override)
        for path, style in discovered
    ]
    results = _parallel_process(
        parse_tasks, _parse_and_extract_corpus_entry, max_workers, label="Broad corpus"
    )
    logger.info(f"Broad search extracted {len(results)} voice groups")
    return results


# ---------------------------------------------------------------------------
# BWV-targeted loading (discovery + parallel parse+extract)
# ---------------------------------------------------------------------------


def _load_by_bwv(
    bwv_numbers: list[int],
    already_loaded: set[str],
    max_workers: int = 1,
    max_source_voices: int = 4,
    max_groups: int = 1,
    voices_override: int | None = None,
) -> list[tuple]:
    """Load specific BWV numbers from music21, skipping already-loaded paths.

    Phase 1: discover new source paths.  Phase 2: parse + extract in parallel.

    Args:
        bwv_numbers: List of BWV catalog numbers to search for.
        already_loaded: Set of sourcePath strings already loaded (mutated
            in-place to include newly discovered paths).
        max_workers: Number of parallel workers.
        max_source_voices: Skip works with more parts than this.
        max_groups: Cap extracted groups per work.
        voices_override: Override detected voice count.

    Returns:
        List of (VoiceComposition, form_str) tuples for newly loaded works.
    """
    # Phase 1: discover
    parse_tasks: list[tuple] = []

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
            parse_tasks.append(
                (path, "bach", max_source_voices, max_groups, voices_override)
            )

    logger.info(f"BWV-targeted: {len(parse_tasks)} new paths to process")

    # Phase 2: parse + extract in parallel
    results = _parallel_process(
        parse_tasks, _parse_and_extract_corpus_entry, max_workers, label="BWV-targeted"
    )
    logger.info(f"BWV-targeted loading extracted {len(results)} voice groups")
    return results


# ---------------------------------------------------------------------------
# Style/filter helpers
# ---------------------------------------------------------------------------


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


def _original_source(source: str) -> str:
    """Strip voice group suffix to recover the original source path.

    ``extract_voice_groups`` appends `` (voices X-Y)`` when extracting
    sub-groups from a score with more parts than requested.  This helper
    strips that suffix so we can deduplicate by original path.
    """
    idx = source.rfind(" (voices ")
    return source[:idx] if idx >= 0 else source


# ---------------------------------------------------------------------------
# User-supplied MIDI / score files (discovery + parallel parse+extract)
# ---------------------------------------------------------------------------


def get_midi_files(
    midi_dir: str | Path = "data/midi",
    accepted: set[str] | None = None,
    max_workers: int = 1,
    max_source_voices: int = 4,
    max_groups: int = 1,
    voices_override: int | None = None,
) -> list[tuple]:
    """Load additional score files from a directory.

    Supports all formats music21 can parse: MIDI, MusicXML, Humdrum kern,
    MuseScore, ABC, LilyPond, etc.  Scans recursively so you can organize
    files into subdirectories (e.g. data/midi/bach/, data/midi/handel/).

    Style is inferred from the first subdirectory name under midi_dir using
    DIR_TO_STYLE.  Falls back to empty string if not recognised.

    Phase 1 (fast, sequential): discover and filter file paths.
    Phase 2 (slow, parallel): parse and extract voice groups.

    Returns:
        List of (VoiceComposition, form_str) tuples.
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

    # Phase 1: discover and filter files (fast, sequential)
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

    # Phase 2: build tasks and run parse+extract in parallel
    parse_tasks: list[tuple] = []
    for f in sorted_files:
        style = files[f]
        desc = str(f.relative_to(midi_path)).rsplit(".", 1)[0]
        parse_tasks.append(
            (str(f), desc, style, max_source_voices, max_groups, voices_override)
        )

    logger.info(f"MIDI files: {len(parse_tasks)} candidates discovered in {midi_dir}")

    results = _parallel_process(
        parse_tasks, _parse_and_extract_file_entry, max_workers, label="MIDI files"
    )

    if results:
        logger.info(
            f"Extracted {len(results)} voice groups from {midi_dir} "
            f"({len(sorted_files)} candidates)"
        )
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def get_all_works(
    composer_filter: list[str] | None = None,
    max_workers: int | None = None,
    max_source_voices: int = 4,
    max_groups_per_work: int = 1,
    voices_override: int | None = None,
) -> list[tuple]:
    """Load and extract works for training, optionally filtered by composer/style.

    Workers parse each score **and** extract voice groups in the same process,
    returning picklable ``VoiceComposition`` objects.  This avoids the
    ``music21.stream.Score`` pickling corruption that previously caused
    silent data loss.

    1. Broad corpus search (composer + title)
    2. Targeted BWV loading from ALL_TARGETED_BWV
    3. User-supplied MIDI files from data/midi/

    All sources deduplicated by sourcePath.

    Args:
        composer_filter: If set, only include works whose style matches one
            of these keys (from DIR_TO_STYLE values or keys, e.g.
            ``["bach"]``, ``["baroque", "renaissance"]``).  Matching is
            case-insensitive.  When ``None``, all works are returned.
        max_workers: Number of parallel workers.  Defaults to
            ``min(os.cpu_count(), 8)`` when ``None``.
        max_source_voices: Skip works with more parts than this.
        max_groups_per_work: Cap extracted groups per work.
        voices_override: Override the auto-detected voice count for extraction.

    Returns:
        List of (VoiceComposition, form_str) tuples.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

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

    logger.info(f"Using {max_workers} worker(s) for parallel processing")

    extraction_kwargs = dict(
        max_source_voices=max_source_voices,
        max_groups=max_groups_per_work,
        voices_override=voices_override,
    )

    # Phase 1: broad search
    works = _search_corpus_broad(max_workers=max_workers, **extraction_kwargs)
    loaded_paths = {_original_source(comp.source) for comp, _ in works}

    # Phase 2: targeted BWV loading
    bwv_works = _load_by_bwv(
        ALL_TARGETED_BWV, loaded_paths, max_workers=max_workers, **extraction_kwargs,
    )
    works.extend(bwv_works)

    # Phase 3: user MIDI files (always additive, no dedup needed)
    works.extend(get_midi_files(
        accepted=accepted, max_workers=max_workers, **extraction_kwargs,
    ))

    # Apply composer filter
    if accepted is not None:
        before = len(works)
        works = [
            (comp, form) for comp, form in works
            if _work_matches_filter(comp.source, comp.style, accepted)
        ]
        logger.info(f"Composer filter kept {len(works)}/{before} voice groups")

    logger.info(f"Total voice groups loaded: {len(works)}")
    return works


def get_all_bach_works() -> list[tuple]:
    """Load all available works for training (unfiltered).

    Convenience wrapper around :func:`get_all_works` for backwards
    compatibility.
    """
    return get_all_works(composer_filter=None)
