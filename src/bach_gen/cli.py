"""CLI interface for bach-gen: generate, train, evaluate, prepare-data."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from bach_gen.utils.constants import (
    FORM_DEFAULTS, VALID_FORMS, DEFAULT_SEQ_LEN,
    METER_MAP, LENGTH_NAMES, METER_NAMES,
    compute_measure_count, length_bucket, DEFAULT_PREPARE_COMPOSER_FILTER,
)

console = Console()
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Bach Multi-Voice Composition Generator."""
    setup_logging(verbose)


def chunk_sequences(
    sequences: list[list[int]],
    max_seq_len: int,
    stride_fraction: float = 0.75,
    bos_token: int = 1,
    tokenizer=None,
    piece_ids: list[str] | None = None,
) -> tuple[list[list[int]], list[str]]:
    """Split long sequences into overlapping chunks.

    - Sequences at or under max_seq_len are kept as-is.
    - Longer sequences are split into windows of max_seq_len with overlap.
    - When a tokenizer is provided, each chunk preserves the full conditioning
      prefix through the KEY token so continuation windows still carry style/
      form/mode/analysis metadata.
    - Fallback behavior (no tokenizer): each continuation chunk is BOS-prefixed.
    - stride = int(max_seq_len * stride_fraction), so 0.75 means 512 tokens
      of overlap on a 2048 window.

    Returns:
        (chunked_sequences, chunked_piece_ids)
    """
    stride = int(max_seq_len * stride_fraction)
    result = []
    result_ids = []

    for seq_idx, seq in enumerate(sequences):
        pid = piece_ids[seq_idx] if piece_ids else ""
        if len(seq) <= max_seq_len:
            result.append(seq)
            result_ids.append(pid)
        else:
            prefix_end = -1
            if tokenizer is not None and hasattr(tokenizer, "token_to_name"):
                for i, tok in enumerate(seq):
                    name = tokenizer.token_to_name.get(tok, "")
                    if name.startswith("KEY_"):
                        prefix_end = i
                        break

            # Preferred path: chunk the event body while preserving conditioning.
            if 0 <= prefix_end < (max_seq_len - 1):
                prefix = seq[:prefix_end + 1]
                body = seq[prefix_end + 1:]
                body_window = max_seq_len - len(prefix)
                body_stride = max(1, int(body_window * stride_fraction))
                start = 0
                while start < len(body):
                    chunk = prefix + body[start:start + body_window]
                    if len(chunk) >= max_seq_len // 4:
                        result.append(chunk)
                        result_ids.append(pid)
                    start += body_stride
                continue

            # Fallback: legacy BOS-prefixed continuation chunking.
            start = 0
            while start < len(seq):
                end = start + max_seq_len
                chunk = seq[start:end]
                if start > 0:
                    chunk = [bos_token] + chunk[:max_seq_len - 1]
                if len(chunk) >= max_seq_len // 4:
                    result.append(chunk)
                    result_ids.append(pid)
                start += stride

    return result, result_ids


def _print_conditioning_histograms(sequences: list[list[int]], tokenizer) -> None:
    """Print distribution of conditioning tokens across sequences."""
    from collections import Counter

    # Only check the first ~30 tokens of each sequence (prefix region — now longer)
    prefix_len = 30
    token_counts: Counter = Counter()
    for seq in sequences:
        for tok in seq[:prefix_len]:
            name = tokenizer.token_to_name.get(tok, "")
            if name.startswith(("STYLE_", "FORM_", "MODE_", "LENGTH_", "METER_",
                                "TEXTURE_", "IMITATION_", "HARMONIC_RHYTHM_",
                                "HARMONIC_TENSION_", "CHROMATICISM_", "ENCODE_")):
                token_counts[name] += 1

    # Group by category
    categories = {
        "Style": {k: v for k, v in token_counts.items() if k.startswith("STYLE_")},
        "Form": {k: v for k, v in token_counts.items() if k.startswith("FORM_")},
        "Mode": {k: v for k, v in token_counts.items() if k.startswith("MODE_")},
        "Length": {k: v for k, v in token_counts.items() if k.startswith("LENGTH_")},
        "Meter": {k: v for k, v in token_counts.items() if k.startswith("METER_")},
        "Texture": {k: v for k, v in token_counts.items() if k.startswith("TEXTURE_")},
        "Imitation": {k: v for k, v in token_counts.items() if k.startswith("IMITATION_")},
        "Harmonic Rhythm": {k: v for k, v in token_counts.items() if k.startswith("HARMONIC_RHYTHM_")},
        "Tension": {k: v for k, v in token_counts.items() if k.startswith("HARMONIC_TENSION_")},
        "Chromaticism": {k: v for k, v in token_counts.items() if k.startswith("CHROMATICISM_")},
        "Encoding": {k: v for k, v in token_counts.items() if k.startswith("ENCODE_")},
    }

    console.print("\n  [bold]Conditioning token distributions:[/bold]")
    for cat_name, counts in categories.items():
        if counts:
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            console.print(f"    {cat_name}: {', '.join(parts)}")
        else:
            console.print(f"    {cat_name}: (none)")


def _composition_signature(comp) -> tuple:
    """Canonical representation of decoded musical content."""
    return (
        comp.key_root,
        comp.key_mode,
        tuple(tuple((start, dur, pitch) for start, dur, pitch in voice) for voice in comp.voices),
    )


def _infer_roundtrip_settings(seq: list[int], tokenizer, default_form: str) -> tuple[str, str]:
    """Infer form and encoding mode from sequence prefix."""
    form = default_form
    encoding_mode = "interleaved"
    for tok in seq[:64]:
        name = tokenizer.token_to_name.get(tok, "")
        if name.startswith("FORM_"):
            form = name[5:].lower()
        elif name == "ENCODE_SEQUENTIAL":
            encoding_mode = "sequential"
        elif name == "ENCODE_INTERLEAVED":
            encoding_mode = "interleaved"
        elif name.startswith("KEY_"):
            break
    return form, encoding_mode


def _build_token_category_map(tokenizer) -> tuple[list[int], list[str]]:
    """Build token_id -> category index mapping for monitoring losses."""
    categories = [
        "pitch",
        "octave",
        "duration",
        "timing",
        "structure",
        "conditioning",
        "other",
    ]
    cat_to_idx = {name: i for i, name in enumerate(categories)}
    other_idx = cat_to_idx["other"]

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return [], categories

    token_to_name = getattr(tokenizer, "token_to_name", {})
    token_map = [other_idx] * vocab_size
    if not isinstance(token_to_name, dict):
        return token_map, categories

    conditioning_prefixes = (
        "STYLE_",
        "FORM_",
        "MODE_",
        "LENGTH_",
        "METER_",
        "TEXTURE_",
        "IMITATION_",
        "HARMONIC_RHYTHM_",
        "HARMONIC_TENSION_",
        "CHROMATICISM_",
        "ENCODE_",
        "KEY_",
    )

    for tok, name in token_to_name.items():
        try:
            tok_id = int(tok)
        except (TypeError, ValueError):
            continue
        if tok_id < 0 or tok_id >= vocab_size or not isinstance(name, str):
            continue

        if name.startswith("DEG_") or name in ("SHARP", "FLAT") or name.startswith("Pitch_"):
            cat_name = "pitch"
        elif name.startswith("OCT_"):
            cat_name = "octave"
        elif name.startswith("Dur_"):
            cat_name = "duration"
        elif name.startswith("TimeShift_"):
            cat_name = "timing"
        elif (
            name == "BAR"
            or name.startswith("BEAT_")
            or name.startswith("VOICE_")
            or name in (
                "VOICE_SEP",
                "SUBJECT_START",
                "SUBJECT_END",
                "CAD_PAC",
                "CAD_IAC",
                "CAD_HC",
                "CAD_DC",
            )
        ):
            cat_name = "structure"
        elif name.startswith(conditioning_prefixes):
            cat_name = "conditioning"
        else:
            cat_name = "other"

        token_map[tok_id] = cat_to_idx[cat_name]

    return token_map, categories


def _tokenize_items(
    items: list,
    forms: list[str],
    tokenizer,
    no_sequential: bool,
    conditioning_phase: str = "none",
    subject_forms: set[str] | None = None,
    cadence_min_confidence: float = 2.0,
    subject_min_quality: float = 0.80,
    subject_min_match_ratio: float = 0.70,
) -> tuple[list[list[int]], list[str]]:
    """Tokenize a batch of compositions with optional dual encoding."""
    from bach_gen.data.analysis import analyze_composition
    from bach_gen.data.conditioning import (
        cadence_token_ids_by_tick,
        detect_cadence_events,
        detect_subject_entries,
        subject_boundary_note_indices,
    )

    sequences: list[list[int]] = []
    piece_ids: list[str] = []
    use_cadence = conditioning_phase in {"cadence", "cadence+subject"}
    use_subject = conditioning_phase == "cadence+subject"
    if subject_forms is None:
        subject_forms = {"fugue", "invention", "sinfonia"}
    subject_forms_normalized = {f.lower() for f in subject_forms}

    for i, item in enumerate(items):
        item_form = forms[i] if i < len(forms) else "chorale"
        item_form_normalized = item_form.lower()
        time_sig = item.time_signature if hasattr(item, "time_signature") else (4, 4)
        num_bars = compute_measure_count(item.voices, time_sig)

        labels = analyze_composition(item, time_sig)
        cadence_map: dict[int, int] | None = None
        subject_start_markers: set[tuple[int, int]] | None = None
        subject_end_markers: set[tuple[int, int]] | None = None

        if use_cadence:
            cadence_events = detect_cadence_events(
                item,
                min_confidence=cadence_min_confidence,
            )
            cadence_map = cadence_token_ids_by_tick(cadence_events, tokenizer.name_to_token)

        if use_subject and item_form_normalized in subject_forms_normalized:
            subject_entries = detect_subject_entries(
                item,
                min_quality=subject_min_quality,
                min_match_ratio=subject_min_match_ratio,
            )
            subject_start_markers, subject_end_markers = subject_boundary_note_indices(subject_entries)

        tokens = tokenizer.encode(
            item, form=item_form, length_bars=num_bars,
            texture=labels["texture"], imitation=labels["imitation"],
            harmonic_rhythm=labels["harmonic_rhythm"],
            harmonic_tension=labels["harmonic_tension"],
            chromaticism=labels["chromaticism"],
            cadence_tokens_by_tick=cadence_map,
            subject_start_markers=subject_start_markers,
            subject_end_markers=subject_end_markers,
        )
        if len(tokens) >= 20:
            sequences.append(tokens)
            piece_ids.append(item.source)

        if not no_sequential:
            tokens_seq = tokenizer.encode_sequential(
                item, form=item_form, length_bars=num_bars,
                texture=labels["texture"], imitation=labels["imitation"],
                harmonic_rhythm=labels["harmonic_rhythm"],
                harmonic_tension=labels["harmonic_tension"],
                chromaticism=labels["chromaticism"],
                cadence_tokens_by_tick=cadence_map,
                subject_start_markers=subject_start_markers,
                subject_end_markers=subject_end_markers,
            )
            if len(tokens_seq) >= 20:
                sequences.append(tokens_seq)
                piece_ids.append(item.source)

    return sequences, piece_ids


def _tokenize_batch_task(
    task: tuple[int, list, list[str], str, bool, str, tuple[str, ...], float, float, float],
) -> tuple[int, list[list[int]], list[str], int]:
    """Worker entrypoint for parallel Step-3 tokenization."""
    (
        batch_idx,
        items,
        forms,
        tokenizer_type,
        no_sequential,
        conditioning_phase,
        subject_forms,
        cadence_min_confidence,
        subject_min_quality,
        subject_min_match_ratio,
    ) = task

    from bach_gen.data.tokenizer import BachTokenizer

    if tokenizer_type == "scale-degree":
        from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
        tokenizer = ScaleDegreeTokenizer()
    else:
        tokenizer = BachTokenizer()

    seqs, pids = _tokenize_items(
        items,
        forms,
        tokenizer,
        no_sequential,
        conditioning_phase=conditioning_phase,
        subject_forms=set(subject_forms),
        cadence_min_confidence=cadence_min_confidence,
        subject_min_quality=subject_min_quality,
        subject_min_match_ratio=subject_min_match_ratio,
    )
    return batch_idx, seqs, pids, len(items)


def _extract_form_from_prefix(seq: list[int], tokenizer) -> str:
    for tok in seq[:64]:
        name = tokenizer.token_to_name.get(tok, "")
        if name.startswith("FORM_"):
            return name[5:].lower()
        if name.startswith("KEY_"):
            break
    return "unknown"


def _is_interleaved_sequence(seq: list[int], tokenizer) -> bool:
    for tok in seq[:64]:
        name = tokenizer.token_to_name.get(tok, "")
        if name == "ENCODE_SEQUENTIAL":
            return False
        if name == "ENCODE_INTERLEAVED":
            return True
        if name.startswith("KEY_"):
            break
    return True


def _print_structural_conditioning_report(
    sequences: list[list[int]],
    tokenizer,
) -> None:
    """Report cadence/subject token statistics from interleaved sequences."""
    from collections import Counter

    cadence_names = {"CAD_PAC", "CAD_IAC", "CAD_HC", "CAD_DC"}
    subject_start_names = {"SUBJECT_START"}

    cadence_by_form: Counter = Counter()
    bars_by_form: Counter = Counter()
    cadence_types: Counter = Counter()

    subj_entries_by_form: Counter = Counter()
    subj_pieces_by_form: Counter = Counter()
    subj_position_bins: Counter = Counter()

    n_interleaved = 0
    for seq in sequences:
        if not _is_interleaved_sequence(seq, tokenizer):
            continue
        n_interleaved += 1
        form = _extract_form_from_prefix(seq, tokenizer)

        bars = 0
        subj_positions: list[int] = []
        for tok in seq:
            name = tokenizer.token_to_name.get(tok, "")
            if name == "BAR":
                bars += 1
            elif name in cadence_names:
                cadence_by_form[form] += 1
                cadence_types[name] += 1
            elif name in subject_start_names:
                subj_positions.append(bars)

        if bars > 0:
            bars_by_form[form] += bars
        if subj_positions:
            subj_entries_by_form[form] += len(subj_positions)
            subj_pieces_by_form[form] += 1
            for pos in subj_positions:
                ratio = pos / max(bars, 1)
                if ratio < 0.33:
                    subj_position_bins["early"] += 1
                elif ratio < 0.66:
                    subj_position_bins["mid"] += 1
                else:
                    subj_position_bins["late"] += 1

    if n_interleaved == 0:
        console.print("  Structural conditioning report: no interleaved sequences found")
        return

    console.print("\n  [bold]Structural conditioning report (interleaved only):[/bold]")

    if cadence_types:
        total_cad = sum(cadence_types.values())
        parts = [f"{name}={count}" for name, count in sorted(cadence_types.items())]
        console.print(f"    Cadence type counts: {', '.join(parts)} (total={total_cad})")
    else:
        console.print("    Cadence type counts: (none)")

    if bars_by_form:
        density_parts = []
        for form, bars in sorted(bars_by_form.items()):
            cad = cadence_by_form.get(form, 0)
            density = cad / max(bars, 1)
            density_parts.append(f"{form}: {density:.3f}/bar ({cad}/{bars})")
        console.print(f"    Cadence density by form: {'; '.join(density_parts)}")

    if subj_entries_by_form:
        entry_parts = []
        for form in sorted(subj_entries_by_form.keys()):
            entries = subj_entries_by_form[form]
            pieces = max(1, subj_pieces_by_form.get(form, 1))
            entry_parts.append(f"{form}: {entries / pieces:.2f} entries/piece")
        console.print(f"    Subject entries by form: {'; '.join(entry_parts)}")
        if subj_position_bins:
            pos_parts = [f"{k}={v}" for k, v in sorted(subj_position_bins.items())]
            console.print(f"    Subject entry positions: {', '.join(pos_parts)}")
    else:
        console.print("    Subject entries by form: (none)")


def _apply_conditioning_dropout_to_sequences(
    sequences: list[list[int]],
    tokenizer,
    dropout_prob: float,
    seed: int,
) -> list[list[int]]:
    from bach_gen.data.conditioning import apply_conditioning_dropout

    cadence_token_ids = {
        tokenizer.name_to_token[name]
        for name in ("CAD_PAC", "CAD_IAC", "CAD_HC", "CAD_DC")
        if name in tokenizer.name_to_token
    }
    subject_start_token_ids = {
        tokenizer.name_to_token[name]
        for name in ("SUBJECT_START",)
        if name in tokenizer.name_to_token
    }
    subject_end_token_ids = {
        tokenizer.name_to_token[name]
        for name in ("SUBJECT_END",)
        if name in tokenizer.name_to_token
    }

    rng = random.Random(seed)
    return [
        apply_conditioning_dropout(
            seq,
            cadence_token_ids=cadence_token_ids,
            subject_start_token_ids=subject_start_token_ids,
            subject_end_token_ids=subject_end_token_ids,
            dropout_prob=dropout_prob,
            rng=rng,
            keep_first_subject_entry=True,
        )
        for seq in sequences
    ]


@cli.command()
@click.option("--mode", "-m", type=click.Choice(VALID_FORMS), default="all",
              help="Composition mode (determines voice count)")
@click.option("--voices", type=int, default=None,
              help="Override number of voices (default: from mode)")
@click.option("--tokenizer", "tokenizer_type",
              type=click.Choice(["absolute", "scale-degree"]), default="scale-degree",
              help="Tokenizer type: absolute (default) or scale-degree (key-agnostic)")
@click.option("--max-seq-len", default=4096, type=int,
              help="Drop sequences longer than this (default: model max_seq_len)")
@click.option("--no-chunk", is_flag=True, default=False,
              help="Drop long sequences instead of chunking them")
@click.option("--data-dir", default=None, type=click.Path(),
              help="Output directory for prepared data (default: data/)")
@click.option(
    "--midi-dir",
    default=None,
    type=click.Path(),
    help=(
        "Local score root to ingest (default: auto — use data/midi/all when "
        "present, else data/midi)"
    ),
)
@click.option("--composer-filter", default=None, type=str,
              help=("Comma-separated composer/style names to include "
                    "(default: bach,baroque,renaissance,classical; use 'all' to disable filtering)"))
@click.option("--no-sequential", is_flag=True, default=False,
              help="Disable dual sequential encoding (skip voice-by-voice training data)")
@click.option("--max-source-voices", default=4, type=int,
              help="Skip works whose raw score has more than this many parts (default: 4)")
@click.option("--max-groups-per-work", default=1, type=int,
              help="Cap extracted N-voice groups per work (default: 1)")
@click.option("--pair-strategy", type=click.Choice(["adjacent+outer", "adjacent-only", "all-combinations"]),
              default="adjacent+outer",
              help="How to derive 2-part pairs from multi-voice works (default: adjacent+outer)")
@click.option("--max-pairs-per-work", default=2, type=int,
              help="Cap extracted 2-part pairs per work (default: 2)")
@click.option("--sonata-policy", type=click.Choice(["counterpoint-safe", "all"]),
              default="counterpoint-safe",
              help="How to treat sonata data in broad training (default: counterpoint-safe)")
@click.option("--workers", default=None, type=int,
              help="Number of parallel workers for file parsing (default: min(cpu_count, 8))")
@click.option("--conditioning-phase", type=click.Choice(["none", "cadence", "cadence+subject"]),
              default="cadence+subject",
              help="Structural conditioning labels to add during tokenization")
@click.option("--conditioning-dropout", default=0.40, type=float,
              help="Drop probability applied to cadence/subject markers (default: 0.40)")
@click.option("--conditioning-seed", default=1337, type=int,
              help="Random seed for conditioning dropout (default: 1337)")
@click.option("--subject-forms", default="fugue,invention,sinfonia", type=str,
              help="Comma-separated forms that receive subject-entry labels")
@click.option("--cadence-min-confidence", default=2.0, type=float,
              help="Minimum cadence detector confidence (default: 2.0)")
@click.option("--subject-min-quality", default=0.80, type=float,
              help="Minimum subject interval-match quality (default: 0.80)")
@click.option("--subject-min-match-ratio", default=0.70, type=float,
              help="Minimum subject match length ratio (default: 0.70)")
def prepare_data(mode: str, voices: int | None, tokenizer_type: str, max_seq_len: int,
                 no_chunk: bool, data_dir: str | None, midi_dir: str | None, composer_filter: str | None,
                 no_sequential: bool, max_source_voices: int,
                 max_groups_per_work: int, pair_strategy: str,
                 max_pairs_per_work: int, sonata_policy: str,
                 workers: int | None, conditioning_phase: str,
                 conditioning_dropout: float, conditioning_seed: int,
                 subject_forms: str, cadence_min_confidence: float,
                 subject_min_quality: float, subject_min_match_ratio: float) -> None:
    """Extract Bach corpus, tokenize, and cache statistics."""
    from collections import Counter, defaultdict
    from bach_gen.data.corpus import get_all_works, _default_local_midi_dir, _original_source
    from bach_gen.data.extraction import (
        is_accompaniment_texture_like_comp,
        accompaniment_texture_severity_comp,
        is_keyboard_like_source,
        VoiceComposition,
    )
    from bach_gen.data.augmentation import augment_to_all_keys
    from bach_gen.data.tokenizer import BachTokenizer
    from bach_gen.data.dataset import compute_corpus_stats

    use_scale_degree = tokenizer_type == "scale-degree"

    is_all_mode = mode == "all"
    if is_all_mode:
        num_voices = voices or 4  # placeholder; actual count detected per piece
        voices_for_extraction = voices  # None → auto-detect
    else:
        num_voices = voices or FORM_DEFAULTS[mode][0]
        voices_for_extraction = num_voices

    # Resolve output directory
    out_dir = Path(data_dir) if data_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= conditioning_dropout <= 1.0):
        console.print("[red]--conditioning-dropout must be in [0, 1][/red]")
        sys.exit(1)
    if not (0.0 <= subject_min_match_ratio <= 1.0):
        console.print("[red]--subject-min-match-ratio must be in [0, 1][/red]")
        sys.exit(1)
    if not (0.0 <= subject_min_quality <= 1.0):
        console.print("[red]--subject-min-quality must be in [0, 1][/red]")
        sys.exit(1)
    parsed_subject_forms = {
        f.strip().lower() for f in subject_forms.split(",") if f.strip()
    }
    if not parsed_subject_forms:
        parsed_subject_forms = {"fugue", "invention", "sinfonia"}
    local_midi_dir = Path(midi_dir) if midi_dir else _default_local_midi_dir()

    # Parse composer/era filter.
    # Default is a curated era set; pass --composer-filter all to disable.
    filter_list = None
    filter_origin = "custom"
    if composer_filter:
        parsed = [c.strip().lower() for c in composer_filter.split(",") if c.strip()]
        if "all" in parsed:
            filter_list = None
            filter_origin = "disabled"
        else:
            filter_list = parsed
    else:
        if local_midi_dir.name.lower() == "all":
            # Consolidated local corpus should be ingested in full by default.
            filter_list = None
            filter_origin = "auto-all"
        else:
            filter_list = list(DEFAULT_PREPARE_COMPOSER_FILTER)
            filter_origin = "default"

    # Step 1: Load and extract works (parse + extract in workers)
    if filter_list:
        suffix = " [default]" if filter_origin == "default" else ""
        filter_desc = f" (filter: {', '.join(filter_list)}{suffix})"
    else:
        filter_desc = " (filter: disabled)"
        if filter_origin == "auto-all":
            filter_desc = " (filter: disabled for consolidated all/ input)"
    console.print(f"[bold]Step 1:[/] Loading and extracting from corpus...{filter_desc}")
    if is_all_mode:
        console.print(f"  Mode: all (auto-detect form and voice count per piece)")
    else:
        console.print(f"  Mode: {mode} ({num_voices} voices)")
    console.print(f"  Tokenizer: {tokenizer_type}")
    console.print(f"  Max source voices: {max_source_voices}")
    console.print(f"  Sonata policy: {sonata_policy}")
    console.print(f"  Local score dir: {local_midi_dir}")
    console.print(
        f"  Conditioning: {conditioning_phase} "
        f"(dropout={conditioning_dropout:.2f}, seed={conditioning_seed})"
    )
    if conditioning_phase == "cadence+subject":
        console.print(f"  Subject forms: {sorted(parsed_subject_forms)}")
        console.print(
            "  Detector thresholds: "
            f"cadence_conf>={cadence_min_confidence:.2f}, "
            f"subject_quality>={subject_min_quality:.2f}, "
            f"subject_match_ratio>={subject_min_match_ratio:.2f}"
        )
    import os as _os
    effective_workers = workers if workers is not None else min(_os.cpu_count() or 1, 8)
    console.print(f"  Workers: {effective_workers}")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading and extracting...", total=None)
        works_with_forms = get_all_works(
            composer_filter=filter_list, max_workers=workers,
            max_source_voices=max_source_voices,
            max_groups_per_work=max_groups_per_work,
            midi_dir=local_midi_dir,
            voices_override=voices_for_extraction,
        )
        progress.update(task, description=f"Extracted {len(works_with_forms)} voice groups")

    if not works_with_forms:
        console.print("[red]No works found. Check music21 corpus installation.[/red]")
        console.print("Run: python -c \"import music21; music21.configure.run()\"")
        sys.exit(1)

    # Apply sonata policy filtering (per-work grouping)
    compositions: list[VoiceComposition] = []
    form_per_comp: list[str] = []
    form_counter: Counter = Counter()
    voice_count_counter: Counter = Counter()
    skipped_by_sonata_policy = 0

    if sonata_policy == "counterpoint-safe":
        # Group by original source for per-work filtering
        work_groups: dict[str, list[tuple[VoiceComposition, str]]] = defaultdict(list)
        for comp, form in works_with_forms:
            work_groups[_original_source(comp.source)].append((comp, form))

        for orig_src, items in work_groups.items():
            representative_form = items[0][1]
            apply_filter = (
                representative_form == "sonata"
                or is_keyboard_like_source(orig_src)
            )
            if apply_filter:
                kept: list[tuple[VoiceComposition, str]] = []
                accompaniment_like: list[tuple[float, VoiceComposition, str]] = []
                for comp, form in items:
                    if is_accompaniment_texture_like_comp(comp):
                        sev = accompaniment_texture_severity_comp(comp)
                        accompaniment_like.append((sev, comp, form))
                        continue
                    kept.append((comp, form))
                # Keep at most one accompaniment-like sample per work
                if accompaniment_like:
                    accompaniment_like.sort(key=lambda x: x[0])
                    kept.append((accompaniment_like[0][1], accompaniment_like[0][2]))
                    skipped_by_sonata_policy += max(0, len(accompaniment_like) - 1)
                items = kept
            for comp, form in items:
                compositions.append(comp)
                # For all mode, use auto-detected form; otherwise use CLI mode
                item_form = form if is_all_mode else mode
                form_per_comp.append(item_form)
                form_counter[form] += 1
                voice_count_counter[comp.num_voices] += 1
    else:
        for comp, form in works_with_forms:
            compositions.append(comp)
            item_form = form if is_all_mode else mode
            form_per_comp.append(item_form)
            form_counter[form] += 1
            voice_count_counter[comp.num_voices] += 1

    console.print(f"  Extracted {len(compositions)} voice groups")
    if skipped_by_sonata_policy:
        console.print(f"  Skipped by sonata policy: {skipped_by_sonata_policy}")
    console.print(f"  Form distribution: {dict(form_counter.most_common())}")
    console.print(f"  Voice count distribution: {dict(voice_count_counter.most_common())}")

    if not compositions:
        console.print("[red]No voice groups extracted.[/red]")
        sys.exit(1)

    # Step 2: Augment (skip for scale-degree tokenizer)
    if use_scale_degree:
        console.print(f"\n[bold]Step 2:[/] Skipping key augmentation (scale-degree mode)")
        items_to_tokenize = compositions
        forms_to_tokenize = form_per_comp
    else:
        console.print(f"\n[bold]Step 2:[/] Augmenting to all 12 keys...")
        items_to_tokenize = augment_to_all_keys(compositions)
        # Replicate forms for each augmented copy (12 keys per original)
        augment_factor = len(items_to_tokenize) // len(compositions) if compositions else 1
        forms_to_tokenize = []
        for f in form_per_comp:
            forms_to_tokenize.extend([f] * augment_factor)
        console.print(f"  Augmented to {len(items_to_tokenize)} compositions")

    # Step 3: Tokenize
    console.print(f"\n[bold]Step 3:[/] Tokenizing...")
    if use_scale_degree:
        from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
        tokenizer = ScaleDegreeTokenizer()
    else:
        tokenizer = BachTokenizer()

    sequences: list[list[int]] = []
    piece_ids: list[str] = []
    tokenize_workers = max(1, effective_workers)
    target_batches = max(1, tokenize_workers * 8)
    batch_size = max(1, (len(items_to_tokenize) + target_batches - 1) // target_batches)
    batch_tasks: list[tuple[int, list, list[str], str, bool, str, tuple[str, ...], float, float, float]] = []
    for batch_idx, start in enumerate(range(0, len(items_to_tokenize), batch_size)):
        end = start + batch_size
        batch_tasks.append(
            (
                batch_idx,
                items_to_tokenize[start:end],
                forms_to_tokenize[start:end],
                tokenizer_type,
                no_sequential,
                conditioning_phase,
                tuple(sorted(parsed_subject_forms)),
                cadence_min_confidence,
                subject_min_quality,
                subject_min_match_ratio,
            )
        )

    used_parallel_tokenization = False
    if tokenize_workers > 1 and len(batch_tasks) > 1:
        console.print(f"  Tokenization workers: {tokenize_workers}")
        try:
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(), TaskProgressColumn(), console=console,
            ) as progress:
                task = progress.add_task("Tokenizing", total=len(items_to_tokenize))
                batch_results: dict[int, tuple[list[list[int]], list[str]]] = {}
                with ProcessPoolExecutor(max_workers=tokenize_workers) as executor:
                    futures = {
                        executor.submit(_tokenize_batch_task, bt): bt[0]
                        for bt in batch_tasks
                    }
                    for future in as_completed(futures):
                        batch_idx, batch_seqs, batch_ids, n_items = future.result()
                        batch_results[batch_idx] = (batch_seqs, batch_ids)
                        progress.advance(task, n_items)

            for batch_idx in range(len(batch_tasks)):
                batch_seqs, batch_ids = batch_results[batch_idx]
                sequences.extend(batch_seqs)
                piece_ids.extend(batch_ids)
            used_parallel_tokenization = True
        except Exception as e:
            logger.warning(
                f"Step 3 tokenization: parallel processing failed ({e}); falling back to sequential"
            )
            sequences = []
            piece_ids = []

    if not used_parallel_tokenization:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Tokenizing", total=len(items_to_tokenize))
            for _, batch_items, batch_forms, _, _, _, _, _, _, _ in batch_tasks:
                batch_seqs, batch_ids = _tokenize_items(
                    batch_items,
                    batch_forms,
                    tokenizer,
                    no_sequential,
                    conditioning_phase=conditioning_phase,
                    subject_forms=parsed_subject_forms,
                    cadence_min_confidence=cadence_min_confidence,
                    subject_min_quality=subject_min_quality,
                    subject_min_match_ratio=subject_min_match_ratio,
                )
                sequences.extend(batch_seqs)
                piece_ids.extend(batch_ids)
                progress.advance(task, len(batch_items))

    console.print(f"  Tokenized {len(sequences)} sequences")
    console.print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Conditioning token distribution histograms
    _print_conditioning_histograms(sequences, tokenizer)
    _print_structural_conditioning_report(sequences, tokenizer)

    if conditioning_phase != "none" and conditioning_dropout > 0.0:
        console.print(
            "  Applying conditioning dropout: "
            f"p={conditioning_dropout:.2f} (seed={conditioning_seed})"
        )
        sequences = _apply_conditioning_dropout_to_sequences(
            sequences,
            tokenizer,
            dropout_prob=conditioning_dropout,
            seed=conditioning_seed,
        )

    # Sequence length distribution (before filtering)
    lengths = [len(s) for s in sequences]
    brackets = {
        "≤512": sum(1 for l in lengths if l <= 512),
        "513-1024": sum(1 for l in lengths if 512 < l <= 1024),
        "1025-2048": sum(1 for l in lengths if 1024 < l <= 2048),
        "2049-4096": sum(1 for l in lengths if 2048 < l <= 4096),
        ">4096": sum(1 for l in lengths if l > 4096),
    }
    console.print(f"  Length distribution: {brackets}")

    # Filter sequences exceeding max_seq_len
    if no_chunk:
        before_count = len(sequences)
        filtered = [(s, pid) for s, pid in zip(sequences, piece_ids) if len(s) <= max_seq_len]
        if filtered:
            sequences, piece_ids = zip(*filtered)
            sequences, piece_ids = list(sequences), list(piece_ids)
        else:
            sequences, piece_ids = [], []
        dropped = before_count - len(sequences)
        if dropped > 0:
            console.print(f"  Dropped {dropped} sequences exceeding {max_seq_len} tokens")
    else:
        short_seqs, short_ids = [], []
        long_seqs, long_ids = [], []
        for s, pid in zip(sequences, piece_ids):
            if len(s) <= max_seq_len:
                short_seqs.append(s)
                short_ids.append(pid)
            else:
                long_seqs.append(s)
                long_ids.append(pid)
        chunked, chunked_ids = chunk_sequences(
            long_seqs, max_seq_len, stride_fraction=0.75,
            bos_token=tokenizer.BOS, tokenizer=tokenizer, piece_ids=long_ids,
        )
        console.print(f"  Kept {len(short_seqs)} sequences under {max_seq_len} tokens")
        console.print(f"  Chunked {len(long_seqs)} long sequences into {len(chunked)} windows")
        sequences = short_seqs + chunked
        piece_ids = short_ids + chunked_ids

    # Sequence length stats (after filtering)
    lengths = [len(s) for s in sequences]
    console.print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
                  f"mean={sum(lengths)/len(lengths):.0f}")

    # Step 4: Save
    console.print(f"\n[bold]Step 4:[/] Saving data...")

    tokenizer.save(out_dir / "tokenizer.json")
    with open(out_dir / "sequences.json", "w") as f:
        json.dump(sequences, f)
    with open(out_dir / "piece_ids.json", "w") as f:
        json.dump(piece_ids, f)

    # Save mode metadata
    with open(out_dir / "mode.json", "w") as f:
        json.dump({
            "mode": mode,
            "num_voices": num_voices,
            "tokenizer_type": tokenizer_type,
            "max_seq_len": max_seq_len,
            "sequential_enabled": not no_sequential,
            "conditioning_phase": conditioning_phase,
            "conditioning_dropout": conditioning_dropout,
            "conditioning_seed": conditioning_seed,
            "subject_forms": sorted(parsed_subject_forms),
            "cadence_min_confidence": cadence_min_confidence,
            "subject_min_quality": subject_min_quality,
            "subject_min_match_ratio": subject_min_match_ratio,
        }, f, indent=2)

    # Compute and save corpus statistics
    console.print("  Computing corpus statistics...")
    stats = compute_corpus_stats(sequences, tokenizer.vocab_size, tokenizer=tokenizer)
    with open(out_dir / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"\n[green]Done![/green] Data saved to {out_dir}/")
    console.print(f"  - tokenizer.json")
    console.print(f"  - sequences.json ({len(sequences)} sequences)")
    console.print(f"  - piece_ids.json ({len(set(piece_ids))} unique pieces)")
    console.print(f"  - corpus_stats.json")
    console.print(f"  - mode.json (mode={mode}, voices={num_voices})")

    # Round-trip verification
    console.print(f"\n[bold]Verification:[/] Round-trip test...")
    test_seq = sequences[0]
    decoded = tokenizer.decode(test_seq)
    default_form = "chorale" if is_all_mode else mode
    roundtrip_form, encoding_mode = _infer_roundtrip_settings(test_seq, tokenizer, default_form)
    if encoding_mode == "sequential" and hasattr(tokenizer, "encode_sequential"):
        re_encoded = tokenizer.encode_sequential(decoded, form=roundtrip_form)
    else:
        re_encoded = tokenizer.encode(decoded, form=roundtrip_form)
    re_decoded = tokenizer.decode(re_encoded)
    is_equivalent = _composition_signature(decoded) == _composition_signature(re_decoded)
    voice_desc = ", ".join(f"v{i+1}={len(v)} notes" for i, v in enumerate(decoded.voices) if v)
    console.print(f"  Original: {len(test_seq)} tokens → Decoded: {voice_desc} "
                  f"→ Re-encoded: {len(re_encoded)} tokens")
    if is_equivalent:
        console.print(f"  [green]Round-trip OK (musical content preserved)[/green]")
    else:
        console.print(f"  [red]Round-trip mismatch (decoded musical content changed)[/red]")


@cli.command()
@click.option("--epochs", default=200, help="Number of training epochs")
@click.option("--lr", default=3e-4, type=float, help="Learning rate")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--seq-len", default=None, type=int,
              help="Max sequence length (default: from mode)")
@click.option("--mode", "-m", type=click.Choice(VALID_FORMS), default=None,
              help="Composition mode (auto-detected from data if not set)")
@click.option("--accumulation-steps", default=1, type=int,
              help="Gradient accumulation steps (effective batch = batch_size * steps)")
@click.option("--resume", default=None, type=click.Path(exists=True),
              help="Resume training from a checkpoint (e.g. models/latest.pt)")
@click.option("--data-dir", default=None, type=click.Path(),
              help="Directory with prepared training data (default: data/)")
@click.option("--curriculum", is_flag=True, default=False,
              help="Three-phase training: pre-train on --data-dir, DroPE recalibrate, then fine-tune on --finetune-data-dir or --finetune style subset")
@click.option("--pretrain-epochs", default=300, type=int,
              help="Epochs for pre-training phase (curriculum mode, default: 300)")
@click.option("--finetune-data-dir", default="data/bach", type=click.Path(),
              help="Data directory for fine-tuning phase (default: data/bach)")
@click.option("--finetune", "finetune_style",
              type=str,
              default=None,
              help="Fine-tune subset from same --data-dir (style token or composer substring, e.g. 'bach' or 'beethoven')")
@click.option("--finetune-lr", default=5e-5, type=float,
              help="Learning rate for fine-tuning phase (default: 5e-5)")
@click.option("--pretrained-checkpoint", default=None, type=click.Path(exists=True),
              help="Skip curriculum pre-train phase and start at DroPE from this checkpoint")
@click.option("--drope/--no-drope", default=True,
              help="Enable/disable DroPE recalibration phase (default: enabled)")
@click.option("--drope-epochs", default=20, type=int,
              help="Maximum DroPE recalibration epochs (default: 20)")
@click.option("--drope-lr", default=1e-3, type=float,
              help="Learning rate for DroPE recalibration (default: 1e-3)")
@click.option("--drope-warmup-epochs", default=1, type=int,
              help="Warmup epochs before DroPE cosine decay (default: 1)")
@click.option("--drope-early-stop/--drope-fixed", default=True,
              help="Enable/disable DroPE early stopping (default: enabled)")
@click.option("--drope-patience", default=5, type=int,
              help="DroPE early-stop patience in epochs (default: 5)")
@click.option("--drope-min-delta", default=1e-4, type=float,
              help="Minimum DroPE metric improvement to reset patience (default: 1e-4)")
@click.option("--drope-min-epochs", default=4, type=int,
              help="Minimum DroPE epochs before early stopping can trigger (default: 4)")
@click.option("--early-stop/--no-early-stop", default=True,
              help="Enable/disable early stopping on val loss plateau (default: enabled)")
@click.option("--es-patience", default=20, type=int,
              help="Pre-train early-stop patience: consecutive non-improving val checks (default: 20)")
@click.option("--es-min-delta", default=1e-4, type=float,
              help="Pre-train minimum val loss improvement to reset patience (default: 1e-4)")
@click.option("--es-min-epochs", default=10, type=int,
              help="Pre-train minimum epochs before early stopping can trigger (default: 10)")
@click.option("--finetune-es-patience", default=None, type=int,
              help="Fine-tune early-stop patience (default: same as --es-patience)")
@click.option("--finetune-es-min-delta", default=None, type=float,
              help="Fine-tune minimum improvement to reset patience (default: same as --es-min-delta)")
@click.option("--finetune-es-min-epochs", default=None, type=int,
              help="Fine-tune minimum epochs before early stopping (default: same as --es-min-epochs)")
@click.option("--log-interval", default=5, type=int,
              help="Training log frequency in epochs (default: 5)")
@click.option("--val-interval", default=None, type=int,
              help="Validation frequency in epochs (default: auto = epochs//20)")
@click.option("--fp16", is_flag=True, default=False,
              help="Enable mixed precision (fp16) training — CUDA only")
@click.option("--pos-encoding", type=click.Choice(["rope", "pope"]),
              default="pope", help="Positional encoding for main training stage")
@click.option("--num-kv-heads", default=None, type=int,
              help="Number of KV heads for GQA (default: same as num_heads = standard MHA)")
def train(epochs: int, lr: float, batch_size: int, seq_len: int | None, mode: str | None,
          accumulation_steps: int, resume: str | None, data_dir: str | None,
          curriculum: bool, pretrain_epochs: int, finetune_data_dir: str,
          finetune_style: str | None,
          finetune_lr: float, pretrained_checkpoint: str | None,
          drope: bool, drope_epochs: int, drope_lr: float, drope_warmup_epochs: int,
          drope_early_stop: bool, drope_patience: int, drope_min_delta: float,
          drope_min_epochs: int,
          early_stop: bool, es_patience: int, es_min_delta: float, es_min_epochs: int,
          finetune_es_patience: int | None, finetune_es_min_delta: float | None,
          finetune_es_min_epochs: int | None,
          log_interval: int,
          val_interval: int | None,
          fp16: bool, pos_encoding: str, num_kv_heads: int | None) -> None:
    """Train the Bach Transformer model."""
    import torch
    from bach_gen.data.dataset import BachDataset, create_dataset
    from bach_gen.data.tokenizer import load_tokenizer
    from bach_gen.model.config import ModelConfig
    from bach_gen.model.architecture import BachTransformer
    from bach_gen.model.trainer import Trainer, get_device

    # Resolve data directory
    train_data_dir = Path(data_dir) if data_dir else DATA_DIR

    # Load data
    seq_path = train_data_dir / "sequences.json"
    if not seq_path.exists():
        console.print(f"[red]No training data found at {train_data_dir}. Run 'bach-gen prepare-data' first.[/red]")
        sys.exit(1)

    # Auto-detect mode from saved metadata
    mode_path = train_data_dir / "mode.json"
    mode_info = {}
    if mode_path.exists():
        with open(mode_path) as f:
            mode_info = json.load(f)
    if mode is None:
        mode = mode_info.get("mode", "all")

    # Set seq_len from mode defaults if not specified
    if seq_len is None:
        if "max_seq_len" in mode_info:
            seq_len = int(mode_info["max_seq_len"])
        else:
            seq_len = FORM_DEFAULTS.get(mode, (2, 768))[1]

    if val_interval is not None and val_interval < 1:
        console.print("[red]--val-interval must be >= 1[/red]")
        sys.exit(1)
    if log_interval < 1:
        console.print("[red]--log-interval must be >= 1[/red]")
        sys.exit(1)
    if drope_warmup_epochs < 0:
        console.print("[red]--drope-warmup-epochs must be >= 0[/red]")
        sys.exit(1)

    console.print(f"[bold]Loading training data from {train_data_dir}...[/bold]")
    console.print(f"  Mode: {mode}, seq_len: {seq_len}")
    with open(seq_path) as f:
        sequences = json.load(f)

    # Load piece IDs for piece-level train/val split (avoids chunk leakage)
    piece_ids_path = train_data_dir / "piece_ids.json"
    piece_ids = None
    if piece_ids_path.exists():
        with open(piece_ids_path) as f:
            piece_ids = json.load(f)
        console.print(f"  Loaded piece IDs ({len(set(piece_ids))} unique pieces)")
    else:
        console.print(f"  [yellow]No piece_ids.json found — using random split (re-run prepare-data to fix)[/yellow]")

    tokenizer = load_tokenizer(train_data_dir / "tokenizer.json")
    token_category_map, token_category_names = _build_token_category_map(tokenizer)

    def _filter_for_finetune_target(
        seqs: list[list[int]],
        pids: list[str] | None,
        target: str,
    ) -> tuple[list[list[int]], list[str] | None]:
        target_l = target.lower().strip()
        tok_name = f"STYLE_{target_l.upper()}"
        tok_id = tokenizer.name_to_token.get(tok_name)
        # Prefer exact style-token filtering when available.
        if tok_id is not None:
            kept_idx = [i for i, s in enumerate(seqs) if tok_id in s]
            filtered_seqs = [seqs[i] for i in kept_idx]
            filtered_pids = [pids[i] for i in kept_idx] if pids is not None else None
            return filtered_seqs, filtered_pids

        # Fallback: filter by piece/source ID substring (composer/work hint).
        if pids is None:
            console.print(
                f"[red]Cannot apply --finetune {target!r}: no piece_ids.json available "
                f"and no matching style token ({tok_name}).[/red]"
            )
            sys.exit(1)

        kept_idx = [i for i, pid in enumerate(pids) if target_l in str(pid).lower()]
        filtered_seqs = [seqs[i] for i in kept_idx]
        filtered_pids = [pids[i] for i in kept_idx] if pids is not None else None
        return filtered_seqs, filtered_pids

    # Create datasets
    train_ds, val_ds = create_dataset(sequences, seq_len=seq_len, piece_ids=piece_ids)
    console.print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    # Create model
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        pos_encoding=pos_encoding,
        num_kv_heads=num_kv_heads,
    )

    model = BachTransformer(config)
    device = get_device()
    console.print(f"\n[bold]Model:[/bold]")
    console.print(f"  Parameters: {model.count_parameters():,}")
    console.print(f"  Device: {device}")
    console.print(f"  Vocab size: {config.vocab_size}")
    attn_desc = f"{config.num_heads}h"
    if config.effective_num_kv_heads < config.num_heads:
        attn_desc += f" (GQA: {config.effective_num_kv_heads} KV heads)"
    console.print(f"  Config: {config.embed_dim}d, {attn_desc}, "
                  f"{config.num_layers}L, {config.ffn_dim}ff")

    # Train
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=lr,
        batch_size=batch_size,
        checkpoint_dir=MODELS_DIR,
        device=device,
        accumulation_steps=accumulation_steps,
        fp16=fp16,
        token_category_map=token_category_map,
        token_category_names=token_category_names,
    )

    if curriculum:
        if resume and pretrained_checkpoint:
            console.print("[red]Use either --resume or --pretrained-checkpoint, not both.[/red]")
            sys.exit(1)

    start_epoch = 1
    if resume:
        start_epoch = trainer.resume_from_checkpoint(resume)
        console.print(f"  Resuming from epoch {start_epoch}")

    def _resolve_phase_checkpoint(candidates: list[Path], phase_label: str) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        console.print(
            f"[red]No checkpoint found for {phase_label}. Checked:[/red] "
            + ", ".join(str(c) for c in candidates)
        )
        sys.exit(1)

    if curriculum:
        # Validate curriculum parameters
        if pretrain_epochs >= epochs:
            console.print(f"[red]--pretrain-epochs ({pretrain_epochs}) must be less than "
                          f"--epochs ({epochs})[/red]")
            sys.exit(1)

        if finetune_style and "--finetune-data-dir" in sys.argv:
            console.print("[red]Use either --finetune <target> or --finetune-data-dir, not both.[/red]")
            sys.exit(1)

        ft_sequences: list[list[int]]
        ft_piece_ids: list[str] | None = None
        ft_source_desc: str
        if finetune_style:
            ft_sequences, ft_piece_ids = _filter_for_finetune_target(sequences, piece_ids, finetune_style)
            if not ft_sequences:
                console.print(
                    f"[red]No sequences found for --finetune '{finetune_style}' "
                    f"in {train_data_dir}.[/red]"
                )
                sys.exit(1)
            ft_source_desc = f"{train_data_dir} (finetune={finetune_style})"
        else:
            ft_data_dir = Path(finetune_data_dir)
            ft_seq_path = ft_data_dir / "sequences.json"
            if not ft_seq_path.exists():
                console.print(f"[red]Fine-tune data not found at {ft_data_dir}. "
                              f"Run 'bach-gen prepare-data --data-dir {ft_data_dir}' first.[/red]")
                sys.exit(1)
            with open(ft_seq_path) as f:
                ft_sequences = json.load(f)

            ft_piece_ids_path = ft_data_dir / "piece_ids.json"
            if ft_piece_ids_path.exists():
                with open(ft_piece_ids_path) as f:
                    ft_piece_ids = json.load(f)
            ft_source_desc = str(ft_data_dir)

        finetune_epochs = epochs - pretrain_epochs
        skip_pretrain = pretrained_checkpoint is not None
        ft_es_patience = finetune_es_patience if finetune_es_patience is not None else es_patience
        ft_es_min_delta = finetune_es_min_delta if finetune_es_min_delta is not None else es_min_delta
        ft_es_min_epochs = finetune_es_min_epochs if finetune_es_min_epochs is not None else es_min_epochs

        console.print(f"\n[bold]Curriculum training:[/bold]")
        if skip_pretrain:
            console.print(f"  Phase 1 (pre-train): skipped, loading {pretrained_checkpoint}")
        else:
            console.print(f"  Phase 1 (pre-train): up to {pretrain_epochs} epochs on {train_data_dir}")
        if drope:
            console.print(f"  Phase 2 (DroPE): up to {drope_epochs} epochs on {train_data_dir}")
        else:
            console.print("  Phase 2 (DroPE): disabled")
        console.print(f"  Phase 3 (fine-tune): up to {finetune_epochs} epochs on {ft_source_desc}")
        console.print(f"  Pre-train LR: {lr}")
        if drope:
            console.print(f"  DroPE LR: {drope_lr}")
        console.print(f"  Fine-tune LR: {finetune_lr}")
        if early_stop:
            console.print(f"  Pre-train early stop: patience={es_patience}, min_delta={es_min_delta}, min_epochs={es_min_epochs}")
            console.print(f"  Fine-tune early stop: patience={ft_es_patience}, min_delta={ft_es_min_delta}, min_epochs={ft_es_min_epochs}")
            if drope and drope_early_stop:
                console.print(f"  DroPE early stop: patience={drope_patience}, min_delta={drope_min_delta}, min_epochs={drope_min_epochs}")
            elif drope:
                console.print("  DroPE early stop: disabled")
        else:
            console.print("  Early stop: disabled for pre-train/fine-tune")

        # --- Phase 1: Pre-train (optional skip via checkpoint) ---
        if skip_pretrain:
            console.print(
                f"\n[bold]Phase 1:[/bold] Skipped pre-training; loading checkpoint {pretrained_checkpoint}"
            )
            trainer.resume_from_checkpoint(pretrained_checkpoint)
            trainer.save_checkpoint("pretrain_final.pt")
            pretrain_ckpt = Path(pretrained_checkpoint)
            pt_history = {"train_loss": [], "val_loss": [], "lr": []}
        else:
            console.print(f"\n[bold]Phase 1: Pre-training for {pretrain_epochs} epochs...[/bold]")
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(), TaskProgressColumn(), console=console,
            ) as progress:
                task = progress.add_task("Pre-training", total=pretrain_epochs - start_epoch + 1)

                def pt_callback(epoch, train_loss, val_loss):
                    desc = f"[PRETRAIN] Epoch {epoch}/{pretrain_epochs} | loss={train_loss:.4f}"
                    if val_loss is not None:
                        desc += f" | val_loss={val_loss:.4f}"
                    progress.update(task, advance=1, description=desc)

                pt_history = trainer.train(
                    epochs=pretrain_epochs,
                    start_epoch=start_epoch,
                    log_interval=log_interval,
                    val_interval=(val_interval if val_interval is not None else max(1, pretrain_epochs // 20)),
                    progress_callback=pt_callback,
                    early_stop=early_stop,
                    patience=es_patience,
                    min_delta=es_min_delta,
                    min_epochs=es_min_epochs,
                    phase_name="PRETRAIN",
                    checkpoint_prefix="pretrain_",
                    use_rope=True,
                )

            if pt_history["train_loss"]:
                console.print(f"  Pre-train final loss: {pt_history['train_loss'][-1]:.4f}")
            if pt_history.get("stop_reason") and pt_history["stop_reason"] != "max_epochs_reached":
                console.print(f"  Pre-train stopped early: {pt_history['stop_reason']} "
                              f"(ran {pt_history.get('epochs_ran', '?')} epochs)")
            pretrain_ckpt = _resolve_phase_checkpoint(
                [MODELS_DIR / "pretrain_best.pt", MODELS_DIR / "pretrain_final.pt"],
                "pre-training transition",
            )

        # Always load the selected pre-training checkpoint before DroPE/fine-tune.
        trainer.resume_from_checkpoint(pretrain_ckpt)
        console.print(f"  Loaded pre-train checkpoint: {pretrain_ckpt}")

        # --- Phase 2: DroPE recalibration on broad (pre-train) corpus ---
        if drope:
            console.print(
                f"\n[bold]Phase 2: DroPE recalibration for up to {drope_epochs} epochs (lr={drope_lr})...[/bold]"
            )
            if drope_early_stop:
                console.print(
                    f"  Early stop: enabled (patience={drope_patience}, "
                    f"min_delta={drope_min_delta}, min_epochs={drope_min_epochs})"
                )
            else:
                console.print("  Early stop: disabled (fixed-length DroPE run)")
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(), TaskProgressColumn(), console=console,
            ) as progress:
                task = progress.add_task("DroPE recalibration", total=drope_epochs)

                drope_history = trainer.recalibrate_drope(
                    epochs=drope_epochs,
                    lr=drope_lr,
                    early_stop=drope_early_stop,
                    patience=drope_patience,
                    min_delta=drope_min_delta,
                    min_epochs=drope_min_epochs,
                    warmup_epochs=drope_warmup_epochs,
                )

                progress.update(
                    task,
                    completed=drope_history.get("epochs_ran", drope_epochs),
                    description="DroPE done!",
                )

            console.print(f"  DroPE final loss: {drope_history['train_loss'][-1]:.4f}")
            console.print(f"  DroPE epochs ran: {drope_history.get('epochs_ran', drope_epochs)}")
            console.print(f"  DroPE stop reason: {drope_history.get('stop_reason', 'unknown')}")
            console.print("  Model marked as drope_trained=True")

            drope_ckpt = _resolve_phase_checkpoint(
                [MODELS_DIR / "drope_best.pt", MODELS_DIR / "drope_final.pt"],
                "DroPE transition",
            )
            trainer.resume_from_checkpoint(drope_ckpt)
            console.print(f"  Loaded DroPE checkpoint: {drope_ckpt}")
        else:
            console.print("\n[bold]Phase 2:[/bold] DroPE disabled; continuing from pre-train checkpoint")

        # --- Phase 3: Fine-tune ---
        console.print(f"\n[bold]Phase 3: Preparing fine-tune data from {ft_source_desc}...[/bold]")

        ft_train_ds, ft_val_ds = create_dataset(ft_sequences, seq_len=seq_len, piece_ids=ft_piece_ids)
        console.print(f"  Fine-tune train: {len(ft_train_ds)}, val: {len(ft_val_ds)}")

        trainer.reset_for_finetuning(
            ft_train_ds,
            ft_val_ds,
            lr=finetune_lr,
            save_checkpoint_name=None,
        )

        console.print(f"\n[bold]Fine-tuning for {finetune_epochs} epochs...[/bold]")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Fine-tuning", total=finetune_epochs)

            def ft_callback(epoch, train_loss, val_loss):
                desc = f"[FINETUNE] Epoch {epoch}/{finetune_epochs} | loss={train_loss:.4f}"
                if val_loss is not None:
                    desc += f" | val_loss={val_loss:.4f}"
                progress.update(task, advance=1, description=desc)

            ft_history = trainer.train(
                epochs=finetune_epochs,
                start_epoch=1,
                log_interval=log_interval,
                val_interval=(val_interval if val_interval is not None else max(1, finetune_epochs // 20)),
                progress_callback=ft_callback,
                early_stop=early_stop,
                patience=ft_es_patience,
                min_delta=ft_es_min_delta,
                min_epochs=ft_es_min_epochs,
                phase_name="FINETUNE",
                checkpoint_prefix="finetune_",
            )

        # Merge histories for reporting
        history = {
            "train_loss": pt_history["train_loss"] + ft_history["train_loss"],
            "val_loss": pt_history["val_loss"] + ft_history["val_loss"],
            "lr": pt_history["lr"] + ft_history["lr"],
        }
        sequences_for_cal = ft_sequences

    else:
        # Standard single-phase training
        console.print(f"\n[bold]Training for {epochs} epochs (starting at {start_epoch})...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training", total=epochs - start_epoch + 1)

            def callback(epoch, train_loss, val_loss):
                desc = f"Epoch {epoch}/{epochs} | loss={train_loss:.4f}"
                if val_loss is not None:
                    desc += f" | val_loss={val_loss:.4f}"
                progress.update(task, advance=1, description=desc)

            history = trainer.train(
                epochs=epochs,
                start_epoch=start_epoch,
                log_interval=log_interval,
                val_interval=(val_interval if val_interval is not None else max(1, epochs // 20)),
                progress_callback=callback,
                early_stop=early_stop,
                patience=es_patience,
                min_delta=es_min_delta,
                min_epochs=es_min_epochs,
                phase_name="TRAIN",
            )
        sequences_for_cal = sequences

    # DroPE recalibration phase for single-phase training
    if drope and not curriculum:
        console.print(
            f"\n[bold]DroPE recalibration (max {drope_epochs} epochs, lr={drope_lr})...[/bold]"
        )
        if drope_early_stop:
            console.print(
                f"  Early stop: enabled (patience={drope_patience}, "
                f"min_delta={drope_min_delta}, min_epochs={drope_min_epochs})"
            )
        else:
            console.print("  Early stop: disabled (fixed-length DroPE run)")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("DroPE recalibration", total=drope_epochs)

            drope_history = trainer.recalibrate_drope(
                epochs=drope_epochs,
                lr=drope_lr,
                early_stop=drope_early_stop,
                patience=drope_patience,
                min_delta=drope_min_delta,
                min_epochs=drope_min_epochs,
                warmup_epochs=drope_warmup_epochs,
            )

            progress.update(
                task,
                completed=drope_history.get("epochs_ran", drope_epochs),
                description="DroPE done!",
            )

        console.print(f"  DroPE final loss: {drope_history['train_loss'][-1]:.4f}")
        console.print(f"  DroPE epochs ran: {drope_history.get('epochs_ran', drope_epochs)}")
        console.print(f"  DroPE stop reason: {drope_history.get('stop_reason', 'unknown')}")
        console.print(f"  Model marked as drope_trained=True")

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        console.print(f"  Best val loss: {min(history['val_loss']):.4f}")
    console.print(f"  Checkpoints saved to {MODELS_DIR}/")

    # Calibrate information-theoretic metrics
    console.print("\n[bold]Calibrating evaluation metrics...[/bold]")
    from bach_gen.evaluation.information import (
        calibrate_from_corpus,
        save_information_calibration,
    )
    cal = calibrate_from_corpus(sequences_for_cal[:50], model)
    console.print(f"  Perplexity range: {cal['perplexity_range']}")
    console.print(f"  Entropy range: {cal['entropy_range']}")
    info_cal_path = MODELS_DIR / "information_calibration.json"
    save_information_calibration(info_cal_path, cal)
    save_information_calibration(train_data_dir / "information_calibration.json", cal)
    console.print(f"  Saved information calibration: {info_cal_path}")


@cli.command()
@click.option("--key", "-k", required=True, help="Key (e.g., 'C minor', 'D major')")
@click.option("--subject", "-s", default=None, help="Subject notes (e.g., 'C4 D4 Eb4 F4')")
@click.option("--candidates", "-n", default=100, help="Number of candidates to generate")
@click.option("--top", "-t", default=3, help="Number of top results to return")
@click.option("--temperature", default=1.15, type=float, help="Sampling temperature")
@click.option("--min-p", default=0.015, type=float,
              help="Min-p sampling threshold (recommended primary control; 0 disables)")
@click.option("--max-length", default=None, type=int,
              help="Max generation length (tokens; default: from mode)")
@click.option("--model-path", default=None, help="Path to model checkpoint")
@click.option("--mode", "-m", type=click.Choice(VALID_FORMS), default=None,
              help="Composition mode (default: auto-detect from data)")
@click.option("--voices", type=int, default=None,
              help="Override number of voices (default: from mode)")
@click.option("--beam-width", "-b", type=int, default=None,
              help="Beam width for beam search (disables sampling when set)")
@click.option("--length-penalty", type=float, default=0.7,
              help="Length penalty alpha for beam search (default: 0.7)")
@click.option("--style", type=click.Choice(["bach", "baroque", "renaissance", "classical"]),
              default="bach", help="Style conditioning (default: bach)")
@click.option("--length", type=click.Choice(["short", "medium", "long", "extended"]),
              default=None, help="Length conditioning (default: infer from form)")
@click.option("--meter", type=click.Choice(["2_4", "3_4", "4_4", "6_8", "3_8", "alla_breve"]),
              default=None, help="Meter conditioning (default: 4/4)")
@click.option("--texture", type=click.Choice(["homophonic", "polyphonic", "mixed"]),
              default=None, help="Texture conditioning")
@click.option("--imitation", type=click.Choice(["none", "low", "high"]),
              default=None, help="Imitation conditioning")
@click.option("--harmonic-rhythm", type=click.Choice(["slow", "moderate", "fast"]),
              default=None, help="Harmonic rhythm conditioning")
@click.option("--tension", type=click.Choice(["low", "moderate", "high"]),
              default=None, help="Harmonic tension conditioning")
@click.option("--chromaticism", type=click.Choice(["low", "moderate", "high"]),
              default=None, help="Chromaticism conditioning")
@click.option("--candidate-batch-size", type=int, default=None,
              help="Candidates decoded in parallel during sampling (auto if omitted)")
@click.option("--voice-by-voice", is_flag=True, default=False,
              help="Use voice-by-voice (sequential) generation")
@click.option("--provide-voice", default=None, type=click.Path(exists=True),
              help="Path to MIDI file for voice 1 (use with --voice-by-voice)")
@click.option("--cadence-density", type=click.Choice(["low", "medium", "high"]),
              default=None, help="Bias cadence marker injection frequency")
@click.option("--min-subject-entries", default=0, type=int,
              help="Minimum prompted subject re-entry markers after exposition")
@click.option("--subject-spacing", default=8, type=int,
              help="Minimum bars between prompted subject re-entry markers")
def generate(
    key: str,
    subject: str | None,
    candidates: int,
    top: int,
    temperature: float,
    min_p: float,
    max_length: int | None,
    model_path: str | None,
    mode: str | None,
    voices: int | None,
    beam_width: int | None,
    length_penalty: float,
    style: str,
    length: str | None,
    meter: str | None,
    texture: str | None,
    imitation: str | None,
    harmonic_rhythm: str | None,
    tension: str | None,
    chromaticism: str | None,
    candidate_batch_size: int | None,
    voice_by_voice: bool,
    provide_voice: str | None,
    cadence_density: str | None,
    min_subject_entries: int,
    subject_spacing: int,
) -> None:
    """Generate Bach-style compositions."""
    from bach_gen.data.tokenizer import load_tokenizer
    from bach_gen.model.trainer import Trainer
    from bach_gen.generation.generator import generate as gen_fn
    from bach_gen.generation.generator import generate_voice_by_voice as gen_vbv_fn
    from bach_gen.evaluation.statistical import load_corpus_stats
    from bach_gen.evaluation.information import load_information_calibration

    # Auto-detect mode
    mode_path = DATA_DIR / "mode.json"
    mode_info = {}
    if mode_path.exists():
        with open(mode_path) as f:
            mode_info = json.load(f)
    if mode is None:
        mode = mode_info.get("mode", "all")
    if mode == "all":
        # "all" is a data-prep mode, not a concrete generation form.
        # Default generation to a stable 4-voice form.
        console.print("[yellow]mode=all in metadata is not directly generative; defaulting to chorale for generation.[/yellow]")
        mode = "chorale"

    num_voices = voices or FORM_DEFAULTS.get(mode, (2, 768))[0]

    if min_subject_entries < 0:
        console.print("[red]--min-subject-entries must be >= 0[/red]")
        sys.exit(1)
    if subject_spacing < 1:
        console.print("[red]--subject-spacing must be >= 1[/red]")
        sys.exit(1)

    if max_length is None:
        max_length = FORM_DEFAULTS.get(mode, (2, 768))[1]

    # Load model
    if model_path is None:
        model_path = MODELS_DIR / "best.pt"
        if not model_path.exists():
            model_path = MODELS_DIR / "latest.pt"
        if not model_path.exists():
            model_path = MODELS_DIR / "final.pt"

    if not Path(model_path).exists():
        console.print(f"[red]No model found at {model_path}. Run 'bach-gen train' first.[/red]")
        sys.exit(1)

    console.print(f"[bold]Loading model from {model_path}...[/bold]")
    model, config = Trainer.load_checkpoint(model_path)

    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")

    # Load corpus stats for evaluation
    load_corpus_stats(DATA_DIR / "corpus_stats.json")
    info_cal = (
        load_information_calibration(Path(model_path).parent / "information_calibration.json")
        or load_information_calibration(DATA_DIR / "information_calibration.json")
    )
    if info_cal:
        console.print(
            f"  Loaded info calibration: ppl={info_cal['perplexity_range']}, "
            f"ent={info_cal['entropy_range']}"
        )

    form_label = {
        "2-part": "2-part invention",
        "sinfonia": "sinfonia (3-part)",
        "chorale": "chorale (4-part)",
        "sonata": "sonata (up to 4 voices)",
        "fugue": f"fugue ({num_voices}-voice)",
    }.get(mode, mode)

    console.print(f"\n[bold]Generating {form_label} in {key}...[/bold]")
    if beam_width is not None and beam_width > 1:
        console.print(f"  Strategy: beam search (width={beam_width}, length_penalty={length_penalty})")
    else:
        console.print(f"  Strategy: sampling ({candidates} candidates)")
        if candidate_batch_size is not None:
            console.print(f"  Candidate batch size: {candidate_batch_size}")
    if cadence_density is not None:
        console.print(f"  Cadence density control: {cadence_density}")
    if min_subject_entries > 0:
        console.print(
            f"  Subject re-entry control: min_entries={min_subject_entries}, "
            f"spacing={subject_spacing} bars"
        )
    if subject:
        console.print(f"  Subject: {subject}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_steps = beam_width if (beam_width is not None and beam_width > 1) else candidates
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Generating", total=total_steps)

        def on_progress(current, total):
            progress.update(task, completed=current,
                           description=f"Candidate {current}/{total}")

        results = gen_fn(
            model=model,
            tokenizer=tokenizer,
            key_str=key,
            subject_str=subject,
            num_candidates=candidates,
            top_k_results=top,
            temperature=temperature,
            min_p=min_p,
            max_length=max_length,
            output_dir=OUTPUT_DIR,
            form=mode,
            num_voices=num_voices if voices else None,
            progress_callback=on_progress,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty,
            style=style,
            length=length,
            meter=meter,
            texture=texture,
            imitation=imitation,
            harmonic_rhythm=harmonic_rhythm,
            harmonic_tension=tension,
            chromaticism=chromaticism,
            candidate_batch_size=candidate_batch_size,
            cadence_density=cadence_density,
            min_subject_entries=min_subject_entries,
            subject_spacing_bars=subject_spacing,
        ) if not voice_by_voice else gen_vbv_fn(
            model=model,
            tokenizer=tokenizer,
            key_str=key,
            num_candidates=candidates,
            top_k_results=top,
            temperature=temperature,
            min_p=min_p,
            max_length=max_length,
            output_dir=OUTPUT_DIR,
            form=mode,
            num_voices=num_voices if voices else None,
            progress_callback=on_progress,
            style=style,
            length=length,
            meter=meter,
            texture=texture,
            imitation=imitation,
            harmonic_rhythm=harmonic_rhythm,
            harmonic_tension=tension,
            chromaticism=chromaticism,
            provided_voice_midi=provide_voice,
            candidate_batch_size=candidate_batch_size,
            cadence_density=cadence_density,
            min_subject_entries=min_subject_entries,
            subject_spacing_bars=subject_spacing,
        )
        progress.update(task, description="Done!")

    if not results:
        console.print("[red]No valid candidates generated.[/red]")
        sys.exit(1)

    # Display results
    table = Table(title=f"Top {len(results)} Results — {key} ({mode})")
    table.add_column("#", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Voice Lead.", justify="right")
    table.add_column("Statistical", justify="right")
    table.add_column("Structural", justify="right")
    table.add_column("Contrap.", justify="right")
    table.add_column("Complete", justify="right")
    table.add_column("Thematic", justify="right")
    table.add_column("File")

    for i, r in enumerate(results):
        s = r.score
        table.add_row(
            str(i + 1),
            f"{s.composite:.3f}",
            f"{s.voice_leading:.3f}",
            f"{s.statistical:.3f}",
            f"{s.structural:.3f}",
            f"{s.contrapuntal:.3f}",
            f"{s.completeness:.3f}",
            f"{s.thematic_recall:.3f}",
            r.midi_path or "—",
        )

    console.print()
    console.print(table)
    console.print(f"\n[green]MIDI files saved to {OUTPUT_DIR}/[/green]")


@cli.command()
@click.argument("midi_file", type=click.Path(exists=True))
@click.option("--mode", "-m", type=click.Choice(VALID_FORMS), default=None,
              help="Composition mode (auto-detected from voice count if not set)")
@click.option("--render-audio/--no-render-audio", default=False,
              help="Render a WAV file from the MIDI using FluidSynth.")
@click.option("--soundfont", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to .sf2/.sf3 soundfont (defaults to auto-detect; prefers Jeux14).")
@click.option("--audio-out", type=click.Path(dir_okay=False), default=None,
              help="Output WAV path (default: <midi_stem>.organ.wav).")
@click.option("--record-speed", type=float, default=0.75, show_default=True,
              help="Tempo scale applied to MIDI before synthesis (0.75 = 75% speed).")
@click.option("--reverb-mix", type=float, default=0.18, show_default=True,
              help="Reverb amount in [0, 1].")
@click.option("--stereo-width", type=float, default=1.20, show_default=True,
              help="Stereo width scalar (1.0 = neutral).")
@click.option("--spatialize-voices/--no-spatialize-voices", default=True, show_default=True,
              help="Apply deterministic per-voice panning (higher voices to the right).")
@click.option("--swap-stereo/--no-swap-stereo", default=False, show_default=True,
              help="Swap left/right channels in rendered audio.")
@click.option("--sample-rate", type=int, default=48000, show_default=True,
              help="Rendered WAV sample rate.")
def evaluate(
    midi_file: str,
    mode: str | None,
    render_audio: bool,
    soundfont: str | None,
    audio_out: str | None,
    record_speed: float,
    reverb_mix: float,
    stereo_width: float,
    spatialize_voices: bool,
    swap_stereo: bool,
    sample_rate: int,
) -> None:
    """Evaluate a MIDI file for Bach-style quality."""
    from bach_gen.utils.midi_io import load_midi, midi_to_note_events
    from bach_gen.data.extraction import VoiceComposition
    from bach_gen.data.tokenizer import BachTokenizer, load_tokenizer
    from bach_gen.evaluation.scorer import score_composition
    from bach_gen.evaluation.statistical import load_corpus_stats
    from bach_gen.evaluation.information import load_information_calibration
    from bach_gen.utils.music_theory import detect_key

    import numpy as np

    # Load corpus stats
    stats_path = DATA_DIR / "corpus_stats.json"
    if stats_path.exists():
        load_corpus_stats(stats_path)
    load_information_calibration(MODELS_DIR / "information_calibration.json")
    load_information_calibration(DATA_DIR / "information_calibration.json")

    console.print(f"[bold]Evaluating:[/bold] {midi_file}")

    mid = load_midi(midi_file)
    tracks = midi_to_note_events(mid)

    if len(tracks) < 2:
        # Try to split single track into upper/lower by pitch
        if len(tracks) == 1:
            all_notes = tracks[0]
            if all_notes:
                median_pitch = np.median([n[2] for n in all_notes])
                upper = [(s, d, p) for s, d, p in all_notes if p >= median_pitch]
                lower = [(s, d, p) for s, d, p in all_notes if p < median_pitch]
                tracks = [upper, lower]

    if len(tracks) < 2:
        console.print("[red]Need at least 2 voices/tracks in the MIDI file.[/red]")
        sys.exit(1)

    # Auto-detect mode from number of tracks
    num_tracks = len(tracks)
    if mode is None:
        if num_tracks == 2:
            mode = "2-part"
        elif num_tracks == 3:
            mode = "sinfonia"
        elif num_tracks >= 4:
            mode = "chorale"
        else:
            mode = "2-part"

    # Detect key from all voices
    pc_counts = np.zeros(12)
    for track in tracks:
        for _, _, p in track:
            pc_counts[p % 12] += 1
    key_root, key_mode, _ = detect_key(pc_counts)

    comp = VoiceComposition(
        voices=tracks,
        key_root=key_root,
        key_mode=key_mode,
        source=midi_file,
    )

    from bach_gen.utils.music_theory import pc_to_note_name
    key_name = f"{pc_to_note_name(key_root)} {key_mode}"
    console.print(f"  Detected key: {key_name}")
    console.print(f"  Mode: {mode} ({comp.num_voices} voices)")
    for i, v in enumerate(comp.voices):
        console.print(f"  Voice {i+1}: {len(v)} notes")

    # Score — use saved tokenizer if available, fall back to BachTokenizer
    tok_path = DATA_DIR / "tokenizer.json"
    if tok_path.exists():
        tokenizer = load_tokenizer(tok_path)
    else:
        tokenizer = BachTokenizer()
    tokens = tokenizer.encode(comp, form=mode)
    score = score_composition(comp, token_sequence=tokens, vocab_size=tokenizer.vocab_size, form=mode)

    # Display
    from bach_gen.evaluation.scorer import get_weights_for_form
    w = get_weights_for_form(mode)

    table = Table(title=f"Evaluation Scores ({mode})")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted", justify="right")

    display_metrics = {
        "Voice Leading": (score.voice_leading, w.get("voice_leading", 0.22)),
        "Statistical Sim.": (score.statistical, w.get("statistical", 0.10)),
        "Structural": (score.structural, w.get("structural", 0.15)),
        "Contrapuntal": (score.contrapuntal, w.get("contrapuntal", 0.18)),
        "Completeness": (score.completeness, w.get("completeness", 0.05)),
        "Thematic Recall": (score.thematic_recall, w.get("thematic_recall", 0.30)),
    }

    for name, (val, wt) in display_metrics.items():
        table.add_row(name, f"{val:.3f}", f"{wt:.2f}", f"{val * wt:.3f}")

    table.add_section()
    table.add_row("[bold]COMPOSITE[/bold]", f"[bold]{score.composite:.3f}[/bold]", "", "")

    console.print()
    console.print(table)

    # Show details
    if score.details:
        console.print("\n[bold]Details:[/bold]")
        for dim, details in score.details.items():
            if isinstance(details, dict):
                console.print(f"\n  {dim}:")
                for k, v in details.items():
                    if isinstance(v, float):
                        console.print(f"    {k}: {v:.4f}")
                    else:
                        console.print(f"    {k}: {v}")

    if not render_audio:
        return

    midi_path = Path(midi_file)
    sf_path = _resolve_soundfont(soundfont)
    if sf_path is None:
        console.print("[red]No usable soundfont found for rendering.[/red]")
        console.print("Provide one with [bold]--soundfont /path/to/file.sf2[/bold]")
        console.print("or install a system GM soundfont for FluidSynth.")
        sys.exit(1)

    out_path = Path(audio_out) if audio_out else midi_path.with_suffix(".organ.wav")
    console.print(f"\n[bold]Rendering audio:[/bold] {midi_path.name}")
    console.print(f"  Soundfont: {sf_path}")
    console.print(
        f"  Render options: speed={record_speed:.2f}, "
        f"reverb_mix={reverb_mix:.2f}, stereo_width={stereo_width:.2f}, "
        f"spatialize_voices={'yes' if spatialize_voices else 'no'}, "
        f"swap_stereo={'yes' if swap_stereo else 'no'}"
    )

    try:
        rendered = _render_midi_to_wav(
            midi_path=midi_path,
            wav_path=out_path,
            soundfont=sf_path,
            speed=record_speed,
            reverb_mix=reverb_mix,
            stereo_width=stereo_width,
            spatialize_voices=spatialize_voices,
            swap_stereo=swap_stereo,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        console.print(f"[red]Audio render failed:[/red] {exc}")
        sys.exit(1)

    console.print(f"[green]Rendered audio:[/green] {rendered}")


@cli.command()
@click.option("--sample-size", "-n", default=50, type=int,
              help="Number of corpus pieces to sample for calibration")
def calibrate(sample_size: int) -> None:
    """Calibrate evaluation scorer on real corpus and degenerate baselines.

    Establishes the score range by evaluating:
    - Real Bach corpus pieces (expected ceiling)
    - Shuffled pieces (notes reordered within each voice)
    - Random token sequences (floor)
    - Repetitive patterns (degenerate monotone)

    Use the results to anchor composite score thresholds.
    """
    import random as rng
    import numpy as np
    from bach_gen.data.tokenizer import load_tokenizer
    from bach_gen.evaluation.scorer import score_composition
    from bach_gen.evaluation.statistical import load_corpus_stats
    from bach_gen.evaluation.information import load_information_calibration
    from bach_gen.data.extraction import VoiceComposition
    from bach_gen.utils.constants import TICKS_PER_QUARTER, DURATION_BINS

    # Load data
    seq_path = DATA_DIR / "sequences.json"
    if not seq_path.exists():
        console.print("[red]No training data found. Run 'bach-gen prepare-data' first.[/red]")
        sys.exit(1)

    console.print("[bold]Loading data...[/bold]")
    with open(seq_path) as f:
        sequences = json.load(f)

    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")
    stats_path = DATA_DIR / "corpus_stats.json"
    if stats_path.exists():
        load_corpus_stats(stats_path)
    load_information_calibration(MODELS_DIR / "information_calibration.json")
    load_information_calibration(DATA_DIR / "information_calibration.json")

    mode_path = DATA_DIR / "mode.json"
    mode_info = {}
    if mode_path.exists():
        with open(mode_path) as f:
            mode_info = json.load(f)
    mode = mode_info.get("mode", "all")

    # Sample corpus sequences
    sample = rng.sample(sequences, min(sample_size, len(sequences)))

    def score_sequences(seqs, label, decode=True):
        """Score a list of sequences and return composite scores + breakdowns."""
        scores = []
        breakdowns = []
        failed = 0
        for seq in seqs:
            try:
                if decode:
                    comp = tokenizer.decode(seq)
                else:
                    comp = seq  # already a VoiceComposition
                tokens = tokenizer.encode(comp, form=mode) if decode else tokenizer.encode(comp, form=mode)
                sb = score_composition(comp, token_sequence=tokens, vocab_size=tokenizer.vocab_size)
                scores.append(sb.composite)
                breakdowns.append(sb)
            except Exception:
                failed += 1
        return scores, breakdowns, failed

    # 1. Real Bach corpus pieces
    console.print(f"\n[bold]1. Scoring {len(sample)} real Bach corpus pieces...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Scoring corpus", total=len(sample))
        corpus_scores = []
        corpus_breakdowns = []
        corpus_failed = 0
        for seq in sample:
            try:
                comp = tokenizer.decode(seq)
                tokens = tokenizer.encode(comp, form=mode)
                sb = score_composition(comp, token_sequence=tokens, vocab_size=tokenizer.vocab_size)
                corpus_scores.append(sb.composite)
                corpus_breakdowns.append(sb)
            except Exception:
                corpus_failed += 1
            progress.advance(task)

    # 2. Shuffled pieces (same notes, shuffled order within each voice)
    console.print(f"\n[bold]2. Scoring shuffled pieces (same notes, random order)...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Scoring shuffled", total=len(sample))
        shuffled_scores = []
        shuffled_breakdowns = []
        shuffled_failed = 0
        for seq in sample:
            try:
                comp = tokenizer.decode(seq)
                # Shuffle note pitches within each voice (keep timing intact)
                shuffled_voices = []
                for voice in comp.voices:
                    if not voice:
                        shuffled_voices.append(voice)
                        continue
                    pitches = [n[2] for n in voice]
                    rng.shuffle(pitches)
                    shuffled_voice = [(n[0], n[1], p) for n, p in zip(voice, pitches)]
                    shuffled_voices.append(shuffled_voice)
                shuffled_comp = VoiceComposition(
                    voices=shuffled_voices,
                    key_root=comp.key_root,
                    key_mode=comp.key_mode,
                    source="shuffled",
                )
                tokens = tokenizer.encode(shuffled_comp, form=mode)
                sb = score_composition(shuffled_comp, token_sequence=tokens,
                                       vocab_size=tokenizer.vocab_size)
                shuffled_scores.append(sb.composite)
                shuffled_breakdowns.append(sb)
            except Exception:
                shuffled_failed += 1
            progress.advance(task)

    # 3. Random token sequences
    console.print(f"\n[bold]3. Scoring random token sequences...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Scoring random", total=len(sample))
        random_scores = []
        random_breakdowns = []
        random_failed = 0
        for seq in sample:
            try:
                # Generate random pitches in valid ranges, same structure as original
                comp = tokenizer.decode(seq)
                rand_voices = []
                for voice in comp.voices:
                    if not voice:
                        rand_voices.append(voice)
                        continue
                    pitches = [n[2] for n in voice]
                    if pitches:
                        lo, hi = min(pitches), max(pitches)
                    else:
                        lo, hi = 48, 72
                    rand_voice = [(n[0], n[1], rng.randint(lo, hi)) for n in voice]
                    rand_voices.append(rand_voice)
                rand_comp = VoiceComposition(
                    voices=rand_voices,
                    key_root=comp.key_root,
                    key_mode=comp.key_mode,
                    source="random",
                )
                tokens = tokenizer.encode(rand_comp, form=mode)
                sb = score_composition(rand_comp, token_sequence=tokens,
                                       vocab_size=tokenizer.vocab_size)
                random_scores.append(sb.composite)
                random_breakdowns.append(sb)
            except Exception:
                random_failed += 1
            progress.advance(task)

    # 4. Repetitive patterns (single note repeated)
    console.print(f"\n[bold]4. Scoring repetitive patterns (monotone voices)...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Scoring repetitive", total=len(sample))
        repetitive_scores = []
        repetitive_breakdowns = []
        repetitive_failed = 0
        for seq in sample:
            try:
                comp = tokenizer.decode(seq)
                rep_voices = []
                base_pitches = [60, 55, 48, 43]  # one note per voice
                for i, voice in enumerate(comp.voices):
                    if not voice:
                        rep_voices.append(voice)
                        continue
                    pitch = base_pitches[i % len(base_pitches)]
                    rep_voice = [(n[0], n[1], pitch) for n in voice]
                    rep_voices.append(rep_voice)
                rep_comp = VoiceComposition(
                    voices=rep_voices,
                    key_root=0,  # C
                    key_mode="major",
                    source="repetitive",
                )
                tokens = tokenizer.encode(rep_comp, form=mode)
                sb = score_composition(rep_comp, token_sequence=tokens,
                                       vocab_size=tokenizer.vocab_size)
                repetitive_scores.append(sb.composite)
                repetitive_breakdowns.append(sb)
            except Exception:
                repetitive_failed += 1
            progress.advance(task)

    # Display results
    def stats(scores):
        if not scores:
            return {"n": 0, "mean": 0, "std": 0, "min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0}
        a = np.array(scores)
        return {
            "n": len(a),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
        }

    def dim_stats(breakdowns, dim):
        vals = [getattr(b, dim) for b in breakdowns]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    console.print("\n" + "=" * 72)
    console.print("[bold]CALIBRATION RESULTS[/bold]")
    console.print("=" * 72)

    # Summary table
    table = Table(title="Composite Score Distribution by Condition")
    table.add_column("Condition", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("P25", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P75", justify="right")
    table.add_column("Max", justify="right")

    for label, scores in [
        ("Real Bach Corpus", corpus_scores),
        ("Shuffled Notes", shuffled_scores),
        ("Random Pitches", random_scores),
        ("Repetitive (monotone)", repetitive_scores),
    ]:
        s = stats(scores)
        table.add_row(
            label,
            str(s["n"]),
            f"{s['mean']:.3f}",
            f"{s['std']:.3f}",
            f"{s['min']:.3f}",
            f"{s['p25']:.3f}",
            f"{s['median']:.3f}",
            f"{s['p75']:.3f}",
            f"{s['max']:.3f}",
        )

    console.print(table)

    # Per-dimension breakdown
    dims = ["voice_leading", "statistical", "structural", "contrapuntal",
            "completeness", "thematic_recall"]
    dim_table = Table(title="Per-Dimension Mean ± Std")
    dim_table.add_column("Dimension", style="bold")
    dim_table.add_column("Bach Corpus", justify="right")
    dim_table.add_column("Shuffled", justify="right")
    dim_table.add_column("Random", justify="right")
    dim_table.add_column("Repetitive", justify="right")

    for dim in dims:
        row = [dim]
        for bds in [corpus_breakdowns, shuffled_breakdowns, random_breakdowns, repetitive_breakdowns]:
            m, s = dim_stats(bds, dim)
            row.append(f"{m:.3f} ± {s:.3f}")
        dim_table.add_row(*row)

    console.print()
    console.print(dim_table)

    # Threshold analysis
    console.print("\n[bold]Threshold Analysis:[/bold]")
    if corpus_scores and shuffled_scores:
        corpus_p10 = float(np.percentile(corpus_scores, 10))
        shuffled_p90 = float(np.percentile(shuffled_scores, 90))
        random_p90 = float(np.percentile(random_scores, 90)) if random_scores else 0
        rep_p90 = float(np.percentile(repetitive_scores, 90)) if repetitive_scores else 0

        console.print(f"  Bach corpus P10 (floor of real music):  {corpus_p10:.3f}")
        console.print(f"  Shuffled P90 (ceiling of shuffled):     {shuffled_p90:.3f}")
        console.print(f"  Random P90 (ceiling of random):         {random_p90:.3f}")
        console.print(f"  Repetitive P90 (ceiling of monotone):   {rep_p90:.3f}")

        # Suggest threshold
        baseline_max = max(shuffled_p90, random_p90, rep_p90)
        suggested = (corpus_p10 + baseline_max) / 2
        console.print(f"\n  Suggested threshold (midpoint corpus_P10 & baseline_max): "
                      f"[bold green]{suggested:.3f}[/bold green]")
        console.print(f"  Current roadmap target: 0.6")

        if suggested > 0.6:
            console.print(f"  [yellow]→ 0.6 may be too easy (below suggested {suggested:.3f})[/yellow]")
        elif suggested < 0.4:
            console.print(f"  [yellow]→ 0.6 may be too hard (above suggested {suggested:.3f})[/yellow]")
        else:
            console.print(f"  [green]→ 0.6 looks reasonable relative to calibration[/green]")

    # Failures
    if corpus_failed or shuffled_failed or random_failed or repetitive_failed:
        console.print(f"\n[dim]Failures: corpus={corpus_failed}, shuffled={shuffled_failed}, "
                      f"random={random_failed}, repetitive={repetitive_failed}[/dim]")

    # Save calibration results
    cal_results = {
        "corpus": stats(corpus_scores),
        "shuffled": stats(shuffled_scores),
        "random": stats(random_scores),
        "repetitive": stats(repetitive_scores),
        "per_dimension": {},
    }
    for dim in dims:
        cal_results["per_dimension"][dim] = {}
        for label, bds in [("corpus", corpus_breakdowns), ("shuffled", shuffled_breakdowns),
                           ("random", random_breakdowns), ("repetitive", repetitive_breakdowns)]:
            m, s = dim_stats(bds, dim)
            cal_results["per_dimension"][dim][label] = {"mean": m, "std": s}

    cal_path = DATA_DIR / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(cal_results, f, indent=2)
    console.print(f"\n[green]Calibration results saved to {cal_path}[/green]")


@cli.command()
@click.option("--sample-size", "-n", default=200, type=int,
              help="Max sequences per form to sample for calibration")
def calibrate_forms(sample_size: int) -> None:
    """Calibrate scorer weights per form (chorale, fugue, invention, etc.).

    Groups training sequences by FORM conditioning token and computes
    per-dimension discrimination against multiple degenerate baselines
    (shuffled, random, repetitive). Then blends data-driven weights with
    small form priors so recalibration stays aligned with listener-facing
    quality (especially for fugue rhetoric and texture flow).
    """
    import random as rng
    import numpy as np
    from bach_gen.data.tokenizer import load_tokenizer
    from bach_gen.evaluation.scorer import score_composition, DEFAULT_WEIGHTS
    from bach_gen.evaluation.statistical import load_corpus_stats
    from bach_gen.evaluation.information import load_information_calibration
    from bach_gen.data.extraction import VoiceComposition

    # Load data
    seq_path = DATA_DIR / "sequences.json"
    if not seq_path.exists():
        console.print("[red]No training data found. Run 'bach-gen prepare-data' first.[/red]")
        sys.exit(1)

    console.print("[bold]Loading data...[/bold]")
    with open(seq_path) as f:
        sequences = json.load(f)

    tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")
    stats_path = DATA_DIR / "corpus_stats.json"
    if stats_path.exists():
        load_corpus_stats(stats_path)
    load_information_calibration(MODELS_DIR / "information_calibration.json")
    load_information_calibration(DATA_DIR / "information_calibration.json")

    # Map form token IDs to form names
    form_token_to_name = {}
    for name, tok_id in tokenizer.FORM_TO_FORM_TOKEN.items():
        form_token_to_name[tok_id] = name

    # Group sequences by form token (scan first ~15 tokens of each sequence)
    form_groups: dict[str, list[list[int]]] = {}
    unclassified = []
    for seq in sequences:
        found_form = None
        for tok in seq[:15]:
            if tok in form_token_to_name:
                found_form = form_token_to_name[tok]
                break
        if found_form:
            form_groups.setdefault(found_form, []).append(seq)
        else:
            unclassified.append(seq)

    console.print(f"\n[bold]Sequence counts by form:[/bold]")
    for form_name, seqs in sorted(form_groups.items(), key=lambda x: -len(x[1])):
        console.print(f"  {form_name}: {len(seqs)}")
    if unclassified:
        console.print(f"  (unclassified): {len(unclassified)}")

    dims = [
        "voice_leading",
        "statistical",
        "structural",
        "contrapuntal",
        "completeness",
        "thematic_recall",
    ]
    baseline_mix = {
        "shuffled": 0.45,
        "random": 0.35,
        "repetitive": 0.20,
    }
    # Soft priors: still data-driven, but prevents domination by any single metric.
    form_priors: dict[str, dict[str, float]] = {
        "fugue": {
            "voice_leading": 0.18,
            "statistical": 0.08,
            "structural": 0.24,
            "contrapuntal": 0.24,
            "completeness": 0.06,
            "thematic_recall": 0.20,
        },
        "invention": {
            "voice_leading": 0.21,
            "statistical": 0.10,
            "structural": 0.23,
            "contrapuntal": 0.23,
            "completeness": 0.07,
            "thematic_recall": 0.16,
        },
        "sinfonia": {
            "voice_leading": 0.20,
            "statistical": 0.10,
            "structural": 0.23,
            "contrapuntal": 0.23,
            "completeness": 0.07,
            "thematic_recall": 0.17,
        },
    }
    prior_mix_by_form = {
        "fugue": 0.45,
        "invention": 0.30,
        "sinfonia": 0.30,
    }

    all_form_weights: dict[str, dict[str, float]] = {}
    all_form_results: dict[str, dict] = {}

    def _normalize(raw: dict[str, float]) -> dict[str, float]:
        positive = {k: max(0.0, float(v)) for k, v in raw.items()}
        total = sum(positive.values())
        if total <= 1e-9:
            uniform = 1.0 / max(1, len(positive))
            return {k: uniform for k in positive}
        return {k: positive[k] / total for k in positive}

    def _shift_weight(
        weights: dict[str, float],
        *,
        from_dim: str,
        to_dims: list[tuple[str, float]],
        amount: float,
    ) -> None:
        """Move weight from one dimension to one or more target dimensions."""
        if amount <= 0:
            return
        take = min(amount, max(0.0, weights.get(from_dim, 0.0)))
        if take <= 0:
            return
        weights[from_dim] = weights.get(from_dim, 0.0) - take
        mix_total = sum(max(0.0, r) for _, r in to_dims)
        if mix_total <= 0:
            return
        for dim, ratio in to_dims:
            if ratio <= 0:
                continue
            weights[dim] = weights.get(dim, 0.0) + take * (ratio / mix_total)

    def _apply_form_constraints(form_name: str, weights: dict[str, float]) -> dict[str, float]:
        """Apply small form-specific limits so calibration remains musically aligned."""
        w = dict(weights)
        form_constraints: dict[str, dict[str, object]] = {
            "fugue": {
                "thematic_cap": 0.30,
                "thematic_shift_mix": [("structural", 0.65), ("contrapuntal", 0.35)],
                "structural_floor": 0.14,
                "donor_floors": [("thematic_recall", 0.16), ("voice_leading", 0.14), ("statistical", 0.05)],
            },
            "invention": {
                "thematic_cap": 0.26,
                "thematic_shift_mix": [("structural", 0.60), ("contrapuntal", 0.40)],
                "structural_floor": 0.16,
                "donor_floors": [("thematic_recall", 0.13), ("voice_leading", 0.14), ("statistical", 0.06)],
            },
            "sinfonia": {
                "thematic_cap": 0.27,
                "thematic_shift_mix": [("structural", 0.58), ("contrapuntal", 0.42)],
                "structural_floor": 0.16,
                "donor_floors": [("thematic_recall", 0.14), ("voice_leading", 0.14), ("statistical", 0.06)],
            },
        }

        cfg = form_constraints.get(form_name)
        if cfg:
            thematic_cap = float(cfg["thematic_cap"])
            # Do not let thematic recall dominate near-tied candidates.
            if w.get("thematic_recall", 0.0) > thematic_cap:
                excess = w["thematic_recall"] - thematic_cap
                _shift_weight(
                    w,
                    from_dim="thematic_recall",
                    to_dims=cfg["thematic_shift_mix"],  # type: ignore[arg-type]
                    amount=excess,
                )

            # Keep structural perception meaningful after recalibration.
            structural_floor = float(cfg["structural_floor"])
            if w.get("structural", 0.0) < structural_floor:
                needed = structural_floor - w["structural"]
                # Pull mostly from dimensions that can over-dominate rankings.
                for donor, donor_floor in cfg["donor_floors"]:  # type: ignore[assignment]
                    if needed <= 1e-9:
                        break
                    available = max(0.0, w.get(donor, 0.0) - donor_floor)
                    moved = min(available, needed)
                    if moved > 0:
                        w[donor] -= moved
                        w["structural"] += moved
                        needed -= moved

        return _normalize(w)

    def _blend_with_prior(form_name: str, data_weights: dict[str, float]) -> dict[str, float]:
        prior = form_priors.get(form_name)
        if not prior:
            return data_weights
        prior_mix = float(prior_mix_by_form.get(form_name, 0.0))
        prior_mix = max(0.0, min(0.90, prior_mix))
        blended = {}
        for dim in dims:
            blended[dim] = (
                (1.0 - prior_mix) * data_weights.get(dim, 0.0)
                + prior_mix * prior.get(dim, data_weights.get(dim, 0.0))
            )
        return _normalize(blended)

    for form_name, form_seqs in sorted(form_groups.items(), key=lambda x: -len(x[1])):
        n = min(sample_size, len(form_seqs))
        if n < 10:
            console.print(f"\n[yellow]Skipping {form_name} (only {n} sequences)[/yellow]")
            continue

        sample = rng.sample(form_seqs, n)
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]{form_name.upper()} ({n} sequences)[/bold]")
        console.print('=' * 60)

        # Score real Bach and baseline perturbations.
        corpus_breakdowns = []
        for seq in sample:
            try:
                comp = tokenizer.decode(seq)
                tokens = tokenizer.encode(comp, form=form_name)
                sb = score_composition(
                    comp,
                    token_sequence=tokens,
                    vocab_size=tokenizer.vocab_size,
                    form=form_name,
                )
                corpus_breakdowns.append(sb)
            except Exception:
                pass

        def _build_shuffled(comp: VoiceComposition) -> VoiceComposition:
            shuffled_voices = []
            for voice in comp.voices:
                if not voice:
                    shuffled_voices.append(voice)
                    continue
                pitches = [n_[2] for n_ in voice]
                rng.shuffle(pitches)
                shuffled_voices.append([(n_[0], n_[1], p) for n_, p in zip(voice, pitches)])
            return VoiceComposition(
                voices=shuffled_voices,
                key_root=comp.key_root,
                key_mode=comp.key_mode,
                source="shuffled",
            )

        def _build_random(comp: VoiceComposition) -> VoiceComposition:
            rand_voices = []
            for voice in comp.voices:
                if not voice:
                    rand_voices.append(voice)
                    continue
                pitches = [n_[2] for n_ in voice]
                lo = min(pitches) if pitches else 48
                hi = max(pitches) if pitches else 72
                rand_voices.append([(n_[0], n_[1], rng.randint(lo, hi)) for n_ in voice])
            return VoiceComposition(
                voices=rand_voices,
                key_root=comp.key_root,
                key_mode=comp.key_mode,
                source="random",
            )

        def _build_repetitive(comp: VoiceComposition) -> VoiceComposition:
            rep_voices = []
            base_pitches = [60, 55, 48, 43]
            for i, voice in enumerate(comp.voices):
                if not voice:
                    rep_voices.append(voice)
                    continue
                pitch = base_pitches[i % len(base_pitches)]
                rep_voices.append([(n_[0], n_[1], pitch) for n_ in voice])
            return VoiceComposition(
                voices=rep_voices,
                key_root=comp.key_root,
                key_mode=comp.key_mode,
                source="repetitive",
            )

        shuffled_breakdowns = []
        random_breakdowns = []
        repetitive_breakdowns = []
        for seq in sample:
            try:
                comp = tokenizer.decode(seq)
                for bucket, variant in [
                    (shuffled_breakdowns, _build_shuffled(comp)),
                    (random_breakdowns, _build_random(comp)),
                    (repetitive_breakdowns, _build_repetitive(comp)),
                ]:
                    try:
                        tokens = tokenizer.encode(variant, form=form_name)
                        sb = score_composition(
                            variant,
                            token_sequence=tokens,
                            vocab_size=tokenizer.vocab_size,
                            form=form_name,
                        )
                        bucket.append(sb)
                    except Exception:
                        pass
            except Exception:
                pass

        if not corpus_breakdowns or not shuffled_breakdowns:
            console.print(f"  [red]Scoring failed for {form_name}[/red]")
            continue

        # Compute discrimination signals per dimension.
        signals = {}
        form_result = {}
        dim_table = Table(title=f"{form_name} — Per-Dimension Discrimination")
        dim_table.add_column("Dimension", style="bold")
        dim_table.add_column("Bach Mean", justify="right")
        dim_table.add_column("Shuffled Mean", justify="right")
        dim_table.add_column("Random Mean", justify="right")
        dim_table.add_column("Repetitive Mean", justify="right")
        dim_table.add_column("Signal", justify="right")
        dim_table.add_column("Discriminates?", justify="center")

        for dim in dims:
            bach_vals = [getattr(b, dim) for b in corpus_breakdowns]
            shuf_vals = [getattr(b, dim) for b in shuffled_breakdowns]
            rand_vals = [getattr(b, dim) for b in random_breakdowns]
            rep_vals = [getattr(b, dim) for b in repetitive_breakdowns]
            bach_mean = float(np.mean(bach_vals))
            shuf_mean = float(np.mean(shuf_vals))
            rand_mean = float(np.mean(rand_vals)) if rand_vals else 0.0
            rep_mean = float(np.mean(rep_vals)) if rep_vals else 0.0
            gap_shuf = bach_mean - shuf_mean
            gap_rand = bach_mean - rand_mean
            gap_rep = bach_mean - rep_mean
            # Multi-baseline signal with square-root compression to avoid domination.
            signal_raw = (
                max(0.0, gap_shuf) * baseline_mix["shuffled"]
                + max(0.0, gap_rand) * baseline_mix["random"]
                + max(0.0, gap_rep) * baseline_mix["repetitive"]
            )
            signal = max(0.001, float(np.sqrt(signal_raw)))
            signals[dim] = signal

            form_result[dim] = {
                "bach_mean": bach_mean,
                "shuffled_mean": shuf_mean,
                "random_mean": rand_mean,
                "repetitive_mean": rep_mean,
                "gap_shuffled": gap_shuf,
                "gap_random": gap_rand,
                "gap_repetitive": gap_rep,
                "signal_raw": signal_raw,
                "signal": signal,
            }

            if signal_raw > 0.15:
                quality = "[bold green]Excellent[/bold green]"
            elif signal_raw > 0.08:
                quality = "[green]Good[/green]"
            elif signal_raw > 0.03:
                quality = "[yellow]Weak[/yellow]"
            else:
                quality = "[red]Dead weight[/red]"

            dim_table.add_row(
                dim,
                f"{bach_mean:.3f}",
                f"{shuf_mean:.3f}",
                f"{rand_mean:.3f}",
                f"{rep_mean:.3f}",
                f"{signal:.3f}",
                quality,
            )

        console.print(dim_table)

        # Derive data-driven weights, then blend with lightweight form prior.
        data_weights = _normalize(signals)
        weights = _blend_with_prior(form_name, data_weights)
        weights = _apply_form_constraints(form_name, weights)

        # Display suggested weights
        weight_table = Table(title=f"{form_name} — Suggested Weights")
        weight_table.add_column("Dimension", style="bold")
        weight_table.add_column("Current", justify="right")
        weight_table.add_column("Data-only", justify="right")
        weight_table.add_column("Suggested", justify="right")
        weight_table.add_column("Change", justify="right")

        for dim in sorted(dims, key=lambda d: -weights[d]):
            current = DEFAULT_WEIGHTS.get(dim, 0.0)
            data_only = data_weights[dim]
            suggested = weights[dim]
            delta = suggested - current
            sign = "+" if delta > 0 else ""
            weight_table.add_row(
                dim,
                f"{current:.3f}",
                f"{data_only:.3f}",
                f"[bold]{suggested:.3f}[/bold]",
                f"{sign}{delta:.3f}",
            )

        console.print(weight_table)

        all_form_weights[form_name] = weights
        all_form_results[form_name] = form_result

    # Summary: all forms side by side
    if len(all_form_weights) > 1:
        console.print(f"\n{'=' * 72}")
        console.print("[bold]SUGGESTED WEIGHTS BY FORM[/bold]")
        console.print('=' * 72)

        summary = Table(title="Optimal Weights Per Form")
        summary.add_column("Dimension", style="bold")
        for form_name in sorted(all_form_weights.keys()):
            summary.add_column(form_name.capitalize(), justify="right")

        for dim in dims:
            row = [dim]
            for form_name in sorted(all_form_weights.keys()):
                w = all_form_weights[form_name].get(dim, 0.0)
                row.append(f"{w:.3f}")
            summary.add_row(*row)

        console.print(summary)

    # Save results
    cal_path = DATA_DIR / "calibration_forms.json"
    output = {
        "weights_by_form": all_form_weights,
        "discrimination_by_form": all_form_results,
        "meta": {
            "calibration_method": "multi_baseline_signal_plus_perceptual_prior_v2",
            "baseline_mix": baseline_mix,
            "prior_mix_by_form": prior_mix_by_form,
            "form_priors": form_priors,
        },
    }
    with open(cal_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[green]Form calibration saved to {cal_path}[/green]")

    # Print copy-paste ready Python dict
    console.print("\n[bold]Copy-paste ready weights:[/bold]")
    console.print("FORM_WEIGHTS = {")
    for form_name in sorted(all_form_weights.keys()):
        w = all_form_weights[form_name]
        items = ", ".join(f'"{d}": {w[d]:.3f}' for d in sorted(w, key=lambda d: -w[d]))
        console.print(f'    "{form_name}": {{{items}}},')
    console.print("}")


def _resolve_soundfont(explicit_path: str | None = None) -> Path | None:
    """Resolve a soundfont path, preferring explicit and Jeux14 defaults."""
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    env_sf = os.environ.get("BACH_GEN_SOUNDFONT")
    if env_sf:
        candidates.append(Path(env_sf))

    # Preferred local organ soundfont.
    candidates.append(Path("/Users/tannerfokkens/Downloads/jeux14/Jeux14.SF2"))

    # Common system locations.
    candidates.extend([
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
        Path("/usr/local/share/fluidsynth/FluidR3_GM.sf2"),
        Path("/opt/homebrew/share/fluidsynth/FluidR3_GM.sf2"),
        Path("/usr/share/sounds/sf2/default-GM.sf2"),
    ])

    # Homebrew Cellar fallback.
    cellar_sf2_dir = Path("/opt/homebrew/Cellar/fluid-synth")
    if cellar_sf2_dir.exists():
        try:
            for sf in sorted(cellar_sf2_dir.rglob("*.sf2")):
                if sf.is_file():
                    candidates.append(sf)
        except Exception:
            pass

    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.expanduser()
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists() and cand.is_file():
            return cand
    return None


def _prepare_tempo_scaled_midi(
    midi_path: Path,
    out_path: Path,
    speed: float,
    spatialize_voices: bool = True,
) -> Path:
    """Write a MIDI with scaled tempo map (no audio time-stretching)."""
    import mido
    import shutil

    if speed <= 0:
        raise ValueError("record speed must be > 0")

    if abs(speed - 1.0) < 1e-6 and not spatialize_voices:
        shutil.copy2(midi_path, out_path)
        return out_path

    mid = mido.MidiFile(str(midi_path))
    found_tempo = False
    for track_idx, track in enumerate(mid.tracks):
        for msg_idx, msg in enumerate(track):
            if msg.type != "set_tempo":
                continue
            found_tempo = True
            new_tempo = max(1, int(round(msg.tempo / speed)))
            if new_tempo != msg.tempo:
                mid.tracks[track_idx][msg_idx] = msg.copy(tempo=new_tempo)

    if not found_tempo:
        tempo_msg = mido.MetaMessage(
            "set_tempo",
            tempo=max(1, int(round(500000 / speed))),
            time=0,
        )
        if mid.tracks:
            mid.tracks[0].insert(0, tempo_msg)
        else:
            new_track = mido.MidiTrack()
            new_track.append(tempo_msg)
            mid.tracks.append(new_track)

    if spatialize_voices:
        # Compute average pitch per MIDI channel to map higher voices to the right.
        pitch_stats: dict[int, tuple[int, int]] = {}
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                    ch = getattr(msg, "channel", None)
                    if ch is None:
                        continue
                    total, count = pitch_stats.get(ch, (0, 0))
                    pitch_stats[ch] = (total + int(msg.note), count + 1)

        if pitch_stats:
            channel_order = sorted(
                pitch_stats.keys(),
                key=lambda ch: (pitch_stats[ch][0] / max(1, pitch_stats[ch][1])),
                reverse=True,  # highest average pitch first
            )

            # Pan range: right-biased highs, left-biased lows.
            # MIDI pan: 0=left, 64=center, 127=right.
            n = len(channel_order)
            if n == 1:
                pan_map = {channel_order[0]: 64}
            else:
                right = 104
                left = 24
                step = (right - left) / (n - 1)
                pan_map = {
                    ch: int(round(right - i * step))
                    for i, ch in enumerate(channel_order)
                }

            assigned_channels: set[int] = set()
            for track in mid.tracks:
                track_channels = sorted({msg.channel for msg in track if hasattr(msg, "channel")})
                for ch in track_channels:
                    if ch in assigned_channels or ch not in pan_map:
                        continue
                    track.insert(
                        0,
                        mido.Message("control_change", channel=ch, control=10, value=pan_map[ch], time=0),
                    )
                    assigned_channels.add(ch)

    mid.save(str(out_path))
    return out_path


def _render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont: Path,
    speed: float = 0.75,
    reverb_mix: float = 0.18,
    stereo_width: float = 1.20,
    spatialize_voices: bool = True,
    swap_stereo: bool = False,
    sample_rate: int = 48000,
) -> Path:
    """Render MIDI to WAV via FluidSynth + FFmpeg post-processing."""
    import shutil
    import subprocess

    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth is required for audio rendering.")

    wav_path = wav_path.expanduser()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    raw_wav = wav_path.with_name(f"{wav_path.stem}.raw.wav")
    scaled_midi = wav_path.with_name(f"{wav_path.stem}.speed.mid")

    render_midi = midi_path
    if abs(speed - 1.0) > 1e-6:
        render_midi = _prepare_tempo_scaled_midi(
            midi_path, scaled_midi, speed, spatialize_voices=spatialize_voices,
        )
    elif spatialize_voices:
        render_midi = _prepare_tempo_scaled_midi(
            midi_path, scaled_midi, 1.0, spatialize_voices=True,
        )

    synth_cmd = [
        "fluidsynth",
        "-ni",
        "-F",
        str(raw_wav),
        "-r",
        str(sample_rate),
        "-g",
        "0.9",
        str(soundfont),
        str(render_midi),
    ]
    subprocess.run(synth_cmd, check=True)
    if not raw_wav.exists():
        raise RuntimeError(
            f"FluidSynth did not produce output file: {raw_wav}. "
            "Check output path permissions and soundfont validity."
        )

    try:
        if shutil.which("ffmpeg") is None:
            if reverb_mix > 0 or abs(stereo_width - 1.0) > 1e-6:
                raise RuntimeError(
                    "ffmpeg is required for reverb/stereo processing. "
                    "Install ffmpeg or render with reverb_mix=0 and stereo_width=1.0."
                )
            raw_wav.replace(wav_path)
            return wav_path

        mix = max(0.0, min(1.0, float(reverb_mix)))
        width = max(0.25, min(2.0, float(stereo_width)))

        filters: list[str] = []
        if mix > 0:
            wet_1 = 0.16 * mix
            wet_2 = 0.08 * mix
            filters.append(f"aecho=0.8:0.88:45|90:{wet_1:.4f}|{wet_2:.4f}")
        if abs(width - 1.0) > 1e-6:
            filters.append(f"stereotools=mlev=1.0:slev={width:.4f}")
        if swap_stereo:
            filters.append("pan=stereo|c0=c1|c1=c0")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_wav),
        ]
        if filters:
            ffmpeg_cmd += ["-af", ",".join(filters)]
        ffmpeg_cmd += [
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            str(wav_path),
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        return wav_path
    finally:
        if raw_wav.exists():
            raw_wav.unlink(missing_ok=True)
        if scaled_midi.exists():
            scaled_midi.unlink(missing_ok=True)


@cli.command()
@click.argument("midi_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output-dir", "-d", default="output", type=click.Path(),
              help="Directory to scan for MIDI files (if no files given)")
@click.option("--tempo", default=120, type=int, help="Playback tempo in BPM")
@click.option("--soundfont", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to .sf2/.sf3 soundfont (defaults to auto-detect; prefers Jeux14).")
@click.option("--list", "list_only", is_flag=True, help="List available MIDI files without playing")
def play(
    midi_files: tuple[str, ...],
    output_dir: str,
    tempo: int,
    soundfont: str | None,
    list_only: bool,
) -> None:
    """Play MIDI files for quick audition.

    If MIDI_FILES are given, plays them in order.
    Otherwise, lists and plays files from the output directory.

    Requires a MIDI player: FluidSynth (recommended), timidity, or macOS afplay.
    """
    import subprocess
    import shutil

    output_path = Path(output_dir)

    # Gather files
    if midi_files:
        files = [Path(f) for f in midi_files]
    else:
        if not output_path.exists():
            console.print(f"[red]Output directory '{output_path}' not found.[/red]")
            sys.exit(1)
        files = sorted(output_path.glob("*.mid")) + sorted(output_path.glob("*.midi"))

    if not files:
        console.print("[yellow]No MIDI files found.[/yellow]")
        sys.exit(0)

    # List mode
    if list_only:
        table = Table(title="Available MIDI Files")
        table.add_column("#", style="bold")
        table.add_column("File")
        table.add_column("Size", justify="right")
        for i, f in enumerate(files, 1):
            size = f.stat().st_size
            table.add_row(str(i), str(f.name), f"{size:,} B")
        console.print(table)
        return

    # Detect available player
    player = None
    player_args = []

    if shutil.which("fluidsynth"):
        player = "fluidsynth"
        sf_path = _resolve_soundfont(soundfont)
        if sf_path:
            player_args = ["fluidsynth", "-ni", str(sf_path)]
            console.print(f"  Soundfont: {sf_path}")
        else:
            console.print("[yellow]FluidSynth found but no soundfont detected.[/yellow]")
            console.print("  Install one: brew install fluid-synth && brew install soundfont-fluid")
            player = None

    if not player and shutil.which("timidity"):
        player = "timidity"
        player_args = ["timidity"]

    if not player and sys.platform == "darwin":
        # macOS: try using the built-in MIDI playback via afplay doesn't work for MIDI,
        # but we can convert via fluidsynth or use a Python MIDI player
        pass

    # Fallback: try pygame
    if not player:
        try:
            import pygame
            player = "pygame"
        except ImportError:
            pass

    if not player:
        console.print("[red]No MIDI player found.[/red]")
        console.print("Install one of:")
        console.print("  brew install fluid-synth    (recommended)")
        console.print("  brew install timidity")
        console.print("  pip install pygame")
        sys.exit(1)

    console.print(f"[bold]Playing {len(files)} MIDI file(s) via {player}[/bold]")
    console.print("  Press Ctrl+C to skip to next, Ctrl+C twice to quit\n")

    for i, f in enumerate(files, 1):
        console.print(f"  [{i}/{len(files)}] [bold]{f.name}[/bold]")

        try:
            if player == "pygame":
                _play_pygame(f, tempo)
            else:
                cmd = player_args + [str(f)]
                subprocess.run(cmd, timeout=120)
        except KeyboardInterrupt:
            console.print("  [dim]Skipped[/dim]")
            continue
        except subprocess.TimeoutExpired:
            console.print("  [dim]Timeout[/dim]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    console.print("\n[green]Done.[/green]")


def _play_pygame(midi_path: Path, tempo: int) -> None:
    """Play MIDI via pygame.mixer.music."""
    import pygame
    import time

    pygame.mixer.init()
    pygame.mixer.music.load(str(midi_path))
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


if __name__ == "__main__":
    cli()
