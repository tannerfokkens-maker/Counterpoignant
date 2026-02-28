"""CLI interface for bach-gen: generate, train, evaluate, prepare-data."""

from __future__ import annotations

import json
import logging
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
            or name in ("VOICE_SEP", "SUBJECT_START", "SUBJECT_END")
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
) -> tuple[list[list[int]], list[str]]:
    """Tokenize a batch of compositions with optional dual encoding."""
    from bach_gen.data.analysis import analyze_composition

    sequences: list[list[int]] = []
    piece_ids: list[str] = []

    for i, item in enumerate(items):
        item_form = forms[i] if i < len(forms) else "chorale"
        time_sig = item.time_signature if hasattr(item, "time_signature") else (4, 4)
        num_bars = compute_measure_count(item.voices, time_sig)

        labels = analyze_composition(item, time_sig)

        tokens = tokenizer.encode(
            item, form=item_form, length_bars=num_bars,
            texture=labels["texture"], imitation=labels["imitation"],
            harmonic_rhythm=labels["harmonic_rhythm"],
            harmonic_tension=labels["harmonic_tension"],
            chromaticism=labels["chromaticism"],
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
            )
            if len(tokens_seq) >= 20:
                sequences.append(tokens_seq)
                piece_ids.append(item.source)

    return sequences, piece_ids


def _tokenize_batch_task(task: tuple[int, list, list[str], str, bool]) -> tuple[int, list[list[int]], list[str], int]:
    """Worker entrypoint for parallel Step-3 tokenization."""
    batch_idx, items, forms, tokenizer_type, no_sequential = task

    from bach_gen.data.tokenizer import BachTokenizer

    if tokenizer_type == "scale-degree":
        from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
        tokenizer = ScaleDegreeTokenizer()
    else:
        tokenizer = BachTokenizer()

    seqs, pids = _tokenize_items(items, forms, tokenizer, no_sequential)
    return batch_idx, seqs, pids, len(items)


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
def prepare_data(mode: str, voices: int | None, tokenizer_type: str, max_seq_len: int,
                 no_chunk: bool, data_dir: str | None, composer_filter: str | None,
                 no_sequential: bool, max_source_voices: int,
                 max_groups_per_work: int, pair_strategy: str,
                 max_pairs_per_work: int, sonata_policy: str,
                 workers: int | None) -> None:
    """Extract Bach corpus, tokenize, and cache statistics."""
    from collections import Counter, defaultdict
    from bach_gen.data.corpus import get_all_works, _original_source
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
        filter_list = list(DEFAULT_PREPARE_COMPOSER_FILTER)
        filter_origin = "default"

    # Step 1: Load and extract works (parse + extract in workers)
    if filter_list:
        suffix = " [default]" if filter_origin == "default" else ""
        filter_desc = f" (filter: {', '.join(filter_list)}{suffix})"
    else:
        filter_desc = " (filter: disabled)"
    console.print(f"[bold]Step 1:[/] Loading and extracting from corpus...{filter_desc}")
    if is_all_mode:
        console.print(f"  Mode: all (auto-detect form and voice count per piece)")
    else:
        console.print(f"  Mode: {mode} ({num_voices} voices)")
    console.print(f"  Tokenizer: {tokenizer_type}")
    console.print(f"  Max source voices: {max_source_voices}")
    console.print(f"  Sonata policy: {sonata_policy}")
    import os as _os
    effective_workers = workers if workers is not None else min(_os.cpu_count() or 1, 8)
    console.print(f"  Workers: {effective_workers}")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading and extracting...", total=None)
        works_with_forms = get_all_works(
            composer_filter=filter_list, max_workers=workers,
            max_source_voices=max_source_voices,
            max_groups_per_work=max_groups_per_work,
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
    batch_tasks: list[tuple[int, list, list[str], str, bool]] = []
    for batch_idx, start in enumerate(range(0, len(items_to_tokenize), batch_size)):
        end = start + batch_size
        batch_tasks.append(
            (
                batch_idx,
                items_to_tokenize[start:end],
                forms_to_tokenize[start:end],
                tokenizer_type,
                no_sequential,
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
            for _, batch_items, batch_forms, _, _ in batch_tasks:
                batch_seqs, batch_ids = _tokenize_items(batch_items, batch_forms, tokenizer, no_sequential)
                sequences.extend(batch_seqs)
                piece_ids.extend(batch_ids)
                progress.advance(task, len(batch_items))

    console.print(f"  Tokenized {len(sequences)} sequences")
    console.print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Conditioning token distribution histograms
    _print_conditioning_histograms(sequences, tokenizer)

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
              help="Two-phase training: pre-train on --data-dir, fine-tune on --finetune-data-dir or --finetune style subset")
@click.option("--pretrain-epochs", default=300, type=int,
              help="Epochs for pre-training phase (curriculum mode, default: 300)")
@click.option("--finetune-data-dir", default="data/bach", type=click.Path(),
              help="Data directory for fine-tuning phase (default: data/bach)")
@click.option("--finetune", "finetune_style",
              type=str,
              default=None,
              help="Fine-tune subset from same --data-dir (style token or composer substring, e.g. 'bach' or 'beethoven')")
@click.option("--finetune-lr", default=1e-4, type=float,
              help="Learning rate for fine-tuning phase (default: 1e-4)")
@click.option("--pretrained-checkpoint", default=None, type=click.Path(exists=True),
              help="Skip curriculum pre-train phase and start fine-tuning from this checkpoint")
@click.option("--drope/--no-drope", default=True,
              help="Enable/disable DroPE recalibration after training (default: enabled)")
@click.option("--drope-epochs", default=10, type=int,
              help="Maximum DroPE recalibration epochs (default: 10)")
@click.option("--drope-lr", default=1e-3, type=float,
              help="Learning rate for DroPE recalibration (default: 1e-3)")
@click.option("--drope-early-stop/--drope-fixed", default=True,
              help="Enable/disable DroPE early stopping (default: enabled)")
@click.option("--drope-patience", default=2, type=int,
              help="DroPE early-stop patience in epochs (default: 2)")
@click.option("--drope-min-delta", default=1e-4, type=float,
              help="Minimum DroPE metric improvement to reset patience (default: 1e-4)")
@click.option("--drope-min-epochs", default=4, type=int,
              help="Minimum DroPE epochs before early stopping can trigger (default: 4)")
@click.option("--early-stop/--no-early-stop", default=True,
              help="Enable/disable early stopping on val loss plateau (default: enabled)")
@click.option("--es-patience", default=20, type=int,
              help="Early-stop patience: consecutive non-improving val checks (default: 20)")
@click.option("--es-min-delta", default=1e-4, type=float,
              help="Minimum val loss improvement to reset patience (default: 1e-4)")
@click.option("--es-min-epochs", default=10, type=int,
              help="Minimum epochs before early stopping can trigger (default: 10)")
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
          drope: bool, drope_epochs: int, drope_lr: float,
          drope_early_stop: bool, drope_patience: int, drope_min_delta: float,
          drope_min_epochs: int,
          early_stop: bool, es_patience: int, es_min_delta: float, es_min_epochs: int,
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

        console.print(f"\n[bold]Curriculum training:[/bold]")
        console.print(f"  Phase 1 (pre-train): epochs 1–{pretrain_epochs} on {train_data_dir}")
        console.print(f"  Phase 2 (fine-tune): epochs {pretrain_epochs+1}–{epochs} on {ft_source_desc}")
        console.print(f"  Fine-tune LR: {finetune_lr}")
        if early_stop:
            console.print(f"  Early stop: enabled (patience={es_patience}, min_delta={es_min_delta}, min_epochs={es_min_epochs})")
        else:
            console.print(f"  Early stop: disabled")

        # --- Phase 1: Pre-train (optional) ---
        if skip_pretrain:
            console.print(
                f"\n[bold]Phase 1:[/bold] Skipped pre-training; loading checkpoint {pretrained_checkpoint}"
            )
            trainer.resume_from_checkpoint(pretrained_checkpoint)
            pt_history = {"train_loss": [], "val_loss": [], "lr": []}
        else:
            console.print(f"\n[bold]Phase 1: Pre-training for {pretrain_epochs} epochs...[/bold]")
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(), TaskProgressColumn(), console=console,
            ) as progress:
                task = progress.add_task("Pre-training", total=pretrain_epochs - start_epoch + 1)

                def pt_callback(epoch, train_loss, val_loss):
                    desc = f"[pretrain] Epoch {epoch}/{pretrain_epochs} | loss={train_loss:.4f}"
                    if val_loss is not None:
                        desc += f" | val_loss={val_loss:.4f}"
                    progress.update(task, advance=1, description=desc)

                pt_history = trainer.train(
                    epochs=pretrain_epochs,
                    start_epoch=start_epoch,
                    log_interval=max(1, pretrain_epochs // 20),
                    val_interval=(val_interval if val_interval is not None else max(1, pretrain_epochs // 20)),
                    progress_callback=pt_callback,
                    early_stop=early_stop,
                    patience=es_patience,
                    min_delta=es_min_delta,
                    min_epochs=es_min_epochs,
                )

            console.print(f"  Pre-train final loss: {pt_history['train_loss'][-1]:.4f}")
            if pt_history.get("stop_reason") and pt_history["stop_reason"] != "max_epochs_reached":
                console.print(f"  Pre-train stopped early: {pt_history['stop_reason']} "
                              f"(ran {pt_history.get('epochs_ran', '?')} epochs)")

        # --- Phase 2: Fine-tune ---
        console.print(f"\n[bold]Phase 2: Preparing fine-tune data from {ft_source_desc}...[/bold]")

        ft_train_ds, ft_val_ds = create_dataset(ft_sequences, seq_len=seq_len, piece_ids=ft_piece_ids)
        console.print(f"  Fine-tune train: {len(ft_train_ds)}, val: {len(ft_val_ds)}")

        trainer.reset_for_finetuning(ft_train_ds, ft_val_ds, lr=finetune_lr)

        console.print(f"\n[bold]Fine-tuning for {finetune_epochs} epochs...[/bold]")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Fine-tuning", total=finetune_epochs)

            def ft_callback(epoch, train_loss, val_loss):
                desc = f"[finetune] Epoch {epoch}/{epochs} | loss={train_loss:.4f}"
                if val_loss is not None:
                    desc += f" | val_loss={val_loss:.4f}"
                progress.update(task, advance=1, description=desc)

            ft_history = trainer.train(
                epochs=epochs,
                start_epoch=pretrain_epochs + 1,
                log_interval=max(1, finetune_epochs // 20),
                val_interval=(val_interval if val_interval is not None else max(1, finetune_epochs // 20)),
                progress_callback=ft_callback,
                early_stop=early_stop,
                patience=es_patience,
                min_delta=es_min_delta,
                min_epochs=pretrain_epochs + es_min_epochs,
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
                log_interval=max(1, epochs // 20),
                val_interval=(val_interval if val_interval is not None else max(1, epochs // 20)),
                progress_callback=callback,
                early_stop=early_stop,
                patience=es_patience,
                min_delta=es_min_delta,
                min_epochs=es_min_epochs,
            )
        sequences_for_cal = sequences

    # DroPE recalibration phase
    if drope:
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
@click.option("--temperature", default=0.9, type=float, help="Sampling temperature")
@click.option("--min-p", default=0.03, type=float,
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
@click.option("--voice-by-voice", is_flag=True, default=False,
              help="Use voice-by-voice (sequential) generation")
@click.option("--provide-voice", default=None, type=click.Path(exists=True),
              help="Path to MIDI file for voice 1 (use with --voice-by-voice)")
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
    voice_by_voice: bool,
    provide_voice: str | None,
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
    table.add_column("Info", justify="right")
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
            f"{s.information:.3f}",
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
def evaluate(midi_file: str, mode: str | None) -> None:
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
    score = score_composition(comp, token_sequence=tokens, vocab_size=tokenizer.vocab_size)

    # Display
    table = Table(title=f"Evaluation Scores ({mode})")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted", justify="right")

    weights = {
        "Voice Leading": (score.voice_leading, 0.25),
        "Statistical Sim.": (score.statistical, 0.15),
        "Structural": (score.structural, 0.15),
        "Information": (score.information, 0.15),
        "Contrapuntal": (score.contrapuntal, 0.10),
        "Completeness": (score.completeness, 0.10),
        "Thematic Recall": (score.thematic_recall, 0.10),
    }

    for name, (val, w) in weights.items():
        table.add_row(name, f"{val:.3f}", f"{w:.2f}", f"{val * w:.3f}")

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
            "information", "completeness", "thematic_recall"]
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
@click.argument("midi_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output-dir", "-d", default="output", type=click.Path(),
              help="Directory to scan for MIDI files (if no files given)")
@click.option("--tempo", default=120, type=int, help="Playback tempo in BPM")
@click.option("--list", "list_only", is_flag=True, help="List available MIDI files without playing")
def play(midi_files: tuple[str, ...], output_dir: str, tempo: int, list_only: bool) -> None:
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
        # Try to find a soundfont
        sf_paths = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/local/share/fluidsynth/FluidR3_GM.sf2",
            "/opt/homebrew/share/fluidsynth/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
        ]
        # Also check Homebrew Cellar for any .sf2 files
        try:
            cellar_sf2_dir = Path("/opt/homebrew/Cellar/fluid-synth")
            if cellar_sf2_dir.exists():
                for sf in cellar_sf2_dir.rglob("*.sf2"):
                    # Skip symlinks with special chars, use the real file
                    if sf.is_file() and not sf.is_symlink():
                        sf_paths.insert(0, str(sf))
        except Exception:
            pass

        soundfont = None
        for sf in sf_paths:
            if Path(sf).exists():
                soundfont = sf
                break

        if soundfont:
            player_args = ["fluidsynth", "-ni", soundfont]
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
