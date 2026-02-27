"""CLI interface for bach-gen: generate, train, evaluate, prepare-data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from bach_gen.utils.constants import (
    FORM_DEFAULTS, VALID_FORMS, DEFAULT_SEQ_LEN,
    METER_MAP, LENGTH_NAMES, METER_NAMES,
    compute_measure_count, length_bucket,
)

console = Console()

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
    piece_ids: list[str] | None = None,
) -> tuple[list[list[int]], list[str]]:
    """Split long sequences into overlapping chunks.

    - Sequences at or under max_seq_len are kept as-is.
    - Longer sequences are split into windows of max_seq_len with overlap.
    - Each chunk after the first gets BOS prepended so the model always sees
      a valid start token.
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
            start = 0
            while start < len(seq):
                end = start + max_seq_len
                chunk = seq[start:end]

                # Prepend BOS to continuation chunks so the model
                # always sees a valid sequence start
                if start > 0:
                    chunk = [bos_token] + chunk[:max_seq_len - 1]

                # Only keep chunks that are at least 25% of max_seq_len
                # to avoid tiny tail fragments
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


@cli.command()
@click.option("--mode", "-m", type=click.Choice(VALID_FORMS), default="2-part",
              help="Composition mode (determines voice count)")
@click.option("--voices", type=int, default=None,
              help="Override number of voices (default: from mode)")
@click.option("--tokenizer", "tokenizer_type",
              type=click.Choice(["absolute", "scale-degree"]), default="absolute",
              help="Tokenizer type: absolute (default) or scale-degree (key-agnostic)")
@click.option("--max-seq-len", default=DEFAULT_SEQ_LEN, type=int,
              help="Drop sequences longer than this (default: model max_seq_len)")
@click.option("--no-chunk", is_flag=True, default=False,
              help="Drop long sequences instead of chunking them")
@click.option("--data-dir", default=None, type=click.Path(),
              help="Output directory for prepared data (default: data/)")
@click.option("--composer-filter", default=None, type=str,
              help="Comma-separated composer/style names to include (e.g. 'bach' or 'bach,baroque')")
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
def prepare_data(mode: str, voices: int | None, tokenizer_type: str, max_seq_len: int,
                 no_chunk: bool, data_dir: str | None, composer_filter: str | None,
                 no_sequential: bool, max_source_voices: int,
                 max_groups_per_work: int, pair_strategy: str,
                 max_pairs_per_work: int) -> None:
    """Extract Bach corpus, tokenize, and cache statistics."""
    from bach_gen.data.corpus import get_all_works
    from bach_gen.data.extraction import extract_voice_pairs, extract_voice_groups, detect_form, VoicePair, VoiceComposition
    from bach_gen.data.augmentation import augment_to_all_keys
    from bach_gen.data.tokenizer import BachTokenizer
    from bach_gen.data.dataset import compute_corpus_stats

    use_scale_degree = tokenizer_type == "scale-degree"

    is_all_mode = mode == "all"
    if is_all_mode:
        num_voices = voices or 4  # placeholder; actual count detected per piece
    else:
        num_voices = voices or FORM_DEFAULTS[mode][0]

    # Resolve output directory
    out_dir = Path(data_dir) if data_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse composer filter
    filter_list = None
    if composer_filter:
        filter_list = [c.strip() for c in composer_filter.split(",") if c.strip()]

    # Step 1: Load works
    filter_desc = f" (filter: {', '.join(filter_list)})" if filter_list else ""
    console.print(f"[bold]Step 1:[/] Loading works from corpus...{filter_desc}")
    if is_all_mode:
        console.print(f"  Mode: all (auto-detect form and voice count per piece)")
    else:
        console.print(f"  Mode: {mode} ({num_voices} voices)")
    console.print(f"  Tokenizer: {tokenizer_type}")
    console.print(f"  Max source voices: {max_source_voices}")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading...", total=None)
        works = get_all_works(composer_filter=filter_list)
        progress.update(task, description=f"Loaded {len(works)} works")

    if not works:
        console.print("[red]No works found. Check music21 corpus installation.[/red]")
        console.print("Run: python -c \"import music21; music21.configure.run()\"")
        sys.exit(1)

    # Step 2: Extract voice groups
    if is_all_mode:
        console.print(f"\n[bold]Step 2:[/] Extracting voice groups (auto-detect per piece)...")
    else:
        console.print(f"\n[bold]Step 2:[/] Extracting {num_voices}-voice groups...")

    if is_all_mode:
        # --mode all: detect form and voice count per piece
        from collections import Counter
        compositions: list[VoiceComposition] = []
        form_per_comp: list[str] = []
        form_counter: Counter = Counter()
        voice_count_counter: Counter = Counter()

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Extracting", total=len(works))
            skipped_by_voice_cap = 0
            for desc, score, style in works:
                try:
                    source_parts = len(list(score.parts))
                except Exception:
                    source_parts = 0
                if source_parts > max_source_voices:
                    skipped_by_voice_cap += 1
                    progress.advance(task)
                    continue
                detected_form, detected_nv = detect_form(score, desc, style)
                if voices is not None:
                    detected_nv = voices  # user override
                extracted = extract_voice_groups(
                    score, num_voices=detected_nv, source=desc, form=detected_form,
                )
                if max_groups_per_work > 0:
                    extracted = extracted[:max_groups_per_work]
                for comp in extracted:
                    comp.style = style
                    compositions.append(comp)
                    form_per_comp.append(detected_form)
                    form_counter[detected_form] += 1
                    voice_count_counter[detected_nv] += 1
                progress.advance(task)

        console.print(f"  Extracted {len(compositions)} voice groups from {len(works)} works")
        if skipped_by_voice_cap:
            console.print(f"  Skipped by source voice cap: {skipped_by_voice_cap}")
        console.print(f"  Form distribution: {dict(form_counter.most_common())}")
        console.print(f"  Voice count distribution: {dict(voice_count_counter.most_common())}")

        if not compositions:
            console.print("[red]No voice groups extracted.[/red]")
            sys.exit(1)

        # Step 3: Augment (skip for scale-degree tokenizer)
        if use_scale_degree:
            console.print(f"\n[bold]Step 3:[/] Skipping key augmentation (scale-degree mode)")
            items_to_tokenize = compositions
            forms_to_tokenize = form_per_comp
        else:
            console.print(f"\n[bold]Step 3:[/] Augmenting to all 12 keys...")
            items_to_tokenize = augment_to_all_keys(compositions)
            # Replicate forms for each augmented copy (12 keys per original)
            augment_factor = len(items_to_tokenize) // len(compositions) if compositions else 1
            forms_to_tokenize = []
            for f in form_per_comp:
                forms_to_tokenize.extend([f] * augment_factor)
            console.print(f"  Augmented to {len(items_to_tokenize)} compositions")

        # Step 4: Tokenize
        console.print(f"\n[bold]Step 4:[/] Tokenizing...")
        if use_scale_degree:
            from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
            tokenizer = ScaleDegreeTokenizer()
        else:
            tokenizer = BachTokenizer()

        sequences: list[list[int]] = []
        piece_ids: list[str] = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Tokenizing", total=len(items_to_tokenize))
            for i, item in enumerate(items_to_tokenize):
                item_form = forms_to_tokenize[i] if i < len(forms_to_tokenize) else "chorale"
                time_sig = item.time_signature if hasattr(item, "time_signature") else (4, 4)
                num_bars = compute_measure_count(item.voices, time_sig)

                # Compute analysis labels for Phase 2 conditioning
                from bach_gen.data.analysis import analyze_composition
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

                # Dual encoding: also produce sequential encoding
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

                progress.advance(task)

    elif num_voices == 2:
        pairs: list[VoicePair] = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Extracting", total=len(works))
            skipped_by_voice_cap = 0
            for desc, score, style in works:
                try:
                    source_parts = len(list(score.parts))
                except Exception:
                    source_parts = 0
                if source_parts > max_source_voices:
                    skipped_by_voice_cap += 1
                    progress.advance(task)
                    continue
                extracted = extract_voice_pairs(
                    score,
                    source=desc,
                    pair_strategy=pair_strategy,
                    max_pairs=max_pairs_per_work,
                )
                for pair in extracted:
                    pair.style = style
                pairs.extend(extracted)
                progress.advance(task)
        console.print(f"  Extracted {len(pairs)} voice pairs from {len(works)} works")
        if skipped_by_voice_cap:
            console.print(f"  Skipped by source voice cap: {skipped_by_voice_cap}")

        if not pairs:
            console.print("[red]No voice pairs extracted.[/red]")
            sys.exit(1)

        # Step 3: Augment (skip for scale-degree tokenizer)
        if use_scale_degree:
            console.print(f"\n[bold]Step 3:[/] Skipping key augmentation (scale-degree mode)")
            items_to_tokenize = pairs
        else:
            console.print(f"\n[bold]Step 3:[/] Augmenting to all 12 keys...")
            items_to_tokenize = augment_to_all_keys(pairs)
            console.print(f"  Augmented to {len(items_to_tokenize)} pairs")

        # Step 4: Tokenize
        console.print(f"\n[bold]Step 4:[/] Tokenizing...")
        if use_scale_degree:
            from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
            tokenizer = ScaleDegreeTokenizer()
        else:
            tokenizer = BachTokenizer()

        sequences: list[list[int]] = []
        piece_ids: list[str] = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Tokenizing", total=len(items_to_tokenize))
            for item in items_to_tokenize:
                # Compute measure count for length conditioning
                time_sig = item.time_signature if hasattr(item, "time_signature") else (4, 4)
                if isinstance(item, VoicePair):
                    item_voices = [item.upper, item.lower]
                else:
                    item_voices = item.voices
                num_bars = compute_measure_count(item_voices, time_sig)

                # Compute analysis labels for Phase 2 conditioning
                from bach_gen.data.analysis import analyze_composition
                from bach_gen.data.extraction import VoiceComposition as VC
                if isinstance(item, VoicePair):
                    analysis_comp = VC.from_voice_pair(item)
                else:
                    analysis_comp = item
                labels = analyze_composition(analysis_comp, time_sig)

                tokens = tokenizer.encode(
                    item, form=mode, length_bars=num_bars,
                    texture=labels["texture"], imitation=labels["imitation"],
                    harmonic_rhythm=labels["harmonic_rhythm"],
                    harmonic_tension=labels["harmonic_tension"],
                    chromaticism=labels["chromaticism"],
                )
                if len(tokens) >= 20:
                    sequences.append(tokens)
                    piece_ids.append(item.source)

                # Dual encoding: also produce sequential encoding
                if not no_sequential:
                    tokens_seq = tokenizer.encode_sequential(
                        item, form=mode, length_bars=num_bars,
                        texture=labels["texture"], imitation=labels["imitation"],
                        harmonic_rhythm=labels["harmonic_rhythm"],
                        harmonic_tension=labels["harmonic_tension"],
                        chromaticism=labels["chromaticism"],
                    )
                    if len(tokens_seq) >= 20:
                        sequences.append(tokens_seq)
                        piece_ids.append(item.source)

                progress.advance(task)
    else:
        compositions: list[VoiceComposition] = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Extracting", total=len(works))
            skipped_by_voice_cap = 0
            for desc, score, style in works:
                try:
                    source_parts = len(list(score.parts))
                except Exception:
                    source_parts = 0
                if source_parts > max_source_voices:
                    skipped_by_voice_cap += 1
                    progress.advance(task)
                    continue
                extracted = extract_voice_groups(score, num_voices=num_voices, source=desc, form=mode)
                if max_groups_per_work > 0:
                    extracted = extracted[:max_groups_per_work]
                for comp in extracted:
                    comp.style = style
                compositions.extend(extracted)
                progress.advance(task)
        console.print(f"  Extracted {len(compositions)} {num_voices}-voice groups from {len(works)} works")
        if skipped_by_voice_cap:
            console.print(f"  Skipped by source voice cap: {skipped_by_voice_cap}")

        if not compositions:
            console.print(f"[red]No {num_voices}-voice groups extracted.[/red]")
            sys.exit(1)

        # Step 3: Augment (skip for scale-degree tokenizer)
        if use_scale_degree:
            console.print(f"\n[bold]Step 3:[/] Skipping key augmentation (scale-degree mode)")
            items_to_tokenize = compositions
        else:
            console.print(f"\n[bold]Step 3:[/] Augmenting to all 12 keys...")
            items_to_tokenize = augment_to_all_keys(compositions)
            console.print(f"  Augmented to {len(items_to_tokenize)} compositions")

        # Step 4: Tokenize
        console.print(f"\n[bold]Step 4:[/] Tokenizing...")
        if use_scale_degree:
            from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
            tokenizer = ScaleDegreeTokenizer()
        else:
            tokenizer = BachTokenizer()

        sequences: list[list[int]] = []
        piece_ids: list[str] = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Tokenizing", total=len(items_to_tokenize))
            for item in items_to_tokenize:
                # Compute measure count for length conditioning
                time_sig = item.time_signature if hasattr(item, "time_signature") else (4, 4)
                num_bars = compute_measure_count(item.voices, time_sig)

                # Compute analysis labels for Phase 2 conditioning
                from bach_gen.data.analysis import analyze_composition
                labels = analyze_composition(item, time_sig)

                tokens = tokenizer.encode(
                    item, form=mode, length_bars=num_bars,
                    texture=labels["texture"], imitation=labels["imitation"],
                    harmonic_rhythm=labels["harmonic_rhythm"],
                    harmonic_tension=labels["harmonic_tension"],
                    chromaticism=labels["chromaticism"],
                )
                if len(tokens) >= 20:
                    sequences.append(tokens)
                    piece_ids.append(item.source)

                # Dual encoding: also produce sequential encoding
                if not no_sequential:
                    tokens_seq = tokenizer.encode_sequential(
                        item, form=mode, length_bars=num_bars,
                        texture=labels["texture"], imitation=labels["imitation"],
                        harmonic_rhythm=labels["harmonic_rhythm"],
                        harmonic_tension=labels["harmonic_tension"],
                        chromaticism=labels["chromaticism"],
                    )
                    if len(tokens_seq) >= 20:
                        sequences.append(tokens_seq)
                        piece_ids.append(item.source)

                progress.advance(task)

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
            bos_token=tokenizer.BOS, piece_ids=long_ids,
        )
        console.print(f"  Kept {len(short_seqs)} sequences under {max_seq_len} tokens")
        console.print(f"  Chunked {len(long_seqs)} long sequences into {len(chunked)} windows")
        sequences = short_seqs + chunked
        piece_ids = short_ids + chunked_ids

    # Sequence length stats (after filtering)
    lengths = [len(s) for s in sequences]
    console.print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
                  f"mean={sum(lengths)/len(lengths):.0f}")

    # Step 5: Save
    console.print(f"\n[bold]Step 5:[/] Saving data...")

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
    roundtrip_form = "chorale" if is_all_mode else mode
    re_encoded = tokenizer.encode(decoded, form=roundtrip_form)
    voice_desc = ", ".join(f"v{i+1}={len(v)} notes" for i, v in enumerate(decoded.voices) if v)
    console.print(f"  Original: {len(test_seq)} tokens → Decoded: {voice_desc} "
                  f"→ Re-encoded: {len(re_encoded)} tokens")
    console.print(f"  [green]Round-trip OK[/green]")


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
              help="Two-phase training: pre-train on --data-dir, fine-tune on --finetune-data-dir")
@click.option("--pretrain-epochs", default=300, type=int,
              help="Epochs for pre-training phase (curriculum mode, default: 300)")
@click.option("--finetune-data-dir", default="data/bach", type=click.Path(),
              help="Data directory for fine-tuning phase (default: data/bach)")
@click.option("--finetune-lr", default=1e-4, type=float,
              help="Learning rate for fine-tuning phase (default: 1e-4)")
@click.option("--drope", is_flag=True, default=False,
              help="Enable DroPE recalibration after training (drop positional embeddings)")
@click.option("--drope-epochs", default=10, type=int,
              help="Number of DroPE recalibration epochs (default: 10)")
@click.option("--drope-lr", default=1e-3, type=float,
              help="Learning rate for DroPE recalibration (default: 1e-3)")
@click.option("--fp16", is_flag=True, default=False,
              help="Enable mixed precision (fp16) training — CUDA only")
@click.option("--pos-encoding", type=click.Choice(["rope", "pope"]),
              default="rope", help="Positional encoding: rope (default) or pope")
@click.option("--num-kv-heads", default=None, type=int,
              help="Number of KV heads for GQA (default: same as num_heads = standard MHA)")
def train(epochs: int, lr: float, batch_size: int, seq_len: int | None, mode: str | None,
          accumulation_steps: int, resume: str | None, data_dir: str | None,
          curriculum: bool, pretrain_epochs: int, finetune_data_dir: str,
          finetune_lr: float, drope: bool, drope_epochs: int, drope_lr: float,
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
        mode = mode_info.get("mode", "2-part")

    # Set seq_len from mode defaults if not specified
    if seq_len is None:
        seq_len = FORM_DEFAULTS.get(mode, (2, 768))[1]
        # Scale-degree sequences are ~50-75% longer; bump seq_len by 1.5x
        if mode_info.get("tokenizer_type") == "scale-degree":
            seq_len = int(seq_len * 1.5)

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
    )

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

        ft_data_dir = Path(finetune_data_dir)
        ft_seq_path = ft_data_dir / "sequences.json"
        if not ft_seq_path.exists():
            console.print(f"[red]Fine-tune data not found at {ft_data_dir}. "
                          f"Run 'bach-gen prepare-data --data-dir {ft_data_dir}' first.[/red]")
            sys.exit(1)

        finetune_epochs = epochs - pretrain_epochs

        console.print(f"\n[bold]Curriculum training:[/bold]")
        console.print(f"  Phase 1 (pre-train): epochs 1–{pretrain_epochs} on {train_data_dir}")
        console.print(f"  Phase 2 (fine-tune): epochs {pretrain_epochs+1}–{epochs} on {ft_data_dir}")
        console.print(f"  Fine-tune LR: {finetune_lr}")

        # --- Phase 1: Pre-train ---
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
                val_interval=max(1, pretrain_epochs // 20),
                progress_callback=pt_callback,
            )

        console.print(f"  Pre-train final loss: {pt_history['train_loss'][-1]:.4f}")

        # --- Phase 2: Fine-tune ---
        console.print(f"\n[bold]Phase 2: Loading fine-tune data from {ft_data_dir}...[/bold]")
        with open(ft_seq_path) as f:
            ft_sequences = json.load(f)

        ft_piece_ids = None
        ft_piece_ids_path = ft_data_dir / "piece_ids.json"
        if ft_piece_ids_path.exists():
            with open(ft_piece_ids_path) as f:
                ft_piece_ids = json.load(f)

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
                val_interval=max(1, finetune_epochs // 20),
                progress_callback=ft_callback,
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
                val_interval=max(1, epochs // 20),
                progress_callback=callback,
            )
        sequences_for_cal = sequences

    # DroPE recalibration phase
    if drope:
        console.print(f"\n[bold]DroPE recalibration for {drope_epochs} epochs (lr={drope_lr})...[/bold]")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("DroPE recalibration", total=drope_epochs)

            drope_history = trainer.recalibrate_drope(epochs=drope_epochs, lr=drope_lr)

            progress.update(task, completed=drope_epochs, description="DroPE done!")

        console.print(f"  DroPE final loss: {drope_history['train_loss'][-1]:.4f}")
        console.print(f"  Model marked as drope_trained=True")

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        console.print(f"  Best val loss: {min(history['val_loss']):.4f}")
    console.print(f"  Checkpoints saved to {MODELS_DIR}/")

    # Calibrate information-theoretic metrics
    console.print("\n[bold]Calibrating evaluation metrics...[/bold]")
    from bach_gen.evaluation.information import calibrate_from_corpus
    cal = calibrate_from_corpus(sequences_for_cal[:50], model)
    console.print(f"  Perplexity range: {cal['perplexity_range']}")
    console.print(f"  Entropy range: {cal['entropy_range']}")


@cli.command()
@click.option("--key", "-k", required=True, help="Key (e.g., 'C minor', 'D major')")
@click.option("--subject", "-s", default=None, help="Subject notes (e.g., 'C4 D4 Eb4 F4')")
@click.option("--candidates", "-n", default=100, help="Number of candidates to generate")
@click.option("--top", "-t", default=3, help="Number of top results to return")
@click.option("--temperature", default=0.9, type=float, help="Sampling temperature")
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

    # Auto-detect mode
    mode_path = DATA_DIR / "mode.json"
    mode_info = {}
    if mode_path.exists():
        with open(mode_path) as f:
            mode_info = json.load(f)
    if mode is None:
        mode = mode_info.get("mode", "2-part")

    num_voices = voices or FORM_DEFAULTS[mode][0]

    if max_length is None:
        max_length = FORM_DEFAULTS.get(mode, (2, 768))[1]
        if mode_info.get("tokenizer_type") == "scale-degree":
            max_length = int(max_length * 1.5)

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

    form_label = {
        "2-part": "2-part invention",
        "sinfonia": "sinfonia (3-part)",
        "chorale": "chorale (4-part)",
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
    from bach_gen.utils.music_theory import detect_key

    import numpy as np

    # Load corpus stats
    stats_path = DATA_DIR / "corpus_stats.json"
    if stats_path.exists():
        load_corpus_stats(stats_path)

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

    mode_path = DATA_DIR / "mode.json"
    mode_info = {}
    if mode_path.exists():
        with open(mode_path) as f:
            mode_info = json.load(f)
    mode = mode_info.get("mode", "chorale")

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
