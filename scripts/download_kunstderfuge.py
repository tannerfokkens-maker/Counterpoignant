#!/usr/bin/env python3
"""Download MIDI files from kunstderfuge.com and triage by quality.

Scrapes composer pages, downloads .mid files, then analyzes each file
to classify it as:
  - clean:   multi-track, quantized, ready for training
  - fixable: single-track or minor issues, needs voice separation
  - bad:     performed (rubato/dynamics), not usable without heavy processing

Resumable — skips files already downloaded.

Requires a Pro account for unlimited downloads. Set credentials via
environment variables or prompted interactively:
  KUNSTDERFUGE_USER=email
  KUNSTDERFUGE_PASS=password
"""

import json
import os
import re
import sys
import time
import urllib.parse
import copy
from pathlib import Path
from html.parser import HTMLParser

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.kunstderfuge.com"
MIDI_ASP = "/-/midi.asp?file="
DIRECT_PREFIX = "/-/mid.files/"

OUT = Path("data/midi/kunstderfuge")
REPORT_FILE = OUT / "_triage_report.json"
FAILED_FILE = OUT / "_failed.json"
CACHE = Path("/tmp/kunstderfuge_cache")

# Composers relevant to tonal/diatonic training data.
# (page_path, output_dirname, era)
# page_path can be a single page or a list of sub-pages.
#
# Excluded (non-tonal / atonal / serial):
#   - Schoenberg (serial/atonal)
#   - Messiaen (modes of limited transposition, non-functional harmony)
#
# Excluded (non-score / performance recordings):
#   - Piano Rolls (piano-rolls.htm) — performed recordings, not scores
#
# Excluded (folk/hymn collections — too simple, no polyphony):
#   - tunes/*.htm (Scottish, Irish, Welsh, English, American, Canadian)
#   - hymns.htm (simple hymn harmonizations)
#   - medieval.htm carols (simple carols)
#   - polish.htm (carols, psalms)
#
COMPOSERS = [
    # ── Medieval ──────────────────────────────────────────────
    ("machault.htm", None, "machault", "medieval"),
    ("anonymous.htm", None, "anonymous", "medieval"),

    # ── Renaissance ──────────────────────────────────────────
    ("brade.htm", None, "brade", "renaissance"),
    ("byrd.htm", None, "byrd", "renaissance"),
    ("cabanilles.htm", None, "cabanilles", "renaissance"),
    ("cima.htm", None, "cima", "renaissance"),
    ("dowland.htm", None, "dowland", "renaissance"),
    ("eccard.htm", None, "eccard", "renaissance"),
    ("gabrieli-a.htm", None, "gabrieli-a", "renaissance"),
    ("gabrieli-g.htm", None, "gabrieli-g", "renaissance"),
    ("galilei.htm", None, "galilei", "renaissance"),
    ("gesualdo.htm", None, "gesualdo", "renaissance"),
    ("guerrero.htm", None, "guerrero", "renaissance"),
    ("hassler.htm", None, "hassler", "renaissance"),
    ("henry-viii.htm", None, "henry-viii", "renaissance"),
    ("holborne.htm", None, "holborne", "renaissance"),
    ("isaac.htm", None, "isaac", "renaissance"),
    ("lasso.htm", None, "lasso", "renaissance"),
    ("monteverdi.htm", None, "monteverdi", "renaissance"),
    ("morales.htm", None, "morales", "renaissance"),
    ("morley.htm", None, "morley", "renaissance"),
    ("palestrina.htm", None, "palestrina", "renaissance"),
    ("pres.htm", None, "josquin", "renaissance"),
    ("praetorius.htm", None, "praetorius", "renaissance"),
    ("ravenscroft.htm", None, "ravenscroft", "renaissance"),
    ("schein.htm", None, "schein", "renaissance"),
    ("schutz.htm", None, "schutz", "renaissance"),
    ("tallis.htm", None, "tallis", "renaissance"),
    ("vecchi.htm", None, "vecchi", "renaissance"),
    ("victoria.htm", None, "victoria", "renaissance"),

    # ── Baroque ──────────────────────────────────────────────
    ("bach.htm", [
        "bach/harpsi.htm", "bach/wtk1.htm", "bach/wtk2.htm",
        "bach/organ.htm", "bach/chamber.htm", "bach/canons.htm",
        "bach/lute.htm", "bach/variae.htm", "bach/chorales.htm",
    ], "bach", "baroque"),
    ("albinoni.htm", None, "albinoni", "baroque"),
    ("anglebert.htm", None, "anglebert", "baroque"),
    ("bach-jc.htm", None, "bach-jc", "baroque"),
    ("balbastre.htm", None, "balbastre", "baroque"),
    ("buxtehude.htm", None, "buxtehude", "baroque"),
    ("couperin.htm", None, "couperin", "baroque"),
    ("dandrieu.htm", None, "dandrieu", "baroque"),
    ("frescobaldi.htm", None, "frescobaldi", "baroque"),
    ("froberger.htm", None, "froberger", "baroque"),
    ("gorczycki.htm", None, "gorczycki", "baroque"),
    ("hammerschmidt.htm", None, "hammerschmidt", "baroque"),
    ("handel.htm", None, "handel", "baroque"),
    ("marcello-b.htm", None, "marcello", "baroque"),
    ("marini.htm", None, "marini", "baroque"),
    ("merula.htm", None, "merula", "baroque"),
    ("pachelbel.htm", None, "pachelbel", "baroque"),
    ("rameau.htm", None, "rameau", "baroque"),
    ("roman.htm", None, "roman", "baroque"),
    ("rosenmuller.htm", None, "rosenmuller", "baroque"),
    ("scarlatti.htm", None, "scarlatti", "baroque"),
    ("soler.htm", None, "soler", "baroque"),
    ("telemann.htm", None, "telemann", "baroque"),
    ("visee.htm", None, "visee", "baroque"),
    ("vivaldi.htm", None, "vivaldi", "baroque"),
    ("walther.htm", None, "walther", "baroque"),
    ("zipoli.htm", None, "zipoli", "baroque"),

    # ── Classical ────────────────────────────────────────────
    ("albrechtsberger.htm", None, "albrechtsberger", "classical"),
    ("beethoven.htm", [
        "beethoven/klavier.htm", "beethoven/sonatas.htm",
        "beethoven/variations.htm", "beethoven/chamber.htm",
        "beethoven/variae.htm", "beethoven/canons.htm",
    ], "beethoven", "classical"),
    ("clementi.htm", None, "clementi", "classical"),
    ("danzi.htm", None, "danzi", "classical"),
    ("dussek.htm", None, "dussek", "classical"),
    ("eberl.htm", None, "eberl", "classical"),
    ("haydn.htm", None, "haydn", "classical"),
    ("hummel.htm", None, "hummel", "classical"),
    ("mozart.htm", None, "mozart", "classical"),
    ("sor.htm", None, "sor", "classical"),

    # ── Romantic ─────────────────────────────────────────────
    ("albeniz.htm", None, "albeniz", "romantic"),
    ("alkan.htm", None, "alkan", "romantic"),
    ("berlioz.htm", None, "berlioz", "romantic"),
    ("bizet.htm", None, "bizet", "romantic"),
    ("boely.htm", None, "boely", "romantic"),
    ("brahms.htm", None, "brahms", "romantic"),
    ("bruckner.htm", None, "bruckner", "romantic"),
    ("busoni.htm", None, "busoni", "romantic"),
    ("chopin.htm", None, "chopin", "romantic"),
    ("dvorak.htm", None, "dvorak", "romantic"),
    ("faure.htm", None, "faure", "romantic"),
    ("franck.htm", None, "franck", "romantic"),
    ("glazunov.htm", None, "glazunov", "romantic"),
    ("glinka.htm", None, "glinka", "romantic"),
    ("godowsky.htm", None, "godowsky", "romantic"),
    ("gottschalk.htm", None, "gottschalk", "romantic"),
    ("grieg.htm", None, "grieg", "romantic"),
    ("liszt.htm", None, "liszt", "romantic"),
    ("mahler.htm", None, "mahler", "romantic"),
    ("medtner.htm", None, "medtner", "romantic"),
    ("mendelssohn.htm", None, "mendelssohn", "romantic"),
    ("mussorgsky.htm", None, "mussorgsky", "romantic"),
    ("rachmaninov.htm", None, "rachmaninov", "romantic"),
    ("raff.htm", None, "raff", "romantic"),
    ("rimskij-korsakov.htm", None, "rimskij-korsakov", "romantic"),
    ("rossini.htm", None, "rossini", "romantic"),
    ("rubinstein.htm", None, "rubinstein", "romantic"),
    ("saintsaens.htm", None, "saintsaens", "romantic"),
    ("schubert.htm", None, "schubert", "romantic"),
    ("schumann.htm", None, "schumann", "romantic"),
    ("sibelius.htm", None, "sibelius", "romantic"),
    ("tchaikovsky.htm", None, "tchaikovsky", "romantic"),
    ("verdi.htm", None, "verdi", "romantic"),
    ("wagner.htm", None, "wagner", "romantic"),

    # ── Romantic organ (especially valuable — fugal tradition) ─
    ("dupre.htm", None, "dupre", "romantic"),
    ("fuehrer.htm", None, "fuehrer", "romantic"),
    ("karg-elert.htm", None, "karg-elert", "romantic"),
    ("reger.htm", None, "reger", "romantic"),
    ("vierne.htm", None, "vierne", "romantic"),
    ("widor.htm", None, "widor", "romantic"),

    # ── Impressionist (tonal but extended harmony) ───────────
    ("debussy.htm", None, "debussy", "impressionist"),
    ("ravel.htm", None, "ravel", "impressionist"),
    ("satie.htm", None, "satie", "impressionist"),

    # ── 20th Century (tonal / neoclassical) ──────────────────
    ("bartok.htm", None, "bartok", "modern"),
    ("gershwin.htm", None, "gershwin", "modern"),
    ("hindemith.htm", None, "hindemith", "modern"),
    ("janacek.htm", None, "janacek", "modern"),
    ("peeters.htm", None, "peeters", "modern"),
    ("poulenc.htm", None, "poulenc", "modern"),
    ("prokofiev.htm", None, "prokofiev", "modern"),
    ("scriabin.htm", None, "scriabin", "modern"),
    ("shostakovitch.htm", None, "shostakovich", "modern"),
    ("stravinsky.htm", None, "stravinsky", "modern"),

    # ── Ragtime (tonal, rhythmically interesting) ────────────
    ("ragtime.htm", None, "ragtime", "ragtime"),

    # ── Misc / uncategorized ─────────────────────────────────
    ("becker.htm", None, "becker", "misc"),
    ("buehler.htm", None, "buehler", "misc"),
    ("maier.htm", None, "maier", "misc"),
    ("rahs.htm", None, "rahs", "misc"),

    # ── New fugal composers (site's own contributors) ────────
    ("new/classical.htm", None, "new-classical", "modern"),
    ("new/pacchioni.htm", None, "pacchioni", "modern"),
]

# Rate limiting
DELAY_BETWEEN_DOWNLOADS = 0.3  # seconds
DELAY_BETWEEN_PAGES = 1.0

# Max downloads for testing (set to None for unlimited)
MAX_DOWNLOADS = None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

class MidiLinkParser(HTMLParser):
    """Extract MIDI download links from kunstderfuge HTML."""

    def __init__(self):
        super().__init__()
        self.midi_links: list[str] = []
        self.sub_pages: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        if MIDI_ASP in href or "midi.asp?file=" in href:
            # Extract the file parameter
            if "file=" in href:
                file_param = href.split("file=", 1)[1]
                # Clean up any HTML encoding
                file_param = file_param.replace("&amp;", "&")
                if file_param.endswith(".mid"):
                    self.midi_links.append(file_param)
        elif href.endswith(".htm") and not href.startswith("http"):
            self.sub_pages.append(href)


def parse_midi_links(html: str) -> tuple[list[str], list[str]]:
    """Parse HTML and return (midi_file_paths, sub_page_links)."""
    parser = MidiLinkParser()
    parser.feed(html)
    return parser.midi_links, parser.sub_pages


# ---------------------------------------------------------------------------
# MIDI quality triage
# ---------------------------------------------------------------------------

def triage_midi(filepath: Path) -> dict:
    """Analyze a MIDI file and classify its quality for training.

    Returns a dict with:
      - status: "clean" | "fixable" | "bad"
      - num_tracks: number of parts/tracks
      - num_notes: total note count
      - pitch_range: (min, max) MIDI pitch
      - unique_durations: count of distinct note durations
      - velocity_range: (min, max) velocity
      - unique_velocities: count of distinct velocities
      - issues: list of detected problems
    """
    try:
        import music21  # noqa: F401
    except ImportError:
        return {"status": "pending", "issues": ["music21 not available — run triage with uv run"], "num_tracks": 0}

    try:
        score = _parse_with_timeout(filepath, timeout=30)
    except Exception as e:
        return {"status": "bad", "issues": [f"parse error: {e}"], "num_tracks": 0}

    parts = score.parts
    num_tracks = len(parts)
    issues = []

    all_pitches = []
    all_durations = []
    all_velocities = []
    total_notes = 0

    for part in parts:
        notes = list(part.recurse().notes)
        total_notes += len(notes)
        for n in notes:
            if hasattr(n, "pitch"):
                all_pitches.append(n.pitch.midi)
            else:
                for p in n.pitches:
                    all_pitches.append(p.midi)
            all_durations.append(round(float(n.duration.quarterLength), 4))
            if hasattr(n, "volume") and n.volume.velocity is not None:
                all_velocities.append(n.volume.velocity)

    if not all_pitches:
        return {"status": "bad", "issues": ["no notes"], "num_tracks": num_tracks}

    pitch_range = (min(all_pitches), max(all_pitches))
    unique_durs = set(all_durations)
    unique_vels = set(all_velocities) if all_velocities else {0}
    vel_range = (min(all_velocities), max(all_velocities)) if all_velocities else (0, 0)

    # Detect performance artifacts (fixable via de-perform)
    needs_deperform = False
    if len(unique_vels) > 10:
        issues.append(f"needs deperform: dynamics ({len(unique_vels)} velocity levels)")
        needs_deperform = True

    if len(unique_durs) > 50:
        issues.append(f"needs deperform: rubato ({len(unique_durs)} unique durations)")
        needs_deperform = True

    clean_grid = {0.0625, 0.125, 0.1667, 0.25, 0.3333, 0.375, 0.5, 0.6667,
                  0.75, 1.0, 1.25, 1.3333, 1.5, 1.6667, 1.75, 2.0, 2.5,
                  2.6667, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0}
    off_grid = [d for d in unique_durs if d > 0 and not any(
        abs(d - g) < 0.02 for g in clean_grid
    )]
    if len(off_grid) > 10:
        issues.append(f"needs deperform: off-grid durations ({len(off_grid)})")
        needs_deperform = True

    # Single track = needs voice separation (the hard problem)
    needs_voice_sep = False
    if num_tracks == 1 and total_notes > 50:
        issues.append("needs voice separation (single track)")
        needs_voice_sep = True

    # Too many voices = needs reduction (>4 tracks with meaningful content)
    needs_voice_reduce = False
    tracks_with_notes = sum(1 for p in parts if len(list(p.recurse().notes)) > 10)
    if tracks_with_notes > 4 and not needs_voice_sep:
        issues.append(f"needs voice reduction ({tracks_with_notes} voices → 4)")
        needs_voice_reduce = True

    # Classify
    if not issues:
        status = "clean"
    elif needs_voice_sep:
        status = "needs_voice_sep"
    elif needs_voice_reduce:
        status = "needs_voice_reduce"
    elif needs_deperform:
        status = "needs_deperform"
    else:
        status = "clean"

    return {
        "status": status,
        "num_tracks": num_tracks,
        "tracks_with_notes": tracks_with_notes,
        "num_notes": total_notes,
        "pitch_range": pitch_range,
        "unique_durations": len(unique_durs),
        "velocity_range": vel_range,
        "unique_velocities": len(unique_vels),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# De-perform: quantize + strip dynamics
# ---------------------------------------------------------------------------

def _repair_overlaps_in_stream(
    events: list[tuple[float, float, "music21.note.NotRest"]],
    grid_unit: float,
) -> list[tuple[float, float, "music21.note.NotRest"]]:
    """Quantize a stream and remove accidental self-overlap.

    Overlap repair is only applied when a later onset would land before the
    previous note's end (start > prev_start and start < prev_end). True
    simultaneities (same onset) are preserved.

    TODO: This overlap repair is still rhythm-centric and not ornament-aware.
    Fast trill/turn passages can still be assigned in ways that sound wrong
    (e.g., rapid alternation splitting across perceived voices). Add a
    dedicated ornament pass that preserves local trill identity in one voice.
    """
    events.sort(key=lambda e: (e[0], e[1]))
    cleaned: list[tuple[float, float, "music21.note.NotRest"]] = []

    for start, end, note_obj in events:
        start = max(0.0, start)
        if end <= start:
            end = start + grid_unit

        if cleaned:
            prev_start, prev_end, prev_note = cleaned[-1]
            if start > prev_start and start < prev_end:
                # Prefer preserving the new attack and shorten the prior note.
                min_prev_end = prev_start + grid_unit
                trimmed_prev_end = max(min_prev_end, start)
                if trimmed_prev_end <= start:
                    cleaned[-1] = (prev_start, trimmed_prev_end, prev_note)
                else:
                    # Prior note is already at minimum duration; move this one.
                    start = prev_end
                    if end <= start:
                        end = start + grid_unit

        cleaned.append((start, end, note_obj))

    return cleaned


def _snap_duration_to_musical_bins(duration_q: float, grid_unit: float) -> float:
    """Snap duration to a conservative musical bin set.

    This avoids odd residual bins (e.g. 1.25q, 1.75q, 2.25q) that often arise
    from de-perform rounding noise and sound irregular in contrapuntal voices.
    """
    if duration_q <= grid_unit:
        return grid_unit

    multipliers = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    bins = [grid_unit * m for m in multipliers]
    # Extend bins upward for unusually long notes.
    while bins[-1] < duration_q:
        bins.append(bins[-1] + grid_unit * 8)
    return min(bins, key=lambda b: abs(b - duration_q))


def deperform_midi(
    src: Path,
    dest: Path,
    grid: int = 24,
    velocity: int = 64,
    duration_grid: int | None = None,
) -> bool:
    """Quantize timing, normalize velocity, strip CCs from a MIDI file.

    Args:
        src: Input MIDI path.
        dest: Output MIDI path.
        grid: Quantization grid as subdivision of a quarter note.
              16 = 16th notes, 12 = triplet 8ths, 24 = 16th triplets.
        velocity: Fixed velocity for all notes.
        duration_grid: Optional separate duration grid subdivision. If None,
                       uses ``grid``. Example: onset grid=16, duration_grid=8.

    TODO: Current deperform is not sufficient for all live/performed fugue
    inputs. It still needs robust trill-aware voice assignment after
    separation to avoid cross-voice ornament bleed.

    Returns:
        True if successful.
    """
    try:
        import music21
        score = music21.converter.parse(str(src))
    except Exception:
        return False

    if grid <= 0:
        return False
    if duration_grid is None:
        duration_grid = grid
    if duration_grid <= 0:
        return False
    onset_grid_unit = 4.0 / grid       # e.g. grid=16 -> 0.25 quarter notes
    duration_grid_unit = 4.0 / duration_grid

    out_score = music21.stream.Score()

    for part in score.parts:
        out_part = music21.stream.Part(id=getattr(part, "id", None))

        # Keep first instrument/time-signature/key-signature/clef context.
        for cls in (
            music21.instrument.Instrument,
            music21.meter.TimeSignature,
            music21.key.KeySignature,
            music21.clef.Clef,
        ):
            first = next(iter(part.recurse().getElementsByClass(cls)), None)
            if first is not None:
                out_part.insert(0, copy.deepcopy(first))

        # Quantize using absolute start/end and repair overlap per voice stream.
        events_by_stream: dict[int, list[tuple[float, float, music21.note.NotRest]]] = {}
        for n in part.recurse().notes:
            try:
                raw_start = float(n.getOffsetInHierarchy(part))
            except Exception:
                raw_start = float(n.offset)
            raw_dur = float(n.duration.quarterLength)
            if raw_dur <= 0:
                continue

            q_start = round(raw_start / onset_grid_unit) * onset_grid_unit
            q_dur = _snap_duration_to_musical_bins(raw_dur, duration_grid_unit)
            q_end = q_start + q_dur
            if q_end <= q_start:
                q_end = q_start + duration_grid_unit

            voice_ctx = n.getContextByClass(music21.stream.Voice)
            stream_key = id(voice_ctx) if voice_ctx is not None else -1
            events_by_stream.setdefault(stream_key, []).append((q_start, q_end, n))

        repaired_events: list[tuple[float, float, music21.note.NotRest]] = []
        for stream_events in events_by_stream.values():
            repaired_events.extend(_repair_overlaps_in_stream(stream_events, onset_grid_unit))
        repaired_events.sort(key=lambda e: (e[0], e[1]))

        for start, end, note_obj in repaired_events:
            new_note = copy.deepcopy(note_obj)
            raw_dur_q = max(duration_grid_unit, end - start)
            new_note.duration.quarterLength = _snap_duration_to_musical_bins(raw_dur_q, duration_grid_unit)
            if hasattr(new_note, "volume"):
                new_note.volume.velocity = velocity
            out_part.insert(start, new_note)

        out_score.insert(0, out_part)

    # Strip source tempo map; write one fixed tempo.
    out_score.insert(0, music21.tempo.MetronomeMark(number=120))

    try:
        out_score.write("midi", fp=str(dest))
        return True
    except Exception:
        return False


def deperform_all(triage_results: dict, base_dir: Path, grid: int = 16) -> int:
    """De-perform all files marked as needs_deperform.

    Processed files are written alongside originals with a .deperformed.mid
    suffix, then re-triaged. Returns count of files processed.
    """
    to_process = [
        (k, v) for k, v in triage_results.items()
        if v.get("status") == "needs_deperform"
    ]
    if not to_process:
        print("  No files need de-performing.")
        return 0

    print(f"  De-performing {len(to_process)} files (grid=1/{grid})...")
    processed = 0

    for key, info in to_process:
        src = base_dir / key
        dest = src.with_suffix(".deperformed.mid")
        if dest.exists():
            processed += 1
            continue

        if deperform_midi(src, dest, grid=grid):
            processed += 1
            # Re-triage the cleaned file
            result = triage_midi(dest)
            new_key = key.replace(".mid", ".deperformed.mid")
            triage_results[new_key] = result
            status_icons = {"clean": "+", "needs_deperform": "~", "needs_voice_sep": "V", "needs_voice_reduce": "R", "bad": "x", "pending": "?"}
            icon = status_icons.get(result["status"], "?")
            print(f"    [{icon}] {dest.name}: {result['status']}")
        else:
            print(f"    FAILED: {src.name}")

    return processed


# ---------------------------------------------------------------------------
# Voice separation via MuseScore
# ---------------------------------------------------------------------------

MUSESCORE_PATHS = [
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    "mscore",
    "musescore",
    "mscore4",
]


def find_musescore() -> str | None:
    """Find the MuseScore binary."""
    import shutil
    for path in MUSESCORE_PATHS:
        if "/" in path:
            if Path(path).exists():
                return path
        else:
            found = shutil.which(path)
            if found:
                return found
    return None


def voice_separate_mscore(src: Path, dest: Path, mscore_bin: str) -> bool:
    """Use MuseScore to import MIDI and export as MusicXML.

    MuseScore's MIDI importer performs voice separation and quantization
    as part of its notation inference. The resulting MusicXML has distinct
    voices/staves that music21 can read as separate parts.

    The MusicXML is then re-exported as MIDI with voices on separate tracks.
    """
    import subprocess
    import tempfile

    # Step 1: MIDI → MusicXML via MuseScore
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        mxml_path = tmp.name

    try:
        result = subprocess.run(
            [mscore_bin, "-o", mxml_path, str(src)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not Path(mxml_path).exists():
            return False

        # Step 2: Read MusicXML with music21 and export as multi-track MIDI
        import music21
        score = music21.converter.parse(mxml_path)

        if len(score.parts) <= 1:
            # MuseScore didn't separate voices — not much we can do
            Path(mxml_path).unlink(missing_ok=True)
            return False

        score.write("midi", fp=str(dest))
        return True

    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        Path(mxml_path).unlink(missing_ok=True)


def voice_separate_all(triage_results: dict, base_dir: Path) -> int:
    """Run MuseScore voice separation on all single-track files.

    Processed files are written with a .separated.mid suffix, then re-triaged.
    Returns count of files successfully processed.
    """
    mscore_bin = find_musescore()
    if not mscore_bin:
        print("  ERROR: MuseScore not found. Install MuseScore 4 or set mscore in PATH.")
        return 0

    print(f"  Using MuseScore: {mscore_bin}")

    to_process = [
        (k, v) for k, v in triage_results.items()
        if v.get("status") == "needs_voice_sep"
    ]
    if not to_process:
        print("  No files need voice separation.")
        return 0

    print(f"  Processing {len(to_process)} files through MuseScore...")
    processed = 0

    for key, info in to_process:
        src = base_dir / key
        if not src.exists():
            continue

        dest = src.with_suffix(".separated.mid")
        if dest.exists():
            processed += 1
            continue

        print(f"    Processing: {src.name}...", end=" ", flush=True)
        if voice_separate_mscore(src, dest, mscore_bin):
            processed += 1
            # Re-triage the result
            result = triage_midi(dest)
            new_key = key.replace(".mid", ".separated.mid")
            triage_results[new_key] = result
            print(f"→ {result['num_tracks']} tracks, {result['status']}")
        else:
            print("failed (MuseScore couldn't separate)")

    return processed


# ---------------------------------------------------------------------------
# Voice reduction: N voices → max 4 (outer + best inner)
# ---------------------------------------------------------------------------

def _rank_parts_for_reduction(
    parts: list,
    max_voices: int = 4,
) -> list[int]:
    """Rank parts and return indices of the ones to keep.

    Strategy:
    1. Always keep the highest voice (soprano) and lowest voice (bass)
       determined by mean pitch, not track order.
    2. For inner voices, rank by a combination of:
       - Note density (more notes = more musically active)
       - Duration coverage (voices spanning the full piece)
       - Register distinctness (prefer voices that fill the middle
         rather than doubling soprano/bass)
       - Melodic range (wider range = more interesting line)
    3. Keep the top (max_voices - 2) inner voices.

    Returns:
        List of part indices to keep, in their original order.
    """
    if len(parts) <= max_voices:
        return list(range(len(parts)))

    # Compute per-part statistics
    stats = []
    for i, part in enumerate(parts):
        notes = list(part.recurse().notes)
        if not notes:
            stats.append(None)
            continue

        pitches = []
        for n in notes:
            if hasattr(n, "pitch"):
                pitches.append(n.pitch.midi)
            else:
                for p in n.pitches:
                    pitches.append(p.midi)

        if not pitches:
            stats.append(None)
            continue

        onsets = []
        for n in notes:
            try:
                onsets.append(float(n.getOffsetInHierarchy(part)))
            except Exception:
                onsets.append(float(n.offset))

        stats.append({
            "idx": i,
            "mean_pitch": sum(pitches) / len(pitches),
            "min_pitch": min(pitches),
            "max_pitch": max(pitches),
            "note_count": len(notes),
            "pitch_range": max(pitches) - min(pitches),
            "first_onset": min(onsets) if onsets else 0,
            "last_onset": max(onsets) if onsets else 0,
            "duration_span": (max(onsets) - min(onsets)) if len(onsets) > 1 else 0,
        })

    valid = [s for s in stats if s is not None]
    if len(valid) <= max_voices:
        return [s["idx"] for s in valid]

    # Identify outer voices by mean pitch
    by_pitch = sorted(valid, key=lambda s: s["mean_pitch"])
    bass_idx = by_pitch[0]["idx"]
    soprano_idx = by_pitch[-1]["idx"]
    keep = {bass_idx, soprano_idx}

    # Rank inner voices
    inner = [s for s in valid if s["idx"] not in keep]
    if not inner:
        return sorted(keep)

    full_duration = max((s["duration_span"] for s in valid if s["duration_span"] > 0), default=1.0)
    max_notes = max(s["note_count"] for s in valid)
    soprano_mean = by_pitch[-1]["mean_pitch"]
    bass_mean = by_pitch[0]["mean_pitch"]
    mid_pitch = (soprano_mean + bass_mean) / 2
    register_range = max(soprano_mean - bass_mean, 1)

    for s in inner:
        density_score = s["note_count"] / max_notes if max_notes > 0 else 0
        coverage_score = s["duration_span"] / full_duration if full_duration > 0 else 0
        dist_from_mid = abs(s["mean_pitch"] - mid_pitch)
        register_score = max(0, 1.0 - dist_from_mid / (register_range / 2))
        range_score = min(1.0, s["pitch_range"] / 24)

        s["inner_score"] = (
            density_score * 0.35
            + coverage_score * 0.30
            + register_score * 0.20
            + range_score * 0.15
        )

    inner.sort(key=lambda s: s["inner_score"], reverse=True)
    for s in inner[:max_voices - 2]:
        keep.add(s["idx"])

    return sorted(keep)


def reduce_voices(
    src: Path,
    dest: Path,
    max_voices: int = 4,
    velocity: int = 64,
) -> bool:
    """Reduce an N-voice MIDI file to at most max_voices.

    Keeps outer voices (soprano + bass by mean pitch) and the most
    active/distinct inner voices. Normalizes velocity.

    Returns True if reduction was performed.
    """
    try:
        import music21
        score = _parse_with_timeout(src, timeout=60)
    except Exception:
        return False

    parts = list(score.parts)
    parts_with_notes = [(i, p) for i, p in enumerate(parts)
                        if len(list(p.recurse().notes)) > 10]

    if len(parts_with_notes) <= max_voices:
        return False

    keep_indices = _rank_parts_for_reduction(
        [p for _, p in parts_with_notes],
        max_voices=max_voices,
    )
    selected_parts = [parts_with_notes[i][1] for i in keep_indices]

    # Sort by mean pitch (high to low) for SATB ordering
    def mean_pitch(part):
        pitches = []
        for n in part.recurse().notes:
            if hasattr(n, "pitch"):
                pitches.append(n.pitch.midi)
            else:
                for p in n.pitches:
                    pitches.append(p.midi)
        return sum(pitches) / len(pitches) if pitches else 60

    selected_parts.sort(key=mean_pitch, reverse=True)

    import music21
    out_score = music21.stream.Score()
    out_score.insert(0, music21.tempo.MetronomeMark(number=120))

    for part in selected_parts:
        out_part = music21.stream.Part()

        for cls in (
            music21.instrument.Instrument,
            music21.meter.TimeSignature,
            music21.key.KeySignature,
            music21.clef.Clef,
        ):
            first = next(iter(part.recurse().getElementsByClass(cls)), None)
            if first is not None:
                out_part.insert(0, copy.deepcopy(first))

        for n in part.recurse().notes:
            new_note = copy.deepcopy(n)
            if hasattr(new_note, "volume"):
                new_note.volume.velocity = velocity
            try:
                offset = float(n.getOffsetInHierarchy(part))
            except Exception:
                offset = float(n.offset)
            out_part.insert(offset, new_note)

        out_score.insert(0, out_part)

    try:
        out_score.write("midi", fp=str(dest))
        return True
    except Exception:
        return False


def reduce_voices_all(
    triage_results: dict,
    base_dir: Path,
    max_voices: int = 4,
) -> int:
    """Reduce all files with >max_voices to max_voices.

    Catches both explicitly flagged files and clean files with >4 tracks
    that were triaged before the reduction status existed.

    Processed files are written with a .reduced.mid suffix, then re-triaged.
    Returns count of files processed.
    """
    to_process = []
    for k, v in triage_results.items():
        if v.get("status") == "needs_voice_reduce":
            to_process.append((k, v))
        elif v.get("tracks_with_notes", v.get("num_tracks", 0)) > max_voices and v.get("status") == "clean":
            to_process.append((k, v))

    if not to_process:
        print("  No files need voice reduction.")
        return 0

    print(f"  Reducing {len(to_process)} files to ≤{max_voices} voices...")
    processed = 0

    for key, info in to_process:
        src = base_dir / key
        if not src.exists():
            continue

        dest = src.with_suffix(".reduced.mid")
        if dest.exists():
            processed += 1
            continue

        original_tracks = info.get("tracks_with_notes", info.get("num_tracks", "?"))
        print(f"    {src.name} ({original_tracks} voices)...", end=" ", flush=True)

        if reduce_voices(src, dest, max_voices=max_voices):
            processed += 1
            result = triage_midi(dest)
            new_key = key.replace(".mid", ".reduced.mid")
            triage_results[new_key] = result
            status_icons = {"clean": "+", "needs_deperform": "~", "needs_voice_sep": "V",
                            "needs_voice_reduce": "R", "bad": "x", "pending": "?"}
            icon = status_icons.get(result["status"], "?")
            print(f"→ {result['num_tracks']} tracks, {result['status']}")
        else:
            print("skipped (already ≤4 or failed)")

    return processed

def _extract_catalog_number(filename: str) -> str | None:
    """Extract a catalog number (BWV, BuxWV, K, Op, etc.) from a filename."""
    import re
    # BWV numbers: bwv-846, bwv_846, bwv846
    m = re.search(r'bwv[_-]?(\d+)', filename, re.IGNORECASE)
    if m:
        return f"bwv-{m.group(1)}"
    # BuxWV
    m = re.search(r'buxwv[_-]?(\d+)', filename, re.IGNORECASE)
    if m:
        return f"buxwv-{m.group(1)}"
    # HWV (Handel)
    m = re.search(r'hwv[_-]?(\d+)', filename, re.IGNORECASE)
    if m:
        return f"hwv-{m.group(1)}"
    # K (Mozart)
    m = re.search(r'(?:^|[_-])k[_-]?(\d+)', filename, re.IGNORECASE)
    if m:
        return f"k-{m.group(1)}"
    # Op (generic)
    m = re.search(r'op(?:us)?[_-]?(\d+)[_-]?(?:no)?[_-]?(\d+)?', filename, re.IGNORECASE)
    if m:
        num = f"op-{m.group(1)}"
        if m.group(2):
            num += f"-{m.group(2)}"
        return num
    return None


def _score_version(filepath: Path, triage_results: dict) -> float:
    """Score a file version for quality ranking. Higher = better."""
    key = str(filepath.relative_to(filepath.parents[1]))
    # Try to find triage data
    # Look through triage results for a matching key
    info = None
    for k, v in triage_results.items():
        if filepath.name in k:
            info = v
            break

    if info is None:
        # Triage it now
        info = triage_midi(filepath)

    score = 0.0
    status = info.get("status", "bad")
    num_tracks = info.get("num_tracks", 0)
    num_notes = info.get("num_notes", 0)

    # Strong preference for clean files
    if status == "clean":
        score += 100
    elif status == "needs_deperform":
        score += 50
    elif status == "needs_voice_reduce":
        score += 40
    elif status == "needs_voice_sep":
        score += 10
    elif status == "bad":
        score += 0

    # Prefer multi-track (voice-separated) but not too many tracks
    # Ideal: 2-4 tracks for counterpoint
    if 2 <= num_tracks <= 4:
        score += 30
    elif num_tracks == 1:
        score += 0
    elif 5 <= num_tracks <= 6:
        score += 20
    else:
        score += 10  # too many tracks (over-split)

    # More notes = more complete (slight tiebreaker)
    score += min(10, num_notes / 500)

    return score


def dedup_internal(base_dir: Path, triage_results: dict) -> tuple[int, list[str]]:
    """Identify duplicate versions of the same piece within kunstderfuge.

    Groups files by catalog number, picks the best version, marks the
    rest as 'internal_dup' in triage_results. Originals are never moved.

    Returns (num_marked, list_of_marked).
    """
    from collections import defaultdict

    # Scan all .mid files (skip derived files)
    all_files = sorted(base_dir.rglob("*.mid"))
    all_files = [f for f in all_files if
                 ".separated." not in f.name and
                 ".deperformed." not in f.name and
                 ".reduced." not in f.name and
                 not f.name.startswith("_") and
                 "_duplicates" not in str(f) and
                 "_internal_dups" not in str(f)]

    # Group by (composer_dir, catalog_number)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    ungrouped = 0
    for f in all_files:
        composer = f.parent.name
        cat = _extract_catalog_number(f.name)
        if cat:
            groups[(composer, cat)].append(f)
        else:
            ungrouped += 1

    # Find groups with multiple versions
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  Found {len(dup_groups)} pieces with multiple versions "
          f"({ungrouped} files without catalog numbers skipped)")

    marked = []

    for (composer, cat), files in sorted(dup_groups.items()):
        # Score each version
        scored = [(f, _score_version(f, triage_results)) for f in files]
        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0]
        rest = scored[1:]

        if len(rest) > 0:
            print(f"    {composer}/{cat}: keeping {best[0].name} (score={best[1]:.0f})")
            for f, s in rest:
                # Mark as internal duplicate in triage results
                key = f"{f.parent.name}/{f.name}"
                if key in triage_results:
                    triage_results[key]["is_internal_dup"] = True
                    triage_results[key]["better_version"] = best[0].name
                marked.append(f"{composer}/{f.name} (score={s:.0f})")

    return len(marked), marked


# ---------------------------------------------------------------------------
# Deduplication against trusted datasets
# ---------------------------------------------------------------------------

TRUSTED_DIRS = [
    Path("data/midi/kernscores"),
    Path("data/midi/jrp"),
    Path("data/midi/openscore_quartets"),
]

FINGERPRINT_CACHE = OUT / "_fingerprints.json"


def _parse_with_timeout(filepath: Path, timeout: int = 30):
    """Parse a music file with a timeout to avoid hanging on large/malformed files."""
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        import music21
        return music21.converter.parse(str(filepath))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def compute_fingerprint(filepath: Path) -> dict | None:
    """Compute a content-based fingerprint for a music file.

    Returns a dict with:
      - pc_hist: pitch class histogram (12 bins, normalized)
      - note_count: total number of notes
      - duration: total duration in quarter notes
      - interval_sig: first 20 melodic intervals (sorted by onset)
    """
    try:
        score = _parse_with_timeout(filepath, timeout=30)
    except Exception:
        return None

    all_notes = []  # (onset, midi_pitch)
    total_dur = 0.0
    pc_counts = [0] * 12

    for part in score.parts:
        for n in part.recurse().notes:
            offset = float(n.offset)
            dur = float(n.duration.quarterLength)
            if hasattr(n, "pitch"):
                midi = n.pitch.midi
                all_notes.append((offset, midi))
                pc_counts[midi % 12] += 1
            else:
                for p in n.pitches:
                    all_notes.append((offset, p.midi))
                    pc_counts[p.midi % 12] += 1
            total_dur = max(total_dur, offset + dur)

    if not all_notes:
        return None

    # Normalize pitch class histogram
    total = sum(pc_counts)
    pc_hist = [round(c / total, 4) for c in pc_counts]

    # First 20 intervals sorted by onset
    all_notes.sort()
    intervals = []
    for i in range(min(20, len(all_notes) - 1)):
        intervals.append(all_notes[i + 1][1] - all_notes[i][1])

    return {
        "pc_hist": pc_hist,
        "note_count": len(all_notes),
        "duration": round(total_dur, 2),
        "interval_sig": intervals,
    }


def fingerprints_match(a: dict, b: dict, threshold: float = 0.98) -> bool:
    """Check if two fingerprints represent the same piece.

    Uses cosine similarity on pitch class histograms, note count and
    duration tolerance, plus interval signature overlap.
    """
    # Cosine similarity on pitch class histograms
    ah, bh = a["pc_hist"], b["pc_hist"]
    dot = sum(x * y for x, y in zip(ah, bh))
    mag_a = sum(x * x for x in ah) ** 0.5
    mag_b = sum(x * x for x in bh) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return False
    cosine = dot / (mag_a * mag_b)

    if cosine < threshold:
        return False

    # Note count within 20%
    nc_a, nc_b = a["note_count"], b["note_count"]
    if nc_a == 0 or nc_b == 0:
        return False
    nc_ratio = min(nc_a, nc_b) / max(nc_a, nc_b)
    if nc_ratio < 0.8:
        return False

    # Duration within 20%
    d_a, d_b = a["duration"], b["duration"]
    if d_a == 0 or d_b == 0:
        return False
    d_ratio = min(d_a, d_b) / max(d_a, d_b)
    if d_ratio < 0.8:
        return False

    # Interval signature: at least 60% of first N intervals must match
    int_a = a.get("interval_sig", [])
    int_b = b.get("interval_sig", [])
    if int_a and int_b:
        n = min(len(int_a), len(int_b))
        if n >= 5:
            matches = sum(1 for i in range(n) if int_a[i] == int_b[i])
            if matches / n < 0.6:
                return False

    return True


def dedup_against_trusted(base_dir: Path, triage_results: dict) -> tuple[int, list[str]]:
    """Identify kunstderfuge files that duplicate trusted dataset content.

    Builds fingerprints for all trusted files, then checks each kunstderfuge
    file against them. Duplicates are marked in triage_results, never moved.

    Returns (num_marked, list_of_marked_paths).
    """
    import music21  # noqa: F401 — ensure available

    # Load or build fingerprint cache
    cache: dict = {}
    if FINGERPRINT_CACHE.exists():
        cache = json.loads(FINGERPRINT_CACHE.read_text())

    # Build trusted fingerprints
    print("  Building trusted dataset fingerprints...")
    trusted_fps: list[tuple[str, dict]] = []
    trusted_files = []
    for d in TRUSTED_DIRS:
        if not d.exists():
            continue
        for ext in ("*.krn", "*.mid", "*.musicxml", "*.mxl"):
            trusted_files.extend(d.rglob(ext))

    print(f"  Found {len(trusted_files)} trusted files")
    for i, f in enumerate(trusted_files):
        key = str(f)
        if key in cache and cache[key] is not None:
            trusted_fps.append((key, cache[key]))
            continue

        fp = compute_fingerprint(f)
        cache[key] = fp
        if fp:
            trusted_fps.append((key, fp))

        if (i + 1) % 100 == 0:
            print(f"    Fingerprinted {i + 1}/{len(trusted_files)} trusted files...")
            # Save cache periodically
            FINGERPRINT_CACHE.write_text(json.dumps(cache))

    print(f"  {len(trusted_fps)} trusted fingerprints computed")

    # Save cache
    FINGERPRINT_CACHE.write_text(json.dumps(cache))

    # Build kunstderfuge fingerprints and check for duplicates
    print("  Checking kunstderfuge files for duplicates...")
    kdf_files = sorted(base_dir.rglob("*.mid"))
    # Skip derived files
    kdf_files = [f for f in kdf_files if
                 ".separated." not in f.name and
                 ".deperformed." not in f.name and
                 ".reduced." not in f.name and
                 not f.name.startswith("_")]

    marked = []

    for i, f in enumerate(kdf_files):
        key = str(f)
        if key in cache and cache[key] is not None:
            kdf_fp = cache[key]
        else:
            kdf_fp = compute_fingerprint(f)
            cache[key] = kdf_fp

        if kdf_fp is None:
            continue

        # Check against all trusted fingerprints
        for trusted_key, trusted_fp in trusted_fps:
            if fingerprints_match(kdf_fp, trusted_fp):
                # Mark as trusted duplicate in triage results
                triage_key = f"{f.parent.name}/{f.name}"
                if triage_key in triage_results:
                    triage_results[triage_key]["is_trusted_dup"] = True
                    triage_results[triage_key]["dup_of"] = Path(trusted_key).name
                marked.append(f"{f.relative_to(base_dir)} ← {Path(trusted_key).name}")
                break

        if (i + 1) % 100 == 0:
            print(f"    Checked {i + 1}/{len(kdf_files)} files ({len(marked)} duplicates found)...")
            FINGERPRINT_CACHE.write_text(json.dumps(cache))

    # Final cache save
    FINGERPRINT_CACHE.write_text(json.dumps(cache))

    return len(marked), marked


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def get_session(user: str | None = None, pwd: str | None = None) -> requests.Session:
    """Create a session. Login if credentials provided."""
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) bach-gen-research/1.0"
    )
    # The site uses cookie-based auth after visiting midi.asp links.
    # Pro accounts get unlimited downloads via session cookies.
    if user and pwd:
        # Attempt login — the exact endpoint may need adjustment
        login_url = f"{BASE_URL}/-/login.asp"
        session.post(login_url, data={"email": user, "password": pwd})
    return session


def fetch_page(session: requests.Session, url: str, cache_path: Path) -> str | None:
    """Fetch a page, using cache if available."""
    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path.read_text(errors="replace")

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
        return resp.text
    except requests.RequestException as e:
        print(f"    WARNING: Could not fetch {url}: {e}")
        return None


def _is_valid_midi(data: bytes) -> bool:
    """Check if data starts with MIDI magic bytes (MThd)."""
    return data[:4] == b"MThd"


def download_midi(session: requests.Session, file_path: str, dest: Path) -> bool:
    """Download a single MIDI file."""
    # Direct URL pattern: the midi.asp redirects to mid.files/
    url = f"{BASE_URL}/{DIRECT_PREFIX}{urllib.parse.quote(file_path, safe='/()')}"
    try:
        resp = session.get(url, timeout=30, allow_redirects=True)
        if resp.status_code == 200 and _is_valid_midi(resp.content):
            dest.write_bytes(resp.content)
            return True
        # Fallback to midi.asp redirect
        url2 = f"{BASE_URL}{MIDI_ASP}{urllib.parse.quote(file_path, safe='/()')}"
        resp = session.get(url2, timeout=30, allow_redirects=True)
        if resp.status_code == 200 and _is_valid_midi(resp.content):
            dest.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def collect_midi_links(
    session: requests.Session,
    composer_page: str,
    sub_pages: list[str] | None,
    dirname: str,
) -> list[str]:
    """Collect all MIDI file paths from a composer's pages."""
    all_links: list[str] = []
    seen = set()

    pages_to_fetch = []
    if sub_pages:
        pages_to_fetch = sub_pages
    else:
        pages_to_fetch = [composer_page]

    for page in pages_to_fetch:
        if page.startswith("http"):
            url = page
        elif "/" in page:
            url = f"{BASE_URL}/{page}"
        else:
            url = f"{BASE_URL}/{page}"

        cache_file = CACHE / page.replace("/", "_")
        html = fetch_page(session, url, cache_file)
        if not html:
            continue

        links, discovered_subs = parse_midi_links(html)
        for link in links:
            if link not in seen:
                seen.add(link)
                all_links.append(link)

        time.sleep(DELAY_BETWEEN_PAGES)

    return all_links


def filename_from_path(file_path: str) -> str:
    """Extract a clean filename from the MIDI file path."""
    return file_path.split("/")[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Collect: gather training-ready files into a clean output directory
# ---------------------------------------------------------------------------

TRAINING_DIR = OUT / "_training"


def collect_training_data(
    triage_results: dict,
    base_dir: Path,
    output_dir: Path | None = None,
) -> int:
    """Copy all training-ready files into a clean output directory.

    For each original file, picks the best available version:
    1. If a .reduced.mid exists and is clean → use that
    2. If a .deperformed.mid exists and is clean → use that
    3. If the original is clean → use that

    Skips files marked as duplicates (internal or trusted).
    Preserves composer subdirectory structure.

    Returns count of files collected.
    """
    import shutil

    if output_dir is None:
        output_dir = TRAINING_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a map of original files → best available version
    # First, find all original files (not derived)
    originals: dict[str, dict] = {}
    for key, info in triage_results.items():
        if any(s in key for s in (".deperformed.", ".separated.", ".reduced.")):
            continue
        originals[key] = info

    collected = 0
    skipped_dup = 0
    skipped_bad = 0

    for key, info in sorted(originals.items()):
        # Skip duplicates
        if info.get("is_internal_dup") or info.get("is_trusted_dup"):
            skipped_dup += 1
            continue

        src = base_dir / key

        # Find best available version (prefer processed over original)
        candidates = []

        # Check for reduced version
        reduced_key = key.replace(".mid", ".reduced.mid")
        if reduced_key in triage_results:
            reduced_info = triage_results[reduced_key]
            if reduced_info.get("status") == "clean":
                candidates.append((base_dir / reduced_key, "reduced"))

        # Check for deperformed version
        deperformed_key = key.replace(".mid", ".deperformed.mid")
        if deperformed_key in triage_results:
            dep_info = triage_results[deperformed_key]
            if dep_info.get("status") == "clean":
                candidates.append((base_dir / deperformed_key, "deperformed"))

        # Check for separated version
        separated_key = key.replace(".mid", ".separated.mid")
        if separated_key in triage_results:
            sep_info = triage_results[separated_key]
            if sep_info.get("status") == "clean":
                candidates.append((base_dir / separated_key, "separated"))

        # Original itself
        if info.get("status") == "clean":
            candidates.append((src, "original"))

        if not candidates:
            skipped_bad += 1
            continue

        # Priority: reduced > deperformed > separated > original
        priority = {"reduced": 0, "deperformed": 1, "separated": 2, "original": 3}
        candidates.sort(key=lambda c: priority.get(c[1], 99))
        best_path, best_type = candidates[0]

        if not best_path.exists():
            skipped_bad += 1
            continue

        # Copy to output dir preserving composer subdirectory
        # Use original filename (without .reduced/.deperformed suffix)
        parts = key.split("/")
        if len(parts) >= 2:
            dest_dir = output_dir / parts[0]
        else:
            dest_dir = output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Name: composer/original_name.mid (even if source is .reduced.mid)
        dest = dest_dir / parts[-1]
        shutil.copy2(best_path, dest)
        collected += 1

    print(f"  Collected {collected} files into {output_dir}/")
    print(f"  Skipped: {skipped_dup} duplicates, {skipped_bad} bad/unavailable")
    return collected


def main():
    # Credentials
    user = os.environ.get("KUNSTDERFUGE_USER")
    pwd = os.environ.get("KUNSTDERFUGE_PASS")
    if not user:
        print("Tip: Set KUNSTDERFUGE_USER and KUNSTDERFUGE_PASS env vars for Pro access.")
        print("     Without Pro, downloads are limited to 5/day.\n")

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    session = get_session(user, pwd)

    # Parse CLI args for test mode
    max_dl = MAX_DOWNLOADS
    if "--test" in sys.argv:
        max_dl = 5
        print("TEST MODE: downloading max 5 files total\n")
    elif "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        max_dl = int(sys.argv[idx + 1])
        print(f"Limited to {max_dl} downloads total\n")

    # Filter composers if specified
    composer_filter = None
    if "--composer" in sys.argv:
        idx = sys.argv.index("--composer")
        composer_filter = sys.argv[idx + 1].lower().split(",")

    # Triage only mode
    triage_only = "--triage-only" in sys.argv

    # De-perform mode
    do_deperform = "--deperform" in sys.argv
    deperform_grid = 16
    if "--grid" in sys.argv:
        idx = sys.argv.index("--grid")
        deperform_grid = int(sys.argv[idx + 1])

    # Voice separation mode
    do_voice_sep = "--voice-sep" in sys.argv

    # Deduplication modes
    do_dedup = "--dedup" in sys.argv
    do_dedup_internal = "--dedup-internal" in sys.argv

    # Voice reduction mode
    do_voice_reduce = "--voice-reduce" in sys.argv
    voice_reduce_max = 4
    if "--max-voices" in sys.argv:
        idx = sys.argv.index("--max-voices")
        voice_reduce_max = int(sys.argv[idx + 1])

    # Collect training data mode
    do_collect = "--collect" in sys.argv

    total_new = 0
    total_skip = 0
    total_fail = 0
    triage_results: dict = {}
    failed_files: set = set()

    # Load previously failed files (paywall/premium content)
    if FAILED_FILE.exists():
        failed_files = set(json.loads(FAILED_FILE.read_text()))

    # Load existing triage report
    if REPORT_FILE.exists():
        triage_results = json.loads(REPORT_FILE.read_text())

    for composer_page, sub_pages, dirname, era in COMPOSERS:
        if composer_filter and dirname not in composer_filter:
            continue

        print()
        print("=" * 60)
        print(f"  {dirname} ({era})")
        print("=" * 60)

        comp_dir = OUT / dirname
        comp_dir.mkdir(parents=True, exist_ok=True)

        if triage_only:
            # Just triage existing files
            midi_files = sorted(comp_dir.glob("*.mid"))
            print(f"  Triaging {len(midi_files)} existing files...")
            for f in midi_files:
                key = f"{dirname}/{f.name}"
                if key in triage_results and triage_results[key].get("status") != "pending":
                    continue
                result = triage_midi(f)
                triage_results[key] = result
                status_icons = {"clean": "+", "needs_deperform": "~", "needs_voice_sep": "V", "needs_voice_reduce": "R", "bad": "x", "pending": "?"}
                status_icon = status_icons.get(result["status"], "?")
                print(f"    [{status_icon}] {f.name}: {result['status']}"
                      + (f" — {', '.join(result['issues'])}" if result["issues"] else ""))
            continue

        # Collect MIDI links from pages
        print("  Scanning pages for MIDI links...")
        midi_links = collect_midi_links(session, composer_page, sub_pages, dirname)
        print(f"  Found {len(midi_links)} MIDI files")

        if not midi_links:
            continue

        # Download
        new = 0
        skip = 0
        fail = 0

        for file_path in midi_links:
            if max_dl is not None and total_new >= max_dl:
                print(f"  Reached download limit ({max_dl}), stopping.")
                break

            fname = filename_from_path(file_path)
            dest = comp_dir / fname

            if dest.exists():
                skip += 1
                continue

            if file_path in failed_files:
                skip += 1
                continue

            if download_midi(session, file_path, dest):
                new += 1
                total_new += 1
                print(f"    Downloaded: {fname}")

                # Triage immediately
                result = triage_midi(dest)
                triage_results[f"{dirname}/{fname}"] = result
                status_icons = {"clean": "+", "needs_deperform": "~", "needs_voice_sep": "V", "needs_voice_reduce": "R", "bad": "x", "pending": "?"}
                status_icon = status_icons.get(result["status"], "?")
                if result["issues"]:
                    print(f"    [{status_icon}] {result['status']}: {', '.join(result['issues'])}")
            else:
                fail += 1
                total_fail += 1
                failed_files.add(file_path)
                print(f"    FAILED: {fname}")

            time.sleep(DELAY_BETWEEN_DOWNLOADS)

        skip_total = skip
        total_skip += skip_total
        actual = len(list(comp_dir.glob("*.mid")))
        print(f"  Result: {new} new, {skip} skipped, {fail} failed ({actual} total)")

        if max_dl is not None and total_new >= max_dl:
            break

    # De-perform fixable files
    if do_deperform:
        print()
        print("=" * 60)
        print("  DE-PERFORMING")
        print("=" * 60)
        n_fixed = deperform_all(triage_results, OUT, grid=deperform_grid)
        print(f"  Processed {n_fixed} files")

    # Voice separation via MuseScore
    if do_voice_sep:
        print()
        print("=" * 60)
        print("  VOICE SEPARATION (MuseScore)")
        print("=" * 60)
        n_sep = voice_separate_all(triage_results, OUT)
        print(f"  Successfully separated {n_sep} files")

    # Voice reduction (N voices → 4)
    if do_voice_reduce:
        print()
        print("=" * 60)
        print(f"  VOICE REDUCTION (→ {voice_reduce_max} voices)")
        print("=" * 60)
        n_reduced = reduce_voices_all(triage_results, OUT, max_voices=voice_reduce_max)
        print(f"  Reduced {n_reduced} files")

    # Internal dedup (pick best version of same piece)
    if do_dedup_internal:
        print()
        print("=" * 60)
        print("  INTERNAL DEDUP (pick best version per piece)")
        print("=" * 60)
        n_marked, marked_list = dedup_internal(OUT, triage_results)
        print(f"  Marked {n_marked} inferior versions (originals preserved)")

    # External dedup (against trusted datasets)
    if do_dedup:
        print()
        print("=" * 60)
        print("  DEDUPLICATION (against trusted datasets)")
        print("=" * 60)
        n_marked, marked_list = dedup_against_trusted(OUT, triage_results)
        print(f"  Marked {n_marked} duplicates (originals preserved)")
        for r in marked_list:
            print(f"    {r}")

    # Collect training-ready files
    if do_collect:
        print()
        print("=" * 60)
        print("  COLLECTING TRAINING DATA")
        print("=" * 60)
        n_collected = collect_training_data(triage_results, OUT)

    # Save triage report
    REPORT_FILE.write_text(json.dumps(triage_results, indent=2))
    if failed_files:
        FAILED_FILE.write_text(json.dumps(sorted(failed_files), indent=2))

    # Summary
    clean = sum(1 for v in triage_results.values() if v.get("status") == "clean")
    needs_deperform = sum(1 for v in triage_results.values() if v.get("status") == "needs_deperform")
    needs_voice_sep = sum(1 for v in triage_results.values() if v.get("status") == "needs_voice_sep")
    needs_voice_reduce = sum(1 for v in triage_results.values() if v.get("status") == "needs_voice_reduce")
    bad = sum(1 for v in triage_results.values() if v.get("status") == "bad")
    pending = sum(1 for v in triage_results.values() if v.get("status") == "pending")

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  New downloads:  {total_new}")
    print(f"  Already had:    {total_skip}")
    print(f"  Failed:         {total_fail}")
    print()
    print(f"  Triage results ({len(triage_results)} files):")
    print(f"    [+] Clean (ready for training):     {clean}")
    print(f"    [~] Needs de-perform (fixable):     {needs_deperform}")
    print(f"    [R] Needs voice reduction (>4 v):   {needs_voice_reduce}")
    print(f"    [V] Needs voice separation (hard):  {needs_voice_sep}")
    print(f"    [x] Bad (corrupt/empty):            {bad}")
    if pending:
        print(f"    [?] Pending (needs triage):         {pending}")
        print()
        print("  To triage pending files, run:")
        print("    uv run python scripts/download_kunstderfuge.py --triage-only")
    print()
    print(f"  Files:   {OUT}/")
    print(f"  Report:  {REPORT_FILE}")
    print()
    print("  Commands:")
    print("    uv run python scripts/download_kunstderfuge.py --triage-only          # re-triage without downloading")
    print("    uv run python scripts/download_kunstderfuge.py --deperform            # quantize + strip dynamics")
    print("    uv run python scripts/download_kunstderfuge.py --deperform --grid 12  # triplet-friendly grid")
    print("    uv run python scripts/download_kunstderfuge.py --voice-sep            # MuseScore voice separation")
    print("    uv run python scripts/download_kunstderfuge.py --voice-reduce         # reduce >4 voice files to 4")
    print("    uv run python scripts/download_kunstderfuge.py --voice-reduce --max-voices 3  # reduce to 3 voices")
    print("    uv run python scripts/download_kunstderfuge.py --dedup-internal        # pick best version of same piece")
    print("    uv run python scripts/download_kunstderfuge.py --dedup                # mark duplicates from trusted datasets")
    print("    uv run python scripts/download_kunstderfuge.py --collect              # copy training-ready files to _training/")
    print()
    print("  [+] Clean files can go straight into prepare-data.")
    print("  [~] De-perform files are fixable (run --deperform).")
    print("  [R] Voice reduction reduces >4 voice files to 4 (run --voice-reduce).")
    print("  [V] Voice separation via MuseScore (run --voice-sep).")
    print("  Dedup marks duplicates in the triage report (originals preserved).")
    print("  Collect copies the best version of each file into _training/.")


if __name__ == "__main__":
    main()
