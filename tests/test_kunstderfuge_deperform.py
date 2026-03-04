"""Tests for deperform rewrite (raw MIDI, ornament-aware)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mido


def _load_download_module():
    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "download_kunstderfuge.py"
    spec = importlib.util.spec_from_file_location("download_kunstderfuge", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_track_from_abs_events(path: Path, tpqn: int, events: list[tuple[int, object]]) -> None:
    mid = mido.MidiFile(ticks_per_beat=tpqn)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    ordered = sorted(enumerate(events), key=lambda x: (int(x[1][0]), x[0]))
    current = 0
    for _, (abs_tick, msg) in ordered:
        delta = int(abs_tick) - current
        track.append(msg.copy(time=max(0, delta)))
        current = int(abs_tick)
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(str(path))


def _extract_notes(mid: mido.MidiFile) -> list[tuple[int, int, int, int, int, int]]:
    notes: list[tuple[int, int, int, int, int, int]] = []
    for track_idx, track in enumerate(mid.tracks):
        current = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for msg in track:
            current += int(msg.time)
            if msg.type == "note_on" and msg.velocity > 0:
                key = (int(getattr(msg, "channel", 0)), int(msg.note))
                active.setdefault(key, []).append((current, int(msg.velocity)))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (int(getattr(msg, "channel", 0)), int(msg.note))
                starts = active.get(key)
                if not starts:
                    continue
                start, vel = starts.pop(0)
                if not starts:
                    del active[key]
                end = current
                if end > start:
                    notes.append((track_idx, start, end, int(msg.note), vel, key[0]))
    notes.sort(key=lambda n: (n[1], n[3], n[0]))
    return notes


def test_deperform_downbeat_magnetism(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "downbeat.mid"
    dest = tmp_path / "downbeat.deperformed.mid"
    tpqn = 480

    events = [
        (0, mido.MetaMessage("set_tempo", tempo=500000, time=0)),
        (
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            ),
        ),
        (475, mido.Message("note_on", note=60, velocity=90, channel=0, time=0)),
        (700, mido.Message("note_off", note=60, velocity=0, channel=0, time=0)),
    ]
    _write_track_from_abs_events(src, tpqn, events)

    mod.DEPERFORM_PROTECT_ORNAMENTS = True
    assert mod.deperform_midi(src, dest, grid=16, velocity=64)

    out_mid = mido.MidiFile(str(dest))
    notes = _extract_notes(out_mid)
    assert len(notes) == 1
    assert notes[0][1] == 480


def test_deperform_preserves_trill_shape(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "trill.mid"
    dest = tmp_path / "trill.deperformed.mid"
    tpqn = 480

    events: list[tuple[int, object]] = [
        (0, mido.MetaMessage("set_tempo", tempo=520000, time=0)),
        (
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            ),
        ),
    ]
    start = 233
    for i in range(8):
        pitch = 72 if i % 2 == 0 else 74
        on = start + i * 60
        off = on + 50
        events.append((on, mido.Message("note_on", note=pitch, velocity=80, channel=0, time=0)))
        events.append((off, mido.Message("note_off", note=pitch, velocity=0, channel=0, time=0)))
    _write_track_from_abs_events(src, tpqn, events)

    mod.DEPERFORM_PROTECT_ORNAMENTS = True
    assert mod.deperform_midi(src, dest, grid=16, velocity=64)

    notes = _extract_notes(mido.MidiFile(str(dest)))
    pitches = [n[3] for n in notes]
    starts = [n[1] for n in notes]

    assert len(notes) == 8
    assert set(pitches) == {72, 74}
    assert all(pitches[i] != pitches[i + 1] for i in range(len(pitches) - 1))
    assert starts[0] % 120 == 0
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))


def test_deperform_strips_cc_pitchbend_preserves_tempo_program(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "meta.mid"
    dest = tmp_path / "meta.deperformed.mid"
    tpqn = 480

    events = [
        (0, mido.MetaMessage("set_tempo", tempo=600000, time=0)),
        (
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=3,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            ),
        ),
        (0, mido.Message("program_change", program=40, channel=3, time=0)),
        (90, mido.Message("control_change", control=64, value=127, channel=3, time=0)),
        (120, mido.Message("note_on", note=62, velocity=30, channel=3, time=0)),
        (130, mido.Message("pitchwheel", pitch=3000, channel=3, time=0)),
        (360, mido.Message("note_off", note=62, velocity=0, channel=3, time=0)),
        (480, mido.MetaMessage("set_tempo", tempo=450000, time=0)),
        (500, mido.Message("note_on", note=65, velocity=110, channel=3, time=0)),
        (760, mido.Message("note_off", note=65, velocity=0, channel=3, time=0)),
    ]
    _write_track_from_abs_events(src, tpqn, events)

    mod.DEPERFORM_PROTECT_ORNAMENTS = True
    assert mod.deperform_midi(src, dest, grid=16, velocity=64)

    out = mido.MidiFile(str(dest))
    assert len(out.tracks) == 1

    tempos = []
    time_sigs = []
    programs = 0
    controls = 0
    pitchwheels = 0
    note_on_vels = []
    for msg in out.tracks[0]:
        if msg.is_meta and msg.type == "set_tempo":
            tempos.append(int(msg.tempo))
        if msg.is_meta and msg.type == "time_signature":
            time_sigs.append((int(msg.numerator), int(msg.denominator)))
        if msg.type == "program_change":
            programs += 1
        if msg.type == "control_change":
            controls += 1
        if msg.type == "pitchwheel":
            pitchwheels += 1
        if msg.type == "note_on" and msg.velocity > 0:
            note_on_vels.append(int(msg.velocity))

    assert tempos == [600000, 450000]
    assert time_sigs == [(3, 4)]
    assert programs == 1
    assert controls == 0
    assert pitchwheels == 0
    assert note_on_vels and all(v == 64 for v in note_on_vels)


def test_deperform_grace_note_kept_before_main_note(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "grace.mid"
    dest = tmp_path / "grace.deperformed.mid"
    tpqn = 480

    events = [
        (0, mido.MetaMessage("set_tempo", tempo=500000, time=0)),
        (
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            ),
        ),
        (450, mido.Message("note_on", note=67, velocity=80, channel=0, time=0)),
        (490, mido.Message("note_off", note=67, velocity=0, channel=0, time=0)),
        (480, mido.Message("note_on", note=72, velocity=90, channel=0, time=0)),
        (760, mido.Message("note_off", note=72, velocity=0, channel=0, time=0)),
    ]
    _write_track_from_abs_events(src, tpqn, events)

    mod.DEPERFORM_PROTECT_ORNAMENTS = True
    assert mod.deperform_midi(src, dest, grid=16, velocity=64)

    notes = _extract_notes(mido.MidiFile(str(dest)))
    by_pitch = {n[3]: n[1] for n in notes}
    assert 67 in by_pitch and 72 in by_pitch
    assert by_pitch[67] == by_pitch[72] - 120


def test_deperform_fast_run_not_collapsed_to_block_chords(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "fast_run.mid"
    dest = tmp_path / "fast_run.deperformed.mid"
    tpqn = 384

    events: list[tuple[int, object]] = [
        (0, mido.MetaMessage("set_tempo", tempo=500000, time=0)),
        (
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            ),
        ),
    ]

    starts = [
        11512, 11584, 11628, 11672, 11720, 11760, 11816, 11860, 11916, 11956,
        12016, 12064, 12124, 12168, 12228, 12272, 12324, 12380, 12436,
    ]
    pitches = [
        81, 80, 78, 80, 81, 80, 81, 80, 81, 80,
        81, 80, 81, 80, 81, 80, 78, 80, 81,
    ]
    for s, p in zip(starts, pitches):
        events.append((s, mido.Message("note_on", note=p, velocity=88, channel=0, time=0)))
        events.append((s + 40, mido.Message("note_off", note=p, velocity=0, channel=0, time=0)))

    _write_track_from_abs_events(src, tpqn, events)

    mod.DEPERFORM_PROTECT_ORNAMENTS = True
    assert mod.deperform_midi(src, dest, grid=16, velocity=64)

    notes = _extract_notes(mido.MidiFile(str(dest)))
    run_notes = [n for n in notes if n[3] >= 78]
    assert len(run_notes) == len(starts)

    run_starts = [n[1] for n in run_notes]
    assert all(run_starts[i] < run_starts[i + 1] for i in range(len(run_starts) - 1))
    assert len(set(run_starts)) == len(run_starts)
