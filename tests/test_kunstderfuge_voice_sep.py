"""Tests for local Kunstderfuge voice separation."""

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


def _write_midi(path: Path, tpqn: int, notes: list[tuple[int, int, int, int, int]]) -> None:
    """Write a single-track MIDI from absolute note tuples."""
    mid = mido.MidiFile(ticks_per_beat=tpqn)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )

    events: list[tuple[int, int, int, int, int]] = []
    for start, end, pitch, velocity, channel in notes:
        events.append((start, 1, pitch, velocity, channel))
        events.append((max(start + 1, end), 0, pitch, 0, channel))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    cur = 0
    for tick, is_on, pitch, velocity, channel in events:
        delta = tick - cur
        if is_on:
            track.append(
                mido.Message(
                    "note_on", note=pitch, velocity=velocity, time=delta, channel=channel
                )
            )
        else:
            track.append(
                mido.Message(
                    "note_off", note=pitch, velocity=0, time=delta, channel=channel
                )
            )
        cur = tick

    mid.save(str(path))


def _extract_note_tracks(mid: mido.MidiFile) -> list[list[tuple[int, int, int]]]:
    note_tracks: list[list[tuple[int, int, int]]] = []
    for track in mid.tracks:
        cur = 0
        active: dict[tuple[int, int], list[int]] = {}
        notes: list[tuple[int, int, int]] = []
        for msg in track:
            cur += int(msg.time)
            if msg.type == "note_on" and msg.velocity > 0:
                key = (int(getattr(msg, "channel", 0)), int(msg.note))
                active.setdefault(key, []).append(cur)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (int(getattr(msg, "channel", 0)), int(msg.note))
                starts = active.get(key)
                if not starts:
                    continue
                start = starts.pop(0)
                if not starts:
                    del active[key]
                dur = cur - start
                if dur > 0:
                    notes.append((start, dur, int(msg.note)))
        if notes:
            notes.sort(key=lambda n: (n[0], n[2]))
            note_tracks.append(notes)
    return note_tracks


def _assert_monophonic(track_notes: list[tuple[int, int, int]]) -> None:
    for (s0, d0, _), (s1, _, _) in zip(track_notes, track_notes[1:]):
        assert s1 >= s0 + d0


def test_voice_separate_basic_monophony(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "poly.mid"
    dest = tmp_path / "poly.separated.mid"
    tpqn = 480

    notes = []
    for i in range(8):
        start = i * tpqn
        notes.append((start, start + tpqn, 72 + (i % 5), 80, 0))
        notes.append((start, start + tpqn, 52 + (i % 5), 70, 0))
    _write_midi(src, tpqn, notes)

    ok, reason = mod.voice_separate_midi(
        src=src,
        dest=dest,
        max_voices=6,
        mode="auto",
        jitter_ratio=0.025,
        on_cap="fail",
        ignore_pedal=True,
    )
    assert ok, reason
    out_mid = mido.MidiFile(str(dest))
    tracks = _extract_note_tracks(out_mid)
    assert len(tracks) >= 2
    for tr in tracks:
        _assert_monophonic(tr)


def test_voice_separate_cap_failure(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "too_many.mid"
    dest = tmp_path / "too_many.separated.mid"
    tpqn = 480

    notes = []
    for pitch in (84, 79, 72, 67, 60):
        notes.append((0, tpqn, pitch, 80, 0))
    _write_midi(src, tpqn, notes)

    ok, reason = mod.voice_separate_midi(
        src=src,
        dest=dest,
        max_voices=4,
        mode="fixed",
        jitter_ratio=0.025,
        on_cap="fail",
        ignore_pedal=True,
    )
    assert not ok
    assert reason is not None and "cluster of size" in reason
    assert not dest.exists()


def test_voice_separate_ignore_pedal_assignment_only(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "pedal.mid"
    dest = tmp_path / "pedal.separated.mid"
    tpqn = 480

    notes = []
    for i in range(8):
        start = i * 4 * tpqn
        long_end = start + 8 * tpqn
        notes.append((start, long_end, 72 + (i % 4), 82, 0))
        notes.append((start, long_end, 50 + (i % 4), 74, 0))
        if i % 2 == 0:
            notes.append((start, long_end, 62 + (i % 3), 78, 0))
    _write_midi(src, tpqn, notes)

    ok, reason = mod.voice_separate_midi(
        src=src,
        dest=dest,
        max_voices=16,
        mode="auto",
        jitter_ratio=0.025,
        on_cap="fail",
        ignore_pedal=True,
    )
    assert ok, reason
    out_mid = mido.MidiFile(str(dest))
    tracks = _extract_note_tracks(out_mid)
    assert 2 <= len(tracks) <= 3

    max_dur = max(d for tr in tracks for _, d, _ in tr)
    assert max_dur >= 8 * tpqn - 1
