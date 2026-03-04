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


def _extract_note_channels_by_track(mid: mido.MidiFile) -> list[set[int]]:
    channels_by_track: list[set[int]] = []
    for track in mid.tracks:
        chs = {
            int(getattr(msg, "channel", 0))
            for msg in track
            if msg.type == "note_on" and int(getattr(msg, "velocity", 0)) > 0
        }
        if chs:
            channels_by_track.append(chs)
    return channels_by_track


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
    channels_by_track = _extract_note_channels_by_track(out_mid)
    assert len(channels_by_track) == len(tracks)
    for chs in channels_by_track:
        assert len(chs) == 1
    uniq_channels = {next(iter(chs)) for chs in channels_by_track}
    assert len(uniq_channels) == len(channels_by_track)


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


def test_voice_separate_auto_two_pass_discovers_four_voices(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "varying_polyphony.mid"
    dest = tmp_path / "varying_polyphony.separated.mid"
    tpqn = 480

    notes = []
    for i in range(8):
        start = i * tpqn
        notes.append((start, start + tpqn, 72, 80, 0))
        notes.append((start, start + tpqn, 48, 70, 0))
    for i in range(8, 16):
        start = i * tpqn
        notes.append((start, start + tpqn, 76, 82, 0))
        notes.append((start, start + tpqn, 69, 78, 0))
        notes.append((start, start + tpqn, 60, 74, 0))
        notes.append((start, start + tpqn, 52, 70, 0))
    _write_midi(src, tpqn, notes)

    ok, reason = mod.voice_separate_midi(
        src=src,
        dest=dest,
        max_voices=8,
        mode="auto",
        jitter_ratio=0.025,
        on_cap="raise-cap",
        ignore_pedal=True,
    )
    assert ok, reason

    tracks = _extract_note_tracks(mido.MidiFile(str(dest)))
    assert len(tracks) == 4
    for tr in tracks:
        _assert_monophonic(tr)


def test_voice_separate_auto_matches_fixed_uniform_three_voice(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "uniform_3v.mid"
    auto_dest = tmp_path / "uniform_3v.auto.mid"
    fixed_dest = tmp_path / "uniform_3v.fixed.mid"
    tpqn = 480

    notes = []
    for i in range(12):
        start = i * tpqn
        notes.append((start, start + tpqn, 76 + (i % 2), 82, 0))
        notes.append((start, start + tpqn, 64 + (i % 2), 76, 0))
        notes.append((start, start + tpqn, 52 + (i % 2), 70, 0))
    _write_midi(src, tpqn, notes)

    ok_auto, reason_auto = mod.voice_separate_midi(
        src=src,
        dest=auto_dest,
        max_voices=16,
        mode="auto",
        jitter_ratio=0.025,
        on_cap="raise-cap",
        ignore_pedal=True,
    )
    assert ok_auto, reason_auto

    ok_fixed, reason_fixed = mod.voice_separate_midi(
        src=src,
        dest=fixed_dest,
        max_voices=3,
        mode="fixed",
        jitter_ratio=0.025,
        on_cap="raise-cap",
        ignore_pedal=True,
    )
    assert ok_fixed, reason_fixed

    auto_tracks = _extract_note_tracks(mido.MidiFile(str(auto_dest)))
    fixed_tracks = _extract_note_tracks(mido.MidiFile(str(fixed_dest)))
    assert auto_tracks == fixed_tracks


def test_voice_separate_auto_pass2_fallback_uses_pass1_result(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "fallback.mid"
    dest = tmp_path / "fallback.separated.mid"
    tpqn = 480

    notes = []
    for i in range(8):
        start = i * tpqn
        notes.append((start, start + tpqn, 72 + (i % 3), 80, 0))
        notes.append((start, start + tpqn, 52 + (i % 3), 70, 0))
    _write_midi(src, tpqn, notes)

    original_run = mod._run_voice_assignment
    calls = {"n": 0}

    def _patched_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return original_run(*args, **kwargs)
        return False, None, "forced pass2 failure"

    mod._run_voice_assignment = _patched_run
    try:
        ok, reason = mod.voice_separate_midi(
            src=src,
            dest=dest,
            max_voices=8,
            mode="auto",
            jitter_ratio=0.025,
            on_cap="raise-cap",
            ignore_pedal=True,
        )
    finally:
        mod._run_voice_assignment = original_run

    assert calls["n"] == 2
    assert ok, reason
    assert dest.exists()


def test_voice_separate_p90_helper_removed() -> None:
    mod = _load_download_module()
    assert not hasattr(mod, "_p90_cluster_size")


def test_voice_separate_all_retries_with_stronger_settings(tmp_path: Path) -> None:
    mod = _load_download_module()
    comp_dir = tmp_path / "chopin"
    comp_dir.mkdir(parents=True, exist_ok=True)
    src = comp_dir / "retry.mid"
    _write_midi(src, 480, [(0, 480, 72, 80, 0), (0, 480, 52, 70, 0)])

    triage_results = {"chopin/retry.mid": {"status": "needs_voice_sep"}}
    calls: list[dict] = []

    orig_sep = mod.voice_separate_midi
    orig_triage = mod.triage_midi

    def _patched_voice_separate_midi(
        src: Path,
        dest: Path,
        max_voices: int = 16,
        mode: str = "auto",
        jitter_ratio: float = 0.025,
        on_cap: str = "raise-cap",
        ignore_pedal: bool = True,
    ):
        calls.append(
            {
                "max_voices": max_voices,
                "mode": mode,
                "jitter_ratio": jitter_ratio,
                "on_cap": on_cap,
                "ignore_pedal": ignore_pedal,
            }
        )
        if len(calls) == 1:
            return False, "forced failure"
        _write_midi(dest, 480, [(0, 480, 72, 80, 0), (0, 480, 52, 70, 1)])
        return True, None

    mod.voice_separate_midi = _patched_voice_separate_midi
    mod.triage_midi = lambda _p: {
        "status": "clean",
        "num_tracks": 2,
        "tracks_with_notes": 2,
        "issues": [],
    }

    try:
        processed = mod.voice_separate_all(
            triage_results=triage_results,
            base_dir=tmp_path,
            max_voices=16,
            mode="auto",
            jitter_ratio=0.025,
            on_cap="raise-cap",
            ignore_pedal=True,
        )
    finally:
        mod.voice_separate_midi = orig_sep
        mod.triage_midi = orig_triage

    assert processed == 1
    assert len(calls) >= 2
    assert calls[0]["mode"] == "auto"
    assert calls[1]["mode"] == "fixed"
    assert calls[1]["max_voices"] == 32
    assert (comp_dir / "retry.separated.mid").exists()
    assert "chopin/retry.separated.mid" in triage_results
