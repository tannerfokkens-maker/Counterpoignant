from __future__ import annotations

import mido

from bach_gen.utils.midi_io import midi_key_signature, midi_time_signature, midi_to_note_events, note_events_to_midi


def test_midi_to_note_events_normalizes_ticks_per_beat() -> None:
    mid = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message("note_on", note=60, velocity=80, time=960))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=62, velocity=80, time=480))
    track.append(mido.Message("note_off", note=62, velocity=0, time=960))

    assert midi_to_note_events(mid) == [[
        (480, 240, 60),
        (960, 480, 62),
    ]]


def test_note_events_to_midi_writes_key_and_time_signature_meta() -> None:
    mid = note_events_to_midi(
        voices=[[(0, 480, 60)], [(0, 480, 48)]],
        key_root=7,
        key_mode="minor",
        time_signature=(3, 4),
    )

    assert midi_key_signature(mid) == "Gm"
    assert midi_time_signature(mid) == (3, 4)
