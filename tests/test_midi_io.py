from __future__ import annotations

import mido

from bach_gen.utils.midi_io import midi_to_note_events


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
