"""MIDI read/write helpers."""

from __future__ import annotations

from pathlib import Path

import mido

from bach_gen.utils.constants import TICKS_PER_QUARTER


def note_events_to_midi(
    upper_notes: list[tuple[int, int, int]] | None = None,
    lower_notes: list[tuple[int, int, int]] | None = None,
    voices: list[list[tuple[int, int, int]]] | None = None,
    ticks_per_quarter: int = TICKS_PER_QUARTER,
    tempo: int = 500000,  # microseconds per beat (120 BPM)
) -> mido.MidiFile:
    """Convert note events to a MIDI file.

    Accepts either:
      - upper_notes + lower_notes (legacy 2-voice interface), or
      - voices: list of N voice note lists (N-voice interface).

    Args:
        upper_notes: List of (start, duration, pitch) for upper voice (legacy).
        lower_notes: List of (start, duration, pitch) for lower voice (legacy).
        voices: List of N voice note lists, each (start, duration, pitch).
        ticks_per_quarter: Ticks per quarter note.
        tempo: Tempo in microseconds per beat.

    Returns:
        mido.MidiFile
    """
    # Build voice list from arguments
    if voices is not None:
        voice_list = voices
    else:
        voice_list = []
        if upper_notes is not None:
            voice_list.append(upper_notes)
        if lower_notes is not None:
            voice_list.append(lower_notes)

    mid = mido.MidiFile(ticks_per_beat=ticks_per_quarter)

    for channel, voice_notes in enumerate(voice_list):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        if channel == 0:
            track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        # Build note-on/note-off events
        events: list[tuple[int, str, int, int]] = []
        for start, dur, pitch in voice_notes:
            events.append((start, "note_on", pitch, 80))
            events.append((start + dur, "note_off", pitch, 0))

        # Sort by time, then note_off before note_on at same time
        events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

        current_tick = 0
        for tick, msg_type, pitch, velocity in events:
            delta = tick - current_tick
            track.append(mido.Message(msg_type, note=pitch, velocity=velocity,
                                      time=delta, channel=channel))
            current_tick = tick

    return mid


def save_midi(mid: mido.MidiFile, path: str | Path) -> None:
    """Save a MIDI file to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def load_midi(path: str | Path) -> mido.MidiFile:
    """Load a MIDI file from disk."""
    return mido.MidiFile(str(path))


def midi_to_note_events(mid: mido.MidiFile) -> list[list[tuple[int, int, int]]]:
    """Extract note events from a MIDI file.

    Returns:
        List of tracks, each track is a list of (start_tick, duration_tick, pitch).
    """
    all_tracks = []
    for track in mid.tracks:
        notes: list[tuple[int, int, int]] = []
        active: dict[int, int] = {}  # pitch -> start_tick
        current_tick = 0

        for msg in track:
            current_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = current_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    start = active.pop(msg.note)
                    dur = current_tick - start
                    if dur > 0:
                        notes.append((start, dur, msg.note))

        if notes:
            notes.sort(key=lambda n: (n[0], n[2]))
            all_tracks.append(notes)

    return all_tracks
