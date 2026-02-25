"""Bisect-based index for fast pitch/activity lookup in a voice."""

from __future__ import annotations

import bisect


class VoiceIndex:
    """Sorted index over a voice's notes for O(log n) time-point queries.

    Args:
        notes: List of (start_tick, duration_ticks, midi_pitch) tuples.
    """

    def __init__(self, notes: list[tuple[int, int, int]]) -> None:
        # Sort by start_tick for bisect
        sorted_notes = sorted(notes, key=lambda n: n[0])
        self._starts: list[int] = [n[0] for n in sorted_notes]
        self._durs: list[int] = [n[1] for n in sorted_notes]
        self._pitches: list[int] = [n[2] for n in sorted_notes]

    def pitch_at(self, t: int) -> int | None:
        """Return the pitch of the note sounding at time *t*, or ``None``.

        Uses ``bisect_right`` to find the candidate note whose start
        is <= *t*, then checks whether *t* falls within its duration.
        Also checks for an exact onset match at the bisect index.
        """
        idx = bisect.bisect_right(self._starts, t) - 1

        # Check exact onset at idx+1 (bisect_right may place t before a
        # note that starts exactly at t when duplicates exist, but the
        # primary candidate is idx).
        right = idx + 1
        if right < len(self._starts) and self._starts[right] == t:
            return self._pitches[right]

        if idx < 0:
            return None

        start = self._starts[idx]
        if start <= t < start + self._durs[idx]:
            return self._pitches[idx]

        return None

    def is_active(self, t: int) -> bool:
        """Return whether any note is sounding at time *t*."""
        return self.pitch_at(t) is not None
