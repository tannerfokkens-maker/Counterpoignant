from __future__ import annotations

from bach_gen.app import SubjectBuilderEvent, build_subject_string


def test_build_subject_string_supports_notes_and_rests() -> None:
    subject = build_subject_string(
        [
            SubjectBuilderEvent("D", 4, "q"),
            SubjectBuilderEvent("A", 4, "e"),
            SubjectBuilderEvent("Rest", 4, "e"),
            SubjectBuilderEvent("Bb", 4, "q."),
        ]
    )

    assert subject == "D4:q A4:e r:e Bb4:q."
