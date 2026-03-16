from __future__ import annotations

from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
from bach_gen.generation.generator import _build_prompt


def test_build_prompt_encodes_subject_in_training_format() -> None:
    tokenizer = ScaleDegreeTokenizer()

    prompt = _build_prompt(
        tokenizer=tokenizer,
        key_root=0,
        key_mode="major",
        key_name="C_major",
        subject_str="C4:q D4:q E4:q F4:q G4:q",
        form="fugue",
        style="bach",
        num_voices=3,
        meter="4_4",
        texture="polyphonic",
        imitation="high",
        harmonic_rhythm="moderate",
        harmonic_tension="high",
        chromaticism="high",
        encoding_mode="interleaved",
    )
    names = [tokenizer.token_to_name[t] for t in prompt]

    key_idx = names.index("KEY_C_major")
    event_names = names[key_idx + 1:]

    assert event_names[:4] == ["BAR", "BEAT_1", "SUBJECT_START", "VOICE_1"]
    assert event_names.count("SUBJECT_START") == 1
    assert event_names.count("SUBJECT_END") == 1
    assert "BEAT_2" in event_names
    assert "BEAT_3" in event_names
    assert "BEAT_4" in event_names
    assert event_names.count("BAR") >= 2
