from __future__ import annotations

from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
from bach_gen.data.tokenizer import BachTokenizer
from bach_gen.generation.live import (
    IncrementalMidiTokenParser,
    PromptPinnedWindow,
    autodetect_channel_base,
    autodetect_midi_output_port,
)
from bach_gen.generation.subject import subject_string_to_events, subject_string_to_note_events


def test_subject_parser_supports_rests_and_natural_duration_names() -> None:
    events = subject_string_to_events("C4:quarter r:e D4:q. rest:1/16 E4:half")

    assert [(event.start_tick, event.duration_ticks, event.midi_note) for event in events] == [
        (0, 480, 60),
        (480, 240, None),
        (720, 720, 62),
        (1440, 120, None),
        (1560, 960, 64),
    ]

    assert subject_string_to_note_events("C4:quarter r:e D4:q. rest:1/16 E4:half") == [
        (0, 480, 60),
        (720, 720, 62),
        (1560, 960, 64),
    ]


def test_incremental_live_parser_emits_voice_split_messages() -> None:
    tokenizer = BachTokenizer()
    parser = IncrementalMidiTokenParser(
        tokenizer,
        key_root=0,
        key_mode="major",
    )

    tokens = [
        tokenizer.name_to_token["KEY_C_major"],
        tokenizer.VOICE_1,
        tokenizer.name_to_token["Pitch_60"],
        tokenizer.name_to_token["Dur_480"],
        tokenizer.name_to_token["TimeShift_480"],
        tokenizer.VOICE_2,
        tokenizer.name_to_token["Pitch_48"],
        tokenizer.name_to_token["Dur_960"],
    ]
    messages = parser.feed_tokens(tokens)

    assert [(msg.tick, msg.priority, msg.message.type, msg.message.note, msg.message.channel) for msg in messages] == [
        (0, 1, "note_on", 60, 0),
        (480, 0, "note_off", 60, 0),
        (480, 1, "note_on", 48, 1),
        (1440, 0, "note_off", 48, 1),
    ]


def test_incremental_live_parser_supports_scale_degree_tokens() -> None:
    tokenizer = ScaleDegreeTokenizer()
    parser = IncrementalMidiTokenParser(
        tokenizer,
        key_root=0,
        key_mode="major",
        single_channel=True,
    )

    tokens = [
        tokenizer.name_to_token["KEY_C_major"],
        tokenizer.VOICE_1,
        tokenizer.name_to_token["OCT_4"],
        tokenizer.name_to_token["SHARP"],
        tokenizer.name_to_token["DEG_1"],
        tokenizer.name_to_token["Dur_480"],
    ]
    messages = parser.feed_tokens(tokens)

    assert [(msg.tick, msg.message.type, msg.message.note, msg.message.channel) for msg in messages] == [
        (0, "note_on", 49, 0),
        (480, "note_off", 49, 0),
    ]


def test_prompt_pinned_window_keeps_context_within_budget() -> None:
    prompt = list(range(40))
    window = PromptPinnedWindow.build(prompt, max_seq_len=64)
    prefill = window.build_prefill(list(range(100, 200)))

    assert len(prefill) <= 64
    assert prefill[: len(window.pinned_prompt)] == window.pinned_prompt
    assert window.rebuild_every_tokens >= 8


def test_autodetect_port_prefers_minilogue_and_channel_defaults_to_one() -> None:
    port = autodetect_midi_output_port(
        [
            "IAC Driver Bus 1",
            "minilogue xd SOUND",
            "Scarlett 4i4 MIDI",
        ]
    )

    assert port == "minilogue xd SOUND"
    assert autodetect_channel_base(port) == 0
