from __future__ import annotations

from bach_gen.data.extraction import VoiceComposition
from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
from bach_gen.data.tokenizer import BachTokenizer


def _simple_comp(num_voices: int, *, style: str) -> VoiceComposition:
    voices: list[list[tuple[int, int, int]]] = []
    for idx in range(num_voices):
        base_pitch = 60 - (idx * 5)
        voices.append(
            [
                (0, 480, base_pitch),
                (480, 480, base_pitch + 2),
                (960, 480, base_pitch + 4),
                (1440, 480, base_pitch + 5),
            ]
        )
    return VoiceComposition(
        voices=voices,
        key_root=0,
        key_mode="major",
        source="unit-test",
        style=style,
        time_signature=(4, 4),
    )


def test_bach_tokenizer_emits_romantic_keyboard_piece_prefix() -> None:
    tokenizer = BachTokenizer()
    comp = _simple_comp(2, style="romantic")

    seq = tokenizer.encode(comp, form="keyboard_piece", style="romantic", length_bars=8)
    prefix = [tokenizer.token_to_name[tok] for tok in seq[:4]]

    assert prefix == ["BOS", "STYLE_ROMANTIC", "FORM_KEYBOARD_PIECE", "MODE_2PART"]


def test_scale_degree_tokenizer_emits_other_orchestral_prefix() -> None:
    tokenizer = ScaleDegreeTokenizer()
    comp = _simple_comp(4, style="other")

    seq = tokenizer.encode(comp, form="orchestral_reduction", style="other", length_bars=24)
    prefix = [tokenizer.token_to_name[tok] for tok in seq[:4]]

    assert prefix == ["BOS", "STYLE_OTHER", "FORM_ORCHESTRAL_REDUCTION", "MODE_4PART"]
