"""Real-time endless MIDI generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import logging
import threading
import time
from pathlib import Path

import mido
import torch

from bach_gen.data.tokenizer import BachTokenizer
from bach_gen.generation.constraints import DecodingConstraints
from bach_gen.generation.generator import (
    _build_prompt,
    _build_structural_control_state,
)
from bach_gen.generation.sampling import sample_next_token
from bach_gen.generation.subject import subject_string_to_note_events
from bach_gen.model.architecture import BachTransformer
from bach_gen.utils.constants import (
    DEFAULT_MAX_GEN_LENGTH,
    DEFAULT_MIN_P,
    DEFAULT_TOP_K_SAMPLING,
    DEFAULT_TOP_P,
    DEFAULT_TEMPERATURE,
    FORM_DEFAULTS,
    TICKS_PER_QUARTER,
    ticks_per_measure,
)
from bach_gen.utils.midi_io import load_midi, midi_to_note_events
from bach_gen.utils.music_theory import (
    get_key_signature_name,
    note_name_to_pc,
    parse_key,
    scale_degree_to_midi,
)

logger = logging.getLogger(__name__)

_PREFERRED_PORT_MARKERS = (
    ("minilogue xd", 100),
    ("minilogue", 60),
    ("korg", 40),
    ("xd", 20),
    ("usb", 5),
)


@dataclass(order=True)
class ScheduledMidiMessage:
    """A MIDI message scheduled at an absolute musical tick."""

    tick: int
    priority: int
    message: mido.Message = field(compare=False)


@dataclass
class PromptPinnedWindow:
    """Rebuild live context so the prompt remains in the active context window."""

    pinned_prompt: list[int]
    max_seq_len: int
    tail_keep_tokens: int
    rebuild_every_tokens: int

    @classmethod
    def build(
        cls,
        prompt_tokens: list[int],
        max_seq_len: int,
        minimum_tail_tokens: int = 16,
    ) -> "PromptPinnedWindow":
        prompt = list(prompt_tokens)
        if len(prompt) > max_seq_len - minimum_tail_tokens:
            pinned_budget = max(minimum_tail_tokens, max_seq_len - minimum_tail_tokens)
            logger.warning(
                "Live prompt exceeds context budget; truncating pinned prompt from %d to %d tokens.",
                len(prompt),
                pinned_budget,
            )
            prompt = prompt[-pinned_budget:]

        tail_capacity = max(1, max_seq_len - len(prompt))
        rebuild_every = max(8, min(128, tail_capacity // 2 if tail_capacity > 1 else 1))
        tail_keep = max(0, tail_capacity - rebuild_every)
        return cls(
            pinned_prompt=prompt,
            max_seq_len=max_seq_len,
            tail_keep_tokens=tail_keep,
            rebuild_every_tokens=rebuild_every,
        )

    def build_prefill(self, continuation_tokens: list[int]) -> list[int]:
        if self.tail_keep_tokens <= 0:
            return list(self.pinned_prompt)
        return list(self.pinned_prompt) + list(continuation_tokens[-self.tail_keep_tokens :])


class IncrementalMidiTokenParser:
    """Convert generated tokens into scheduled MIDI note-on/off messages."""

    def __init__(
        self,
        tokenizer: BachTokenizer,
        *,
        key_root: int,
        key_mode: str,
        channel_base: int = 0,
        single_channel: bool = False,
        velocity: int = 80,
    ) -> None:
        self.tokenizer = tokenizer
        self.key_root = key_root
        self.key_mode = key_mode
        self.channel_base = int(channel_base)
        self.single_channel = bool(single_channel)
        self.velocity = max(1, min(127, int(velocity)))

        self.current_tick = 0
        self.current_voice = 1
        self.pending_pitch: int | None = None
        self.pending_octave: int | None = None
        self.pending_degree: int | None = None
        self.pending_accidental = ""

    def feed_tokens(self, tokens: list[int]) -> list[ScheduledMidiMessage]:
        messages: list[ScheduledMidiMessage] = []
        for token in tokens:
            messages.extend(self.feed_token(token))
        return messages

    def feed_token(self, token: int) -> list[ScheduledMidiMessage]:
        name = self.tokenizer.token_to_name.get(token, "")

        if not name or name in {
            "PAD",
            "BOS",
            "EOS",
            "SUBJECT_START",
            "SUBJECT_END",
            "CAD_PAC",
            "CAD_IAC",
            "CAD_HC",
            "CAD_DC",
            "BAR",
            "BEAT_1",
            "BEAT_2",
            "BEAT_3",
            "BEAT_4",
            "BEAT_5",
            "BEAT_6",
            "MODE_2PART",
            "MODE_3PART",
            "MODE_4PART",
            "MODE_FUGUE",
            "SHARP",
            "FLAT",
        } or name.startswith(
            (
                "STYLE_",
                "FORM_",
                "LENGTH_",
                "METER_",
                "TEXTURE_",
                "IMITATION_",
                "HARMONIC_RHYTHM_",
                "HARMONIC_TENSION_",
                "CHROMATICISM_",
                "ENCODE_",
            )
        ):
            if name == "SHARP":
                self.pending_accidental = "sharp"
            elif name == "FLAT":
                self.pending_accidental = "flat"
            return []

        if name == "VOICE_SEP":
            self.current_tick = 0
            self._clear_pending_pitch()
            return []

        if name.startswith("VOICE_"):
            self.current_voice = int(name[-1])
            self._clear_pending_pitch()
            return []

        if name.startswith("KEY_"):
            key_name = name[4:]
            parts = key_name.rsplit("_", 1)
            if len(parts) == 2:
                self.key_mode = parts[1]
                try:
                    self.key_root = note_name_to_pc(parts[0].replace("s", "#"))
                except ValueError:
                    pass
            return []

        if name.startswith("Pitch_"):
            self.pending_pitch = int(name[6:])
            self.pending_octave = None
            self.pending_degree = None
            self.pending_accidental = ""
            return []

        if name.startswith("OCT_"):
            self.pending_octave = int(name[4:])
            self.pending_degree = None
            self.pending_accidental = ""
            self.pending_pitch = None
            return []

        if name.startswith("DEG_"):
            self.pending_degree = int(name[4:])
            return []

        if name.startswith("Dur_"):
            duration = int(name[4:])
            pitch = self._resolve_pending_pitch()
            self._clear_pending_pitch()
            if pitch is None:
                return []
            channel = self.channel_base if self.single_channel else self.channel_base + self.current_voice - 1
            return [
                ScheduledMidiMessage(
                    tick=self.current_tick,
                    priority=1,
                    message=mido.Message("note_on", note=pitch, velocity=self.velocity, channel=channel, time=0),
                ),
                ScheduledMidiMessage(
                    tick=self.current_tick + duration,
                    priority=0,
                    message=mido.Message("note_off", note=pitch, velocity=0, channel=channel, time=0),
                ),
            ]

        if name.startswith("TimeShift_"):
            self.current_tick += int(name[10:])
            self._clear_pending_pitch()
            return []

        return []

    def _resolve_pending_pitch(self) -> int | None:
        if self.pending_pitch is not None:
            return self.pending_pitch
        if self.pending_octave is None or self.pending_degree is None:
            return None
        return scale_degree_to_midi(
            self.pending_octave,
            self.pending_degree,
            self.pending_accidental,
            self.key_root,
            self.key_mode,
        )

    def _clear_pending_pitch(self) -> None:
        self.pending_pitch = None
        self.pending_octave = None
        self.pending_degree = None
        self.pending_accidental = ""


@dataclass
class _LiveQueueState:
    heap: list[ScheduledMidiMessage] = field(default_factory=list)
    max_scheduled_tick: int = 0
    playback_tick: int = 0
    done: bool = False
    error: Exception | None = None

    def buffered_ahead_ticks(self) -> int:
        return max(0, self.max_scheduled_tick - self.playback_tick)


def list_midi_output_ports() -> list[str]:
    """Return available MIDI output port names."""
    try:
        return list(mido.get_output_names())
    except Exception:
        return []


def autodetect_midi_output_port(port_names: list[str]) -> str | None:
    """Pick the most likely MIDI output port for a connected synth."""
    if not port_names:
        return None
    if len(port_names) == 1:
        return port_names[0]

    def _score(port_name: str) -> tuple[int, int]:
        lowered = port_name.lower()
        score = 0
        for needle, points in _PREFERRED_PORT_MARKERS:
            if needle in lowered:
                score += points
        return score, -len(port_name)

    best = max(port_names, key=_score)
    return best


def autodetect_channel_base(_port_name: str | None = None) -> int:
    """Return the safest default MIDI channel base.

    Plain output MIDI does not reliably expose a hardware synth's configured
    receive channel, so the auto path uses MIDI channel 1 unless overridden.
    """
    return 0


def load_subject_midi_note_events(subject_midi_path: str | Path) -> list[tuple[int, int, int]]:
    """Load a monophonic-ish subject from the first non-empty MIDI track."""
    mid = load_midi(subject_midi_path)
    tracks = midi_to_note_events(mid)
    if not tracks:
        return []
    subject_notes = min((track for track in tracks if track), key=lambda track: (len(track), track[0][0]))
    if not subject_notes:
        return []
    min_start = min(start for start, _dur, _pitch in subject_notes)
    if min_start == 0:
        return subject_notes
    return [(start - min_start, dur, pitch) for start, dur, pitch in subject_notes]


def _build_live_constraints(
    tokenizer: BachTokenizer,
    *,
    key_root: int,
    key_mode: str,
    form: str,
    num_voices: int,
):
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer

    if isinstance(tokenizer, ScaleDegreeTokenizer):
        from bach_gen.generation.scale_degree_constraints import ScaleDegreeDecodingConstraints

        return ScaleDegreeDecodingConstraints(
            tokenizer=tokenizer,
            key_root=key_root,
            key_mode=key_mode,
            enforce_range=True,
            form=form,
            num_voices=num_voices,
        )
    return DecodingConstraints(
        tokenizer=tokenizer,
        key_root=key_root,
        key_mode=key_mode,
        enforce_key=True,
        enforce_range=True,
        form=form,
        num_voices=num_voices,
    )


def _ticks_to_seconds(ticks: int, tempo_bpm: float) -> float:
    return float(ticks) * 60.0 / (float(tempo_bpm) * float(TICKS_PER_QUARTER))


def _seconds_to_ticks(seconds: float, tempo_bpm: float) -> int:
    return max(0, int(round(seconds * float(tempo_bpm) * float(TICKS_PER_QUARTER) / 60.0)))


def _all_notes_off(port: mido.ports.BaseOutput, *, channel_base: int, single_channel: bool) -> None:
    channels = [channel_base] if single_channel else list(range(channel_base, min(channel_base + 4, 16)))
    for channel in channels:
        port.send(mido.Message("control_change", channel=channel, control=123, value=0, time=0))
        port.send(mido.Message("control_change", channel=channel, control=120, value=0, time=0))


def _pop_due_messages(shared: _LiveQueueState, current_tick: int) -> list[mido.Message]:
    due: list[mido.Message] = []
    while shared.heap and shared.heap[0].tick <= current_tick:
        due.append(heapq.heappop(shared.heap).message)
    return due


def run_live_generation(
    *,
    model: BachTransformer,
    tokenizer: BachTokenizer,
    midi_out: mido.ports.BaseOutput,
    key_str: str,
    subject_str: str | None = None,
    subject_midi: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K_SAMPLING,
    top_p: float = DEFAULT_TOP_P,
    min_p: float = DEFAULT_MIN_P,
    max_live_tokens: int | None = None,
    form: str = "fugue",
    num_voices: int | None = None,
    style: str = "bach",
    length: str | None = None,
    meter: str | None = None,
    texture: str | None = None,
    imitation: str | None = None,
    harmonic_rhythm: str | None = None,
    harmonic_tension: str | None = None,
    chromaticism: str | None = None,
    cadence_density: str | None = None,
    subject_remind_bars: int | None = None,
    tempo_bpm: float = 120.0,
    prebuffer_bars: int = 4,
    low_water_bars: int = 2,
    channel_base: int = 0,
    single_channel: bool = False,
    velocity: int = 80,
    panic_on_exit: bool = True,
    external_stop_event: threading.Event | None = None,
    status_callback: callable | None = None,
) -> None:
    """Stream endless generation to a live MIDI output port."""
    if subject_str and subject_midi:
        raise ValueError("Specify at most one of subject_str or subject_midi.")

    model.eval()
    device = next(model.parameters()).device
    use_rope = not getattr(model.config, "drope_trained", False)

    if num_voices is None:
        num_voices = FORM_DEFAULTS.get(form, (2, DEFAULT_MAX_GEN_LENGTH))[0]

    key_root, key_mode = parse_key(key_str)
    subject_note_events = None
    if subject_midi:
        subject_note_events = load_subject_midi_note_events(subject_midi)
    elif subject_str:
        subject_note_events = subject_string_to_note_events(subject_str)

    prompt_tokens = _build_prompt(
        tokenizer=tokenizer,
        key_root=key_root,
        key_mode=key_mode,
        key_name=get_key_signature_name(key_root, key_mode),
        subject_str=None if subject_note_events is not None else subject_str,
        subject_note_events=subject_note_events,
        form=form,
        style=style,
        num_voices=num_voices,
        length=length,
        meter=meter,
        texture=texture,
        imitation=imitation,
        harmonic_rhythm=harmonic_rhythm,
        harmonic_tension=harmonic_tension,
        chromaticism=chromaticism,
    )

    constraints = _build_live_constraints(
        tokenizer,
        key_root=key_root,
        key_mode=key_mode,
        form=form,
        num_voices=num_voices,
    )
    parser = IncrementalMidiTokenParser(
        tokenizer,
        key_root=key_root,
        key_mode=key_mode,
        channel_base=channel_base,
        single_channel=single_channel,
        velocity=velocity,
    )
    window = PromptPinnedWindow.build(prompt_tokens, model.config.max_seq_len)

    time_signature = {
        "2_4": (2, 4),
        "3_4": (3, 4),
        "4_4": (4, 4),
        "6_8": (6, 8),
        "3_8": (3, 8),
        "alla_breve": (2, 2),
    }.get(meter or "4_4", (4, 4))
    measure_ticks = ticks_per_measure(time_signature)
    high_water_ticks = max(measure_ticks, int(prebuffer_bars) * measure_ticks)
    low_water_ticks = max(measure_ticks // 2, int(low_water_bars) * measure_ticks)
    low_water_ticks = min(low_water_ticks, high_water_ticks)

    shared = _LiveQueueState()
    condition = threading.Condition()
    stop_event = external_stop_event or threading.Event()
    initial_messages = parser.feed_tokens(prompt_tokens)
    if initial_messages:
        with condition:
            for message in initial_messages:
                heapq.heappush(shared.heap, message)
                shared.max_scheduled_tick = max(shared.max_scheduled_tick, message.tick)

    min_subject_entries = 0
    subject_spacing_bars = 8
    if subject_remind_bars is not None and subject_remind_bars > 0:
        min_subject_entries = 1_000_000
        subject_spacing_bars = int(subject_remind_bars)

    def _mask_eos(logits: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[tokenizer.EOS] = float("-inf")
        return masked

    def _producer() -> None:
        continuation_tokens: list[int] = []
        state = constraints.initial_state(prompt_tokens)
        control_state = _build_structural_control_state(
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            cadence_density=cadence_density,
            min_subject_entries=min_subject_entries,
            subject_spacing_bars=subject_spacing_bars,
        )
        tokens_since_rebuild = 0

        @torch.no_grad()
        def _prefill() -> tuple[object, torch.Tensor]:
            prefill_tokens = window.build_prefill(continuation_tokens)
            prompt_ids = torch.tensor([prefill_tokens], dtype=torch.long, device=device)
            logits, kv_cache = model(prompt_ids, use_rope=use_rope, use_cache=True)
            return kv_cache, logits[0, -1, :]

        try:
            kv_cache, raw_next_logits = _prefill()

            while not stop_event.is_set():
                if max_live_tokens is not None and len(continuation_tokens) >= max_live_tokens:
                    break

                with condition:
                    while (
                        not stop_event.is_set()
                        and shared.buffered_ahead_ticks() >= high_water_ticks
                    ):
                        condition.wait(timeout=0.05)
                    if stop_event.is_set():
                        break

                constrained = constraints.apply(raw_next_logits, state)
                forced_tok = control_state.maybe_force_token(tokenizer) if control_state is not None else None
                if forced_tok is not None:
                    next_token = forced_tok
                else:
                    endless = max_live_tokens is None
                    sample_logits = _mask_eos(constrained) if endless else constrained
                    fallback_logits = _mask_eos(raw_next_logits) if endless else raw_next_logits
                    next_token = sample_next_token(
                        sample_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        fallback_logits=fallback_logits,
                    )

                continuation_tokens.append(next_token)
                state = constraints.update_state(state, next_token)
                if control_state is not None:
                    control_state.update(next_token, tokenizer)

                emitted = parser.feed_token(next_token)
                with condition:
                    for message in emitted:
                        heapq.heappush(shared.heap, message)
                        shared.max_scheduled_tick = max(shared.max_scheduled_tick, message.tick)
                    condition.notify_all()

                if next_token == tokenizer.EOS and max_live_tokens is not None:
                    break

                tokens_since_rebuild += 1
                if tokens_since_rebuild >= window.rebuild_every_tokens:
                    del kv_cache
                    kv_cache, raw_next_logits = _prefill()
                    tokens_since_rebuild = 0
                else:
                    step_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    logits, kv_cache = model(
                        step_ids,
                        use_rope=use_rope,
                        use_cache=True,
                        kv_cache=kv_cache,
                    )
                    raw_next_logits = logits[0, -1, :]
        except Exception as exc:  # pragma: no cover - exercised via CLI behavior
            logger.exception("Live generation failed.")
            with condition:
                shared.error = exc
                shared.done = True
                condition.notify_all()
            return

        with condition:
            shared.done = True
            condition.notify_all()

    producer = threading.Thread(target=_producer, name="bach-gen-live-producer", daemon=True)
    producer.start()

    try:
        if status_callback is not None:
            status_callback("Buffering live stream...")
        with condition:
            while (
                not shared.done
                and shared.error is None
                and shared.max_scheduled_tick < high_water_ticks
            ):
                condition.wait(timeout=0.05)
        if shared.error is not None:
            raise shared.error

        playback_start = time.monotonic() + 0.05
        if status_callback is not None:
            status_callback("Playing")
        while True:
            now = time.monotonic()
            current_tick = 0 if now < playback_start else _seconds_to_ticks(now - playback_start, tempo_bpm)
            with condition:
                shared.playback_tick = current_tick
                due = _pop_due_messages(shared, current_tick)
                done = shared.done and not shared.heap
                next_tick = shared.heap[0].tick if shared.heap else None
                condition.notify_all()
            for message in due:
                midi_out.send(message)
            if done:
                break

            if next_tick is None:
                time.sleep(0.01)
                continue

            ticks_until_next = max(0, next_tick - current_tick)
            sleep_for = min(0.01, max(0.001, _ticks_to_seconds(ticks_until_next, tempo_bpm)))
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        logger.info("Stopping live generation.")
    finally:
        stop_event.set()
        with condition:
            condition.notify_all()
        producer.join(timeout=1.0)
        if status_callback is not None:
            status_callback("Stopped")
        if panic_on_exit:
            _all_notes_off(
                midi_out,
                channel_base=channel_base,
                single_channel=single_channel,
            )
