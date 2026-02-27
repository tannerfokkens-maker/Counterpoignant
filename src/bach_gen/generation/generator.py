"""Generation orchestrator: generate N candidates, score, return top K."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from bach_gen.data.tokenizer import BachTokenizer
from bach_gen.data.extraction import VoicePair, VoiceComposition
from bach_gen.evaluation.scorer import ScoreBreakdown, score_composition
from bach_gen.generation.sampling import sample_next_token
from bach_gen.generation.constraints import DecodingConstraints
from bach_gen.generation.subject import (
    parse_subject_string, generate_subject,
    parse_subject_string_sd, generate_subject_sd,
)
from bach_gen.model.architecture import BachTransformer
from bach_gen.model.trainer import get_device
from bach_gen.utils.constants import (
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_TOP_K_RESULTS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K_SAMPLING,
    DEFAULT_TOP_P,
    DEFAULT_MAX_GEN_LENGTH,
    FORM_DEFAULTS,
)
from bach_gen.utils.music_theory import parse_key, get_key_signature_name
from bach_gen.utils.midi_io import note_events_to_midi, save_midi

logger = logging.getLogger(__name__)


@dataclass
class BeamHypothesis:
    """A single beam search hypothesis."""

    tokens: list[int]
    log_prob: float = 0.0
    is_finished: bool = False
    constraint_state: object | None = None

    def normalized_score(self, alpha: float = 0.7) -> float:
        """Wu et al. length normalization: lp = (5 + len)^alpha / 6^alpha."""
        length_penalty = ((5 + len(self.tokens)) ** alpha) / (6 ** alpha)
        return self.log_prob / length_penalty


@dataclass
class GenerationResult:
    """Result of a single generation."""

    composition: VoiceComposition
    tokens: list[int]
    score: ScoreBreakdown
    midi_path: str | None = None

    @property
    def pair(self) -> VoicePair:
        """Backward-compatible access as VoicePair."""
        return self.composition.to_voice_pair()


def generate(
    model: BachTransformer,
    tokenizer: BachTokenizer,
    key_str: str,
    subject_str: str | None = None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    top_k_results: int = DEFAULT_TOP_K_RESULTS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K_SAMPLING,
    top_p: float = DEFAULT_TOP_P,
    max_length: int = DEFAULT_MAX_GEN_LENGTH,
    output_dir: str | Path = "output",
    enforce_key: bool = True,
    enforce_range: bool = True,
    form: str = "2-part",
    num_voices: int | None = None,
    progress_callback: callable | None = None,
    beam_width: int | None = None,
    length_penalty_alpha: float = 0.7,
    style: str = "bach",
    length: str | None = None,
    meter: str | None = None,
    texture: str | None = None,
    imitation: str | None = None,
    harmonic_rhythm: str | None = None,
    harmonic_tension: str | None = None,
    chromaticism: str | None = None,
) -> list[GenerationResult]:
    """Generate Bach-style compositions and return top results.

    Args:
        model: Trained BachTransformer.
        tokenizer: BachTokenizer.
        key_str: Key string (e.g., "C minor").
        subject_str: Optional subject (e.g., "C4 D4 Eb4 F4").
        num_candidates: How many candidates to generate (sampling mode).
        top_k_results: How many best results to return.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        max_length: Maximum generation length in tokens.
        output_dir: Directory to save MIDI files.
        enforce_key: Apply key constraints during generation.
        enforce_range: Apply voice range constraints.
        form: Composition form ("2-part", "sinfonia", "chorale", "fugue").
        num_voices: Override number of voices (defaults from form).
        progress_callback: Callback(current, total) for progress updates.
        beam_width: If set, use beam search instead of sampling.
        length_penalty_alpha: Length normalization exponent for beam search.
        style: Style conditioning token name (e.g. "bach", "baroque").
        length: Length conditioning (short/medium/long/extended). Default: infer from form.
        meter: Meter conditioning (e.g. "4_4", "3_4"). Default: METER_4_4.
        texture: Texture conditioning (homophonic/polyphonic/mixed).
        imitation: Imitation conditioning (none/low/high).
        harmonic_rhythm: Harmonic rhythm conditioning (slow/moderate/fast).
        harmonic_tension: Harmonic tension conditioning (low/moderate/high).
        chromaticism: Chromaticism conditioning (low/moderate/high).

    Returns:
        List of top GenerationResult, sorted by score.
    """
    device = next(model.parameters()).device

    # Resolve num_voices from form
    if num_voices is None:
        num_voices = FORM_DEFAULTS.get(form, (2, 768))[0]

    # Parse key
    key_root, key_mode = parse_key(key_str)
    key_name = get_key_signature_name(key_root, key_mode)

    # Build prompt
    prompt_tokens = _build_prompt(
        tokenizer, key_root, key_mode, key_name, subject_str, form, style,
        length=length, meter=meter, texture=texture, imitation=imitation,
        harmonic_rhythm=harmonic_rhythm, harmonic_tension=harmonic_tension,
        chromaticism=chromaticism,
    )

    # Constraints — dispatch based on tokenizer type
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
    if isinstance(tokenizer, ScaleDegreeTokenizer):
        from bach_gen.generation.scale_degree_constraints import ScaleDegreeDecodingConstraints
        constraints = ScaleDegreeDecodingConstraints(
            tokenizer=tokenizer,
            key_root=key_root,
            key_mode=key_mode,
            enforce_range=enforce_range,
            form=form,
            num_voices=num_voices,
        )
    else:
        constraints = DecodingConstraints(
            tokenizer=tokenizer,
            key_root=key_root,
            key_mode=key_mode,
            enforce_key=enforce_key,
            enforce_range=enforce_range,
            form=form,
            num_voices=num_voices,
        )

    # Generate candidates
    candidates: list[GenerationResult] = []

    # Form label for filenames
    form_label = form.replace("-", "")

    if beam_width is not None and beam_width > 1:
        # --- Beam search mode ---
        logger.info(f"Using beam search (width={beam_width}, alpha={length_penalty_alpha})")
        beam_sequences = _beam_search_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            constraints=constraints,
            beam_width=beam_width,
            max_length=max_length,
            device=device,
            length_penalty_alpha=length_penalty_alpha,
        )

        for i, tokens in enumerate(beam_sequences):
            comp = tokenizer.decode(tokens)
            comp.key_root = key_root
            comp.key_mode = key_mode
            comp.source = f"beam_{i+1}"

            non_empty = sum(1 for v in comp.voices if v)
            if non_empty < 2:
                continue

            score = score_composition(comp, token_sequence=tokens, model=model, tokenizer=tokenizer)
            candidates.append(GenerationResult(
                composition=comp,
                tokens=tokens,
                score=score,
            ))

            if progress_callback:
                progress_callback(i + 1, len(beam_sequences))
    else:
        # --- Sampling mode ---
        logger.info(f"Using sampling (candidates={num_candidates})")
        for i in range(num_candidates):
            tokens = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                constraints=constraints,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
                device=device,
            )

            # Decode to composition
            comp = tokenizer.decode(tokens)
            comp.key_root = key_root
            comp.key_mode = key_mode
            comp.source = f"generated_{i+1}"

            # Skip empty generations — need at least 2 non-empty voices
            non_empty = sum(1 for v in comp.voices if v)
            if non_empty < 2:
                continue

            # Score
            score = score_composition(comp, token_sequence=tokens, model=model, tokenizer=tokenizer)

            candidates.append(GenerationResult(
                composition=comp,
                tokens=tokens,
                score=score,
            ))

            if progress_callback:
                progress_callback(i + 1, num_candidates)

    # Sort by composite score (descending)
    candidates.sort(key=lambda r: r.score.composite, reverse=True)

    # Take top K
    top_results = candidates[:top_k_results]

    # Save MIDI files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(top_results):
        filename = f"{form_label}_{key_name}_{i+1}.mid"
        midi_path = output_path / filename
        mid = note_events_to_midi(voices=result.composition.voices)
        save_midi(mid, midi_path)
        result.midi_path = str(midi_path)
        logger.info(f"Saved {midi_path} (score: {result.score.composite:.3f})")

    return top_results


def _build_prompt(
    tokenizer: BachTokenizer,
    key_root: int,
    key_mode: str,
    key_name: str,
    subject_str: str | None,
    form: str = "2-part",
    style: str = "bach",
    length: str | None = None,
    meter: str | None = None,
    texture: str | None = None,
    imitation: str | None = None,
    harmonic_rhythm: str | None = None,
    harmonic_tension: str | None = None,
    chromaticism: str | None = None,
    encoding_mode: str | None = None,
) -> list[int]:
    """Build the prompt token sequence.

    Prefix order: BOS STYLE FORM MODE LENGTH METER TEXTURE IMITATION
                  HARMONIC_RHYTHM ENCODE_* KEY [SUBJECT]
    """
    tokens = [tokenizer.BOS]

    # Style conditioning token (right after BOS)
    if hasattr(tokenizer, "STYLE_TO_TOKEN") and style in tokenizer.STYLE_TO_TOKEN:
        tokens.append(tokenizer.STYLE_TO_TOKEN[style])

    # Form token (what kind of piece)
    if hasattr(tokenizer, "FORM_TO_FORM_TOKEN") and form in tokenizer.FORM_TO_FORM_TOKEN:
        tokens.append(tokenizer.FORM_TO_FORM_TOKEN[form])

    # Voice-count token (how many voices)
    if form in tokenizer.FORM_TO_MODE_TOKEN:
        tokens.append(tokenizer.FORM_TO_MODE_TOKEN[form])

    # Length conditioning token
    # Default: infer from form if not specified
    if length is None:
        _form_to_default_length = {
            "chorale": "short", "invention": "medium", "sinfonia": "medium",
            "trio_sonata": "medium", "2-part": "medium",
            "fugue": "long", "quartet": "long", "sonata": "long", "motet": "long",
        }
        length = _form_to_default_length.get(form)
    if length and hasattr(tokenizer, "LENGTH_TO_TOKEN") and length in tokenizer.LENGTH_TO_TOKEN:
        tokens.append(tokenizer.LENGTH_TO_TOKEN[length])

    # Meter conditioning token
    # Default: 4/4 if not specified
    if meter is None:
        meter = "4_4"
    if meter and hasattr(tokenizer, "METER_TO_TOKEN") and meter in tokenizer.METER_TO_TOKEN:
        tokens.append(tokenizer.METER_TO_TOKEN[meter])

    # Texture conditioning token
    if texture and hasattr(tokenizer, "TEXTURE_TO_TOKEN") and texture in tokenizer.TEXTURE_TO_TOKEN:
        tokens.append(tokenizer.TEXTURE_TO_TOKEN[texture])

    # Imitation conditioning token
    if imitation and hasattr(tokenizer, "IMITATION_TO_TOKEN") and imitation in tokenizer.IMITATION_TO_TOKEN:
        tokens.append(tokenizer.IMITATION_TO_TOKEN[imitation])

    # Harmonic rhythm conditioning token
    if harmonic_rhythm and hasattr(tokenizer, "HARMONIC_RHYTHM_TO_TOKEN") and harmonic_rhythm in tokenizer.HARMONIC_RHYTHM_TO_TOKEN:
        tokens.append(tokenizer.HARMONIC_RHYTHM_TO_TOKEN[harmonic_rhythm])

    # Harmonic tension conditioning token
    if harmonic_tension and hasattr(tokenizer, "HARMONIC_TENSION_TO_TOKEN") and harmonic_tension in tokenizer.HARMONIC_TENSION_TO_TOKEN:
        tokens.append(tokenizer.HARMONIC_TENSION_TO_TOKEN[harmonic_tension])

    # Chromaticism conditioning token
    if chromaticism and hasattr(tokenizer, "CHROMATICISM_TO_TOKEN") and chromaticism in tokenizer.CHROMATICISM_TO_TOKEN:
        tokens.append(tokenizer.CHROMATICISM_TO_TOKEN[chromaticism])

    # Encoding mode token
    if encoding_mode is None:
        encoding_mode = "interleaved"
    if hasattr(tokenizer, "ENCODING_MODE_TO_TOKEN") and encoding_mode in tokenizer.ENCODING_MODE_TO_TOKEN:
        tokens.append(tokenizer.ENCODING_MODE_TO_TOKEN[encoding_mode])

    # Key token
    key_token_name = f"KEY_{key_name}"
    if key_token_name in tokenizer.name_to_token:
        tokens.append(tokenizer.name_to_token[key_token_name])

    # Subject — dispatch based on tokenizer type
    # Skip subject in sequential mode (subjects are interleaved-mode only)
    if encoding_mode == "sequential":
        return tokens

    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
    if isinstance(tokenizer, ScaleDegreeTokenizer):
        if subject_str:
            subject_tokens = parse_subject_string_sd(
                subject_str, tokenizer, key_root, key_mode,
            )
        else:
            subject_tokens = generate_subject_sd(key_root, key_mode, tokenizer)
    else:
        if subject_str:
            subject_tokens = parse_subject_string(subject_str, tokenizer)
        else:
            subject_tokens = generate_subject(key_root, key_mode, tokenizer)

    tokens.extend(subject_tokens)
    return tokens


def generate_voice_by_voice(
    model: BachTransformer,
    tokenizer: BachTokenizer,
    key_str: str,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    top_k_results: int = DEFAULT_TOP_K_RESULTS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K_SAMPLING,
    top_p: float = DEFAULT_TOP_P,
    max_length: int = DEFAULT_MAX_GEN_LENGTH,
    output_dir: str | Path = "output",
    enforce_range: bool = True,
    form: str = "2-part",
    num_voices: int | None = None,
    progress_callback: callable | None = None,
    style: str = "bach",
    length: str | None = None,
    meter: str | None = None,
    texture: str | None = None,
    imitation: str | None = None,
    harmonic_rhythm: str | None = None,
    harmonic_tension: str | None = None,
    chromaticism: str | None = None,
    provided_voice_midi: str | None = None,
) -> list[GenerationResult]:
    """Generate compositions voice-by-voice using sequential encoding.

    1. Builds prompt with encoding_mode="sequential" + conditioning + KEY
    2. If provided_voice_midi given: loads MIDI, encodes voice 1 notes, appends VOICE_SEP
    3. Generates remaining voices autoregressively (model emits VOICE_SEP between voices)
    4. Decodes full sequence, scores, returns top K
    """
    device = next(model.parameters()).device

    if num_voices is None:
        num_voices = FORM_DEFAULTS.get(form, (2, 768))[0]

    key_root, key_mode = parse_key(key_str)
    key_name = get_key_signature_name(key_root, key_mode)

    # Build prompt with sequential encoding mode
    prompt_tokens = _build_prompt(
        tokenizer, key_root, key_mode, key_name,
        subject_str=None, form=form, style=style,
        length=length, meter=meter, texture=texture, imitation=imitation,
        harmonic_rhythm=harmonic_rhythm, harmonic_tension=harmonic_tension,
        chromaticism=chromaticism,
        encoding_mode="sequential",
    )

    # If a MIDI file is provided for voice 1, encode it and append
    if provided_voice_midi:
        from bach_gen.utils.midi_io import load_midi, midi_to_note_events
        mid = load_midi(provided_voice_midi)
        tracks = midi_to_note_events(mid)
        if tracks:
            voice1_notes = tracks[0]  # Use first track as voice 1
            prompt_tokens.append(tokenizer.VOICE_1)

            from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
            time_sig = (4, 4)  # Default; could be detected from MIDI
            if isinstance(tokenizer, ScaleDegreeTokenizer):
                tokenizer._encode_key_root = key_root
                tokenizer._encode_key_mode = key_mode
            voice_tokens = tokenizer._serialize_single_voice(voice1_notes, time_sig)
            prompt_tokens.extend(voice_tokens)
            prompt_tokens.append(tokenizer.VOICE_SEP)

    # Constraints
    from bach_gen.data.scale_degree_tokenizer import ScaleDegreeTokenizer
    if isinstance(tokenizer, ScaleDegreeTokenizer):
        from bach_gen.generation.scale_degree_constraints import ScaleDegreeDecodingConstraints
        constraints = ScaleDegreeDecodingConstraints(
            tokenizer=tokenizer,
            key_root=key_root,
            key_mode=key_mode,
            enforce_range=enforce_range,
            form=form,
            num_voices=num_voices,
        )
    else:
        constraints = DecodingConstraints(
            tokenizer=tokenizer,
            key_root=key_root,
            key_mode=key_mode,
            enforce_key=True,
            enforce_range=enforce_range,
            form=form,
            num_voices=num_voices,
        )

    candidates: list[GenerationResult] = []
    form_label = form.replace("-", "")

    logger.info(f"Voice-by-voice generation (candidates={num_candidates})")
    for i in range(num_candidates):
        tokens = _generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            constraints=constraints,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            device=device,
        )

        comp = tokenizer.decode(tokens)
        comp.key_root = key_root
        comp.key_mode = key_mode
        comp.source = f"vbv_generated_{i+1}"

        non_empty = sum(1 for v in comp.voices if v)
        if non_empty < 2:
            continue

        score = score_composition(comp, token_sequence=tokens, model=model, tokenizer=tokenizer)
        candidates.append(GenerationResult(
            composition=comp,
            tokens=tokens,
            score=score,
        ))

        if progress_callback:
            progress_callback(i + 1, num_candidates)

    candidates.sort(key=lambda r: r.score.composite, reverse=True)
    top_results = candidates[:top_k_results]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(top_results):
        filename = f"{form_label}_{key_name}_vbv_{i+1}.mid"
        midi_path = output_path / filename
        mid = note_events_to_midi(voices=result.composition.voices)
        save_midi(mid, midi_path)
        result.midi_path = str(midi_path)
        logger.info(f"Saved {midi_path} (score: {result.score.composite:.3f})")

    return top_results


@torch.no_grad()
def _generate_one(
    model: BachTransformer,
    tokenizer: BachTokenizer,
    prompt: list[int],
    constraints: DecodingConstraints,
    temperature: float,
    top_k: int,
    top_p: float,
    max_length: int,
    device: torch.device,
) -> list[int]:
    """Generate a single token sequence."""
    model.eval()
    tokens = list(prompt)

    # Initialise constraint state from prompt
    state = constraints.initial_state(tokens)

    for _ in range(max_length):
        # Truncate input to max sequence length
        input_tokens = tokens[-model.config.max_seq_len:]
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        logits = model(input_ids)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply constraints using cached state
        next_logits = constraints.apply(next_logits, state)

        # Sample
        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        tokens.append(next_token)
        state = constraints.update_state(state, next_token)

        # Stop on EOS
        if next_token == tokenizer.EOS:
            break

    return tokens


@torch.no_grad()
def _beam_search_generate(
    model: BachTransformer,
    tokenizer: BachTokenizer,
    prompt: list[int],
    constraints: DecodingConstraints,
    beam_width: int,
    max_length: int,
    device: torch.device,
    length_penalty_alpha: float = 0.7,
) -> list[list[int]]:
    """Generate token sequences using beam search.

    Maintains ``beam_width`` active hypotheses and expands them in parallel.
    Finished beams (those producing EOS) are collected separately. Stops when
    ``beam_width`` finished beams exist and the best active beam cannot beat
    the worst finished beam, or ``max_length`` is reached.

    Args:
        model: Trained BachTransformer.
        tokenizer: BachTokenizer.
        prompt: Prompt token sequence.
        constraints: Decoding constraints.
        beam_width: Number of beams to maintain.
        max_length: Maximum generation length per beam.
        device: Torch device.
        length_penalty_alpha: Length normalization exponent.

    Returns:
        List of token sequences sorted by normalized score (best first).
    """
    model.eval()
    max_seq_len = model.config.max_seq_len

    # Initialise constraint state from prompt
    initial_constraint_state = constraints.initial_state(prompt)

    # Initialise with a single beam containing the prompt
    active: list[BeamHypothesis] = [
        BeamHypothesis(
            tokens=list(prompt),
            log_prob=0.0,
            constraint_state=initial_constraint_state,
        ),
    ]
    finished: list[BeamHypothesis] = []

    expand_k = 2 * beam_width  # candidates per beam

    for _step in range(max_length):
        if not active:
            break

        # --- Batched forward pass ---
        # Truncate each beam's tokens to model context window
        batch_tokens = [h.tokens[-max_seq_len:] for h in active]
        max_len_in_batch = max(len(t) for t in batch_tokens)

        # Pad to equal length (left-pad with PAD token)
        padded = []
        actual_lengths = []
        for t in batch_tokens:
            pad_len = max_len_in_batch - len(t)
            padded.append([tokenizer.PAD] * pad_len + t)
            actual_lengths.append(len(t))

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Build attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != tokenizer.PAD).long()

        logits = model(input_ids, attention_mask=attention_mask)
        # logits: (batch, seq_len, vocab_size)

        # --- Expand candidates ---
        all_candidates: list[BeamHypothesis] = []

        for beam_idx, hyp in enumerate(active):
            # Read logits from the last *real* position of this beam
            last_pos = max_len_in_batch - 1  # after left-padding, last real is always rightmost
            beam_logits = logits[beam_idx, last_pos, :]  # (vocab_size,)

            # Apply constraints per-beam using cached state
            beam_logits = constraints.apply(beam_logits, hyp.constraint_state)

            # Log-softmax for scoring
            log_probs = F.log_softmax(beam_logits, dim=-1)

            # Top-k expansion per beam
            topk_log_probs, topk_indices = torch.topk(log_probs, min(expand_k, log_probs.size(-1)))

            for j in range(topk_indices.size(0)):
                tok = topk_indices[j].item()
                tok_log_prob = topk_log_probs[j].item()

                new_tokens = hyp.tokens + [tok]
                new_log_prob = hyp.log_prob + tok_log_prob
                new_constraint_state = constraints.update_state(
                    hyp.constraint_state, tok,
                )

                new_hyp = BeamHypothesis(
                    tokens=new_tokens,
                    log_prob=new_log_prob,
                    is_finished=(tok == tokenizer.EOS),
                    constraint_state=new_constraint_state,
                )
                all_candidates.append(new_hyp)

        # --- Separate finished from active ---
        new_active: list[BeamHypothesis] = []
        for cand in all_candidates:
            if cand.is_finished:
                finished.append(cand)
            else:
                new_active.append(cand)

        # --- Prune active to beam_width by normalized score ---
        new_active.sort(key=lambda h: h.normalized_score(length_penalty_alpha), reverse=True)
        active = new_active[:beam_width]

        # --- Early stopping ---
        if len(finished) >= beam_width and active:
            worst_finished = min(
                f.normalized_score(length_penalty_alpha) for f in finished
            )
            best_active = active[0].normalized_score(length_penalty_alpha)
            if best_active <= worst_finished:
                break

    # --- Collect results ---
    # If no beam finished, use best active beams as fallback
    if not finished:
        finished = active

    finished.sort(key=lambda h: h.normalized_score(length_penalty_alpha), reverse=True)
    return [h.tokens for h in finished]
