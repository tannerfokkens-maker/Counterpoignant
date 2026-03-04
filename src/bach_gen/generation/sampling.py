"""Sampling strategies: temperature + min-p (with optional top-k/top-p)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_probs_from_logits(logits: torch.Tensor) -> torch.Tensor | None:
    """Return a normalized probability vector, or None if invalid."""
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = probs.sum()

    if not torch.isfinite(total).item():
        return None
    if total.item() <= 0.0:
        return None

    return probs / total


def _safe_argmax(logits: torch.Tensor) -> int:
    """Argmax that tolerates NaN/Inf; falls back to token 0 if needed."""
    safe_logits = torch.nan_to_num(
        logits, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"),
    )
    if not torch.any(torch.isfinite(safe_logits)).item():
        return 0
    return int(torch.argmax(safe_logits).item())


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.15,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.015,
    fallback_logits: torch.Tensor | None = None,
) -> int:
    """Sample the next token from logits using temperature + min-p.

    Args:
        logits: (vocab_size,) raw logits for next token.
        temperature: Sampling temperature (higher = more random).
        top_k: Keep only top-k tokens (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        min_p: Keep tokens with prob >= min_p * max_prob (0 = disabled).
        fallback_logits: Optional pre-constraint logits for safety fallback.

    Returns:
        Sampled token ID.
    """
    logits = logits.clone()

    # Temperature scaling
    if temperature > 0:
        logits = logits / temperature
        pre_filter_logits = logits.clone()
        scaled_fallback = (fallback_logits / temperature) if fallback_logits is not None else None
    else:
        # Greedy
        greedy_source = logits
        if not torch.any(torch.isfinite(greedy_source)).item() and fallback_logits is not None:
            greedy_source = fallback_logits
        return _safe_argmax(greedy_source)

    # Top-k filtering (optional)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering (optional)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # Min-p filtering (recommended):
    # keep tokens with p_i >= min_p * max_j p_j
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs)
        threshold = min_p * max_prob
        keep_mask = probs >= threshold
        # Ensure at least one valid token remains
        if torch.any(keep_mask):
            logits = logits.masked_fill(~keep_mask, float("-inf"))

    # Sample from distribution, with safety fallback when all probability mass
    # was filtered out or became invalid.
    probs = _safe_probs_from_logits(logits)
    if probs is None:
        fallback_source = scaled_fallback if scaled_fallback is not None else pre_filter_logits
        probs = _safe_probs_from_logits(fallback_source)
        if probs is None:
            return _safe_argmax(fallback_source)

    token = torch.multinomial(probs, num_samples=1)

    return token.item()
