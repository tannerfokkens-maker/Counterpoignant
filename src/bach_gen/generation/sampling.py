"""Sampling strategies: temperature + min-p (with optional top-k/top-p)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
) -> int:
    """Sample the next token from logits using temperature + min-p.

    Args:
        logits: (vocab_size,) raw logits for next token.
        temperature: Sampling temperature (higher = more random).
        top_k: Keep only top-k tokens (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        min_p: Keep tokens with prob >= min_p * max_prob (0 = disabled).

    Returns:
        Sampled token ID.
    """
    # Temperature scaling
    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy
        return logits.argmax().item()

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

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)

    return token.item()
