"""Sampling strategies: temperature, top-k, top-p (nucleus)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
) -> int:
    """Sample the next token from logits using temperature + top-k + top-p.

    Args:
        logits: (vocab_size,) raw logits for next token.
        temperature: Sampling temperature (higher = more random).
        top_k: Keep only top-k tokens (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).

    Returns:
        Sampled token ID.
    """
    # Temperature scaling
    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy
        return logits.argmax().item()

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
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

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)

    return token.item()
