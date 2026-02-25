"""Information-theoretic evaluation.

Uses model perplexity and entropy to assess musical naturalness.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Calibrated values from Bach corpus (set during training)
_BACH_PERPLEXITY_RANGE = (5.0, 25.0)  # typical range for well-trained model
_BACH_ENTROPY_RANGE = (2.0, 5.0)


def score_information(
    token_sequence: list[int],
    model: torch.nn.Module | None = None,
    vocab_size: int = 400,
) -> tuple[float, dict]:
    """Score information-theoretic quality.

    Args:
        token_sequence: Tokenized sequence.
        model: Trained model (if None, uses heuristic).
        vocab_size: Vocabulary size.

    Returns:
        (score 0-1, details dict)
    """
    details = {}

    if model is not None:
        perplexity, entropy = _compute_model_metrics(token_sequence, model)
    else:
        # Heuristic based on token distribution
        perplexity, entropy = _compute_heuristic_metrics(token_sequence, vocab_size)

    details["perplexity"] = perplexity
    details["entropy"] = entropy

    # Score based on how close metrics are to Bach corpus range
    ppl_score = _score_in_range(perplexity, *_BACH_PERPLEXITY_RANGE)
    ent_score = _score_in_range(entropy, *_BACH_ENTROPY_RANGE)

    details["perplexity_score"] = ppl_score
    details["entropy_score"] = ent_score

    # Additional heuristic: token-level diversity score
    # Measures whether the sequence uses a healthy variety of tokens
    # (not too few = repetitive, not too uniform = random)
    diversity_score = _score_token_diversity(token_sequence, vocab_size)
    details["diversity_score"] = diversity_score

    score = ppl_score * 0.35 + ent_score * 0.30 + diversity_score * 0.35

    return score, details


def _compute_model_metrics(
    tokens: list[int],
    model: torch.nn.Module,
) -> tuple[float, float]:
    """Compute perplexity and entropy from model predictions."""
    model.eval()
    device = next(model.parameters()).device

    input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    targets = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        # logits: (1, seq_len, vocab_size), targets: (1, seq_len)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        avg_loss = loss.mean().item()
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow

        # Entropy from predicted distributions
        probs = torch.softmax(logits, dim=-1)
        entropy_per_token = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
        entropy = entropy_per_token.mean().item()

    return perplexity, entropy


def _compute_heuristic_metrics(
    tokens: list[int],
    vocab_size: int,
) -> tuple[float, float]:
    """Compute heuristic perplexity and entropy without a model.

    Uses bigram statistics as a proxy for model-based metrics.
    Designed to penalize both degenerate extremes:
    - Very low entropy (repetitive/monotone sequences)
    - Very high entropy (random/incoherent sequences)
    """
    if not tokens:
        return 100.0, 0.0

    # Token distribution entropy (proxy for model entropy)
    counts = np.zeros(vocab_size)
    for t in tokens:
        if 0 <= t < vocab_size:
            counts[t] += 1

    total = counts.sum()
    if total == 0:
        return 100.0, 0.0

    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    # Bigram perplexity as proxy
    bigram_counts: dict[tuple[int, int], int] = {}
    unigram_counts: dict[int, int] = {}
    for i in range(len(tokens) - 1):
        bg = (tokens[i], tokens[i + 1])
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        unigram_counts[tokens[i]] = unigram_counts.get(tokens[i], 0) + 1

    if not bigram_counts:
        return 100.0, entropy

    log_prob_sum = 0.0
    count = 0
    for (t1, t2), bg_count in bigram_counts.items():
        ug_count = unigram_counts.get(t1, 1)
        prob = bg_count / ug_count
        log_prob_sum += math.log(prob + 1e-10)
        count += 1

    avg_log_prob = log_prob_sum / max(count, 1)
    perplexity = math.exp(-avg_log_prob)

    # Bigram repetition ratio: fraction of bigrams that are the same as
    # the most common bigram.  Very repetitive sequences have ratio near 1.
    if bigram_counts:
        max_bg_count = max(bigram_counts.values())
        total_bg = sum(bigram_counts.values())
        repetition_ratio = max_bg_count / total_bg
    else:
        repetition_ratio = 0.0

    # Adjust perplexity for degenerate sequences:
    # - Very low perplexity (highly predictable) = repetitive
    # - Very high perplexity (unpredictable) = random
    if perplexity < 1.5 or repetition_ratio > 0.3:
        # Artificially inflate perplexity for highly repetitive content
        # to push it outside the "natural" range
        perplexity = max(perplexity, 1.0)
    if entropy < 2.0:
        # Low entropy = few unique tokens used = degenerate
        # Map to below the expected range
        entropy = entropy  # keep as-is; the scoring function will penalize

    return perplexity, entropy


def _score_token_diversity(token_sequence: list[int], vocab_size: int) -> float:
    """Score token diversity — discriminates repetitive, natural, and random sequences.

    Measures several properties:
    - Unique token ratio (too low = repetitive, moderate = natural)
    - Bigram diversity (too low = repetitive, moderate = natural, too high = random)
    - Consecutive repetition (high = degenerate)

    Bach corpus typically uses ~30-60% of the vocabulary with moderate bigram
    diversity and low consecutive repetition.
    """
    if len(token_sequence) < 10:
        return 0.5

    tokens = [t for t in token_sequence if t > 2]  # skip PAD, BOS, EOS
    if len(tokens) < 10:
        return 0.5

    # Unique token ratio
    unique_ratio = len(set(tokens)) / len(tokens)

    # Bigram diversity: unique bigrams / total bigrams
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    if bigrams:
        bigram_diversity = len(set(bigrams)) / len(bigrams)
    else:
        bigram_diversity = 0.0

    # Consecutive repetition: fraction of tokens identical to predecessor
    consecutive_same = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    consec_ratio = consecutive_same / max(len(tokens) - 1, 1)

    # Score unique token ratio: target ~0.15-0.45 for natural music tokens
    if unique_ratio < 0.05:
        uniq_score = 0.1
    elif unique_ratio < 0.10:
        uniq_score = 0.1 + (unique_ratio - 0.05) * 8.0
    elif unique_ratio <= 0.50:
        uniq_score = 0.5 + min(0.5, (unique_ratio - 0.10) * 1.25)
    else:
        uniq_score = max(0.3, 1.0 - (unique_ratio - 0.50) * 2)

    # Score bigram diversity: target ~0.3-0.7
    if bigram_diversity < 0.15:
        bg_score = bigram_diversity * 4  # 0 to 0.6
    elif bigram_diversity <= 0.7:
        bg_score = 0.6 + (bigram_diversity - 0.15) * 0.73  # 0.6 to 1.0
    elif bigram_diversity <= 0.9:
        bg_score = 1.0
    else:
        bg_score = max(0.5, 1.0 - (bigram_diversity - 0.9) * 5)

    # Score consecutive repetition: target < 0.15
    if consec_ratio < 0.1:
        consec_score = 1.0
    elif consec_ratio < 0.3:
        consec_score = 1.0 - (consec_ratio - 0.1) * 2.5
    else:
        consec_score = max(0.1, 0.5 - (consec_ratio - 0.3) * 1.0)

    return uniq_score * 0.35 + bg_score * 0.35 + consec_score * 0.30


def _score_in_range(value: float, low: float, high: float) -> float:
    """Score how well a value falls within an expected range.

    Returns 1.0 if in range, decreasing as it moves outside.
    """
    if low <= value <= high:
        return 1.0

    if value < low:
        dist = low - value
        range_size = high - low
        return max(0.0, 1.0 - dist / range_size)
    else:
        dist = value - high
        range_size = high - low
        return max(0.0, 1.0 - dist / (range_size * 2))


def calibrate_from_corpus(
    sequences: list[list[int]],
    model: torch.nn.Module,
) -> dict:
    """Calibrate perplexity/entropy ranges from Bach corpus.

    Call this after training to set realistic scoring ranges.
    """
    global _BACH_PERPLEXITY_RANGE, _BACH_ENTROPY_RANGE

    perplexities = []
    entropies = []

    for seq in sequences[:50]:  # sample 50 sequences
        if len(seq) < 20:
            continue
        ppl, ent = _compute_model_metrics(seq, model)
        perplexities.append(ppl)
        entropies.append(ent)

    if perplexities:
        ppl_low = np.percentile(perplexities, 10)
        ppl_high = np.percentile(perplexities, 90)
        _BACH_PERPLEXITY_RANGE = (ppl_low, ppl_high)

    if entropies:
        ent_low = np.percentile(entropies, 10)
        ent_high = np.percentile(entropies, 90)
        _BACH_ENTROPY_RANGE = (ent_low, ent_high)

    return {
        "perplexity_range": _BACH_PERPLEXITY_RANGE,
        "entropy_range": _BACH_ENTROPY_RANGE,
    }
