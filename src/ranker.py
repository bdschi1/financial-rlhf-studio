import uuid
import time
from datetime import datetime
from typing import List, Dict
from itertools import combinations
from collections import Counter

from src.configs import GenerationConfig
from src.generator import generate_draft


def _estimate_delay(context_limit: int, k: int, provider_counts: Dict[str, int]) -> float:
    """
    Estimate inter-call delay to stay under typical rate limits.

    Most aggressive limit is Anthropic's free/low-tier: 30k input tokens/min.
    At ~4 chars per token, a 60k-char context ≈ 15k tokens → 2 calls/min max.

    Returns seconds to sleep between calls to the same provider.
    """
    approx_tokens = context_limit / 4
    # Anthropic low-tier: 30k TPM; OpenAI: 60k+ TPM; Gemini: generous
    # Use the most conservative estimate
    max_tpm = 30_000
    calls_per_min = max_tpm / max(approx_tokens, 1)
    if calls_per_min >= k:
        return 0  # All K calls fit in one minute window
    interval = 60.0 / max(calls_per_min, 0.1)
    return min(interval, 90)  # Cap at 90s


def generate_k_outputs(
    context: str,
    user_prompt: str,
    configs: List[GenerationConfig],
    api_key: str = None,
    progress_callback=None,
    context_limit: int = 60_000,
) -> Dict[str, str]:
    """
    Generate K outputs, one per config.

    Spaces out calls to the same provider to avoid rate limits.

    Args:
        progress_callback: Optional callable(i, k, status) invoked after each
                           generation to update a progress bar.  Falls back to
                           callable(i, k) for backward compat.
        context_limit: Max chars for context passed to each generation call.

    Returns:
        OrderedDict-like dict mapping label -> output text.
    """
    # Count how many calls go to each provider
    provider_counts = Counter(c.provider for c in configs)
    delay = _estimate_delay(context_limit, len(configs), provider_counts)

    # Track last-call timestamp per provider
    last_call: Dict[str, float] = {}

    outputs = {}
    for i, config in enumerate(configs):
        # Throttle: wait if we'd exceed the rate limit for this provider
        provider = config.provider
        if provider in last_call and delay > 0:
            elapsed = time.time() - last_call[provider]
            remaining = delay - elapsed
            if remaining > 0:
                # Update progress bar with wait status
                _notify(progress_callback, i, len(configs),
                        f"Rate-limit pause ({remaining:.0f}s) before {config.label}...")
                time.sleep(remaining)

        _notify(progress_callback, i, len(configs),
                f"Generating {config.label} ({config.model})...")

        last_call[provider] = time.time()
        outputs[config.label] = generate_draft(
            context, user_prompt, config, api_key, context_limit=context_limit,
        )

        _notify(progress_callback, i + 1, len(configs),
                f"Completed {config.label}")

    return outputs


def _notify(callback, completed: int, total: int, status: str = ""):
    """Call progress callback with flexible signature."""
    if not callback:
        return
    try:
        callback(completed, total, status)
    except TypeError:
        callback(completed, total)


def extract_pairwise_preferences(
    ranked_labels: List[str],
    outputs: Dict[str, str],
    configs_by_label: Dict[str, GenerationConfig],
    prompt: str,
    source: str,
    tags: List[str],
    session_id: str = None,
) -> List[dict]:
    """
    Given a ranking (index 0 = best, index K-1 = worst), produce all
    K(K-1)/2 pairwise preference pairs.

    Example for K=4 ranked [A, C, B, D]:
        (A>C), (A>B), (A>D), (C>B), (C>D), (B>D) = 6 pairs
    """
    session_id = session_id or str(uuid.uuid4())
    total_k = len(ranked_labels)
    pairs = []

    for i, j in combinations(range(total_k), 2):
        chosen_label = ranked_labels[i]
        rejected_label = ranked_labels[j]

        pair = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "chosen": outputs[chosen_label],
            "rejected": outputs[rejected_label],
            "tags": tags,
            "source": source,
            "mode": "ranking",
            "ranking_metadata": {
                "session_id": session_id,
                "total_k": total_k,
                "chosen_rank": i + 1,
                "rejected_rank": j + 1,
                "rank_margin": j - i,
                "chosen_config": configs_by_label[chosen_label].to_dict(),
                "rejected_config": configs_by_label[rejected_label].to_dict(),
                "full_ranking": list(ranked_labels),
            },
        }
        pairs.append(pair)

    return pairs


def count_pairs(k: int) -> int:
    """Number of pairwise comparisons extractable from a K-way ranking."""
    return k * (k - 1) // 2
