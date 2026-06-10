#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import TypeVar
from collections.abc import Callable, Iterator
import logging

from .math import estimate_unique_prob_good_turing

__all__ = [
    "sample_uniques",
]

logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")


def sample_uniques(
    draw: Callable[[int], Iterator[Sample]], num: int, prob: float = 1 / 1000
) -> list[Sample]:
    """
    Algorithm to sample from an unknown design space size
    unique values given some draw function with no prerequisite.

    The draw function may be any function returning an Iterator of Hashable
    function of at most some given number of samples.

    The draw function may not give any guarantee of:
    - uniqueness,
    - non-empty return,
    - exact return count,
    - uniformity of sample distributions.

    This algorithm will return:
    - unique samples,
    - empty samples when the draw function is sparse and stop condition reached.

    The algorithms proceed by rounds of round_num draws, where
    round_num is initialized to num and multiplied by a factor of 2
    if no sample is drawn for a round.

    The rounds stop when either:
    - the requested num of sample is reached, or
    - the probability estimate of a new unique draw is lowerthan the stop probability.

    The stop probability defaults to 1/1000, i.e. predicting 1 new over 1000 draws.

    Args:
        draw: the draw function taking an int and returning an iterator of hashable
        num: the requested number of samples
        prob: stop condition probability of 1 new unique

    Returns:
        a list of unique such samples of at most num elements
    """

    round_num = num
    samples: list[Sample] = []
    seen: set[Sample] = set()
    once: set[Sample] = set()
    twice: set[Sample] = set()
    draws = 0
    requests = 0
    p_valid: float = 1.0
    p_unseen: float = 1.0
    p_next: float = p_valid * p_unseen
    while len(samples) < num and p_next > prob:
        logger.debug(
            "sample_uniques round: count: %d/%d, p_unseen: %g, p_valid: %g, %g > %g, "
            "draws: %d, requests: %d",
            len(samples),
            num,
            p_unseen,
            p_valid,
            p_next,
            prob,
            draws,
            requests,
        )
        round_samples = list(draw(round_num))
        requests += round_num
        draws += len(round_samples)
        round_uniques = []
        for sample in round_samples:
            if sample not in seen:
                round_uniques.append(sample)
                seen.add(sample)
                once.add(sample)
            elif sample in once:
                once.remove(sample)
                twice.add(sample)
            elif sample in twice:
                twice.remove(sample)
        samples.extend(round_uniques)
        p_valid = max(1 / (requests + 1), draws / requests)
        p_unseen = estimate_unique_prob_good_turing(draws, len(once))
        p_next = p_unseen * p_valid
        if len(round_samples) == 0:
            round_num *= 2
    samples = samples[:num]
    logger.debug(
        "sample_uniques sampled: count: %d/%d, p_unseen: %g, p_valid: %g, %g, "
        "draws: %d, requests: %d",
        len(samples),
        num,
        p_unseen,
        p_valid,
        p_next,
        draws,
        requests,
    )
    return samples
