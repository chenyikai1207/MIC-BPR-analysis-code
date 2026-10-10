"""Actual approved target allocation implementation; frozen seed20261032.
Names retain historical proposed_* spelling; the approved producer used this
unchanged implementation before outcomes. No actual codebooks are included.
"""
from __future__ import annotations
import numpy as np

PROPOSED_RULE = "participant_session_true_category_balanced_three_code_v1"

PROPOSED_SEED_ENTROPY = 20261032

N_MONTE_CARLO_CODES = 199

MIN_STRATUM_SIZE = 6

def _checked_strata(true_y: np.ndarray, session: np.ndarray):
    y = np.asarray(true_y)
    s = np.asarray(session)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("Nonempty aligned one-dimensional metadata is required")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("True categories must use integer identifiers 0,1,2")
    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise ValueError("All three true categories must be represented")
    # np.unique gives a canonical order; the caller freezes original pair order.
    strata = []
    for session_id in np.unique(s):
        for category in range(3):
            index = np.flatnonzero((s == session_id) & (y == category))
            if len(index) < MIN_STRATUM_SIZE:
                raise ValueError(
                    f"Stratum session={session_id!r}, true_category={category} "
                    f"has {len(index)} eligible pairs; at least 6 are required. "
                    "Pause for method review; do not remove people, pool strata, "
                    "or redraw seeds to repair support."
                )
            strata.append(index)
    return y, s, tuple(strata)

def draw_proposed_codes(
    true_y: np.ndarray,
    session: np.ndarray,
    participant_number: int,
    *,
    n_codes: int = N_MONTE_CARLO_CODES,
    seed_entropy: int = PROPOSED_SEED_ENTROPY,
) -> np.ndarray:
    """Return [Monte Carlo draw, original pair] target codes for method review.

    One code is shared by the P/I phases of the original pair. For each
    participant x session x true-category stratum, remainder-code selection is
    uniform and the multiset is uniformly permuted across its pairs. Each code
    count is floor(n/3) or ceil(n/3). Do not call on actual data before approval.
    Source supervised labels stay true categories; this function changes only
    target calibration/evaluation labels, and accepts no signals or outcomes.
    """
    if not isinstance(participant_number, int) or participant_number < 0:
        raise ValueError("Participant number must be a nonnegative integer")
    if not isinstance(n_codes, int) or n_codes < 1:
        raise ValueError("n_codes must be a positive integer")
    y, _s, strata = _checked_strata(true_y, session)
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed_entropy, participant_number]))
    )
    codes = np.empty((n_codes, len(y)), dtype=np.int8)
    for draw in range(n_codes):
        for index in strata:
            base, remainder = divmod(len(index), 3)
            counts = np.full(3, base, dtype=np.int64)
            if remainder:
                extra = generator.choice(3, size=remainder, replace=False)
                counts[extra] += 1
            multiset = np.repeat(np.arange(3, dtype=np.int8), counts)
            codes[draw, index] = generator.permutation(multiset)
    assert_proposed_support(y, _s, codes)
    return codes

def assert_proposed_support(true_y, session, codes) -> None:
    """Metadata-only support check; never scores a model or selects new draws."""
    y, _s, strata = _checked_strata(true_y, session)
    z = np.asarray(codes)
    if z.ndim != 2 or z.shape[1] != len(y):
        raise ValueError("codes must be [draw, original pair]")
    if not np.issubdtype(z.dtype, np.integer) or np.any((z < 0) | (z > 2)):
        raise ValueError("All target codes must be integer 0,1,2")
    for row in z:
        for index in strata:
            counts = np.bincount(row[index], minlength=3)
            low, remainder = divmod(len(index), 3)
            if counts.min() < 2 or counts.min() != low:
                raise AssertionError("Insufficient donor support or altered balancing rule")
            if counts.max() > low + 1 or int(np.sum(counts == low + 1)) != remainder:
                raise AssertionError("Stratum allocation is not the stated uniform rule")

def expected_cross_counts(true_counts) -> np.ndarray:
    """Analytic expectation under uniform remainder selection, not observed data."""
    counts = np.asarray(true_counts, dtype=np.int64)
    if counts.shape != (3,) or np.any(counts < MIN_STRATUM_SIZE):
        raise ValueError("Three stratum counts of at least 6 are required")
    return np.repeat((counts / 3.0)[:, None], 3, axis=1)
