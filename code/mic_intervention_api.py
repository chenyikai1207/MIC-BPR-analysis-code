"""MIC matched interventions/statistics on explicit qualified arrays.

No IO, download, fit or random-number generation occurs on import. Units for
contrasts and source sensitivity are percentage points (pp).
"""
import numpy as np
from scipy.stats import t
from meg_binary import loo_coefficients, projection
from mic_interventions import masked_binary_lda, intervention_batches, loo_statistics

ARMS = ('OWN_C', 'OTHER_C', 'OWN_X', 'OTHER_X')
LAMBDA_GRID = np.array([0., .25, .5, .75, 1.])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binary_labels(labels, ndim=None):
    a = np.asarray(labels)
    require(a.dtype.kind in 'iu' and (ndim is None or a.ndim == ndim), 'Integer binary labels required')
    require(a.size > 0 and np.isin(a, [0, 1]).all(), 'Labels must be 0 or 1')
    return a


def strict_factorial_plan(labels, true, half, onset, original_donor=None):
    """Strict melody/half donor, nearest log time then original row index.

    No fallback tier, new code draw or silent person/code deletion. Frozen
    codebooks are inputs. Unsupported cells carry reason 1 (no same-code
    nonself donor) or 2 (no opposite-code partner), with that priority.
    """
    labels, true, half = binary_labels(labels, 2), binary_labels(true, 1), binary_labels(half, 1)
    onset = np.asarray(onset, dtype=float)
    q, n = labels.shape
    require(n == len(true) == len(half) and onset.shape == (n,), 'Metadata row mismatch')
    require(np.isfinite(onset).all(), 'Behavior-log onset seconds must be finite')
    require(q <= 199 and n <= 500, 'Single-person bound: at most 199 codes and 500 rows')
    require(all(np.all((labels == c).sum(1) >= 3) for c in [0, 1]), 'N-1 needs two per training class')
    strata = (true[:, None] == true[None, :]) & (half[:, None] == half[None, :])
    equal = labels[:, :, None] == labels[:, None, :]
    same = equal & strata[None, :, :] & ~np.eye(n, dtype=bool)[None, :, :]
    opposite = ~equal & strata[None, :, :]
    has_same, has_opposite = same.any(2), opposite.any(2)
    eligible = has_same & has_opposite
    reason = np.where(~has_same, 1, np.where(~has_opposite, 2, 0)).astype(np.int8)
    distance = np.abs(onset[:, None] - onset[None, :])
    k = np.argmin(np.where(same, distance[None, :, :], np.inf), axis=2)
    l = np.argmin(np.where(opposite, distance[None, :, :], np.inf), axis=2)
    if original_donor is not None:
        old = np.asarray(original_donor)
        require(old.dtype.kind in 'iu' and old.shape == labels.shape, 'Original donor shape/dtype mismatch')
        require(np.array_equal(old[eligible], k[eligible]), 'Frozen donor differs on common support')
    k, l = np.where(has_same, k, -1), np.where(has_opposite, l, -1)
    require(all(np.all((eligible & (labels == c)).sum(1) > 0) for c in [0, 1]),
            'Every code must score both classes; do not silently omit it')
    return {'k': k, 'l': l, 'eligible': eligible, 'reason_code': reason}


def training_cell(labels, k, l, code, trial, arm):
    """Return P-training labels/deletion only; I scoring labels are not input."""
    labels = binary_labels(labels, 2)
    require(arm in ARMS, 'Unknown four-cell arm')
    y = labels[code].copy()
    donor, partner = int(k[code, trial]), int(l[code, trial])
    require(len({trial, donor, partner}) == 3 and min(donor, partner) >= 0, 'Unsupported cell')
    require(y[trial] == y[donor] and y[trial] != y[partner], 'Donor/partner label mismatch')
    own = arm.startswith('OWN')
    deleted, slot = (donor, trial) if own else (trial, donor)
    if arm.endswith('X'):
        y[slot], y[partner] = y[partner], y[slot]
    return y, deleted


def score_four_cell(perception, imagery, labels, plan, budget=lambda: None):
    """Complete N-1 refits. Positive score predicts 1; zero predicts 0.

    I is used only after fitting. Scoring labels are always original. Eligible
    cells alone are evaluated; unsupported output cells are NaN.
    """
    xp, xi = np.asarray(perception, float), np.asarray(imagery, float)
    labels = binary_labels(labels, 2)
    q, n = labels.shape
    require(xp.shape == xi.shape == (n, 16), 'P/I must be aligned N by 16')
    require(np.isfinite(xp).all() and np.isfinite(xi).all(), 'Nonfinite projected features')
    require(q <= 199 and n <= 500, 'Target bound exceeded')
    eligible, k, l = (np.asarray(plan[s]) for s in ['eligible', 'k', 'l'])
    require(eligible.dtype == bool and eligible.shape == k.shape == l.shape == labels.shape, 'Invalid plan shape')
    require(np.all((k[eligible] >= 0) & (k[eligible] < n) & (l[eligible] >= 0) & (l[eligible] < n)), 'Invalid plan indices')
    original = labels.copy()
    evidence = np.full((q, n, 4), np.nan)
    for first in range(0, q, 8):
        last = min(first + 8, q)
        w, a = loo_coefficients(xp, labels[first:last])
        rows = np.arange(last - first)[:, None]
        donor = np.where(eligible[first:last], k[first:last], 0)
        evidence[first:last, :, 0] = np.einsum('qnd,nd->qn', w[rows, donor], xi) + a[rows, donor]
        evidence[first:last, :, 1] = np.einsum('qnd,nd->qn', w, xi) + a
        budget()
    for qb, jb, ab, train, deleted in intervention_batches(labels, k, l, eligible):
        w, a = masked_binary_lda(xp, train, deleted)
        evidence[qb, jb, ab + 2] = np.einsum('bd,bd->b', w, xi[jb]) + a
        budget()
    evidence[~eligible] = np.nan
    require(np.array_equal(labels, original), 'Scoring labels changed')
    require(np.isfinite(evidence[eligible]).all(), 'Nonfinite eligible decisions')
    return evidence


def balanced_code_accuracies(labels, eligible, evidence):
    """Class means within code; then caller averages codes within person."""
    labels = binary_labels(labels, 2)
    eligible, ev = np.asarray(eligible), np.asarray(evidence, dtype=float)
    require(eligible.dtype == bool and eligible.shape == labels.shape, 'Support mask mismatch')
    require(ev.ndim >= 3 and ev.shape[:2] == labels.shape, 'Decision axes must start code, trial')
    require(np.isfinite(ev[eligible]).all(), 'Nonfinite eligible decisions')
    pred, trailing = ev > 0, (None,) * (ev.ndim - 2)
    out = np.zeros((len(labels),) + ev.shape[2:])
    for c in [0, 1]:
        mask, den = eligible & (labels == c), (eligible & (labels == c)).sum(1)
        require(np.all(den > 0), 'Each code must retain both score classes')
        out += .5 * ((pred == c) & mask[(...,) + trailing]).sum(1) / den[(...,) + trailing]
    return out


def factorial_person_endpoints(code_ba, readouts=('B4', 'PCA16')):
    ba = np.asarray(code_ba, dtype=float)
    require(ba.ndim == 3 and ba.shape[1:] == (len(readouts), 4), 'Expected code, readout, four-arm BA')
    require(ba.shape[0] > 0 and np.isfinite(ba).all() and ((ba >= 0) & (ba <= 1)).all(), 'Invalid BA')
    require(len(set(readouts)) == len(readouts), 'Duplicate readout')
    result = {}
    for r, name in enumerate(readouts):
        for a, arm in enumerate(ARMS):
            result[f'{name}_{arm}_BA'] = float(ba[:, r, a].mean())
        ec, ex = ba[:, r, 0] - ba[:, r, 1], ba[:, r, 2] - ba[:, r, 3]
        result[f'{name}_E_C_pp'], result[f'{name}_E_X_pp'] = float(100 * ec.mean()), float(100 * ex.mean())
        result[f'{name}_J_INTERACTION_pp'] = float(100 * (ec - ex).mean())
    if 'B4' in readouts and 'PCA16' in readouts:
        result['J_B4_MINUS_PCA16_pp'] = result['B4_J_INTERACTION_pp'] - result['PCA16_J_INTERACTION_pp']
    return result


def participant_bootstrap(values, frozen_indices):
    """Paired whole-person percentile CI conditional on fixed source.

    Frozen resamples are an input. No seeds/resamples are generated here.
    Columns may be paired endpoints/contrasts from the same person rows.
    """
    values, idx = np.asarray(values, dtype=float), np.asarray(frozen_indices)
    require(values.ndim in [1, 2] and len(values) >= 2 and np.isfinite(values).all(), 'Invalid person values')
    require(idx.dtype.kind in 'iu' and idx.ndim == 2 and idx.shape[1] == len(values) and len(idx) >= 2, 'Resamples need N person indices per row')
    require(idx.min() >= 0 and idx.max() < len(values), 'Bootstrap person index out of range')
    draws = values[idx].mean(axis=1)
    return {'mean': values.mean(axis=0), 'CI95': np.quantile(draws, [.025, .975], axis=0, method='linear'),
            'people': len(values), 'resamples': len(idx), 'scope': 'CONDITIONAL_ON_FIXED_SOURCE'}


def paired_endpoint_difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Pair endpoint axes before subtraction')
    return a - b


def source_target_sensitivity(full_person_contrasts, deleted_source_person_contrasts):
    """Complete source-delete refits + full-source target variance, in pp.

    Every row is a complete source refit on the same target/support. Correlated
    maps are not independent replicates. This function cannot certify fits.
    """
    full, deleted = np.asarray(full_person_contrasts, float), np.asarray(deleted_source_person_contrasts, float)
    require(full.ndim == 1 and deleted.ndim == 2 and deleted.shape[1] == len(full), 'Source by target axes mismatch')
    require(len(full) >= 2 and len(deleted) >= 2 and np.isfinite(full).all() and np.isfinite(deleted).all(), 'Missing/nonfinite deletion or person')
    s, nt = deleted.shape
    theta, means = float(full.mean()), deleted.mean(axis=1)
    vs = float((s - 1) / s * np.square(means - means.mean()).sum())
    vt = float(np.var(full, ddof=1) / nt)
    radius = float(t.ppf(.975, s - 1) * np.sqrt(vs + vt))
    return {'estimate_pp': theta, 'source_variance_pp2': vs, 'target_variance_pp2': vt,
            'approx_source_plus_target_CI95_pp': [theta - radius, theta + radius], 't_df': s - 1,
            'delete_group_estimates_pp': means, 'source_deletions': s, 'target_people': nt,
            'scope': 'APPROXIMATE_ADDITIVE_FIRST_ORDER_SOURCE_PLUS_TARGET_SENSITIVITY',
            'assumptions': 'disjoint independent source/target roles; stable complete refits; finite variance; additive first-order influences',
            'independent_replication': False, 'exact_coverage_established': False}


def source_direction(raw_person_class_means, model):
    """Equal-person true-class P means through the applicable source map.

    Input: source person, two classes, raw features. No imagery/target values.
    Source membership/UIDs must be independently verified before this call.
    """
    means = np.asarray(raw_person_class_means, float)
    require(means.ndim == 3 and means.shape[1] == 2 and len(means) >= 2 and np.isfinite(means).all(), 'Invalid source class means')
    rep = (projection(means.mean(axis=0), model) - model['p_mean']) @ model['B4_P_weights']
    delta = rep[1] - rep[0]
    require(delta.shape == (16,) and np.isfinite(delta).all() and np.linalg.norm(delta) > 1e-10, 'Degenerate source direction')
    return delta


def direction_calibrators(perception, true_labels, source_delta, lambdas):
    """Own P/label excluded in every target model; no I-fitting argument."""
    xp, y = np.asarray(perception, float), binary_labels(true_labels, 1)
    sd, lam = np.asarray(source_delta, float), np.asarray(lambdas, float)
    require(xp.shape == (len(y), 16) and np.isfinite(xp).all(), 'N by 16 P features required')
    require(all(np.sum(y == c) >= 3 for c in [0, 1]), 'N-1 needs two per class')
    require(sd.shape == (16,) and np.isfinite(sd).all(), 'Finite matched source direction required')
    require(lam.ndim == 1 and lam.size > 0 and np.isfinite(lam).all() and ((lam >= 0) & (lam <= 1)).all(), 'Invalid direction weights')
    cov, means, counts = loo_statistics(xp, y)
    cov, means, counts = cov[0], means[0], counts[0]
    delta, middle = means[:, 1] - means[:, 0], means.mean(axis=1)
    prior = np.log(counts[:, 1] / counts[:, 0])
    directions = np.stack([(1 - a) * delta + a * sd[None, :] for a in lam], axis=2)
    w = np.linalg.solve(cov, directions).transpose(0, 2, 1)
    intercept = -np.einsum('nmd,nd->nm', w, middle) + prior[:, None]
    diag = np.diagonal(cov, axis1=1, axis2=2)
    require((diag > 0).all(), 'Singular diagonal covariance')
    wd = delta / diag
    ad = -np.einsum('nd,nd->n', wd, middle) + prior
    require(np.isfinite(w).all() and np.isfinite(intercept).all(), 'Invalid direction discriminants')
    return w, intercept, wd, ad


def select_source_lambda(source_person_ba, lambda_grid=LAMBDA_GRID):
    """Source-only equal-person selection; <=1e-12 tie takes smaller lambda."""
    ba, grid = np.asarray(source_person_ba, float), np.asarray(lambda_grid, float)
    require(np.array_equal(grid, LAMBDA_GRID), 'Keep the fixed five-weight grid')
    require(ba.ndim == 2 and ba.shape[1] == len(grid) and len(ba) >= 2, 'Source person by lambda BA required')
    require(np.isfinite(ba).all() and ((ba >= 0) & (ba <= 1)).all(), 'Invalid source BA')
    average = ba.mean(axis=0)
    chosen = int(np.flatnonzero(average >= average.max() - 1e-12)[0])
    return {'lambda_grid': grid, 'mean_source_LOPO_BA': average, 'chosen_lambda': float(grid[chosen]),
            'target_scores_used': 0, 'scope': 'SOURCE_ONLY_SELECTION_EXPLORATORY_HISTORICAL_EXPOSURE_NOT_REMOVED'}
