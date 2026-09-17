"""Window ranks, first common-set entry, and reconstruction-loss comparator."""
import numpy as np

def ranks_by_recovery(q, theta_ids):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError('Expected finite window-by-candidate recovery scores')
    ids = np.asarray(theta_ids, dtype=str)
    ranks = np.empty(q.shape, dtype=np.int32)
    for i, row in enumerate(q):
        order = np.lexsort((ids, -row))
        ranks[i, order] = np.arange(1, len(ids) + 1)
    return ranks

def consensus(q, theta_ids):
    ranks = ranks_by_recovery(q, theta_ids)
    entry = ranks.max(axis=0) / ranks.shape[1]
    # The released global and pyocyanin minima are both unique. The secondary
    # ordering below only gives a reproducible listing of subsequent entrants.
    order = np.lexsort((np.asarray(theta_ids, dtype=str), -np.mean(q, axis=0), entry))
    return ranks, entry, order

def loss_comparators(residuals, theta_ids):
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or not np.isfinite(residuals).all():
        raise ValueError('Expected finite window-by-candidate residuals')
    ids = np.asarray(theta_ids, dtype=str)
    return np.array([np.lexsort((ids, row))[0] for row in residuals], dtype=int)
