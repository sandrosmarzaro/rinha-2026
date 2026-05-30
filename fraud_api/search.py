import numpy as np

from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5


def brute_force_score(
    query: np.ndarray,
    vectors: np.ndarray,
    labels: np.ndarray,
    k: int = K_NEIGHBORS,
) -> float:
    """Float32 brute-force KNN over the full reference set (parity oracle)."""
    k_eff = min(k, len(vectors))
    diffs = vectors - query
    sq_dists = np.einsum('ij,ij->i', diffs, diffs)
    nearest = np.argpartition(sq_dists, k_eff - 1)[:k_eff]
    return float(labels[nearest].sum()) / k_eff


def partitioned_score(
    query: np.ndarray,
    key: int,
    index: PartitionedIndex,
    k: int = K_NEIGHBORS,
) -> float:
    """Single-IVF KNN with homogeneous-partition early-exit.

    The query's partition key is used only to short-circuit homogeneous partitions
    (where all reference labels match). For non-homogeneous queries, a single
    faiss.search call on the global IVF index returns the K nearest neighbors —
    avoids the Python overhead of multiple per-partition search calls + bbox sweep.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q = query[None, :]
    _dists, ids = index.global_index.search(q, k)
    return float(index.labels[ids[0]].sum()) / k
