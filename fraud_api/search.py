import numpy as np

from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
NPROBE: int = 12


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
    """Pure numpy IVF KNN with homogeneous-partition early-exit.

    Distances use the norm-expansion identity:
        ||a - b||² = ||a||² + ||b||² - 2·a·b
    so each scan is a matmul + a precomputed-norms add; no per-query
    `(block - q)` allocation is needed.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q_norm = float(query @ query)
    centroid_dists = index.centroid_norms + q_norm - 2.0 * (index.centroids @ query)
    cells = np.argpartition(centroid_dists, NPROBE - 1)[:NPROBE]

    offsets = index.cluster_offsets
    starts = offsets[cells]
    ends = offsets[cells + 1]
    cluster_sizes = ends - starts
    total = int(cluster_sizes.sum())
    if total == 0:
        return 1.0
    if total <= k:
        global_ids = np.concatenate(
            [np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends, strict=False)]
        )
        return float(index.cluster_labels[global_ids].sum()) / k

    blocks = [index.vectors[s:e] for s, e in zip(starts, ends, strict=False)]
    norms = [index.vec_norms[s:e] for s, e in zip(starts, ends, strict=False)]
    block = np.concatenate(blocks, axis=0)
    block_norms = np.concatenate(norms)
    dists = block_norms + q_norm - 2.0 * (block @ query)

    top = np.argpartition(dists, k - 1)[:k]

    cum = np.empty(starts.shape[0] + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(cluster_sizes, out=cum[1:])
    cluster_of = np.searchsorted(cum[1:], top, side='right')
    local_in_cluster = top - cum[cluster_of]
    global_ids = starts[cluster_of] + local_in_cluster
    return float(index.cluster_labels[global_ids].sum()) / k
