import numpy as np

from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
NPROBE: int = 12
# Two-stage re-probe: when top-5 vote is fc ∈ {2, 3} the verdict is one
# neighbor flip away. Re-scan with a wider probe to recover IVF approximation
# misses on truly ambiguous queries. Sim 0.16 CPU saturates here (p99 explodes);
# real 0.40 CPU is 2.5x looser and is the lever's only chance.
NPROBE_REFINE: int = 32
BORDERLINE_LO: int = 2
BORDERLINE_HI: int = 3


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


def _knn_fraud_count(  # noqa: PLR0913
    query: np.ndarray,
    q_norm: float,
    centroid_dists: np.ndarray,
    index: PartitionedIndex,
    nprobe: int,
    k: int,
) -> int:
    cells = np.argpartition(centroid_dists, nprobe - 1)[:nprobe]
    offsets = index.cluster_offsets
    starts = offsets[cells]
    ends = offsets[cells + 1]
    cluster_sizes = ends - starts
    total = int(cluster_sizes.sum())
    if total == 0:
        return k
    if total <= k:
        global_ids = np.concatenate(
            [np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends, strict=False)]
        )
        return int(index.cluster_labels[global_ids].sum())

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
    return int(index.cluster_labels[global_ids].sum())


def partitioned_score(
    query: np.ndarray,
    key: int,
    index: PartitionedIndex,
    k: int = K_NEIGHBORS,
) -> float:
    """Pure numpy IVF KNN with two-stage borderline refinement.

    Stage 1: scan NPROBE closest clusters. If top-5 vote fc ∈ {BORDERLINE_LO,
    BORDERLINE_HI}, the verdict is one neighbor flip away — re-scan with
    NPROBE_REFINE to recover IVF approximation misses.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q_norm = float(query @ query)
    centroid_dists = index.centroid_norms + q_norm - 2.0 * (index.centroids @ query)
    fc = _knn_fraud_count(query, q_norm, centroid_dists, index, NPROBE, k)
    if BORDERLINE_LO <= fc <= BORDERLINE_HI:
        fc = _knn_fraud_count(query, q_norm, centroid_dists, index, NPROBE_REFINE, k)
    return float(fc) / k
