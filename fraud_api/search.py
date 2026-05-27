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
    """Faiss IVF/Flat KNN restricted to the partition (or Hamming-1 fallback)."""
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous
    start = int(index.boundaries[real_key])
    end = int(index.boundaries[real_key + 1])
    part_labels = index.labels[start:end]
    idx = index.faiss_indices[real_key]
    assert idx is not None  # non-homogeneous partitions always have an index
    q = np.ascontiguousarray(query[None, :], dtype=np.float32)
    _, neighbors = idx.search(q, k)
    return float(part_labels[neighbors[0]].sum()) / k
