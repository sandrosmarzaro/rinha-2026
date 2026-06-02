import numpy as np

from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
NPROBE: int = 12  # must match the Faiss training nprobe to keep recall calibrated


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
    """Hybrid IVF KNN: Faiss for centroid lookup, numpy for cluster scan + top-K.

    Steps:
        1. homogeneous-partition early-exit (cheap label lookup)
        2. Faiss quantizer.search → top-nprobe nearest centroids
        3. numpy scan of vectors in those clusters → top-K KNN
        4. label majority → fraud_score
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q = query[None, :]
    _dc, cells = index.quantizer.search(q, NPROBE)
    cells_row = cells[0]

    vectors = index.vectors
    offsets = index.cluster_offsets
    starts = offsets[cells_row]
    ends = offsets[cells_row + 1]

    # Concatenate the nprobe cluster slices and compute squared L2 in one pass.
    # The slices stay contiguous in memory (vectors_sorted by cluster at build time),
    # so this is a sequence of cache-friendly streaming reads.
    pieces = [vectors[s:e] for s, e in zip(starts, ends, strict=False)]
    if not pieces:
        return 1.0
    block = np.concatenate(pieces, axis=0)
    diff = block - q
    dists = np.einsum('ij,ij->i', diff, diff)
    if dists.shape[0] <= k:
        return (
            float(
                index.cluster_labels[
                    np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=False)])[
                        np.argsort(dists)[:k]
                    ]
                ].sum()
            )
            / k
        )
    top = np.argpartition(dists, k - 1)[:k]

    # Map local indices in `block` back to global ids in `cluster_labels`.
    # Build per-cluster cumulative offsets in `block` for a single fancy index.
    cluster_sizes = ends - starts
    cum = np.empty(len(starts) + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(cluster_sizes, out=cum[1:])
    # For each candidate, find which cluster bucket it belongs to and translate.
    cluster_of = np.searchsorted(cum[1:], top, side='right')
    local_in_cluster = top - cum[cluster_of]
    global_ids = starts[cluster_of] + local_in_cluster
    return float(index.cluster_labels[global_ids].sum()) / k
