import numba
import numpy as np

from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
NPROBE: int = 12

# Pre-allocated scratch buffers (single-threaded RSGI worker → safe as globals).
_BUF_SIZE: int = 50_000
_DISTS_BUF: np.ndarray = np.empty(_BUF_SIZE, dtype=np.float32)
_IDS_BUF: np.ndarray = np.empty(_BUF_SIZE, dtype=np.int64)


@numba.njit(cache=True, fastmath=True, boundscheck=False)  # type: ignore[untyped-decorator]
def _knn_dists_numba(  # noqa: PLR0913
    query: np.ndarray,
    q_norm: np.float32,
    vectors: np.ndarray,
    vec_norms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    dists_out: np.ndarray,
    ids_out: np.ndarray,
) -> int:
    """Compute ||q - v||² for each vector in the probed cluster ranges, filling
    pre-allocated output buffers. Manually unrolled dim-14 inner loop so numba
    emits AVX2 FMA without scalar fallback.
    """
    offset = 0
    n_cells = starts.shape[0]
    for c in range(n_cells):
        s = starts[c]
        e = ends[c]
        for i in range(s, e):
            acc = (
                vectors[i, 0] * query[0]
                + vectors[i, 1] * query[1]
                + vectors[i, 2] * query[2]
                + vectors[i, 3] * query[3]
                + vectors[i, 4] * query[4]
                + vectors[i, 5] * query[5]
                + vectors[i, 6] * query[6]
                + vectors[i, 7] * query[7]
                + vectors[i, 8] * query[8]
                + vectors[i, 9] * query[9]
                + vectors[i, 10] * query[10]
                + vectors[i, 11] * query[11]
                + vectors[i, 12] * query[12]
                + vectors[i, 13] * query[13]
            )
            dists_out[offset] = vec_norms[i] + q_norm - np.float32(2.0) * acc
            ids_out[offset] = i
            offset += 1
    return offset


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

    Hot loop runs through a numba @njit kernel that emits AVX2 FMA for the
    14-dim dot product directly, bypassing BLAS call overhead (significant for
    matmuls this small) and the temporary block concatenation.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q_norm = float(query @ query)
    centroid_dists = index.centroid_norms + q_norm - 2.0 * (index.centroids @ query)
    cells = np.argpartition(centroid_dists, NPROBE - 1)[:NPROBE]

    offsets = index.cluster_offsets
    starts = offsets[cells].astype(np.int64)
    ends = offsets[cells + 1].astype(np.int64)

    n_filled = _knn_dists_numba(
        query,
        np.float32(q_norm),
        index.vectors,
        index.vec_norms,
        starts,
        ends,
        _DISTS_BUF,
        _IDS_BUF,
    )
    if n_filled == 0:
        return 1.0
    if n_filled <= k:
        return float(index.cluster_labels[_IDS_BUF[:n_filled]].sum()) / k

    dists = _DISTS_BUF[:n_filled]
    top = np.argpartition(dists, k - 1)[:k]
    global_ids = _IDS_BUF[top]
    return float(index.cluster_labels[global_ids].sum()) / k
