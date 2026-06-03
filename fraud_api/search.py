import numpy as np

import knn_simd
from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
NPROBE: int = 12
QUANT_SCALE: float = 10_000.0
PADDED_DIM: int = 16

# Pre-allocated scratch for the quantized query (single-threaded RSGI worker).
_QUERY_INT16: np.ndarray = np.zeros(PADDED_DIM, dtype=np.int16)


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
    """IVF KNN with Rust+AVX2 SIMD inner kernel.

    Centroid selection still runs in numpy (fp32 BLAS over 2048 x 14 - small).
    The hot cluster scan calls into `knn_simd.knn_top5_fraud_count` which uses
    `_mm256_madd_epi16` to compute squared distances over int16-quantized refs:
    16 i16 muls + 8 i32 adds per AVX2 instruction.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q_norm = float(query @ query)
    centroid_dists = index.centroid_norms + q_norm - 2.0 * (index.centroids @ query)
    cells = np.argpartition(centroid_dists, NPROBE - 1)[:NPROBE].astype(np.int64)

    # Quantize query into the pre-allocated int16 buffer (last 2 lanes stay 0).
    np.rint(query * QUANT_SCALE, out=_QUERY_INT16[:14], casting='unsafe')
    fc = knn_simd.knn_top5_fraud_count(
        _QUERY_INT16,
        index.vectors_int16,
        index.cluster_labels,
        index.cluster_offsets,
        cells,
    )
    return float(fc) / k
