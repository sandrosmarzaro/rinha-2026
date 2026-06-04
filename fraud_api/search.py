import numpy as np

import knn_simd
from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
NPROBE: int = 32
QUANT_SCALE: float = 10_000.0
INV_QUANT_SCALE: float = 1.0 / QUANT_SCALE
PADDED_DIM: int = 16


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
    q_i16: np.ndarray,
    key: int,
    index: PartitionedIndex,
    k: int = K_NEIGHBORS,
) -> float:
    """IVF KNN with Rust+AVX2 SIMD inner kernel.

    The query arrives already quantized to int16 (from `knn_simd.vectorize_to_i16`).
    Centroid selection dequantizes to f32 since centroids are stored in f32.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q_f32 = q_i16[:14].astype(np.float32) * INV_QUANT_SCALE
    q_norm = float(q_f32 @ q_f32)
    centroid_dists = index.centroid_norms + q_norm - 2.0 * (index.centroids @ q_f32)
    cells = np.argpartition(centroid_dists, NPROBE - 1)[:NPROBE].astype(np.int64)

    fc = knn_simd.knn_top5_fraud_count(
        q_i16,
        index.vectors_int16,
        index.cluster_labels,
        index.cluster_offsets,
        cells,
    )
    return float(fc) / k
