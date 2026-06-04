import os

import numpy as np

import knn_simd
from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5
QUANT_SCALE: float = 10_000.0
PADDED_DIM: int = 16
# Early-exit threshold for cross-partition scan: if primary partition's worst-of-top-5
# squared distance is below this, accept without scanning the rest. Sim sweep showed
# the safe range is ≤ 2_000_000 (det stays at FP=29 FN=0); but real Haswell preview
# #8479 with EL=2M regressed -57 - early-exit widens per-query variance (easy 50us,
# hard 200us) and the p99 tail loses what the average gained. Default disabled.
EARLY_LIMIT: int = int(os.environ.get('RINHA_EARLY_LIMIT') or (1 << 63) - 1)


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
    """Exact KNN over a KD-tree with AVX2-pruned branch-and-bound.

    The homogeneous_score fast-exit covers ~9.5% of boundary queries that land
    in a partition where every reference shares the same label — no KNN needed.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    fc = knn_simd.knn_top5_kdtree_partitioned(
        q_i16,
        index.vectors_kd,
        index.labels_kd,
        index.kd_nodes_min,
        index.kd_nodes_max,
        index.kd_nodes_left,
        index.kd_nodes_right,
        index.kd_nodes_start,
        index.kd_nodes_len,
        index.partition_roots,
        index.partition_bbox_min,
        index.partition_bbox_max,
        real_key,
        EARLY_LIMIT,
    )
    return float(fc) / k
