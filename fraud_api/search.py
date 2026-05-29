import faiss
import numpy as np

from fraud_api.index import PartitionedIndex

K_NEIGHBORS: int = 5

# Hard cap on extra partitions visited per query (after the primary). Bounds tail
# latency under saturation; cap=8 found optimal — the bbox lb_sq < worst check
# self-limits before reaching this value in practice.
MAX_EXTRA_PARTITIONS: int = 8

# Asymmetric nprobe for the cross-partition extras: extras are already filtered by the
# bbox lower-bound to be candidates worth visiting, so it pays to scan deeper inside them
# than in the primary (where unanimous-exit handles the easy bulk). 3 is the peak vs
# primary=2: beyond 3 the latency cost overtakes the recall gain.
EXTRAS_NPROBE: int = 3


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
    """Cross-partition KNN with axis-aligned bbox lower-bound pruning.

    Searches the query's own partition first, then expands to any other partition
    whose bbox lower-bound (squared L2) is below the current top-K worst distance.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    # query from vectorize() is already float32, C-contig, dim=14 — view as 2D directly
    q = query[None, :]
    idx_primary = index.faiss_indices[real_key]
    assert idx_primary is not None
    start = int(index.boundaries[real_key])
    part_labels = index.labels[start : int(index.boundaries[real_key + 1])]
    dists_p, ids_p = idx_primary.search(q, k)

    ids0 = ids_p[0]
    # Fancy-index uint8 labels then check unanime via min/max equality on the K-element
    # numpy slice — avoids Python list comp + per-element int() casts of the old path.
    labels_top = part_labels[ids0]
    if labels_top.min() == labels_top.max():
        return float(labels_top[0])

    cand_dists: list[float] = dists_p[0].tolist()
    cand_labels: list[int] = labels_top.tolist()
    worst = max(cand_dists)

    diff_high = np.maximum(0.0, query - index.bbox_max)
    diff_low = np.maximum(0.0, index.bbox_min - query)
    lb_sq = np.einsum('ij,ij->i', diff_high, diff_high) + np.einsum(
        'ij,ij->i',
        diff_low,
        diff_low,
    )
    order = np.argsort(lb_sq)

    extra_visited = 0
    for p_int in order:
        if extra_visited >= MAX_EXTRA_PARTITIONS:
            break
        p = int(p_int)
        if p == real_key:
            continue
        if float(lb_sq[p]) >= worst:
            break
        idx_q = index.faiss_indices[p]
        if idx_q is None:
            continue
        start_q = int(index.boundaries[p])
        labels_q = index.labels[start_q : int(index.boundaries[p + 1])]
        if isinstance(idx_q, faiss.IndexIVF):
            idx_q.nprobe = EXTRAS_NPROBE
        dists_q, ids_q = idx_q.search(q, k)
        cand_dists.extend(dists_q[0].tolist())
        cand_labels.extend(labels_q[ids_q[0]].tolist())
        merged = sorted(zip(cand_dists, cand_labels, strict=True))[:k]
        cand_dists = [d for d, _ in merged]
        cand_labels = [lbl for _, lbl in merged]
        worst = cand_dists[-1]
        extra_visited += 1

    return float(sum(cand_labels)) / k
