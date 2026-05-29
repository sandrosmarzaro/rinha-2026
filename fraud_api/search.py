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
    Equivalent to exact KNN over the full reference set when pruning is tight, but
    only visits a handful of partitions per query in practice.
    """
    real_key = int(index.fallbacks[key])
    homogeneous = float(index.homogeneous_score[real_key])
    if homogeneous >= 0.0:
        return homogeneous

    q = np.ascontiguousarray(query[None, :], dtype=np.float32)
    idx_primary = index.faiss_indices[real_key]
    assert idx_primary is not None  # non-homogeneous partitions always have an index
    start = int(index.boundaries[real_key])
    end = int(index.boundaries[real_key + 1])
    part_labels = index.labels[start:end]
    dists_p, ids_p = idx_primary.search(q, k)

    cand_dists: list[float] = [float(d) for d in dists_p[0]]
    cand_labels: list[int] = [int(part_labels[i]) for i in ids_p[0]]

    # Unanimous primary vote — skip the cross-partition bbox sweep for high-confidence
    # queries. The vast majority hit homogeneous-ish neighborhoods; only ambiguous ones
    # pay the bbox computation + extra partition searches.
    first = cand_labels[0]
    if all(lbl == first for lbl in cand_labels):
        return float(first)

    worst = max(cand_dists)

    # bbox lower-bound (squared L2) from query to every partition's axis-aligned box
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
            break  # sorted ascending — every remaining partition is too far
        idx_q = index.faiss_indices[p]
        if idx_q is None:
            continue  # homogeneous or empty — no faiss index to query
        start_q = int(index.boundaries[p])
        end_q = int(index.boundaries[p + 1])
        labels_q = index.labels[start_q:end_q]
        if isinstance(idx_q, faiss.IndexIVF):
            idx_q.nprobe = EXTRAS_NPROBE
        dists_q, ids_q = idx_q.search(q, k)
        for di, ii in zip(dists_q[0], ids_q[0], strict=True):
            cand_dists.append(float(di))
            cand_labels.append(int(labels_q[ii]))
        merged = sorted(zip(cand_dists, cand_labels, strict=True))[:k]
        cand_dists = [d for d, _ in merged]
        cand_labels = [lbl for _, lbl in merged]
        worst = cand_dists[-1]
        extra_visited += 1

    return float(sum(cand_labels)) / k
