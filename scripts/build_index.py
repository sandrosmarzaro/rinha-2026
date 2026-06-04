"""Build per-partition KD-tree indices from references.json.gz.

Output:
    data/index/
      labels.npy              uint8 (N,) partition-sorted reference labels
      vectors_kd.npy          int16 (N, 16) KD-reordered i16 vectors
      labels_kd.npy           uint8 (N,) KD-reordered labels
      kd_nodes_{min,max}.npy  int16 (n_nodes, 16) bbox per node
      kd_nodes_{left,right}.npy int32 (n_nodes,) child indices (-1 if leaf)
      kd_nodes_{start,len}.npy uint32 (n_nodes,) leaf vector range
      partition_roots.npy     int32 (256,) root node per partition (-1 if empty)
      partition_bbox_{min,max}.npy int16 (256, 16) bbox per partition root
      meta.json               {boundaries, fallbacks, homogeneous_score}

Idempotent: skips rebuild when index files are newer than the source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import msgspec
import numpy as np
from loguru import logger

from fraud_api.index import load_references
from fraud_api.partition import N_PARTITIONS, compute_fallbacks, partition_keys_batch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
INDEX_DIR = DATA_DIR / 'index'
REFERENCES_PATH = DATA_DIR / 'references.json.gz'
LABELS_PATH = INDEX_DIR / 'labels.npy'
META_PATH = INDEX_DIR / 'meta.json'
KD_NODES_MIN_PATH = INDEX_DIR / 'kd_nodes_min.npy'
KD_NODES_MAX_PATH = INDEX_DIR / 'kd_nodes_max.npy'
KD_NODES_LEFT_PATH = INDEX_DIR / 'kd_nodes_left.npy'
KD_NODES_RIGHT_PATH = INDEX_DIR / 'kd_nodes_right.npy'
KD_NODES_START_PATH = INDEX_DIR / 'kd_nodes_start.npy'
KD_NODES_LEN_PATH = INDEX_DIR / 'kd_nodes_len.npy'
VECTORS_KD_PATH = INDEX_DIR / 'vectors_kd.npy'
LABELS_KD_PATH = INDEX_DIR / 'labels_kd.npy'
PARTITION_ROOTS_PATH = INDEX_DIR / 'partition_roots.npy'
PARTITION_BBOX_MIN_PATH = INDEX_DIR / 'partition_bbox_min.npy'
PARTITION_BBOX_MAX_PATH = INDEX_DIR / 'partition_bbox_max.npy'

VECTOR_DIM = 14
PADDED_DIM = 16  # 14 dims + 2 zero-pad lanes → fits one 256-bit AVX2 register
QUANT_SCALE = 10_000  # lossless: refs are round4'd to 4 decimals
KDTREE_LEAF_SIZE = 64

ARTIFACTS = (
    LABELS_PATH,
    META_PATH,
    KD_NODES_MIN_PATH,
    KD_NODES_MAX_PATH,
    KD_NODES_LEFT_PATH,
    KD_NODES_RIGHT_PATH,
    KD_NODES_START_PATH,
    KD_NODES_LEN_PATH,
    VECTORS_KD_PATH,
    LABELS_KD_PATH,
    PARTITION_ROOTS_PATH,
    PARTITION_BBOX_MIN_PATH,
    PARTITION_BBOX_MAX_PATH,
)


def build_kdtree_partitioned(
    vectors_int16: np.ndarray,
    labels: np.ndarray,
    boundaries: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build one KD-sub-tree per non-empty partition, concatenated into shared arrays.

    Each partition's root is the first node of its sub-tree; child indices and
    leaf vector offsets are biased into the global concatenated address space.
    `partition_roots[k] = -1` marks empty partitions (caller falls back via the
    Hamming-nearest map).
    """
    n_partitions = len(boundaries) - 1
    partition_roots = np.full(n_partitions, -1, dtype=np.int32)
    partition_bbox_min = np.zeros((n_partitions, PADDED_DIM), dtype=np.int16)
    partition_bbox_max = np.zeros((n_partitions, PADDED_DIM), dtype=np.int16)

    chunks_min: list[np.ndarray] = []
    chunks_max: list[np.ndarray] = []
    chunks_left: list[np.ndarray] = []
    chunks_right: list[np.ndarray] = []
    chunks_start: list[np.ndarray] = []
    chunks_len: list[np.ndarray] = []
    chunks_vecs: list[np.ndarray] = []
    chunks_labels: list[np.ndarray] = []
    nodes_offset = 0
    vectors_offset = 0

    for p in range(n_partitions):
        s, e = int(boundaries[p]), int(boundaries[p + 1])
        if s == e:
            continue
        sub = build_kdtree(vectors_int16[s:e], labels[s:e])
        n_sub = len(sub['nodes_left'])
        sub_left = sub['nodes_left'].copy()
        sub_left[sub_left >= 0] += nodes_offset
        sub_right = sub['nodes_right'].copy()
        sub_right[sub_right >= 0] += nodes_offset
        sub_start = sub['nodes_start'].copy() + vectors_offset

        chunks_min.append(sub['nodes_min'])
        chunks_max.append(sub['nodes_max'])
        chunks_left.append(sub_left)
        chunks_right.append(sub_right)
        chunks_start.append(sub_start)
        chunks_len.append(sub['nodes_len'])
        chunks_vecs.append(sub['vectors_kd'])
        chunks_labels.append(sub['labels_kd'])

        partition_roots[p] = nodes_offset
        partition_bbox_min[p] = sub['nodes_min'][0]
        partition_bbox_max[p] = sub['nodes_max'][0]

        nodes_offset += n_sub
        vectors_offset += e - s

    return {
        'partition_roots': partition_roots,
        'partition_bbox_min': partition_bbox_min,
        'partition_bbox_max': partition_bbox_max,
        'nodes_min': np.concatenate(chunks_min, axis=0),
        'nodes_max': np.concatenate(chunks_max, axis=0),
        'nodes_left': np.concatenate(chunks_left),
        'nodes_right': np.concatenate(chunks_right),
        'nodes_start': np.concatenate(chunks_start),
        'nodes_len': np.concatenate(chunks_len),
        'vectors_kd': np.concatenate(chunks_vecs, axis=0),
        'labels_kd': np.concatenate(chunks_labels),
    }


def build_kdtree(
    vectors_int16: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a KD-tree over the i16 vectors.

    Returns parallel-array layout: nodes_min/max (n, 16) i16, nodes_left/right
    (n,) i32, nodes_start/len (n,) u32, vectors_kd (n_vec, 16) i16, labels_kd
    (n_vec,) u8. Internal nodes have len=0; leaves carry a contiguous range
    into the KD-reordered arrays. Split picks the widest bbox dim and
    partitions on the median.
    """
    n = len(vectors_int16)
    perm = np.arange(n, dtype=np.int64)
    n_leaves_max = (n + KDTREE_LEAF_SIZE - 1) // KDTREE_LEAF_SIZE
    max_nodes = max(8, 1 << (n_leaves_max.bit_length() + 2))

    nodes_min = np.zeros((max_nodes, PADDED_DIM), dtype=np.int16)
    nodes_max = np.zeros((max_nodes, PADDED_DIM), dtype=np.int16)
    nodes_left = np.full(max_nodes, -1, dtype=np.int32)
    nodes_right = np.full(max_nodes, -1, dtype=np.int32)
    nodes_start = np.zeros(max_nodes, dtype=np.uint32)
    nodes_len = np.zeros(max_nodes, dtype=np.uint32)
    next_idx = 1

    stack: list[tuple[int, int, int]] = [(0, n, 0)]
    while stack:
        s, e, ni = stack.pop()
        seg = vectors_int16[perm[s:e]]
        mn = seg.min(axis=0)
        mx = seg.max(axis=0)
        nodes_min[ni] = mn
        nodes_max[ni] = mx
        if e - s <= KDTREE_LEAF_SIZE:
            nodes_start[ni] = s
            nodes_len[ni] = e - s
            continue
        split_dim = int(np.argmax((mx - mn)[:VECTOR_DIM]))
        vals = vectors_int16[perm[s:e], split_dim]
        mid = (e - s) // 2
        order = np.argpartition(vals, mid)
        perm[s:e] = perm[s:e][order]
        left, right = next_idx, next_idx + 1
        next_idx += 2
        nodes_left[ni] = left
        nodes_right[ni] = right
        # Push right first so left runs first (DFS, cache-friendly).
        stack.append((s + mid, e, right))
        stack.append((s, s + mid, left))

    return {
        'nodes_min': nodes_min[:next_idx].copy(),
        'nodes_max': nodes_max[:next_idx].copy(),
        'nodes_left': nodes_left[:next_idx].copy(),
        'nodes_right': nodes_right[:next_idx].copy(),
        'nodes_start': nodes_start[:next_idx].copy(),
        'nodes_len': nodes_len[:next_idx].copy(),
        'vectors_kd': np.ascontiguousarray(vectors_int16[perm], dtype=np.int16),
        'labels_kd': np.ascontiguousarray(labels[perm], dtype=np.uint8),
    }


def _is_fresh() -> bool:
    if not all(p.exists() for p in ARTIFACTS):
        return False
    src_mtime = REFERENCES_PATH.stat().st_mtime
    return all(p.stat().st_mtime >= src_mtime for p in ARTIFACTS)


def main() -> int:  # noqa: PLR0915
    if _is_fresh():
        logger.info('index is fresh, skipping rebuild')
        return 0

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info('loading references from {}', REFERENCES_PATH)
    vectors_f32, labels = load_references(REFERENCES_PATH)
    logger.info('loaded {} vectors', len(vectors_f32))

    logger.info('computing partition keys')
    keys = partition_keys_batch(vectors_f32)

    logger.info('sorting by partition key')
    order = np.argsort(keys, kind='stable')
    vectors_sorted = vectors_f32[order]
    labels_sorted = labels[order].astype(np.uint8)
    keys_sorted = keys[order]

    boundaries = np.searchsorted(keys_sorted, np.arange(N_PARTITIONS + 1)).astype(np.uint32)
    fallbacks = compute_fallbacks(boundaries)

    counts = np.diff(boundaries)
    n_non_empty = int((counts > 0).sum())
    logger.info(
        '{} non-empty partitions (min={}, max={})',
        n_non_empty,
        int(counts[counts > 0].min()),
        int(counts.max()),
    )

    homogeneous_score = np.full(N_PARTITIONS, -1.0, dtype=np.float32)
    for k in range(N_PARTITIONS):
        start, end = int(boundaries[k]), int(boundaries[k + 1])
        if start == end:
            continue
        fraud_count = int(labels_sorted[start:end].sum())
        if fraud_count == 0:
            homogeneous_score[k] = 0.0
        elif fraud_count == (end - start):
            homogeneous_score[k] = 1.0
    n_homogeneous = int((homogeneous_score >= 0).sum())
    logger.info('{} homogeneous partitions (early-exit eligible)', n_homogeneous)

    # Quantize partition-sorted vectors to int16 with PADDED_DIM lanes for AVX2.
    # Refs are round4'd by the data generator → multiplying by 10000 is lossless.
    logger.info(
        'quantizing vectors to int16 (scale={}, padded {} → {} lanes)',
        QUANT_SCALE,
        VECTOR_DIM,
        PADDED_DIM,
    )
    vectors_int16 = np.zeros((len(vectors_sorted), PADDED_DIM), dtype=np.int16)
    vectors_int16[:, :VECTOR_DIM] = np.rint(vectors_sorted * QUANT_SCALE).astype(np.int16)

    logger.info('building per-partition KD-trees (leaf_size={})', KDTREE_LEAF_SIZE)
    kd = build_kdtree_partitioned(vectors_int16, labels_sorted, boundaries)
    n_nonempty = int((kd['partition_roots'] >= 0).sum())
    logger.info(
        'kdtree: {} nodes ({:.1f} MB) across {} non-empty partitions, {} leaves',
        len(kd['nodes_left']),
        len(kd['nodes_left']) * 80 / 1e6,
        n_nonempty,
        int((kd['nodes_len'] > 0).sum()),
    )

    logger.info('writing numpy artifacts')
    np.save(LABELS_PATH, labels_sorted)
    np.save(KD_NODES_MIN_PATH, kd['nodes_min'])
    np.save(KD_NODES_MAX_PATH, kd['nodes_max'])
    np.save(KD_NODES_LEFT_PATH, kd['nodes_left'])
    np.save(KD_NODES_RIGHT_PATH, kd['nodes_right'])
    np.save(KD_NODES_START_PATH, kd['nodes_start'])
    np.save(KD_NODES_LEN_PATH, kd['nodes_len'])
    np.save(VECTORS_KD_PATH, kd['vectors_kd'])
    np.save(LABELS_KD_PATH, kd['labels_kd'])
    np.save(PARTITION_ROOTS_PATH, kd['partition_roots'])
    np.save(PARTITION_BBOX_MIN_PATH, kd['partition_bbox_min'])
    np.save(PARTITION_BBOX_MAX_PATH, kd['partition_bbox_max'])

    meta = {
        'n_partitions': N_PARTITIONS,
        'total_vectors': len(vectors_sorted),
        'boundaries': boundaries.tolist(),
        'fallbacks': fallbacks.tolist(),
        'homogeneous_score': homogeneous_score.tolist(),
    }
    META_PATH.write_bytes(msgspec.json.encode(meta))

    total_bytes = sum(p.stat().st_size for p in ARTIFACTS)
    logger.info('index built ({:.1f} MB on disk)', total_bytes / 1e6)
    return 0


if __name__ == '__main__':
    sys.exit(main())
