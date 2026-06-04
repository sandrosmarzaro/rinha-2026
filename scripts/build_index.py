"""Build the global IVF index from references.json.gz.

Output:
    data/index/
      labels.npy            uint8 (N,) global label array (sorted by partition key)
      labels_cluster.npy    uint8 (N,) labels aligned to vectors.npy (cluster-sorted)
      vectors.npy           fp32 (N, 14) all reference vectors sorted by IVF cluster
      vec_norms.npy         fp32 (N,) precomputed ||vec||² per vector
      centroids.npy         fp32 (nlist, 14) k-means centroids
      centroid_norms.npy    fp32 (nlist,) precomputed ||centroid||²
      cluster_offsets.npy   int64 (nlist+1,) cluster start offsets in vectors.npy
      meta.json             {boundaries, fallbacks, homogeneous_score, ivf_nprobe, ivf_nlist}

Faiss is used at BUILD time to run k-means (`IndexIVFFlat.train`); at RUNTIME the
search uses only numpy (centroid scan + cluster scan + norms-expansion distance).

Idempotent: skips rebuild when index files are newer than the source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
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
VECTORS_PATH = INDEX_DIR / 'vectors.npy'
VEC_NORMS_PATH = INDEX_DIR / 'vec_norms.npy'
LABELS_CLUSTER_PATH = INDEX_DIR / 'labels_cluster.npy'
CENTROIDS_PATH = INDEX_DIR / 'centroids.npy'
CENTROID_NORMS_PATH = INDEX_DIR / 'centroid_norms.npy'
CLUSTER_OFFSETS_PATH = INDEX_DIR / 'cluster_offsets.npy'
VECTORS_INT16_PATH = INDEX_DIR / 'vectors_int16.npy'
KD_NODES_MIN_PATH = INDEX_DIR / 'kd_nodes_min.npy'
KD_NODES_MAX_PATH = INDEX_DIR / 'kd_nodes_max.npy'
KD_NODES_LEFT_PATH = INDEX_DIR / 'kd_nodes_left.npy'
KD_NODES_RIGHT_PATH = INDEX_DIR / 'kd_nodes_right.npy'
KD_NODES_START_PATH = INDEX_DIR / 'kd_nodes_start.npy'
KD_NODES_LEN_PATH = INDEX_DIR / 'kd_nodes_len.npy'
VECTORS_KD_PATH = INDEX_DIR / 'vectors_kd.npy'
LABELS_KD_PATH = INDEX_DIR / 'labels_kd.npy'

VECTOR_DIM = 14
PADDED_DIM = 16  # 14 dims + 2 zero-pad lanes → fits one 256-bit AVX2 register
QUANT_SCALE = 10_000  # lossless: refs are round4'd to 4 decimals
GLOBAL_NLIST = 2048
GLOBAL_NPROBE = 12
KDTREE_LEAF_SIZE = 64

ARTIFACTS = (
    LABELS_PATH,
    META_PATH,
    VECTORS_PATH,
    VEC_NORMS_PATH,
    LABELS_CLUSTER_PATH,
    CENTROIDS_PATH,
    CENTROID_NORMS_PATH,
    CLUSTER_OFFSETS_PATH,
    VECTORS_INT16_PATH,
    KD_NODES_MIN_PATH,
    KD_NODES_MAX_PATH,
    KD_NODES_LEFT_PATH,
    KD_NODES_RIGHT_PATH,
    KD_NODES_START_PATH,
    KD_NODES_LEN_PATH,
    VECTORS_KD_PATH,
    LABELS_KD_PATH,
)


def build_kdtree(  # noqa: PLR0914
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

    logger.info('training k-means centroids (nlist={}, niter=25 nredo=4)', GLOBAL_NLIST)
    train_vectors = np.ascontiguousarray(vectors_sorted, dtype=np.float32)
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    ivf = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, GLOBAL_NLIST, faiss.METRIC_L2)
    ivf.cp.niter = 25
    ivf.cp.nredo = 4
    ivf.train(train_vectors)

    logger.info('extracting centroids + assigning vectors to clusters')
    centroids = np.empty((GLOBAL_NLIST, VECTOR_DIM), dtype=np.float32)
    for c in range(GLOBAL_NLIST):
        centroids[c] = ivf.quantizer.reconstruct(c)
    _d, assigns = ivf.quantizer.search(train_vectors, 1)
    assigns = assigns[:, 0]

    cluster_counts = np.bincount(assigns, minlength=GLOBAL_NLIST).astype(np.int64)
    cluster_offsets = np.empty(GLOBAL_NLIST + 1, dtype=np.int64)
    cluster_offsets[0] = 0
    np.cumsum(cluster_counts, out=cluster_offsets[1:])
    cluster_order = np.argsort(assigns, kind='stable')
    vectors_cluster = np.ascontiguousarray(train_vectors[cluster_order], dtype=np.float32)
    labels_cluster = labels_sorted[cluster_order]
    vec_norms = np.einsum('ij,ij->i', vectors_cluster, vectors_cluster).astype(np.float32)
    centroid_norms = np.einsum('ij,ij->i', centroids, centroids).astype(np.float32)
    logger.info(
        'cluster layout: min={} p50={} p99={} max={}',
        int(cluster_counts.min()),
        int(np.percentile(cluster_counts, 50)),
        int(np.percentile(cluster_counts, 99)),
        int(cluster_counts.max()),
    )

    # Quantize cluster-sorted vectors to int16 with PADDED_DIM lanes for AVX2.
    # Refs are round4'd by the data generator → multiplying by 10000 is lossless.
    logger.info(
        'quantizing vectors to int16 (scale={}, padded {} → {} lanes)',
        QUANT_SCALE,
        VECTOR_DIM,
        PADDED_DIM,
    )
    vectors_int16 = np.zeros((len(vectors_cluster), PADDED_DIM), dtype=np.int16)
    vectors_int16[:, :VECTOR_DIM] = np.rint(vectors_cluster * QUANT_SCALE).astype(np.int16)

    logger.info('building KD-tree (leaf_size={})', KDTREE_LEAF_SIZE)
    kd = build_kdtree(vectors_int16, labels_cluster)
    logger.info(
        'kdtree: {} nodes ({:.1f} MB), {} leaves',
        len(kd['nodes_left']),
        len(kd['nodes_left']) * 80 / 1e6,
        int((kd['nodes_len'] > 0).sum()),
    )

    logger.info('writing numpy artifacts')
    np.save(LABELS_PATH, labels_sorted)
    np.save(VECTORS_PATH, vectors_cluster)
    np.save(VEC_NORMS_PATH, vec_norms)
    np.save(LABELS_CLUSTER_PATH, labels_cluster)
    np.save(CENTROIDS_PATH, centroids)
    np.save(CENTROID_NORMS_PATH, centroid_norms)
    np.save(CLUSTER_OFFSETS_PATH, cluster_offsets)
    np.save(VECTORS_INT16_PATH, vectors_int16)
    np.save(KD_NODES_MIN_PATH, kd['nodes_min'])
    np.save(KD_NODES_MAX_PATH, kd['nodes_max'])
    np.save(KD_NODES_LEFT_PATH, kd['nodes_left'])
    np.save(KD_NODES_RIGHT_PATH, kd['nodes_right'])
    np.save(KD_NODES_START_PATH, kd['nodes_start'])
    np.save(KD_NODES_LEN_PATH, kd['nodes_len'])
    np.save(VECTORS_KD_PATH, kd['vectors_kd'])
    np.save(LABELS_KD_PATH, kd['labels_kd'])

    meta = {
        'n_partitions': N_PARTITIONS,
        'total_vectors': len(vectors_sorted),
        'boundaries': boundaries.tolist(),
        'fallbacks': fallbacks.tolist(),
        'homogeneous_score': homogeneous_score.tolist(),
        'ivf_nprobe': GLOBAL_NPROBE,
        'ivf_nlist': GLOBAL_NLIST,
    }
    META_PATH.write_bytes(msgspec.json.encode(meta))

    total_bytes = sum(p.stat().st_size for p in ARTIFACTS)
    logger.info('index built ({:.1f} MB on disk)', total_bytes / 1e6)
    return 0


if __name__ == '__main__':
    sys.exit(main())
