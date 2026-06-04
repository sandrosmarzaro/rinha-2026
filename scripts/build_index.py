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
CLUSTER_BBOX_PATH = INDEX_DIR / 'cluster_bbox.npy'

VECTOR_DIM = 14
PADDED_DIM = 16  # 14 dims + 2 zero-pad lanes → fits one 256-bit AVX2 register
QUANT_SCALE = 10_000  # lossless: refs are round4'd to 4 decimals
GLOBAL_NLIST = 2048
GLOBAL_NPROBE = 12

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
    CLUSTER_BBOX_PATH,
)


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

    # Per-cluster bounding box (min/max per dim) in int16 space. Used at query
    # time to compute a sound lower bound on the closest-possible vector
    # distance — clusters whose lb > current kth_dist² are pruned entirely.
    # Layout: (nlist, 2, 16) — [c, 0, :] = min, [c, 1, :] = max.
    logger.info('computing per-cluster bbox for lower-bound prune')
    cluster_bbox = np.zeros((GLOBAL_NLIST, 2, PADDED_DIM), dtype=np.int16)
    # Sentinel: empty cluster bbox stays all-zero (lb²=0 = no prune). We won't
    # probe empty clusters since they have no centroid-distance match anyway.
    for c in range(GLOBAL_NLIST):
        s, e = int(cluster_offsets[c]), int(cluster_offsets[c + 1])
        if s == e:
            continue
        cluster_bbox[c, 0, :VECTOR_DIM] = vectors_int16[s:e, :VECTOR_DIM].min(axis=0)
        cluster_bbox[c, 1, :VECTOR_DIM] = vectors_int16[s:e, :VECTOR_DIM].max(axis=0)

    logger.info('writing numpy artifacts')
    np.save(LABELS_PATH, labels_sorted)
    np.save(VECTORS_PATH, vectors_cluster)
    np.save(VEC_NORMS_PATH, vec_norms)
    np.save(LABELS_CLUSTER_PATH, labels_cluster)
    np.save(CENTROIDS_PATH, centroids)
    np.save(CENTROID_NORMS_PATH, centroid_norms)
    np.save(CLUSTER_OFFSETS_PATH, cluster_offsets)
    np.save(VECTORS_INT16_PATH, vectors_int16)
    np.save(CLUSTER_BBOX_PATH, cluster_bbox)

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
