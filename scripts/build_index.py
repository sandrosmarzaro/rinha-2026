"""Build the global IVF index from references.json.gz.

Output:
    data/index/
      labels.npy            uint8 (N,) global label array (sorted by partition key)
      meta.json             {boundaries, fallbacks, homogeneous_score, ivf_nprobe}
      global.faiss          single IndexIVFScalarQuantizer (fp16) over all N vectors

The faiss index is reloaded at runtime via `faiss.read_index(path, IO_FLAG_MMAP)`
so two API workers can share the same page cache.

Partition data (boundaries, fallbacks, homogeneous_score) is kept only for the
fast-path homogeneous-partition early-exit; KNN itself runs on the single global
index. Single-call faiss.search avoids ~150us of Python crossing overhead vs the
former per-partition + bbox sweep approach.

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
GLOBAL_INDEX_PATH = INDEX_DIR / 'global.faiss'

VECTOR_DIM = 14
GLOBAL_NLIST = 2048  # coarse clusters across the full 3M-vector index
GLOBAL_NPROBE = 12  # lists scanned per query — peak of nprobe sweep at sim 0.17 CPU


def _is_fresh() -> bool:
    if not (LABELS_PATH.exists() and META_PATH.exists() and GLOBAL_INDEX_PATH.exists()):
        return False
    src_mtime = REFERENCES_PATH.stat().st_mtime
    return all(p.stat().st_mtime >= src_mtime for p in (LABELS_PATH, META_PATH, GLOBAL_INDEX_PATH))


def main() -> int:
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

    logger.info('building single global IVF index (nlist={}, fp16)', GLOBAL_NLIST)
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    global_idx = faiss.IndexIVFScalarQuantizer(
        quantizer,
        VECTOR_DIM,
        GLOBAL_NLIST,
        faiss.ScalarQuantizer.QT_fp16,
        faiss.METRIC_L2,
    )
    train_vectors = np.ascontiguousarray(vectors_sorted, dtype=np.float32)
    global_idx.train(train_vectors)
    global_idx.add(train_vectors)
    global_idx.nprobe = GLOBAL_NPROBE
    faiss.write_index(global_idx, str(GLOBAL_INDEX_PATH))

    logger.info('writing {}', LABELS_PATH)
    np.save(LABELS_PATH, labels_sorted)
    logger.info('writing {}', META_PATH)
    meta = {
        'n_partitions': N_PARTITIONS,
        'total_vectors': len(vectors_sorted),
        'boundaries': boundaries.tolist(),
        'fallbacks': fallbacks.tolist(),
        'homogeneous_score': homogeneous_score.tolist(),
        'ivf_nprobe': GLOBAL_NPROBE,
    }
    META_PATH.write_bytes(msgspec.json.encode(meta))

    total_bytes = sum(p.stat().st_size for p in (LABELS_PATH, META_PATH, GLOBAL_INDEX_PATH))
    logger.info('index built ({:.1f} MB on disk)', total_bytes / 1e6)
    return 0


if __name__ == '__main__':
    sys.exit(main())
