"""Build the partitioned IVF index from references.json.gz.

Output:
    data/index/
      labels.npy            uint8 (N,) global label array (reordered by partition key)
      meta.json             {boundaries, fallbacks, homogeneous_score, …}
      faiss/<key>.faiss     one Faiss index per non-empty partition (IndexIVFFlat
                            or IndexFlatL2 for small partitions)

The Faiss indices are reloaded at runtime via `faiss.read_index(path, IO_FLAG_MMAP)`
so two API workers can share the same page cache for the bulk of the data.

Idempotent: skips rebuild when index files are newer than the source.
"""

import shutil
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
FAISS_DIR = INDEX_DIR / 'faiss'
REFERENCES_PATH = DATA_DIR / 'references.json.gz'
LABELS_PATH = INDEX_DIR / 'labels.npy'
META_PATH = INDEX_DIR / 'meta.json'

VECTOR_DIM = 14
IVF_NLIST_DIVISOR = 400
IVF_NLIST_MAX = 1024
IVF_NPROBE = 1


def _ivf_nlist(n_vectors: int) -> int:
    return max(1, min(IVF_NLIST_MAX, n_vectors // IVF_NLIST_DIVISOR))


def _build_partition_index(part_vectors: np.ndarray, path: Path) -> None:
    n = len(part_vectors)
    nlist = _ivf_nlist(n)
    if n < max(40, 8 * nlist):
        idx = faiss.IndexFlatL2(VECTOR_DIM)
        idx.add(part_vectors)
    else:
        quantizer = faiss.IndexFlatL2(VECTOR_DIM)
        idx = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, nlist)
        idx.train(part_vectors)
        idx.add(part_vectors)
        idx.nprobe = IVF_NPROBE
    faiss.write_index(idx, str(path))


def _is_fresh() -> bool:
    if not (LABELS_PATH.exists() and META_PATH.exists() and FAISS_DIR.exists()):
        return False
    src_mtime = REFERENCES_PATH.stat().st_mtime
    paths = [LABELS_PATH, META_PATH, *FAISS_DIR.glob('*.faiss')]
    return all(p.stat().st_mtime >= src_mtime for p in paths)


def main() -> int:  # noqa: PLR0915 - linear pipeline, splitting would obscure the flow
    if _is_fresh():
        logger.info('index is fresh, skipping rebuild')
        return 0

    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    FAISS_DIR.mkdir(parents=True)

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

    logger.info('building Faiss indices…')
    # Per-partition axis-aligned bbox (min/max in each dim) for cross-partition
    # lower-bound pruning at query time. Empty partitions get inert bbox.
    bbox_min = np.full((N_PARTITIONS, VECTOR_DIM), np.inf, dtype=np.float32)
    bbox_max = np.full((N_PARTITIONS, VECTOR_DIM), -np.inf, dtype=np.float32)
    for k in range(N_PARTITIONS):
        start, end = int(boundaries[k]), int(boundaries[k + 1])
        if start == end:
            continue
        block = vectors_sorted[start:end]
        bbox_min[k] = block.min(axis=0)
        bbox_max[k] = block.max(axis=0)
        if homogeneous_score[k] >= 0:
            continue
        part_vectors = np.ascontiguousarray(block, dtype=np.float32)
        _build_partition_index(part_vectors, FAISS_DIR / f'{k:03d}.faiss')

    logger.info('writing {}', LABELS_PATH)
    np.save(LABELS_PATH, labels_sorted)
    logger.info('writing {}', META_PATH)
    meta = {
        'n_partitions': N_PARTITIONS,
        'total_vectors': len(vectors_sorted),
        'boundaries': boundaries.tolist(),
        'fallbacks': fallbacks.tolist(),
        'homogeneous_score': homogeneous_score.tolist(),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'ivf_nprobe': IVF_NPROBE,
    }
    META_PATH.write_bytes(msgspec.json.encode(meta))

    total_bytes = sum(p.stat().st_size for p in (LABELS_PATH, META_PATH))
    total_bytes += sum(p.stat().st_size for p in FAISS_DIR.glob('*.faiss'))
    logger.info('index built ({:.1f} MB on disk)', total_bytes / 1e6)
    return 0


if __name__ == '__main__':
    sys.exit(main())
