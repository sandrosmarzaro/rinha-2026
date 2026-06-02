import contextlib
import gzip
import mmap
from dataclasses import dataclass
from pathlib import Path

import faiss
import msgspec
import numpy as np

from fraud_api.vectorize import VECTOR_DIM

FRAUD_LABEL = 'fraud'

LABELS_FILENAME = 'labels.npy'
META_FILENAME = 'meta.json'
GLOBAL_INDEX_FILENAME = 'global.faiss'
VECTORS_FILENAME = 'vectors.npy'
LABELS_CLUSTER_FILENAME = 'labels_cluster.npy'
CLUSTER_OFFSETS_FILENAME = 'cluster_offsets.npy'


def load_references(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(path) as f:
        raw = f.read()
    records = msgspec.json.decode(raw)
    vectors = np.array([r['vector'] for r in records], dtype=np.float32)
    labels = np.array([r['label'] == FRAUD_LABEL for r in records], dtype=bool)
    if vectors.shape[1] != VECTOR_DIM:
        msg = f'expected {VECTOR_DIM} dims, got shape {vectors.shape}'
        raise ValueError(msg)
    return vectors, labels


def load_mcc_risk(path: Path | str) -> dict[str, float]:
    return msgspec.json.decode(Path(path).read_bytes())


@dataclass(slots=True, frozen=True)
class PartitionedIndex:
    labels: np.ndarray  # (N,) uint8, mmapped, sorted by partition key (legacy)
    boundaries: np.ndarray  # (N_PARTITIONS + 1,) uint32, partition start offsets
    fallbacks: np.ndarray  # (N_PARTITIONS,) uint8, redirect empties to nearest non-empty
    homogeneous_score: np.ndarray  # (N_PARTITIONS,) float32, ≥0 if all labels match
    quantizer: faiss.Index  # coarse quantizer (IndexFlatL2) for centroid lookup
    vectors: np.ndarray  # (N, 14) float32, mmapped, sorted by IVF cluster
    cluster_labels: np.ndarray  # (N,) uint8, mmapped, aligned to `vectors` ordering
    cluster_offsets: np.ndarray  # (nlist + 1,) int64, cluster start offsets in `vectors`
    ivf_nprobe: int


def load_partitioned_index(index_dir: Path | str) -> PartitionedIndex:
    index_dir = Path(index_dir)
    meta = msgspec.json.decode((index_dir / META_FILENAME).read_bytes())
    labels = np.load(index_dir / LABELS_FILENAME, mmap_mode='r')
    with contextlib.suppress(AttributeError, OSError, ValueError):
        labels._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)

    nprobe = int(meta.get('ivf_nprobe', 8))
    quantizer = faiss.read_index(str(index_dir / GLOBAL_INDEX_FILENAME), faiss.IO_FLAG_MMAP)

    vectors = np.load(index_dir / VECTORS_FILENAME, mmap_mode='r')
    cluster_labels = np.load(index_dir / LABELS_CLUSTER_FILENAME, mmap_mode='r')
    cluster_offsets = np.load(index_dir / CLUSTER_OFFSETS_FILENAME)
    with contextlib.suppress(AttributeError, OSError, ValueError):
        vectors._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)
        cluster_labels._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)

    return PartitionedIndex(
        labels=labels,
        boundaries=np.asarray(meta['boundaries'], dtype=np.uint32),
        fallbacks=np.asarray(meta['fallbacks'], dtype=np.uint8),
        homogeneous_score=np.asarray(meta['homogeneous_score'], dtype=np.float32),
        quantizer=quantizer,
        vectors=vectors,
        cluster_labels=cluster_labels,
        cluster_offsets=np.asarray(cluster_offsets, dtype=np.int64),
        ivf_nprobe=nprobe,
    )
