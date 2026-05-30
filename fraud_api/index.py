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
    labels: np.ndarray  # (N,) uint8, mmapped, sorted by partition key (global ids)
    boundaries: np.ndarray  # (N_PARTITIONS + 1,) uint32, partition start offsets
    fallbacks: np.ndarray  # (N_PARTITIONS,) uint8, redirect empties to nearest non-empty
    homogeneous_score: np.ndarray  # (N_PARTITIONS,) float32, ≥0 if all labels match
    global_index: faiss.Index  # single IVF over all 3M vectors (mmapped)
    ivf_nprobe: int


def load_partitioned_index(index_dir: Path | str) -> PartitionedIndex:
    index_dir = Path(index_dir)
    meta = msgspec.json.decode((index_dir / META_FILENAME).read_bytes())
    labels = np.load(index_dir / LABELS_FILENAME, mmap_mode='r')
    with contextlib.suppress(AttributeError, OSError, ValueError):
        labels._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)

    nprobe = int(meta.get('ivf_nprobe', 8))
    global_index = faiss.read_index(
        str(index_dir / GLOBAL_INDEX_FILENAME),
        faiss.IO_FLAG_MMAP,
    )
    if isinstance(global_index, faiss.IndexIVF):
        global_index.nprobe = nprobe

    return PartitionedIndex(
        labels=labels,
        boundaries=np.asarray(meta['boundaries'], dtype=np.uint32),
        fallbacks=np.asarray(meta['fallbacks'], dtype=np.uint8),
        homogeneous_score=np.asarray(meta['homogeneous_score'], dtype=np.float32),
        global_index=global_index,
        ivf_nprobe=nprobe,
    )
