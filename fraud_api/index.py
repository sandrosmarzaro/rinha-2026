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
FAISS_SUBDIR = 'faiss'


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
    labels: np.ndarray  # (N,) uint8, mmapped, reordered by partition key
    boundaries: np.ndarray  # (N_PARTITIONS + 1,) uint32, partition start offsets
    fallbacks: np.ndarray  # (N_PARTITIONS,) uint8, redirect empties to nearest non-empty
    homogeneous_score: np.ndarray  # (N_PARTITIONS,) float32, ≥0 if all labels match
    faiss_indices: tuple[faiss.Index | None, ...]  # one Faiss index per partition (mmapped)
    ivf_nprobe: int


def load_partitioned_index(
    index_dir: Path | str,
    nprobe_override: int | None = None,
) -> PartitionedIndex:
    index_dir = Path(index_dir)
    meta = msgspec.json.decode((index_dir / META_FILENAME).read_bytes())
    labels = np.load(index_dir / LABELS_FILENAME, mmap_mode='r')
    with contextlib.suppress(AttributeError, OSError, ValueError):
        labels._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)

    faiss_dir = index_dir / FAISS_SUBDIR
    nprobe = nprobe_override if nprobe_override is not None else int(meta.get('ivf_nprobe', 8))
    faiss_indices: list[faiss.Index | None] = [None] * meta['n_partitions']
    for path in faiss_dir.glob('*.faiss'):
        key = int(path.stem)
        idx = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
        with contextlib.suppress(AttributeError, RuntimeError):
            idx.nprobe = nprobe
        faiss_indices[key] = idx

    return PartitionedIndex(
        labels=labels,
        boundaries=np.asarray(meta['boundaries'], dtype=np.uint32),
        fallbacks=np.asarray(meta['fallbacks'], dtype=np.uint8),
        homogeneous_score=np.asarray(meta['homogeneous_score'], dtype=np.float32),
        faiss_indices=tuple(faiss_indices),
        ivf_nprobe=nprobe,
    )
