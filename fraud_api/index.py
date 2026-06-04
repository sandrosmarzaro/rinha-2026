import contextlib
import gzip
import mmap
from dataclasses import dataclass
from pathlib import Path

import msgspec
import numpy as np

from fraud_api.vectorize import VECTOR_DIM

FRAUD_LABEL = 'fraud'

LABELS_FILENAME = 'labels.npy'
META_FILENAME = 'meta.json'
VECTORS_FILENAME = 'vectors.npy'
VEC_NORMS_FILENAME = 'vec_norms.npy'
LABELS_CLUSTER_FILENAME = 'labels_cluster.npy'
CENTROIDS_FILENAME = 'centroids.npy'
CENTROID_NORMS_FILENAME = 'centroid_norms.npy'
CLUSTER_OFFSETS_FILENAME = 'cluster_offsets.npy'
VECTORS_INT16_FILENAME = 'vectors_int16.npy'
KD_NODES_MIN_FILENAME = 'kd_nodes_min.npy'
KD_NODES_MAX_FILENAME = 'kd_nodes_max.npy'
KD_NODES_LEFT_FILENAME = 'kd_nodes_left.npy'
KD_NODES_RIGHT_FILENAME = 'kd_nodes_right.npy'
KD_NODES_START_FILENAME = 'kd_nodes_start.npy'
KD_NODES_LEN_FILENAME = 'kd_nodes_len.npy'
VECTORS_KD_FILENAME = 'vectors_kd.npy'
LABELS_KD_FILENAME = 'labels_kd.npy'


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
    labels: np.ndarray  # (N,) uint8, partition-sorted (used by homogeneous_score)
    boundaries: np.ndarray  # (N_PARTITIONS + 1,) uint32
    fallbacks: np.ndarray  # (N_PARTITIONS,) uint8
    homogeneous_score: np.ndarray  # (N_PARTITIONS,) float32
    vectors_kd: np.ndarray  # (N, 16) int16, mmapped, KD-tree-ordered
    labels_kd: np.ndarray  # (N,) uint8, mmapped, aligned to vectors_kd
    kd_nodes_min: np.ndarray  # (n_nodes, 16) int16, in-RAM
    kd_nodes_max: np.ndarray  # (n_nodes, 16) int16, in-RAM
    kd_nodes_left: np.ndarray  # (n_nodes,) int32, in-RAM
    kd_nodes_right: np.ndarray  # (n_nodes,) int32, in-RAM
    kd_nodes_start: np.ndarray  # (n_nodes,) uint32, in-RAM
    kd_nodes_len: np.ndarray  # (n_nodes,) uint32, in-RAM (0 = internal, >0 = leaf)


def load_partitioned_index(index_dir: Path | str) -> PartitionedIndex:
    index_dir = Path(index_dir)
    meta = msgspec.json.decode((index_dir / META_FILENAME).read_bytes())

    labels = np.load(index_dir / LABELS_FILENAME, mmap_mode='r')
    vectors_kd = np.load(index_dir / VECTORS_KD_FILENAME, mmap_mode='r')
    labels_kd = np.load(index_dir / LABELS_KD_FILENAME, mmap_mode='r')
    for arr in (labels, vectors_kd, labels_kd):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            arr._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)

    kd_nodes_min = np.ascontiguousarray(np.load(index_dir / KD_NODES_MIN_FILENAME), dtype=np.int16)
    kd_nodes_max = np.ascontiguousarray(np.load(index_dir / KD_NODES_MAX_FILENAME), dtype=np.int16)
    kd_nodes_left = np.ascontiguousarray(
        np.load(index_dir / KD_NODES_LEFT_FILENAME), dtype=np.int32
    )
    kd_nodes_right = np.ascontiguousarray(
        np.load(index_dir / KD_NODES_RIGHT_FILENAME), dtype=np.int32
    )
    kd_nodes_start = np.ascontiguousarray(
        np.load(index_dir / KD_NODES_START_FILENAME), dtype=np.uint32
    )
    kd_nodes_len = np.ascontiguousarray(np.load(index_dir / KD_NODES_LEN_FILENAME), dtype=np.uint32)

    return PartitionedIndex(
        labels=labels,
        boundaries=np.asarray(meta['boundaries'], dtype=np.uint32),
        fallbacks=np.asarray(meta['fallbacks'], dtype=np.uint8),
        homogeneous_score=np.asarray(meta['homogeneous_score'], dtype=np.float32),
        vectors_kd=vectors_kd,
        labels_kd=labels_kd,
        kd_nodes_min=kd_nodes_min,
        kd_nodes_max=kd_nodes_max,
        kd_nodes_left=kd_nodes_left,
        kd_nodes_right=kd_nodes_right,
        kd_nodes_start=kd_nodes_start,
        kd_nodes_len=kd_nodes_len,
    )
