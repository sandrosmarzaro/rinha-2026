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
CLUSTER_BBOX_FILENAME = 'cluster_bbox.npy'


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
    vectors: np.ndarray  # (N, 14) fp32, mmapped, cluster-sorted
    vec_norms: np.ndarray  # (N,) fp32, mmapped, ||vec||² per row
    cluster_labels: np.ndarray  # (N,) uint8, mmapped, aligned to `vectors`
    centroids: np.ndarray  # (nlist, 14) fp32, in-RAM (≈115 KB)
    centroid_norms: np.ndarray  # (nlist,) fp32, in-RAM
    cluster_offsets: np.ndarray  # (nlist + 1,) int64, in-RAM
    ivf_nprobe: int
    vectors_int16: np.ndarray  # (N, 16) int16, mmapped — for AVX2 SIMD kernel
    cluster_bbox: np.ndarray  # (nlist, 2, 16) int16, in-RAM — bbox prune (min/max per dim)


def load_partitioned_index(index_dir: Path | str) -> PartitionedIndex:
    index_dir = Path(index_dir)
    meta = msgspec.json.decode((index_dir / META_FILENAME).read_bytes())

    labels = np.load(index_dir / LABELS_FILENAME, mmap_mode='r')
    vectors = np.load(index_dir / VECTORS_FILENAME, mmap_mode='r')
    vec_norms = np.load(index_dir / VEC_NORMS_FILENAME, mmap_mode='r')
    cluster_labels = np.load(index_dir / LABELS_CLUSTER_FILENAME, mmap_mode='r')
    vectors_int16 = np.load(index_dir / VECTORS_INT16_FILENAME, mmap_mode='r')
    for arr in (labels, vectors, vec_norms, cluster_labels, vectors_int16):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            arr._mmap.madvise(mmap.MADV_HUGEPAGE | mmap.MADV_WILLNEED)

    centroids = np.ascontiguousarray(np.load(index_dir / CENTROIDS_FILENAME), dtype=np.float32)
    centroid_norms = np.ascontiguousarray(
        np.load(index_dir / CENTROID_NORMS_FILENAME), dtype=np.float32
    )
    cluster_offsets = np.ascontiguousarray(
        np.load(index_dir / CLUSTER_OFFSETS_FILENAME), dtype=np.int64
    )
    cluster_bbox = np.ascontiguousarray(np.load(index_dir / CLUSTER_BBOX_FILENAME), dtype=np.int16)

    return PartitionedIndex(
        labels=labels,
        boundaries=np.asarray(meta['boundaries'], dtype=np.uint32),
        fallbacks=np.asarray(meta['fallbacks'], dtype=np.uint8),
        homogeneous_score=np.asarray(meta['homogeneous_score'], dtype=np.float32),
        vectors=vectors,
        vec_norms=vec_norms,
        cluster_labels=cluster_labels,
        centroids=centroids,
        centroid_norms=centroid_norms,
        cluster_offsets=cluster_offsets,
        ivf_nprobe=int(meta.get('ivf_nprobe', 12)),
        vectors_int16=vectors_int16,
        cluster_bbox=cluster_bbox,
    )
