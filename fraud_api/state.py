import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from loguru import logger

from fraud_api.index import (
    PartitionedIndex,
    load_mcc_risk,
    load_partitioned_index,
)
from fraud_api.partition import N_PARTITIONS, compute_fallbacks, partition_keys_batch
from fraud_api.vectorize import VECTOR_DIM

ENV_DATA_DIR: Final = 'RINHA_DATA_DIR'
INDEX_SUBDIR: Final = 'index'
MCC_RISK_FILENAME: Final = 'mcc_risk.json'

SYNTHETIC_N: Final = 1000
SYNTHETIC_FRAUD_RATE: Final = 0.3
SYNTHETIC_SEED: Final = 42
SYNTHETIC_NPROBE: Final = 4


@dataclass(slots=True, frozen=True)
class AppData:
    mcc_risk: dict[str, float]
    index: PartitionedIndex


def _build_synthetic_index() -> PartitionedIndex:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    vectors_f32 = rng.random((SYNTHETIC_N, VECTOR_DIM), dtype=np.float32)
    labels = (rng.random(SYNTHETIC_N) < SYNTHETIC_FRAUD_RATE).astype(np.uint8)
    keys = partition_keys_batch(vectors_f32)
    order = np.argsort(keys, kind='stable')
    sorted_vectors = vectors_f32[order]
    sorted_labels = labels[order]
    sorted_keys = keys[order]
    boundaries = np.searchsorted(sorted_keys, np.arange(N_PARTITIONS + 1)).astype(np.uint32)

    homogeneous_score = np.full(N_PARTITIONS, -1.0, dtype=np.float32)
    for k in range(N_PARTITIONS):
        start, end = int(boundaries[k]), int(boundaries[k + 1])
        if start == end:
            continue
        fraud_count = int(sorted_labels[start:end].sum())
        if fraud_count == 0:
            homogeneous_score[k] = 0.0
        elif fraud_count == (end - start):
            homogeneous_score[k] = 1.0

    # Synthetic mode is for parity tests only — collapse everything into 1 cluster
    # so the numpy IVF search degenerates to brute-force KNN over the small set.
    vec = np.ascontiguousarray(sorted_vectors, dtype=np.float32)
    centroids = vec.mean(axis=0, keepdims=True).astype(np.float32)
    centroid_norms = np.einsum('ij,ij->i', centroids, centroids).astype(np.float32)
    vec_norms = np.einsum('ij,ij->i', vec, vec).astype(np.float32)
    cluster_offsets = np.array([0, SYNTHETIC_N], dtype=np.int64)

    return PartitionedIndex(
        labels=sorted_labels,
        boundaries=boundaries,
        fallbacks=compute_fallbacks(boundaries),
        homogeneous_score=homogeneous_score,
        vectors=vec,
        vec_norms=vec_norms,
        cluster_labels=sorted_labels,
        centroids=centroids,
        centroid_norms=centroid_norms,
        cluster_offsets=cluster_offsets,
        ivf_nprobe=SYNTHETIC_NPROBE,
        vectors_int16=np.zeros((SYNTHETIC_N, 16), dtype=np.int16),
    )


def _from_disk(data_dir: Path) -> AppData:
    index = load_partitioned_index(data_dir / INDEX_SUBDIR)
    mcc_risk = load_mcc_risk(data_dir / MCC_RISK_FILENAME)
    logger.info(
        'mmap index loaded: {} labels, nprobe={}',
        len(index.labels),
        index.ivf_nprobe,
    )
    return AppData(mcc_risk=mcc_risk, index=index)


def _synthetic() -> AppData:
    logger.info('using synthetic index ({} vectors)', SYNTHETIC_N)
    return AppData(mcc_risk={}, index=_build_synthetic_index())


def build_app_data() -> AppData:
    data_dir = os.environ.get(ENV_DATA_DIR)
    if data_dir:
        return _from_disk(Path(data_dir))
    return _synthetic()
