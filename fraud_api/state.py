import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import faiss
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
    bbox_min = np.full((N_PARTITIONS, VECTOR_DIM), np.inf, dtype=np.float32)
    bbox_max = np.full((N_PARTITIONS, VECTOR_DIM), -np.inf, dtype=np.float32)
    faiss_indices: list[faiss.Index | None] = [None] * N_PARTITIONS
    for k in range(N_PARTITIONS):
        start, end = int(boundaries[k]), int(boundaries[k + 1])
        if start == end:
            continue
        block = sorted_vectors[start:end]
        bbox_min[k] = block.min(axis=0)
        bbox_max[k] = block.max(axis=0)
        fraud_count = int(sorted_labels[start:end].sum())
        if fraud_count == 0:
            homogeneous_score[k] = 0.0
            continue
        if fraud_count == (end - start):
            homogeneous_score[k] = 1.0
            continue
        idx = faiss.IndexFlatL2(VECTOR_DIM)
        idx.add(np.ascontiguousarray(block, dtype=np.float32))
        faiss_indices[k] = idx

    return PartitionedIndex(
        labels=sorted_labels,
        boundaries=boundaries,
        fallbacks=compute_fallbacks(boundaries),
        homogeneous_score=homogeneous_score,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        faiss_indices=tuple(faiss_indices),
        ivf_nprobe=SYNTHETIC_NPROBE,
    )


def _from_disk(data_dir: Path) -> AppData:
    index = load_partitioned_index(data_dir / INDEX_SUBDIR)
    mcc_risk = load_mcc_risk(data_dir / MCC_RISK_FILENAME)
    n_built = sum(1 for i in index.faiss_indices if i is not None)
    logger.info(
        'mmap index loaded: {} faiss partitions, {} labels, nprobe={}',
        n_built,
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
