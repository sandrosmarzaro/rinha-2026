import os
from pathlib import Path

import numpy as np
import pytest

from fraud_api.index import load_references
from fraud_api.partition import partition_key
from fraud_api.search import K_NEIGHBORS, brute_force_score, partitioned_score
from fraud_api.state import AppData, build_app_data

PARITY_SAMPLES = 1000
DISAGREEMENT_THRESHOLD = 0.01
FRAUD_THRESHOLD = 0.6
RNG_SEED = 42
MIN_VECTORS_FOR_PARITY = 100_000


@pytest.fixture(scope='module')
def data() -> AppData:
    return build_app_data()


@pytest.fixture(scope='module')
def oracle() -> tuple[np.ndarray, np.ndarray]:
    data_dir = os.environ.get('RINHA_DATA_DIR')
    if not data_dir:
        return np.empty((0, 14), dtype=np.float32), np.empty(0, dtype=bool)
    return load_references(Path(data_dir) / 'references.json.gz')


def test_partitioned_matches_brute_force(
    data: AppData,
    oracle: tuple[np.ndarray, np.ndarray],
) -> None:
    oracle_vectors, oracle_labels = oracle
    n = len(oracle_vectors)
    if n < MIN_VECTORS_FOR_PARITY:
        pytest.skip(f'parity needs >={MIN_VECTORS_FOR_PARITY} vectors; got {n}')
    n_samples = min(PARITY_SAMPLES, n)
    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(n, size=n_samples, replace=False)

    disagreements = 0
    for i in sample_idx:
        q_f32 = oracle_vectors[i]
        oracle_score = brute_force_score(
            q_f32,
            oracle_vectors,
            oracle_labels,
            k=K_NEIGHBORS,
        )
        actual_score = partitioned_score(
            q_f32,
            partition_key(q_f32),
            data.index,
            k=K_NEIGHBORS,
        )
        if (oracle_score < FRAUD_THRESHOLD) != (actual_score < FRAUD_THRESHOLD):
            disagreements += 1

    rate = disagreements / n_samples
    assert rate < DISAGREEMENT_THRESHOLD, (
        f'{disagreements}/{n_samples} disagreed ({rate * 100:.2f}%)'
    )
