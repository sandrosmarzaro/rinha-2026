"""Lightweight in-process profiler for the request hot path.

Per-stage timing is recorded into fixed-size numpy ring buffers; reading /profile
computes percentiles. Overhead per call is ~50ns * #stages -> ~300ns/request,
negligible at 900 RPS.

Enabled when RINHA_PROFILE=1. When disabled, all functions are no-ops (zero cost).
"""

import os
from time import perf_counter_ns
from typing import Final

import numpy as np

ENABLED: Final = os.environ.get('RINHA_PROFILE') == '1'
BUFFER_SIZE: Final = 100_000

# Stage names — fixed order, must match what the handler records
STAGES: Final = (
    'body_read',
    'decode',
    'fast_path_check',
    'vectorize',
    'partition_key',
    'faiss_search',
    'response',
    'total',
)

# Per-stage ring buffers (ns durations) + per-bucket counters
_buffers: dict[str, np.ndarray] = {s: np.zeros(BUFFER_SIZE, dtype=np.int64) for s in STAGES}
_idx: dict[str, int] = dict.fromkeys(STAGES, 0)
_count: dict[str, int] = dict.fromkeys(STAGES, 0)

# Per-path counters (fast-path-legit, fast-path-fraud, boundary, error)
_path_counts: dict[str, int] = {
    'fast_legit': 0,
    'fast_fraud': 0,
    'boundary': 0,
    'error': 0,
}


def record(stage: str, duration_ns: int) -> None:
    """Record a stage timing. Wraps the ring buffer index."""
    if not ENABLED:
        return
    i = _idx[stage]
    _buffers[stage][i] = duration_ns
    _idx[stage] = (i + 1) % BUFFER_SIZE
    _count[stage] += 1


def mark_path(path: str) -> None:
    if not ENABLED:
        return
    _path_counts[path] = _path_counts.get(path, 0) + 1


def now_ns() -> int:
    return perf_counter_ns()


def summary() -> dict:
    """Compute per-stage p50/p90/p99/max/avg from the ring buffers."""
    out: dict = {'enabled': ENABLED, 'paths': dict(_path_counts), 'stages': {}}
    for stage in STAGES:
        cnt = _count[stage]
        if cnt == 0:
            continue
        # Only use the filled portion of the buffer
        n_filled = min(cnt, BUFFER_SIZE)
        data = _buffers[stage][:n_filled]
        out['stages'][stage] = {
            'count': cnt,
            'avg_us': float(data.mean() / 1000),
            'p50_us': float(np.percentile(data, 50) / 1000),
            'p90_us': float(np.percentile(data, 90) / 1000),
            'p99_us': float(np.percentile(data, 99) / 1000),
            'max_us': float(data.max() / 1000),
        }
    return out
