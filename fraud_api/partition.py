from typing import Final

import numpy as np

N_PARTITIONS: Final = 256

# Bucket cuts on the already-normalized vector space (see fraud_api/vectorize.py).
# Each cut array is sorted; bucket index = np.searchsorted(cuts, value, side='right').
# Amount: cuts at 50, 200, 1000 BRL → vector space (amount/10000): 0.005, 0.020, 0.100.
AMOUNT_CUTS: Final = np.array([0.005, 0.020, 0.100], dtype=np.float32)
# mcc_risk lives at v[12] already in [0, 1]; cuts pick 8 buckets across the band.
MCC_CUTS: Final = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=np.float32)

DIM_AMOUNT: Final = 0
DIM_IS_ONLINE: Final = 9
DIM_CARD_PRESENT: Final = 10
DIM_UNKNOWN_MERCHANT: Final = 11
DIM_MCC_RISK: Final = 12

SHIFT_CARD_PRESENT: Final = 1
SHIFT_UNKNOWN_MERCHANT: Final = 2
SHIFT_AMOUNT: Final = 3
SHIFT_MCC: Final = 5


def partition_key(vector: np.ndarray) -> int:  # noqa: C901
    """Map a 14-dim normalized vector to a partition id in [0, 256).

    Hot path: inline if/elif buckets (avoids numpy.searchsorted's call overhead on
    tiny 3- and 7-element arrays, which dominated this function in profiling).
    Thresholds inline as numeric literals for branch-prediction friendliness; they
    match the AMOUNT_CUTS / MCC_CUTS arrays used in the batch helper below.
    """
    amount = float(vector[DIM_AMOUNT])
    if amount < 0.005:  # noqa: PLR2004
        amount_bucket = 0
    elif amount < 0.020:  # noqa: PLR2004
        amount_bucket = 1
    elif amount < 0.100:  # noqa: PLR2004
        amount_bucket = 2
    else:
        amount_bucket = 3
    mcc = float(vector[DIM_MCC_RISK])
    if mcc < 0.1:  # noqa: PLR2004
        mcc_bucket = 0
    elif mcc < 0.2:  # noqa: PLR2004
        mcc_bucket = 1
    elif mcc < 0.3:  # noqa: PLR2004
        mcc_bucket = 2
    elif mcc < 0.4:  # noqa: PLR2004
        mcc_bucket = 3
    elif mcc < 0.5:  # noqa: PLR2004
        mcc_bucket = 4
    elif mcc < 0.6:  # noqa: PLR2004
        mcc_bucket = 5
    elif mcc < 0.8:  # noqa: PLR2004
        mcc_bucket = 6
    else:
        mcc_bucket = 7
    return (
        int(vector[DIM_IS_ONLINE])
        | (int(vector[DIM_CARD_PRESENT]) << SHIFT_CARD_PRESENT)
        | (int(vector[DIM_UNKNOWN_MERCHANT]) << SHIFT_UNKNOWN_MERCHANT)
        | (amount_bucket << SHIFT_AMOUNT)
        | (mcc_bucket << SHIFT_MCC)
    )


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_fallbacks(boundaries: np.ndarray) -> np.ndarray:
    """For each empty partition, return the nearest non-empty by Hamming distance.

    Ties broken by partition size (prefer larger for better KNN coverage).
    """
    counts = np.diff(boundaries)
    non_empty = np.flatnonzero(counts > 0)
    fallbacks = np.arange(N_PARTITIONS, dtype=np.uint8)
    if len(non_empty) == 0:
        return fallbacks
    for k in range(N_PARTITIONS):
        if counts[k] > 0:
            continue
        distances = np.array([_hamming_distance(k, int(ne)) for ne in non_empty])
        order = np.lexsort((-counts[non_empty], distances))
        fallbacks[k] = int(non_empty[order[0]])
    return fallbacks


def partition_keys_batch(vectors: np.ndarray) -> np.ndarray:
    """Vectorized partition_key for the full dataset (build-time)."""
    is_online = vectors[:, DIM_IS_ONLINE].astype(np.uint16)
    card_present = vectors[:, DIM_CARD_PRESENT].astype(np.uint16)
    unknown_merchant = vectors[:, DIM_UNKNOWN_MERCHANT].astype(np.uint16)
    amount_bucket = np.searchsorted(AMOUNT_CUTS, vectors[:, DIM_AMOUNT], side='right').astype(
        np.uint16,
    )
    mcc_bucket = np.searchsorted(MCC_CUTS, vectors[:, DIM_MCC_RISK], side='right').astype(
        np.uint16,
    )
    return (
        is_online
        | (card_present << SHIFT_CARD_PRESENT)
        | (unknown_merchant << SHIFT_UNKNOWN_MERCHANT)
        | (amount_bucket << SHIFT_AMOUNT)
        | (mcc_bucket << SHIFT_MCC)
    )
