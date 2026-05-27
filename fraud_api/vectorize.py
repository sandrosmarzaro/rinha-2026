import datetime as dt
from typing import Final

import numpy as np

from fraud_api.schemas import FraudRequest

# Normalization constants — mirror of data/normalization.json
MAX_AMOUNT: Final = 10_000.0
MAX_INSTALLMENTS: Final = 12.0
AMOUNT_VS_AVG_RATIO: Final = 10.0
MAX_MINUTES: Final = 1440.0
MAX_KM: Final = 1000.0
MAX_TX_COUNT_24H: Final = 20.0
MAX_MERCHANT_AVG_AMOUNT: Final = 10_000.0

HOURS_DIVISOR: Final = 23.0
WEEKDAY_DIVISOR: Final = 6.0
SECONDS_PER_MINUTE: Final = 60.0
DEFAULT_MCC_RISK: Final = 0.5
MISSING_SENTINEL: Final = -1.0
VECTOR_DIM: Final = 14

QUANTIZATION_SCALE: Final = 10_000
QUANTIZATION_MIN: Final = -10_000
QUANTIZATION_MAX: Final = 10_000


def quantize(vector: np.ndarray) -> np.ndarray:
    """Convert a float32 14-dim vector to int16 by multiplying by QUANTIZATION_SCALE."""
    return (
        np.rint(vector * QUANTIZATION_SCALE)
        .clip(QUANTIZATION_MIN, QUANTIZATION_MAX)
        .astype(
            np.int16,
        )
    )


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def vectorize(req: FraudRequest, mcc_risk: dict[str, float]) -> np.ndarray:
    tx = req.transaction
    cust = req.customer
    merch = req.merchant
    term = req.terminal

    when = _parse_iso(tx.requested_at)
    avg = cust.avg_amount if cust.avg_amount > 0 else 1.0

    v = np.empty(VECTOR_DIM, dtype=np.float32)
    v[0] = _clamp_unit(tx.amount / MAX_AMOUNT)
    v[1] = _clamp_unit(tx.installments / MAX_INSTALLMENTS)
    v[2] = _clamp_unit((tx.amount / avg) / AMOUNT_VS_AVG_RATIO)
    v[3] = when.hour / HOURS_DIVISOR
    v[4] = when.weekday() / WEEKDAY_DIVISOR

    if req.last_transaction is None:
        v[5] = MISSING_SENTINEL
        v[6] = MISSING_SENTINEL
    else:
        last_at = _parse_iso(req.last_transaction.timestamp)
        minutes = (when - last_at).total_seconds() / SECONDS_PER_MINUTE
        v[5] = _clamp_unit(minutes / MAX_MINUTES)
        v[6] = _clamp_unit(req.last_transaction.km_from_current / MAX_KM)

    v[7] = _clamp_unit(term.km_from_home / MAX_KM)
    v[8] = _clamp_unit(cust.tx_count_24h / MAX_TX_COUNT_24H)
    v[9] = 1.0 if term.is_online else 0.0
    v[10] = 1.0 if term.card_present else 0.0
    v[11] = 0.0 if merch.id in cust.known_merchants else 1.0
    v[12] = mcc_risk.get(merch.mcc, DEFAULT_MCC_RISK)
    v[13] = _clamp_unit(merch.avg_amount / MAX_MERCHANT_AVG_AMOUNT)
    return v
