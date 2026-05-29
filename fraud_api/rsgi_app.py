"""Native Granian RSGI app — bypasses Starlette/ASGI overhead.

RSGI is Granian's native protocol: scope is a struct, protocol gives bytes-in/bytes-out
without the ASGI message-loop layer. Saves a function-call layer + ASGI dict allocs
per request.
"""

from typing import Final

import msgspec
import numpy as np
from loguru import logger

from fraud_api.partition import partition_key
from fraud_api.schemas import FraudRequest, FraudResponse
from fraud_api.search import K_NEIGHBORS, partitioned_score
from fraud_api.state import build_app_data
from fraud_api.vectorize import VECTOR_DIM, vectorize

FRAUD_THRESHOLD: Final = 0.6
LEGIT_AMT_RATIO_THRESHOLD: Final = 0.971
FRAUD_AMOUNT_THRESHOLD: Final = 2996.0

_DECODER = msgspec.json.Decoder(FraudRequest)
_ENCODER = msgspec.json.Encoder()

# Pre-rendered response bodies. Only K+1 possible scores → K+1 fixed payloads.
_BODIES: Final = tuple(
    _ENCODER.encode(
        FraudResponse(approved=(c / K_NEIGHBORS) < FRAUD_THRESHOLD, fraud_score=c / K_NEIGHBORS),
    )
    for c in range(K_NEIGHBORS + 1)
)
_HEADERS: Final = [('content-type', 'application/json')]


class FraudApp:
    __slots__ = ('data',)

    def __init__(self) -> None:
        self.data = build_app_data()
        dummy = np.zeros((1, VECTOR_DIM), dtype=np.float32)
        warmed = 0
        for idx in self.data.index.faiss_indices:
            if idx is not None:
                idx.search(dummy, K_NEIGHBORS)
                warmed += 1
        logger.info('rsgi app ready: {} indices warmed', warmed)

    async def __rsgi__(self, scope, proto) -> None:
        if scope.path == '/ready':
            proto.response_empty(200, [])
            return
        try:
            body = await proto()
            payload = _DECODER.decode(body)
            amount = payload.transaction.amount
            avg = payload.customer.avg_amount
            if avg > 0 and amount / avg <= LEGIT_AMT_RATIO_THRESHOLD:
                proto.response_bytes(200, _HEADERS, _BODIES[0])
                return
            if amount > FRAUD_AMOUNT_THRESHOLD:
                proto.response_bytes(200, _HEADERS, _BODIES[K_NEIGHBORS])
                return
            q = vectorize(payload, self.data.mcc_risk)
            key = partition_key(q)
            score = partitioned_score(q, key, self.data.index)
            proto.response_bytes(200, _HEADERS, _BODIES[round(score * K_NEIGHBORS)])
        except Exception:  # noqa: BLE001
            proto.response_bytes(200, _HEADERS, _BODIES[K_NEIGHBORS])


app = FraudApp()
