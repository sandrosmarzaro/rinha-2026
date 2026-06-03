"""Native Granian RSGI app — bypasses Starlette/ASGI overhead.

RSGI is Granian's native protocol: scope is a struct, protocol gives bytes-in/bytes-out
without the ASGI message-loop layer. Saves a function-call layer + ASGI dict allocs
per request.
"""

from typing import Final

import msgspec
import numpy as np
from loguru import logger

from fraud_api import profile
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
_PROFILE_HEADERS: Final = [('content-type', 'application/json')]


class FraudApp:
    __slots__ = ('data',)

    def __init__(self) -> None:
        self.data = build_app_data()
        dummy = np.zeros(VECTOR_DIM, dtype=np.float32)
        # Warm caches: pre-touch the centroid arrays and the vectors mmap.
        partitioned_score(dummy, 0, self.data.index)
        logger.info('rsgi app ready (profile={})', profile.ENABLED)

    async def __rsgi__(self, scope, proto) -> None:  # noqa: PLR0915
        if scope.path == '/ready':
            proto.response_empty(200, [])
            return
        if scope.path == '/profile':
            proto.response_bytes(200, _PROFILE_HEADERS, _ENCODER.encode(profile.summary()))
            return
        t0 = profile.now_ns()
        try:
            body = await proto()
            t1 = profile.now_ns()
            payload = _DECODER.decode(body)
            t2 = profile.now_ns()
            amount = payload.transaction.amount
            avg = payload.customer.avg_amount
            if avg > 0 and amount / avg <= LEGIT_AMT_RATIO_THRESHOLD:
                t3 = profile.now_ns()
                proto.response_bytes(200, _HEADERS, _BODIES[0])
                t_end = profile.now_ns()
                profile.mark_path('fast_legit')
                profile.record('body_read', t1 - t0)
                profile.record('decode', t2 - t1)
                profile.record('fast_path_check', t3 - t2)
                profile.record('response', t_end - t3)
                profile.record('total', t_end - t0)
                return
            if amount > FRAUD_AMOUNT_THRESHOLD:
                t3 = profile.now_ns()
                proto.response_bytes(200, _HEADERS, _BODIES[K_NEIGHBORS])
                t_end = profile.now_ns()
                profile.mark_path('fast_fraud')
                profile.record('body_read', t1 - t0)
                profile.record('decode', t2 - t1)
                profile.record('fast_path_check', t3 - t2)
                profile.record('response', t_end - t3)
                profile.record('total', t_end - t0)
                return
            t3 = profile.now_ns()
            q = vectorize(payload, self.data.mcc_risk)
            t4 = profile.now_ns()
            key = partition_key(q)
            t5 = profile.now_ns()
            score = partitioned_score(q, key, self.data.index)
            t6 = profile.now_ns()
            proto.response_bytes(200, _HEADERS, _BODIES[round(score * K_NEIGHBORS)])
            t_end = profile.now_ns()
            profile.mark_path('boundary')
            profile.record('body_read', t1 - t0)
            profile.record('decode', t2 - t1)
            profile.record('fast_path_check', t3 - t2)
            profile.record('vectorize', t4 - t3)
            profile.record('partition_key', t5 - t4)
            profile.record('faiss_search', t6 - t5)
            profile.record('response', t_end - t6)
            profile.record('total', t_end - t0)
        except Exception:  # noqa: BLE001
            profile.mark_path('error')
            proto.response_bytes(200, _HEADERS, _BODIES[K_NEIGHBORS])


app = FraudApp()
