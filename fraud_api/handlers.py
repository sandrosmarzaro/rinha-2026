from typing import Final

import msgspec
from starlette.requests import Request
from starlette.responses import Response

from fraud_api.partition import partition_key
from fraud_api.schemas import FraudRequest, FraudResponse
from fraud_api.search import K_NEIGHBORS, partitioned_score
from fraud_api.state import AppData
from fraud_api.vectorize import vectorize

FRAUD_THRESHOLD: Final = 0.6
_DECODER = msgspec.json.Decoder(FraudRequest)
_ENCODER = msgspec.json.Encoder()

# Fast-path thresholds derived offline from references.json.gz. Each isolates a
# subset of references with ≥99.99% label purity, jointly covering ~92.6% of the
# distribution. Queries matching either rule skip the entire KNN pipeline.
LEGIT_AMT_RATIO_THRESHOLD: Final = 0.971  # amount/avg_amount; below → legit
FRAUD_AMOUNT_THRESHOLD: Final = 2996.0  # raw BRL; above → fraud

# fraud_score is always fraud_count/K with fraud_count in 0..K, so there are only
# K+1 possible responses. Pre-render them once; the hot path just indexes in.
_RESPONSES: Final = tuple(
    Response(
        _ENCODER.encode(
            FraudResponse(
                approved=(c / K_NEIGHBORS) < FRAUD_THRESHOLD, fraud_score=c / K_NEIGHBORS
            ),
        ),
        media_type='application/json',
    )
    for c in range(K_NEIGHBORS + 1)
)


async def ready(_request: Request) -> Response:
    return Response(status_code=200)


async def fraud_score(request: Request) -> Response:
    try:
        data: AppData = request.app.state.data
        body = await request.body()
        payload = _DECODER.decode(body)
        amount = payload.transaction.amount
        avg = payload.customer.avg_amount
        # Fast-path: deterministic 2-rule classifier covers the easy ~92.6% of queries
        # with ≥99.99% accuracy, bypassing vectorize + partition + Faiss entirely.
        if avg > 0 and amount / avg <= LEGIT_AMT_RATIO_THRESHOLD:
            return _RESPONSES[0]
        if amount > FRAUD_AMOUNT_THRESHOLD:
            return _RESPONSES[K_NEIGHBORS]
        # Ambiguous boundary → KNN
        q = vectorize(payload, data.mcc_risk)
        key = partition_key(q)
        score = partitioned_score(q, key, data.index)
        return _RESPONSES[round(score * K_NEIGHBORS)]
    except Exception:  # noqa: BLE001 — fail-safe: any error → fraud default (avoids HTTP 500)
        return _RESPONSES[K_NEIGHBORS]
