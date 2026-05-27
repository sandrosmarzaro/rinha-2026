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
    data: AppData = request.app.state.data
    body = await request.body()
    payload = _DECODER.decode(body)
    q = vectorize(payload, data.mcc_risk)
    key = partition_key(q)
    score = partitioned_score(q, key, data.index)
    return _RESPONSES[round(score * K_NEIGHBORS)]
