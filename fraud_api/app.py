from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from loguru import logger
from starlette.applications import Starlette
from starlette.routing import Route

from fraud_api.handlers import fraud_score, ready
from fraud_api.search import K_NEIGHBORS
from fraud_api.state import AppData, build_app_data
from fraud_api.vectorize import VECTOR_DIM

routes = [
    Route('/ready', ready, methods=['GET']),
    Route('/fraud-score', fraud_score, methods=['POST']),
]


def _warmup(data: AppData) -> None:
    """Touch every Faiss index once to fault in mmap pages and prime internals."""
    dummy = np.zeros((1, VECTOR_DIM), dtype=np.float32)
    warmed = 0
    for idx in data.index.faiss_indices:
        if idx is not None:
            idx.search(dummy, K_NEIGHBORS)
            warmed += 1
    logger.info('warmup done: {} indices touched', warmed)


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    logger.info('lifespan startup')
    data = build_app_data()
    _warmup(data)
    app.state.data = data
    logger.info('lifespan ready, labels {}', data.index.labels.shape)
    yield
    logger.info('lifespan shutdown')


app = Starlette(routes=routes, lifespan=lifespan)
