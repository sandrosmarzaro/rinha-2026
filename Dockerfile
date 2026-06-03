FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.21 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY fraud_api/ /app/fraud_api/
COPY scripts/ /app/scripts/
COPY data/references.json.gz data/mcc_risk.json data/normalization.json /app/data/

RUN uv run python scripts/build_index.py

# Pre-compile numba JIT kernels so first request isn't blocked by compilation
RUN RINHA_DATA_DIR=/app/data uv run python -c "from fraud_api.state import build_app_data; from fraud_api.search import partitioned_score; import numpy as np; d=build_app_data(); partitioned_score(np.zeros(14, dtype=np.float32), 0, d.index)"


FROM python:3.14-slim AS production

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    RINHA_DATA_DIR=/app/data \
    RINHA_SOCKET=/tmp/sockets/api.sock \
    PYTHONOPTIMIZE=2

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
