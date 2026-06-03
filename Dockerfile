## Stage 1: build the Rust SIMD wheel
FROM rust:1.89-slim AS rust_builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Build with -march=haswell so the AVX2/FMA paths are unconditionally inlined.
ENV RUSTFLAGS="-C target-cpu=haswell"

WORKDIR /build
COPY knn_simd /build/knn_simd

RUN python3 -m venv /opt/maturin-venv && \
    /opt/maturin-venv/bin/pip install --no-cache-dir maturin==1.13.3
RUN cd knn_simd && /opt/maturin-venv/bin/maturin build --release --out /wheels


## Stage 2: prepare Python venv + index
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

# Install the Rust extension wheel into the venv.
COPY --from=rust_builder /wheels /wheels
RUN uv pip install /wheels/knn_simd-*.whl

COPY fraud_api/ /app/fraud_api/
COPY scripts/ /app/scripts/
COPY data/references.json.gz data/mcc_risk.json data/normalization.json /app/data/

RUN uv run python scripts/build_index.py


## Stage 3: minimal production
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
