ARG DEVICE=cu121

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG DEVICE
ENV UV_PYTHON=3.13
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${DEVICE}

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY .dvc .dvc/
COPY .git .git
COPY data/processed.dvc data/processed.dvc
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["sh", "-c", "uv run dvc pull && uv run src/dogs_classification/train.py"]
