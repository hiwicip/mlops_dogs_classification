ARG DEVICE=cpu

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

ARG DEVICE
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${DEVICE}

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/dogs_classification/train.py"]
