FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

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
