FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project --extra serve --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src/
COPY data/processed/classes.json data/processed/classes.json
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models

RUN uv sync --frozen --extra serve --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE $PORT

ENTRYPOINT ["sh", "-c", "uv run uvicorn dogs_classification.api:app --host 0.0.0.0 --port ${PORT}"]
