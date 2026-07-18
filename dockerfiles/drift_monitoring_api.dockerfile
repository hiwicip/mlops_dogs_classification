FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project --extra serve --extra cloud --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src/
COPY data/processed/classes.json data/processed/classes.json
COPY README.md README.md
COPY LICENSE LICENSE

COPY .dvc .dvc/
COPY data/processed.dvc data/processed.dvc

RUN mkdir -p models

RUN uv sync --frozen --extra serve --extra cloud --extra-index-url https://download.pytorch.org/whl/cpu

RUN uv run dvc pull data/processed

EXPOSE $PORT

CMD exec uv run uvicorn dogs_classification.drift_monitoring_api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
