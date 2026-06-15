FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

EXPOSE 8000

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY models models/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.dogs_classification.api:app", "--host", "0.0.0.0", "--port", "$PORT"]
