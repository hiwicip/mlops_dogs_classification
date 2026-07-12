FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN mkdir -p data

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --extra train --no-install-project --extra train

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen --extra train

ENTRYPOINT ["uv", "run", "python", "src/dogs_classification/data.py"]
