FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

EXPOSE 8000

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models

RUN uv sync --frozen --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE $PORT

ENTRYPOINT ["sh", "-c", "uv run python -c \"from google.cloud import storage; storage.Client(project='mlopsdogclassification').bucket('mlops-dog-data-euwest4').blob('models/best_model.pt').download_to_filename('models/best_model.pt')\" && uvicorn src.dogs_classification.api:app --host 0.0.0.0 --port ${PORT}"]
