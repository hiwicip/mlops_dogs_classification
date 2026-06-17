FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

EXPOSE 8000

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

EXPOSE $PORT

ENTRYPOINT ["sh", "-c", "uv run python -c \"from google.cloud import storage; storage.Client(project='mlopsdogclassification').bucket('mlops-dog-data-euwest4').blob('models/best_model.pt').download_to_filename('models/best_model.pt')\" && uvicorn src.dogs_classification.api:app --host 0.0.0.0 --port ${PORT}"]
