FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models data/processed
COPY data/processed/classes.json data/processed/classes.json

RUN uv run python -c "from google.cloud import storage; storage.Client(project='mlopsdogclassification').bucket('mlops-dog-data-euwest4').blob('models/dog_classifier.onnx').download_to_filename('models/dog_classifier.onnx')"

RUN uv sync --frozen --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["uv", "run", "bentoml", "serve", "src.dogs_classification.bentoml:DogBreedClassificationService"]
