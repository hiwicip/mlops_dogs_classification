FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/dogs_classification/bentoml.py .

RUN mkdir -p models

# COPY models/dog_classifier.onnx models/dog_classifier.onnx

RUN uv sync --frozen --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["uv", "run", "bentoml", "serve", "src.dogs_classification.bentoml:DogBreedClassificationService"]
