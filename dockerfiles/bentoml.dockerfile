FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project --extra serve --extra cloud --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models data/processed
COPY data/processed/classes.json data/processed/classes.json

COPY models/dog_classifier.onnx models/dog_classifier.onnx
COPY models/dog_classifier.onnx.data models/dog_classifier.onnx.data

RUN uv sync --frozen --extra serve --extra cloud --extra-index-url https://download.pytorch.org/whl/cpu

RUN uv run python -c "from transformers import AutoImageProcessor; AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')"

CMD ["uv", "run", "bentoml", "serve", "src.dogs_classification.bentoml:DogBreedClassificationService"]
