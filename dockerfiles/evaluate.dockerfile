ARG DEVICE=cu121

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src/ src/
COPY configs/ configs/
COPY .dvc/config .dvc/config
COPY data/processed.dvc data/processed.dvc

ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.13
ARG DEVICE
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${DEVICE}
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
RUN mkdir -p models data

#For pulling data and best model from the cloud
ENTRYPOINT ["sh", "-c", "uv run python -c \"from google.cloud import storage; storage.Client(project='mlopsdogclassification').bucket('mlops-dog-data-euwest4').blob('models/best_model.pt').download_to_filename('models/best_model.pt')\" && uv run dvc pull && uv run src/dogs_classification/evaluate.py"]
#ENTRYPOINT ["uv", "run", "src/dogs_classification/evaluate.py"]
