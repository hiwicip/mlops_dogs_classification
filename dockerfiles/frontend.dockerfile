FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --extra frontend --no-install-project --extra frontend

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen --extra frontend

EXPOSE 8501

CMD ["sh", "-c", "uv run streamlit run src/dogs_classification/frontend.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
