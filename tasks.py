import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "dogs_classification"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context, data_path: str = "data/raw/Images", output_folder: str = "data/processed") -> None:
    """Preprocess data."""
    cmd = f"uv run src/{PROJECT_NAME}/data.py --data-path {data_path} --output-folder {output_folder}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, config_path: str = "../../configs", config_name: str = "config.yaml") -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py --config-path {config_path} --config-name {config_name}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
