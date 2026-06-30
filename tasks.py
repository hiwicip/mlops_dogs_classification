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
def train(
    ctx: Context,
    config_path: str = "../../configs",
    config_name: str = "config.yaml",
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
) -> None:
    """Train model."""

    overrides = []
    if epochs is not None:
        overrides.append(f"training.epochs={epochs}")
    if lr is not None:
        overrides.append(f"training.lr={lr}")
    if batch_size is not None:
        overrides.append(f"training.batch_size={batch_size}")

    command = (
        f"uv run src/{PROJECT_NAME}/train.py "
        f"--config-path {config_path} "
        f"--config-name {config_name} "
        f"{' '.join(overrides)}"
    )

    ctx.run(
        command,
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluate(
    ctx: Context,
    config_path: str = "../../configs",
    config_name: str = "config.yaml",
) -> None:
    """Evaluate model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py " f"--config-path {config_path} " f"--config-name {config_name}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(
    ctx: Context,
    image_name: str,
    dockerfile: str = "dockerfiles/train.dockerfile",
    progress: str = "plain",
    platform: str = "linux/amd64",
    device: str = "cpu",
    push: bool = False,
) -> None:
    """Build docker image."""
    tag = image_name
    command = (
        f"docker build"
        f" --platform {platform}"
        f" -t {tag}"
        f" --build-arg DEVICE={device}"
        f" . -f {dockerfile}"
        f" --progress={progress}"
    )
    if push:
        command += " --push"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def docker_run(ctx: Context, image: str) -> None:
    """Run docker containers."""
    ctx.run(f"docker run -rm {image}", echo=True, pty=not WINDOWS)


@task
def bentoml(ctx: Context) -> None:
    """Run bentoml service."""
    ctx.run(
        "uv run bentoml serve src.dogs_classification.bentoml:DogBreedClassificationService", echo=True, pty=not WINDOWS
    )


@task
def data_drift(ctx: Context) -> None:
    """Run data drift detection."""
    ctx.run("uv run src/dogs_classification/data_drift.py", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
