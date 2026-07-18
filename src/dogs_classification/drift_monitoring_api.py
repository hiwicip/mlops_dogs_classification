import anyio
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from dogs_classification.data_drift import (
    BUCKET_NAME,
    PREDICTIONS_PREFIX,
    build_reference_df,
    download_predictions_from_gcs,
    run_analysis,
)


def lifespan(app: FastAPI):
    """Load the reference data before the application starts."""
    global reference_data
    reference_data = build_reference_df()

    yield

    del reference_data


reference_data: pd.DataFrame | None = None

app = FastAPI(lifespan=lifespan)


def load_latest_files(n: int) -> pd.DataFrame:
    """Fetch the latest n prediction records from the GCP bucket."""
    return download_predictions_from_gcs(BUCKET_NAME, PREDICTIONS_PREFIX, n=n)


@app.get("/report")
async def get_report(n: int = 10):
    """Generate and return the drift report."""
    if reference_data is None:
        return HTMLResponse(content="<h1>Reference data not loaded.</h1>", status_code=503)

    prediction_data = load_latest_files(n=n)
    if prediction_data.empty:
        return HTMLResponse(content="<h1>No predictions found in the bucket.</h1>", status_code=200)

    local_path = run_analysis(reference_data.copy(), prediction_data)

    async with await anyio.open_file(local_path, encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
