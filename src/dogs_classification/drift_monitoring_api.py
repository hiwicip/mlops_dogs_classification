from contextlib import asynccontextmanager

import anyio
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from dogs_classification.data_drift import (
    BUCKET_NAME,
    PREDICTIONS_PREFIX,
    download_predictions_from_gcs,
    load_or_build_reference_report_df,
    run_analysis,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load (or compute and cache) the reference report before the application starts."""
    global reference_report_df
    reference_report_df = load_or_build_reference_report_df()

    yield

    del reference_report_df


reference_report_df: pd.DataFrame | None = None

app = FastAPI(lifespan=lifespan)


def load_latest_files(n: int) -> pd.DataFrame:
    """Fetch the latest n prediction records from the GCP bucket."""
    return download_predictions_from_gcs(BUCKET_NAME, PREDICTIONS_PREFIX, n=n)


@app.get("/report")
async def get_report(n: int = 10):
    """Generate and return the drift report."""
    ref_df = reference_report_df
    if ref_df is None:
        return HTMLResponse(content="<h1>Reference data not loaded.</h1>", status_code=503)

    prediction_data = load_latest_files(n=n)
    if prediction_data.empty:
        return HTMLResponse(content="<h1>No predictions found in the bucket.</h1>", status_code=200)

    local_path = run_analysis(ref_df, prediction_data)

    async with await anyio.open_file(local_path, encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
