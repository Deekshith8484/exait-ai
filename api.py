"""FastAPI backend for EXRT readiness scoring.

Production-ready REST API for readiness predictions.
Supports batch ECG processing and real-time scoring.
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))

from analysis.models.inference import ReadinessModel

# Initialize FastAPI app
app = FastAPI(
    title="EXRT Readiness API",
    description="ECG-based readiness scoring system",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None


@app.on_event("startup")
def startup_event():
    """Load model on application startup."""
    global model
    try:
        model = ReadinessModel()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model not found. Run train_and_save_model.py first.")


# Request/Response models
class ReadinessRequest(BaseModel):
    """Request body for readiness prediction."""

    ecg: List[float]
    fs: int = 700
    window_sec: int = 90
    step_sec: int = 15


class WindowResult(BaseModel):
    """Single window prediction result."""

    window_idx: int
    center_time_sec: float
    readiness: float


class ReadinessSummary(BaseModel):
    """Summary statistics."""

    avg_readiness: float
    min_readiness: float
    max_readiness: float
    std_readiness: float
    n_windows: int


class ReadinessResponse(BaseModel):
    """Full readiness prediction response."""

    summary: ReadinessSummary
    window_results: List[WindowResult]
    overall_state: str


# API Endpoints


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/readiness", response_model=ReadinessResponse, tags=["Predictions"])
async def predict_readiness(request: ReadinessRequest):
    """
    Predict readiness scores for uploaded ECG data.

    **Parameters:**
    - `ecg`: Array of ECG samples
    - `fs`: Sampling frequency (default: 700 Hz)
    - `window_sec`: Window duration in seconds (default: 90)
    - `step_sec`: Slide step in seconds (default: 15)

    **Returns:**
    - Summary statistics (avg, min, max readiness)
    - Per-window predictions
    - Overall state classification
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.ecg or len(request.ecg) == 0:
        raise HTTPException(status_code=400, detail="ECG array is empty")

    if request.step_sec > request.window_sec:
        raise HTTPException(status_code=400, detail="step_sec must be ≤ window_sec")

    try:
        # Convert to numpy array
        ecg = np.array(request.ecg, dtype=np.float32)

        # Batch predict
        results_df = model.batch_predict_ecg(
            ecg,
            fs=request.fs,
            window_sec=request.window_sec,
            step_sec=request.step_sec,
        )

        if results_df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid windows produced. Check ECG quality or parameters.",
            )

        # Compute summary
        avg_readiness = float(results_df["readiness"].mean())
        min_readiness = float(results_df["readiness"].min())
        max_readiness = float(results_df["readiness"].max())
        std_readiness = float(results_df["readiness"].std())

        # Classify overall state
        if avg_readiness >= 70:
            overall_state = "ready"
        elif avg_readiness >= 40:
            overall_state = "neutral"
        else:
            overall_state = "fatigued"

        # Format window results
        window_results = [
            WindowResult(
                window_idx=int(row["window_idx"]),
                center_time_sec=float(row["center_time_sec"]),
                readiness=float(row["readiness"]),
            )
            for _, row in results_df.iterrows()
        ]

        return ReadinessResponse(
            summary=ReadinessSummary(
                avg_readiness=avg_readiness,
                min_readiness=min_readiness,
                max_readiness=max_readiness,
                std_readiness=std_readiness,
                n_windows=len(results_df),
            ),
            window_results=window_results,
            overall_state=overall_state,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/readiness/features", tags=["Predictions"])
async def predict_from_features(features: dict):
    """
    Predict readiness directly from HRV features dictionary.

    **Example features:**
    ```json
    {
      "RMSSD": 45.2,
      "SDNN": 35.1,
      "pNN50": 15.3,
      "HR_est": 62.0,
      "SQI": 0.95,
      ...
    }
    ```

    **Returns:**
    - Readiness score (0-100)
    - State classification
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        readiness = model.predict_from_features(features)
        state = "ready" if readiness >= 70 else "neutral" if readiness >= 40 else "fatigued"
        return {"readiness": float(readiness), "state": state}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Feature error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "pca_components": int(model.pca.n_components_),
        "gmm_components": model.gmm.n_components,
        "ready_cluster": model.ready_cluster,
        "n_features": len(model.feature_cols),
        "feature_names": model.feature_cols,
    }


@app.get("/", tags=["Info"])
async def root():
    """API root with documentation links."""
    return {
        "name": "EXRT Readiness API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /readiness": "Predict readiness from ECG array",
            "POST /readiness/features": "Predict from HRV features",
            "GET /model/info": "Get model information",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
