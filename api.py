"""FastAPI backend for EXRT readiness scoring - Full working version with upload."""

import io
import sys
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))

from analysis.models.inference import ReadinessModel

# Initialize FastAPI app
app = FastAPI(
    title="EXRT Readiness API",
    description="ECG-based readiness scoring system with file upload support",
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
    filename: Optional[str] = None
    samples: Optional[int] = None
    duration_sec: Optional[float] = None
    sampling_rate: Optional[int] = None
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


@app.post("/upload/signal", tags=["Upload"])
async def upload_signal(file: UploadFile = File(...)):
    """
    Upload a signal file (.csv, .pkl, .json) for readiness analysis.
    
    **Supported formats:**
    - `.csv`: Must have 'ecg' column or single column of ECG values
    - `.pkl`: Pickled numpy array or dict with 'ecg' key
    - `.json`: JSON array or object with 'ecg' key
    
    **Returns:**
    - Readiness analysis results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    filename = file.filename.lower()
    content = await file.read()
    
    print(f"📁 Received file: {filename} ({len(content)} bytes)")
    
    try:
        ecg_data = None
        fs = 700  # Default sampling frequency
        
        # Parse based on file extension
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            # Try to find ECG column
            if 'ecg' in df.columns:
                ecg_data = df['ecg'].values
            elif 'ECG' in df.columns:
                ecg_data = df['ECG'].values
            elif 'signal' in df.columns:
                ecg_data = df['signal'].values
            elif len(df.columns) == 1:
                ecg_data = df.iloc[:, 0].values
            else:
                # Try first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ecg_data = df[numeric_cols[0]].values
                else:
                    raise ValueError("Could not find ECG data in CSV file")
            
            # Check for fs column
            if 'fs' in df.columns:
                fs = int(df['fs'].iloc[0])
                
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            data = pickle.loads(content)
            if isinstance(data, np.ndarray):
                ecg_data = data.flatten()
            elif isinstance(data, dict):
                if 'ecg' in data:
                    ecg_data = np.array(data['ecg']).flatten()
                elif 'signal' in data:
                    ecg_data = np.array(data['signal']).flatten()
                else:
                    # Try first array-like value
                    for v in data.values():
                        if isinstance(v, (list, np.ndarray)):
                            ecg_data = np.array(v).flatten()
                            break
                if 'fs' in data:
                    fs = int(data['fs'])
            elif isinstance(data, list):
                ecg_data = np.array(data).flatten()
            else:
                raise ValueError("Unsupported pickle format")
                
        elif filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                ecg_data = np.array(data).flatten()
            elif isinstance(data, dict):
                if 'ecg' in data:
                    ecg_data = np.array(data['ecg']).flatten()
                elif 'signal' in data:
                    ecg_data = np.array(data['signal']).flatten()
                else:
                    raise ValueError("JSON object must have 'ecg' or 'signal' key")
                if 'fs' in data:
                    fs = int(data['fs'])
            else:
                raise ValueError("Unsupported JSON format")
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Use .csv, .pkl, or .json"
            )
        
        if ecg_data is None or len(ecg_data) == 0:
            raise HTTPException(status_code=400, detail="No valid ECG data found in file")
        
        # Convert to float32
        ecg_data = ecg_data.astype(np.float32)
        
        # Run analysis
        results_df = model.batch_predict_ecg(
            ecg_data,
            fs=fs,
            window_sec=90,
            step_sec=15,
        )
        
        if results_df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid windows produced. Check signal quality or length.",
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
            {
                "window_idx": int(row["window_idx"]),
                "center_time_sec": float(row["center_time_sec"]),
                "readiness": float(row["readiness"]),
            }
            for _, row in results_df.iterrows()
        ]
        
        return {
            "filename": file.filename,
            "samples": len(ecg_data),
            "duration_sec": float(len(ecg_data) / fs),
            "sampling_rate": fs,
            "summary": {
                "avg_readiness": round(avg_readiness, 2),
                "min_readiness": round(min_readiness, 2),
                "max_readiness": round(max_readiness, 2),
                "std_readiness": round(std_readiness, 2),
                "n_windows": len(results_df),
            },
            "overall_state": overall_state,
            "window_results": window_results,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Error processing file: {error_msg}")
        import traceback
        import sys as _sys
        traceback.print_exc(file=_sys.stderr)
        raise HTTPException(status_code=400, detail=f"Error processing file: {error_msg}")


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


@app.get("/", tags=["Info"])
async def root():
    """API root with documentation links."""
    return {
        "name": "EXRT Readiness API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /upload/signal": "Upload and analyze ECG file",
            "POST /readiness": "Predict readiness from ECG array",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
