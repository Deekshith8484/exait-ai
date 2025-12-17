#!/usr/bin/env python3
"""Minimal FastAPI server for testing."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None

@app.on_event("startup")
def startup():
    global model
    try:
        from analysis.models.inference import ReadinessModel
        model = ReadinessModel()
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/gemini-key")
def get_gemini_key():
    """Get Gemini API key from environment."""
    api_key = os.getenv("Gemini_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    return {"api_key": api_key}

@app.get("/")
def root():
    return {"message": "EXRT Backend"}

@app.post("/upload/signal")
async def upload_signal(file: UploadFile = File(...)):
    """Upload and analyze ECG signal file."""
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
            if 'ecg' in df.columns:
                ecg_data = df['ecg'].values
            elif 'ECG' in df.columns:
                ecg_data = df['ECG'].values
            elif len(df.columns) == 1:
                ecg_data = df.iloc[:, 0].values
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ecg_data = df[numeric_cols[0]].values
                    
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            data = pickle.loads(content)
            if isinstance(data, np.ndarray):
                ecg_data = data.flatten()
            elif isinstance(data, dict):
                if 'ecg' in data:
                    ecg_data = np.array(data['ecg']).flatten()
                elif 'signal' in data:
                    ecg_data = np.array(data['signal']).flatten()
                if 'fs' in data:
                    fs = int(data['fs'])
            elif isinstance(data, list):
                ecg_data = np.array(data).flatten()
                
        elif filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                ecg_data = np.array(data).flatten()
            elif isinstance(data, dict):
                if 'ecg' in data:
                    ecg_data = np.array(data['ecg']).flatten()
                elif 'signal' in data:
                    ecg_data = np.array(data['signal']).flatten()
                if 'fs' in data:
                    fs = int(data['fs'])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        if ecg_data is None or len(ecg_data) == 0:
            raise HTTPException(status_code=400, detail="No valid ECG data found")
        
        ecg_data = ecg_data.astype(np.float32)
        
        # Run analysis
        results_df = model.batch_predict_ecg(ecg_data, fs=fs, window_sec=90, step_sec=15)
        
        if results_df.empty:
            raise HTTPException(status_code=400, detail="No valid windows produced")
        
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
        
        # Format response
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
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
