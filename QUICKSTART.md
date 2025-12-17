<<<<<<< HEAD
# 🚀 EXRT MVP – Quick Start Guide

## Prerequisites

```powershell
pip install pandas numpy scikit-learn neurokit2 streamlit fastapi uvicorn pydantic plotly
```

---

## Step 1: Train the Model

Navigate to the project directory and run:

```powershell
cd "g:\exait ai\analysis\models"
python train_and_save_model.py
```

**Expected output:**
```
Found 8 subjects: S1, S2, ...
Extracting HRV features...
Total usable windows: 4,200+
Training PCA + GMM model...
✓ Model saved to G:\EXAIT AI\analysis\models\readiness_model.pkl
✓ Metadata saved to G:\EXAIT AI\analysis\models\model_metadata.json
```

**Time:** ~5-10 minutes (depending on your system)

---

## Step 2: Verify the Model

Test that inference works:

```powershell
cd "g:\exait ai\analysis\models"
python inference.py
```

**Expected output:**
```
✓ Model loaded from G:\EXAIT AI\analysis\models\readiness_model.pkl
  Features: 24
  PCA components: 5
  GMM clusters: 8

Model is ready for inference.
Ready cluster: 3
Feature columns: ['RMSSD', 'SDNN', 'pNN50', ...]
```

---

## Step 3: Launch the Streamlit App

This provides a user-friendly interface for uploading ECG files:

```powershell
cd "g:\exait ai"
streamlit run streamlit_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  URL: http://localhost:8501
```

Open `http://localhost:8501` in your browser. You can now:
- Upload ECG files (.csv or .npy)
- Adjust window parameters
- Get real-time readiness scores
- Download results as CSV

---

## Step 4: Start the FastAPI Backend

For programmatic access to readiness predictions:

```powershell
cd "g:\exait ai"
python -m uvicorn api:app --reload --port 8000
```

**Expected output:**
```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

---

## API Usage Examples

### Example 1: Predict Readiness from ECG Array

```bash
curl -X POST "http://localhost:8000/readiness" \
  -H "Content-Type: application/json" \
  -d '{
    "ecg": [0.1, 0.2, -0.1, 0.3, ...],
    "fs": 700,
    "window_sec": 90,
    "step_sec": 15
  }'
```

**Response:**
```json
{
  "summary": {
    "avg_readiness": 68.5,
    "min_readiness": 42.1,
    "max_readiness": 89.3,
    "std_readiness": 15.2,
    "n_windows": 42
  },
  "window_results": [
    {
      "window_idx": 0,
      "center_time_sec": 45.0,
      "readiness": 72.3
    },
    ...
  ],
  "overall_state": "ready"
}
```

### Example 2: Predict from HRV Features

```bash
curl -X POST "http://localhost:8000/readiness/features" \
  -H "Content-Type: application/json" \
  -d '{
    "RMSSD": 45.2,
    "SDNN": 35.1,
    "pNN50": 15.3,
    "HR_est": 62.0,
    "SQI": 0.95
  }'
```

**Response:**
```json
{
  "readiness": 72.3,
  "state": "ready"
}
```

### Example 3: Get Model Info

```bash
curl "http://localhost:8000/model/info"
```

**Response:**
```json
{
  "pca_components": 5,
  "gmm_components": 8,
  "ready_cluster": 3,
  "n_features": 24,
  "feature_names": ["RMSSD", "SDNN", "pNN50", ...]
}
```

---

## Python Client Example

```python
import requests
import numpy as np

# Load your ECG data
ecg = np.array([...])  # Your ECG signal

# Call API
response = requests.post(
    "http://localhost:8000/readiness",
    json={
        "ecg": ecg.tolist(),
        "fs": 700,
        "window_sec": 90,
        "step_sec": 15
    }
)

# Parse result
result = response.json()
print(f"Avg Readiness: {result['summary']['avg_readiness']:.1f}%")
print(f"State: {result['overall_state']}")
```

---

## File Structure

```
g:\exait ai\
├── analysis/
│   ├── ecg_pipeline.py              # Core HRV extraction logic
│   ├── ecg.ipynb                    # Experimental notebook
│   ├── models/
│   │   ├── train_and_save_model.py  # Training script ✅
│   │   ├── inference.py             # Inference wrapper ✅
│   │   ├── readiness_model.pkl      # Trained model (after step 1)
│   │   └── model_metadata.json      # Metadata (after step 1)
├── data/
│   └── ppg+dalia/                   # PPG-Dalia dataset
├── streamlit_app.py                 # UI demo ✅
├── api.py                           # FastAPI backend ✅
└── PIPELINE_SUMMARY.md              # This file
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ecg_pipeline'"
- Ensure you're running scripts from the correct directory
- Both `train_and_save_model.py` and `inference.py` add parent directories to `sys.path`

### "Model not found"
- Run `train_and_save_model.py` first (Step 1)
- Check that `readiness_model.pkl` exists in `analysis/models/`

### "No valid windows produced"
- ECG signal may be too short (<90 seconds)
- Sampling rate may be incorrect
- ECG quality is too poor (low SQI)

### Streamlit stuck on "Loading ECG..."
- Large files (>100MB) may take time
- Check system memory

### FastAPI won't start
- Port 8000 might be in use: `python -m uvicorn api:app --port 8001`
- Check firewall settings

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./analysis/models/readiness_model.pkl:/app/analysis/models/readiness_model.pkl
```

### Heroku Deployment

```bash
heroku create exrt-readiness
git push heroku main
```

---

## Performance Metrics

- **Training time:** ~5-10 minutes (8 subjects, 4,200+ windows)
- **Inference (90s ECG):** ~500ms per window
- **API latency:** ~50-100ms (excluding HRV computation)
- **Model size:** ~2-5 MB

---

## What's Next?

✅ **Phase 1 (Complete):** Model training, inference, UI, API
📋 **Phase 2:** Real-time wearable integration (BLE/WiFi)
📋 **Phase 3:** Multi-user dashboard with history
📋 **Phase 4:** Coach recommendations based on readiness trends
📋 **Phase 5:** Mobile app (iOS/Android)

---

**Happy readiness scoring! 🎉**
=======
# 🚀 EXRT MVP – Quick Start Guide

## Prerequisites

```powershell
pip install pandas numpy scikit-learn neurokit2 streamlit fastapi uvicorn pydantic plotly
```

---

## Step 1: Train the Model

Navigate to the project directory and run:

```powershell
cd "g:\exait ai\analysis\models"
python train_and_save_model.py
```

**Expected output:**
```
Found 8 subjects: S1, S2, ...
Extracting HRV features...
Total usable windows: 4,200+
Training PCA + GMM model...
✓ Model saved to G:\EXAIT AI\analysis\models\readiness_model.pkl
✓ Metadata saved to G:\EXAIT AI\analysis\models\model_metadata.json
```

**Time:** ~5-10 minutes (depending on your system)

---

## Step 2: Verify the Model

Test that inference works:

```powershell
cd "g:\exait ai\analysis\models"
python inference.py
```

**Expected output:**
```
✓ Model loaded from G:\EXAIT AI\analysis\models\readiness_model.pkl
  Features: 24
  PCA components: 5
  GMM clusters: 8

Model is ready for inference.
Ready cluster: 3
Feature columns: ['RMSSD', 'SDNN', 'pNN50', ...]
```

---

## Step 3: Launch the Streamlit App

This provides a user-friendly interface for uploading ECG files:

```powershell
cd "g:\exait ai"
streamlit run streamlit_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  URL: http://localhost:8501
```

Open `http://localhost:8501` in your browser. You can now:
- Upload ECG files (.csv or .npy)
- Adjust window parameters
- Get real-time readiness scores
- Download results as CSV

---

## Step 4: Start the FastAPI Backend

For programmatic access to readiness predictions:

```powershell
cd "g:\exait ai"
python -m uvicorn api:app --reload --port 8000
```

**Expected output:**
```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

---

## API Usage Examples

### Example 1: Predict Readiness from ECG Array

```bash
curl -X POST "http://localhost:8000/readiness" \
  -H "Content-Type: application/json" \
  -d '{
    "ecg": [0.1, 0.2, -0.1, 0.3, ...],
    "fs": 700,
    "window_sec": 90,
    "step_sec": 15
  }'
```

**Response:**
```json
{
  "summary": {
    "avg_readiness": 68.5,
    "min_readiness": 42.1,
    "max_readiness": 89.3,
    "std_readiness": 15.2,
    "n_windows": 42
  },
  "window_results": [
    {
      "window_idx": 0,
      "center_time_sec": 45.0,
      "readiness": 72.3
    },
    ...
  ],
  "overall_state": "ready"
}
```

### Example 2: Predict from HRV Features

```bash
curl -X POST "http://localhost:8000/readiness/features" \
  -H "Content-Type: application/json" \
  -d '{
    "RMSSD": 45.2,
    "SDNN": 35.1,
    "pNN50": 15.3,
    "HR_est": 62.0,
    "SQI": 0.95
  }'
```

**Response:**
```json
{
  "readiness": 72.3,
  "state": "ready"
}
```

### Example 3: Get Model Info

```bash
curl "http://localhost:8000/model/info"
```

**Response:**
```json
{
  "pca_components": 5,
  "gmm_components": 8,
  "ready_cluster": 3,
  "n_features": 24,
  "feature_names": ["RMSSD", "SDNN", "pNN50", ...]
}
```

---

## Python Client Example

```python
import requests
import numpy as np

# Load your ECG data
ecg = np.array([...])  # Your ECG signal

# Call API
response = requests.post(
    "http://localhost:8000/readiness",
    json={
        "ecg": ecg.tolist(),
        "fs": 700,
        "window_sec": 90,
        "step_sec": 15
    }
)

# Parse result
result = response.json()
print(f"Avg Readiness: {result['summary']['avg_readiness']:.1f}%")
print(f"State: {result['overall_state']}")
```

---

## File Structure

```
g:\exait ai\
├── analysis/
│   ├── ecg_pipeline.py              # Core HRV extraction logic
│   ├── ecg.ipynb                    # Experimental notebook
│   ├── models/
│   │   ├── train_and_save_model.py  # Training script ✅
│   │   ├── inference.py             # Inference wrapper ✅
│   │   ├── readiness_model.pkl      # Trained model (after step 1)
│   │   └── model_metadata.json      # Metadata (after step 1)
├── data/
│   └── ppg+dalia/                   # PPG-Dalia dataset
├── streamlit_app.py                 # UI demo ✅
├── api.py                           # FastAPI backend ✅
└── PIPELINE_SUMMARY.md              # This file
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ecg_pipeline'"
- Ensure you're running scripts from the correct directory
- Both `train_and_save_model.py` and `inference.py` add parent directories to `sys.path`

### "Model not found"
- Run `train_and_save_model.py` first (Step 1)
- Check that `readiness_model.pkl` exists in `analysis/models/`

### "No valid windows produced"
- ECG signal may be too short (<90 seconds)
- Sampling rate may be incorrect
- ECG quality is too poor (low SQI)

### Streamlit stuck on "Loading ECG..."
- Large files (>100MB) may take time
- Check system memory

### FastAPI won't start
- Port 8000 might be in use: `python -m uvicorn api:app --port 8001`
- Check firewall settings

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./analysis/models/readiness_model.pkl:/app/analysis/models/readiness_model.pkl
```

### Heroku Deployment

```bash
heroku create exrt-readiness
git push heroku main
```

---

## Performance Metrics

- **Training time:** ~5-10 minutes (8 subjects, 4,200+ windows)
- **Inference (90s ECG):** ~500ms per window
- **API latency:** ~50-100ms (excluding HRV computation)
- **Model size:** ~2-5 MB

---

## What's Next?

✅ **Phase 1 (Complete):** Model training, inference, UI, API
📋 **Phase 2:** Real-time wearable integration (BLE/WiFi)
📋 **Phase 3:** Multi-user dashboard with history
📋 **Phase 4:** Coach recommendations based on readiness trends
📋 **Phase 5:** Mobile app (iOS/Android)

---

**Happy readiness scoring! 🎉**
>>>>>>> afeb0a9 (stimulation poc)
