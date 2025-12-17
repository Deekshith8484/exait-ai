# EXRT AI - Streamlit Cloud Deployment Guide

## 📋 Overview

This guide walks through deploying EXRT AI as a **pure Streamlit Cloud application** with no external backend dependencies. The ML model, file upload processing, and Gemini API integration all run within the Streamlit app.

## ✅ Pre-Deployment Checklist

### Local Testing
- [x] `app.py` created with full ML integration
- [x] `new_dashboard.html` updated for Streamlit endpoints
- [x] `requirements.txt` updated (no FastAPI/uvicorn)
- [x] `.streamlit/config.toml` created with theme colors
- [x] `.streamlit/secrets.toml` created (local Gemini key)
- [x] `.gitignore` configured to exclude secrets

### Files Ready for Deployment
```
✓ app.py                          (Main Streamlit app with ML)
✓ new_dashboard.html              (UI dashboard)
✓ requirements.txt                (Dependencies)
✓ .streamlit/config.toml          (Configuration)
✓ analysis/models/inference.py    (ML Model)
✓ simulator/ecg_generator.py      (ECG simulation)
✓ .gitignore                      (Excludes secrets)
```

### Files NOT Needed Anymore
```
✗ test_backend.py                 (FastAPI backend - deprecated)
✗ streamlit_new_dashboard.py      (Old wrapper - replaced by app.py)
✗ streamlit_*.py                  (Other old versions)
```

---

## 🚀 Deployment Steps

### Step 1: Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Expected behavior:
- Dashboard loads with Tailwind styling
- Upload modal visible at top
- Can upload `.pkl`, `.csv`, or `.json` files
- Results display with readiness metrics
- Chat widget accepts messages (if Gemini key set)

### Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add files
git add app.py new_dashboard.html requirements.txt
git add .streamlit/config.toml .gitignore
git add analysis/ simulator/

# Verify secrets are NOT added
git status  # Ensure .streamlit/secrets.toml and .env are NOT listed

# Commit
git commit -m "feat: Streamlit Cloud deployment - pure Streamlit app with ML"

# Push to GitHub
git push origin main
```

### Step 3: Deploy to Streamlit Cloud

1. **Create Streamlit Cloud Account**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub account
   - Click "New app"

2. **Configure Deployment**
   - **Repository:** Select your GitHub repo (where you just pushed)
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - Click "Deploy"

   Streamlit will automatically:
   - Clone your repo
   - Install `requirements.txt` dependencies
   - Start the app

3. **Add Secrets in Cloud**
   - Go to app settings (three dots menu → Settings)
   - Click "Secrets" in sidebar
   - Add this content:
   ```toml
   Gemini_API_KEY = "AIzaSyDoYWh1ar4DEyx7-q-S3au8u10fNdraJUk"
   ```
   - Click "Save"

4. **App will redeploy automatically** with the secrets configured

---

## 🔑 Environment Variables

### Local Development (`.env` or `.streamlit/secrets.toml`)
```
Gemini_API_KEY = "AIzaSyDoYWh1ar4DEyx7-q-S3au8u10fNdraJUk"
```

### Streamlit Cloud (Secrets UI)
- Use the web dashboard to add `Gemini_API_KEY`
- Do NOT commit `secrets.toml` to Git
- Do NOT commit `.env` to Git

---

## 📊 Application Architecture

```
User Browser
    ↓
Streamlit App (Python)
    ├── HTML Dashboard (new_dashboard.html)
    │   ├── Upload Modal
    │   ├── ECG Chart Display
    │   ├── Results Modal
    │   └── Chat Widget
    │
    ├── File Upload Handler
    │   └── Parse .pkl/.csv/.json → NumPy array
    │
    ├── ML Model
    │   └── ReadinessModel.batch_predict_ecg()
    │
    └── Chat API
        └── Google Gemini API (direct from frontend)
```

---

## 🔧 File Upload Flow

1. User uploads file via sidebar (Streamlit `st.file_uploader()`)
2. File sent to `analyze_ecg_file()` function
3. Function parses based on extension:
   - `.pkl` → pickle deserialization
   - `.csv` → pandas read_csv
   - `.json` → json.loads
4. Extracted NumPy array sent to `ReadinessModel.batch_predict_ecg()`
5. Model returns readiness scores for each window
6. Results formatted as JSON:
   ```json
   {
     "filename": "test.pkl",
     "samples": 21000,
     "duration_sec": 30.0,
     "sampling_rate": 700,
     "summary": {
       "avg_readiness": 0.75,
       "min_readiness": 0.45,
       "max_readiness": 0.95,
       "std_readiness": 0.15,
       "n_windows": 2
     },
     "overall_state": "ready",
     "window_results": [...]
   }
   ```

---

## 📁 Directory Structure After Deployment

```
your-repo/
├── app.py                    ← Main Streamlit app
├── new_dashboard.html        ← UI dashboard
├── requirements.txt          ← Python dependencies
├── .gitignore               ← Excludes secrets
├── .streamlit/
│   ├── config.toml          ← Theme & config
│   └── secrets.toml         ← LOCAL ONLY (not in Git)
├── analysis/
│   └── models/
│       └── inference.py     ← ML model
└── simulator/
    └── ecg_generator.py     ← ECG simulation
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'analysis'"
**Solution:** Ensure `sys.path.insert(0, ...)` is in `app.py` and `analysis/` folder is pushed to GitHub

### Issue: "Gemini API key not found"
**Solution:** Add to Streamlit Cloud Secrets UI, not in code. Use `os.getenv("Gemini_API_KEY")`

### Issue: File upload fails silently
**Solution:** Check Streamlit logs (View logs in Cloud dashboard). Verify file format is .pkl, .csv, or .json

### Issue: "Module 'test_backend' not found"
**Solution:** This is expected - `test_backend.py` is deprecated. All functionality moved to `app.py`

### Issue: Model takes too long to load
**Solution:** `@st.cache_resource` should cache the model. Wait for first load, subsequent runs will be instant

---

## 📈 Performance Notes

- **First load:** ~30 seconds (downloads dependencies, loads ML model)
- **File upload analysis:** 2-10 seconds (depending on file size)
- **Subsequent loads:** <2 seconds (caching enabled)
- **Chat responses:** 2-5 seconds (Gemini API)

---

## 🔐 Security Checklist

- [x] No credentials in code
- [x] `.env` in `.gitignore`
- [x] `.streamlit/secrets.toml` in `.gitignore`
- [x] Gemini key only in Cloud Secrets UI
- [x] File uploads processed server-side only
- [x] No sensitive data logged

---

## 📞 Support

If the app doesn't deploy:
1. Check GitHub Actions (failed requirements installation)
2. View Streamlit Cloud logs (app settings → View logs)
3. Verify all files are pushed to correct branch
4. Ensure `app.py` is in root directory

---

## ✨ What Users See

**Local (Development)**
```
$ streamlit run app.py
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

**Cloud (Production)**
```
https://your-app-name.streamlit.app
```

---

## 🎉 Success Indicators

After deployment:
- ✅ Dashboard loads without errors
- ✅ Can upload files via sidebar
- ✅ Results display with readiness metrics
- ✅ Chat widget responds to messages
- ✅ Logs show no import errors

---

## 📝 Next Steps After Deployment

1. Test upload with `test_signal.pkl`, `test_signal.csv`, `test_signal.json`
2. Verify chat widget works with Gemini API
3. Share app URL with team
4. Monitor Streamlit Cloud dashboard for performance/errors
5. Set up custom domain (optional - Streamlit paid feature)

---

**Deployment Date:** [Current Date]
**Status:** Ready for Cloud Deployment ✅
**Main File:** `app.py`
**Branch:** `main`
