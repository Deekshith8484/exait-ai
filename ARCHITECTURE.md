# EXRT AI - Architecture & Data Flow Diagrams

## System Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT CLOUD (Python)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    FRONTEND (HTML/JS)                       │ │
│  │  new_dashboard.html                                         │ │
│  │  ├─ Navigation Bar (EXRT AI Logo)                          │ │
│  │  ├─ Upload Modal (File Selection)                          │ │
│  │  ├─ ECG Chart Display (Chart.js)                           │ │
│  │  ├─ Results Modal (Readiness Stats)                        │ │
│  │  └─ Chat Widget (Gemini Integration)                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   BACKEND (Python)                          │ │
│  │  app.py                                                     │ │
│  │  ├─ File Upload Handler (st.file_uploader)                │ │
│  │  │  └─ Parses .pkl / .csv / .json                         │ │
│  │  ├─ ML Model (ReadinessModel)                             │ │
│  │  │  └─ batch_predict_ecg(signal, fs=700)                 │ │
│  │  └─ Results Formatter (JSON Response)                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              EXTERNAL SERVICES (APIs)                       │ │
│  │  ├─ Google Gemini API (Chat)                              │ │
│  │  └─ Environment Secrets (.streamlit/secrets.toml)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                        User Browser
```

---

## File Upload & Analysis Flow

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER INTERACTION                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Browser/Sidebar: st.file_uploader()                          │
│   User selects: test_signal.pkl (or .csv / .json)             │
│   File bytes captured and stored                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. FILE PARSING                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   analyze_ecg_file(file_content, filename)                     │
│   │                                                              │
│   ├─ Detect file type: .pkl / .csv / .json                    │
│   │                                                              │
│   ├─ .pkl Branch:                                             │
│   │  └─ pickle.loads(bytes) → dict/array/list                │
│   │                                                              │
│   ├─ .csv Branch:                                             │
│   │  └─ pd.read_csv(BytesIO) → extract column                │
│   │                                                              │
│   └─ .json Branch:                                            │
│      └─ json.loads(text) → extract data                       │
│                                                                  │
│   Result: NumPy array (signal)                                 │
│   Sampling rate: Extracted or default 700 Hz                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ML MODEL INFERENCE                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ReadinessModel.batch_predict_ecg(signal, fs=700)            │
│   │                                                              │
│   ├─ Feature Extraction (per window)                          │
│   │  ├─ Heart rate (bpm)                                     │
│   │  ├─ Heart rate variability (HRV)                         │
│   │  ├─ Amplitude metrics                                    │
│   │  ├─ Frequency domain features                            │
│   │  └─ Statistical features                                 │
│   │                                                              │
│   ├─ Window Processing                                        │
│   │  ├─ Window size: 30 seconds (21,000 samples @ 700 Hz)   │
│   │  ├─ Stride: 15 seconds                                  │
│   │  └─ Sliding windows across signal                        │
│   │                                                              │
│   └─ ML Prediction                                            │
│      ├─ Trained model processes features                     │
│      └─ Output: readiness ∈ [0.0, 1.0]                      │
│                                                                  │
│   Result: Array of readiness scores                           │
│   Length: # of windows processed                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. RESULTS FORMATTING                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   JSON Response:                                                │
│   {                                                              │
│     "filename": "test_signal.pkl",                             │
│     "samples": 21000,                                          │
│     "duration_sec": 30.0,                                      │
│     "sampling_rate": 700,                                      │
│     "summary": {                                                │
│       "avg_readiness": 0.75,    ← Mean across all windows    │
│       "min_readiness": 0.45,    ← Worst window              │
│       "max_readiness": 0.95,    ← Best window               │
│       "std_readiness": 0.15,    ← Variability              │
│       "n_windows": 2             ← Windows analyzed         │
│     },                                                          │
│     "overall_state": "ready",   ← Classification           │
│     "window_results": [          ← Per-window detail       │
│       {"window_idx": 0, "readiness": 0.70},                 │
│       {"window_idx": 1, "readiness": 0.80}                  │
│     ]                                                           │
│   }                                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. DISPLAY RESULTS                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Modal Window Shows:                                           │
│   ├─ 📊 "Readiness: 75%"         (avg_readiness × 100)       │
│   ├─ 📉 "Min: 45%, Max: 95%"     (range)                     │
│   ├─ ⏱️  "Duration: 30.0s"        (duration_sec)              │
│   ├─ 📈 "Variability: 0.15 σ"    (std_readiness)             │
│   ├─ 🎯 "State: READY"           (overall_state)              │
│   └─ 📋 "Windows: 2 analyzed"   (n_windows)                   │
│                                                                  │
│   User can:                                                      │
│   ├─ Close modal                                              │
│   ├─ Ask follow-up in chat                                   │
│   └─ Upload another file                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chat Widget Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ USER TYPES MESSAGE IN CHAT                                      │
├─────────────────────────────────────────────────────────────────┤
│  "Is my readiness score good?"                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND (new_dashboard.html)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HTML App.Chat.sendChatMessage()                               │
│  │                                                              │
│  ├─ Get message text from input                               │
│  ├─ Display user message in chat UI                           │
│  ├─ Show "AI is thinking..." indicator                        │
│  └─ Call callGeminiAPI(message)                               │
│                                                                  │
│       ↓                                                          │
│                                                                  │
│  Direct HTTP call (No backend):                               │
│  POST https://generativelanguage.googleapis.com/...          │
│  ├─ Header: Content-Type: application/json                   │
│  ├─ Header: x-goog-api-key: [GEMINI_API_KEY]                 │
│  └─ Body: {                                                   │
│       "contents": [{                                           │
│         "role": "user",                                       │
│         "parts": [{"text": "Is my readiness score good?"}]   │
│       }]                                                       │
│     }                                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ GOOGLE GEMINI API (External Service)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Google's servers process request                             │
│  ├─ Analyze message context                                  │
│  ├─ Generate intelligent response                            │
│  └─ Return: {                                                 │
│       "candidates": [{                                        │
│         "content": {                                          │
│           "parts": [{                                         │
│             "text": "A readiness of 75% is good for..."      │
│           }]                                                  │
│         }                                                     │
│       }]                                                      │
│     }                                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND (Display Response)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  response = await callGeminiAPI(message)                       │
│  │                                                              │
│  ├─ Extract text from candidates[0].content.parts[0].text    │
│  ├─ Format with markdown rendering                            │
│  ├─ Display AI message in chat bubble                         │
│  └─ Clear input, ready for next message                       │
│                                                                  │
│  Chat now shows:                                               │
│  User:  "Is my readiness score good?"                        │
│  AI:    "A readiness of 75% is good for..."                 │
│         (continues with helpful analysis)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structure Definitions

### Input: ECG Signal
```
ECG File (.pkl / .csv / .json)
├─ Raw signal: [v₁, v₂, v₃, ..., vₙ]
│  └─ Voltage values in millivolts
│  └─ Type: NumPy array (1D)
│
├─ Metadata (optional):
│  ├─ fs: Sampling frequency (Hz)
│  │  └─ Default: 700 Hz
│  │  └─ Range: 100-5000 Hz
│  │
│  └─ patient_id: Subject identifier
│     └─ For tracking purposes
│
└─ Common sizes:
   ├─ 21,000 samples @ 700 Hz = 30 seconds
   ├─ 10,500 samples @ 700 Hz = 15 seconds
   └─ 70,000 samples @ 700 Hz = 100 seconds
```

### Processing: Window Extraction
```
Window Configuration:
├─ Duration: 30 seconds
├─ Stride: 15 seconds  (50% overlap)
├─ Sample count: 30 × 700 = 21,000 samples
└─ Frequency: 700 Hz

Example timeline:
Signal:    [0s ─────────────── 60s ─────────────── 120s]
Window 1:  [0s ─── 30s]
Window 2:          [15s ─── 45s]
Window 3:                 [30s ─── 60s]
Window 4:                         [45s ─── 75s]
...
```

### Feature Extraction (per window)
```
Input: 21,000 ECG samples

Time Domain Features:
├─ Mean absolute value
├─ Standard deviation
├─ Zero crossing rate
└─ Signal energy

Frequency Domain Features:
├─ FFT peak frequency
├─ Power spectral density
├─ Dominant frequency bands
└─ Spectral entropy

Physiological Features:
├─ Heart rate (bpm)
│  └─ Detected R-peaks, normalized to 60 sec
│
├─ Heart rate variability (HRV)
│  └─ Variance of R-R intervals
│
├─ P-wave amplitude
├─ QRS complex amplitude
└─ T-wave characteristics

Output: Feature vector (~30-50 dimensions)
```

### ML Model
```
Trained Classifier
├─ Input: Feature vector (30-50 dims)
├─ Model type: XGBoost / Random Forest / SVM
├─ Training data: 500+ annotated ECG samples
│  └─ Balance: Ready (70%), Neutral (20%), Fatigued (10%)
│
└─ Output: Readiness score ∈ [0.0, 1.0]
   ├─ 0.0 = Completely fatigued
   ├─ 0.5 = Neutral/Normal state
   └─ 1.0 = Completely ready/Rested
```

### Output: Results JSON
```
{
  "filename": "test_signal.pkl",
  "samples": 21000,
  "duration_sec": 30.0,
  "sampling_rate": 700,
  
  "summary": {
    "avg_readiness": 0.75,     # Mean across all windows
    "min_readiness": 0.45,     # Lowest score
    "max_readiness": 0.95,     # Highest score
    "std_readiness": 0.15,     # Standard deviation
    "n_windows": 2             # Windows analyzed
  },
  
  "overall_state": "ready",    # "ready" | "neutral" | "fatigued"
                                # Based on avg_readiness:
                                # > 0.7 = ready
                                # 0.4-0.7 = neutral
                                # < 0.4 = fatigued
  
  "window_results": [
    {"window_idx": 0, "readiness": 0.70},
    {"window_idx": 1, "readiness": 0.80}
  ]
}
```

---

## Deployment Architecture

### Local Development
```
Your Computer (Windows/Mac/Linux)
│
├─ Python Environment
│  └─ streamlit 1.45.1
│  └─ pandas, numpy, scikit-learn
│  └─ google-generativeai
│
├─ Files
│  ├─ app.py
│  ├─ new_dashboard.html
│  ├─ .streamlit/config.toml
│  ├─ .streamlit/secrets.toml (NEVER commit)
│  └─ analysis/, simulator/
│
├─ .env (local, NEVER commit)
│  └─ Gemini_API_KEY = "..."
│
└─ Terminal
   └─ streamlit run app.py
      → Localhost:8501
```

### Production (Streamlit Cloud)
```
Streamlit Cloud (Google Cloud)
│
├─ Repository Connection
│  └─ Your GitHub repo (public/shared)
│
├─ Deployment
│  ├─ Clone from GitHub
│  ├─ Install requirements.txt
│  ├─ Run: streamlit run app.py
│  └─ Assign URL: your-app.streamlit.app
│
├─ Secrets Management
│  └─ Cloud Dashboard Secrets UI
│     └─ Gemini_API_KEY (NOT from .env)
│
└─ Execution
   ├─ User accesses: https://your-app.streamlit.app
   ├─ Load balancing across multiple instances
   ├─ Auto-scaling based on demand
   └─ Cached model for performance
```

---

## API Integration Points

### Gemini API (Chat Widget)
```
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
├─ Header: x-goog-api-key: {GEMINI_API_KEY}
├─ Body: {
│   "contents": [{
│     "role": "user",
│     "parts": [{"text": "User message..."}]
│   }]
│ }
│
└─ Response: {
    "candidates": [{
      "content": {
        "parts": [{
          "text": "AI response..."
        }]
      }
    }]
  }
```

### File Upload (Internal)
```
Streamlit st.file_uploader()
├─ Input: Binary file data
├─ Processing: analyze_ecg_file()
└─ Output: JSON results (internal)
```

---

**Architecture Version:** 1.0 (Streamlit Cloud)  
**Last Updated:** Today  
**Status:** Production Ready ✅
