"""EXRT AI - Streamlit deployment app (standalone, no external backend required)"""

import streamlit as st
import os
import json
import pickle
import io
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="EXRT AI - Sports Performance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome for cleaner UI
st.markdown("""
<style>
  #MainMenu {visibility: hidden;}
  header {visibility: hidden;}
  footer {visibility: hidden;}
  [data-testid="stAppViewContainer"] {padding: 0 !important;}
  .block-container {padding: 0 !important; max-width: 100% !important;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ML MODEL (cached for performance)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the readiness model once at startup."""
    try:
        from analysis.models.inference import ReadinessModel
        model = ReadinessModel()
        st.write("✓ ML Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ============================================================================
# PROCESS FILE UPLOAD (called from frontend)
# ============================================================================

def analyze_ecg_file(file_content, filename):
    """Analyze ECG file using the loaded model."""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        filename_lower = filename.lower()
        ecg_data = None
        fs = 700  # Default sampling frequency
        
        # Parse based on file extension
        if filename_lower.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
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
                    
        elif filename_lower.endswith('.pkl') or filename_lower.endswith('.pickle'):
            data = pickle.loads(file_content)
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
                
        elif filename_lower.endswith('.json'):
            data = json.loads(file_content.decode('utf-8'))
            if isinstance(data, list):
                ecg_data = np.array(data).flatten()
            elif isinstance(data, dict):
                if 'ecg' in data:
                    ecg_data = np.array(data['ecg']).flatten()
        
        if ecg_data is None or len(ecg_data) == 0:
            return {"error": "No valid ECG data found in file"}
        
        # Use model to analyze
        results = model.batch_predict_ecg(
            signal=ecg_data,
            fs=fs,
            window_duration=30,
            stride=15
        )
        
        # Format response
        return {
            'filename': filename,
            'samples': len(ecg_data),
            'duration_sec': len(ecg_data) / fs,
            'sampling_rate': fs,
            'summary': {
                'avg_readiness': float(np.mean(results)),
                'min_readiness': float(np.min(results)),
                'max_readiness': float(np.max(results)),
                'std_readiness': float(np.std(results)),
                'n_windows': len(results)
            },
            'overall_state': 'ready' if np.mean(results) > 0.7 else 'neutral' if np.mean(results) > 0.4 else 'fatigued',
            'window_results': [
                {
                    'window_idx': i,
                    'readiness': float(r)
                }
                for i, r in enumerate(results)
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# LOAD AND EMBED HTML DASHBOARD
# ============================================================================

html_path = Path(__file__).parent / "new_dashboard.html"

if html_path.exists():
    html_content = html_path.read_text(encoding="utf-8")
    
    # Inject Gemini API key into HTML
    gemini_key = os.getenv("Gemini_API_KEY", "")
    if gemini_key:
        # Add script to inject API key before HTML loads
        injection_script = f"""
        <script>
            window.GEMINI_API_KEY = "{gemini_key}";
            
            // Override fetch for upload endpoint
            const originalFetch = window.fetch;
            window.fetch = async function(url, options) {{
                if (url.includes('?action=upload_analyze')) {{
                    // Handle file upload through Streamlit
                    const formData = options.body;
                    const file = formData.get('file');
                    
                    // Read file and send to Streamlit
                    const reader = new FileReader();
                    reader.onload = async function(e) {{
                        const content = new Uint8Array(e.target.result);
                        // Send via POST to Streamlit
                        try {{
                            const response = await originalFetch('/', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{
                                    filename: file.name,
                                    content: Array.from(content)
                                }})
                            }});
                        }} catch (err) {{
                            console.error('Upload error:', err);
                        }}
                    }};
                    reader.readAsArrayBuffer(file);
                    
                    return new Response(JSON.stringify({{error: 'Use sidebar file upload'}}), {{status: 200}});
                }}
                return originalFetch.apply(this, arguments);
            }};
        </script>
        """
        html_content = injection_script + html_content
    
    # Render HTML
    st.components.v1.html(html_content, height=5200, scrolling=True)
else:
    st.error("❌ new_dashboard.html not found")

# ============================================================================
# SIDEBAR FOR FILE UPLOAD (Streamlit integration)
# ============================================================================

with st.sidebar:
    st.header("⬆️ File Upload")
    uploaded_file = st.file_uploader(
        "Upload ECG Signal File",
        type=['pkl', 'csv', 'json'],
        help="Upload .pkl, .csv, or .json files"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 File: {uploaded_file.name}")
        st.info(f"📊 Size: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🔍 Analyze Signal", use_container_width=True):
            with st.spinner("Analyzing..."):
                file_content = uploaded_file.read()
                result = analyze_ecg_file(file_content, uploaded_file.name)
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Analysis Complete!")
                    st.json(result)
    
    st.markdown("---")
    st.info("💡 Supported formats: .pkl, .csv, .json")
