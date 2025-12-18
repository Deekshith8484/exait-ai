"""Streamlit wrapper that renders new_dashboard.html exactly.

This keeps the UI identical to the HTML version (including JavaScript),
while allowing the page to call the FastAPI backend.
Backend URL: http://4.206.202.59:8000 (production) or http://localhost:8000 (local dev)
"""

from pathlib import Path
import sys

# IMPORTANT:
# This repo contains a local file named `streamlit.py` which can shadow the real
# Streamlit package. We remove the workspace root from sys.path before importing
# Streamlit to ensure we import the installed package.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if str(Path(p).resolve()) != str(Path(_script_dir).resolve())]

import streamlit as st


st.set_page_config(
    page_title="EXRT AI - Sports Performance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome so the embedded dashboard looks like a real web app.
st.markdown(
    """
<style>
  #MainMenu {visibility: hidden;}
  header {visibility: hidden;}
  footer {visibility: hidden;}
  [data-testid="stAppViewContainer"] {padding: 0 !important;}
  .block-container {padding: 0 !important; max-width: 100% !important;}
</style>
""",
    unsafe_allow_html=True,
)

html_path = Path(__file__).parent / "new_dashboard.html"

if not html_path.exists():
    st.error("❌ new_dashboard.html not found next to streamlit_new_dashboard.py")
    st.stop()

html_content = html_path.read_text(encoding="utf-8")

# Render the HTML inside an iframe so JS runs and styling matches the HTML exactly.
st.components.v1.html(html_content, height=5200, scrolling=True)

# Keep the session alive (Streamlit requires this to stay running)
_ = st.empty()
