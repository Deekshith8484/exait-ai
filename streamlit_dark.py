import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="EXRT AI - Health Dashboard",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Dark Theme
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background: #0a0e1a !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #0a0e1a !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Dark text */
    h1, h2, h3, p, span, div {
        color: white !important;
    }
    
    /* Dark header */
    .dark-header {
        background: #0a0e1a;
        padding: 1.5rem 0;
        color: white;
    }
    
    .greeting {
        font-size: 1.75rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        color: #e0e0e0;
    }
    
    /* Navigation tabs */
    div[data-testid="stHorizontalBlock"] {
        background: transparent !important;
        padding: 0 !important;
        margin: 1.5rem 0 !important;
        gap: 2rem !important;
        border-bottom: 1px solid #1a2332;
    }
    
    div[data-testid="stHorizontalBlock"] label {
        color: #6b7280 !important;
        padding: 0.75rem 0 !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 500 !important;
        background: transparent !important;
    }
    
    div[data-testid="stHorizontalBlock"] label:has(input:checked) {
        color: white !important;
        border-bottom-color: white !important;
    }
    
    /* Dark cards */
    .dark-card {
        background: #141b2d;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        color: white;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.75rem;
        font-weight: 500;
    }
    
    .card-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .card-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .status-connected {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
    }
    
    .status-normal {
        background: rgba(99, 102, 241, 0.15);
        color: #6366F1;
    }
    
    /* Metric boxes */
    .metric-box {
        background: #1a2332;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    /* Info text */
    .info-text {
        color: #6b7280;
        font-size: 0.875rem;
        line-height: 1.6;
        margin: 1.5rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 24px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Bottom navigation */
    .bottom-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #1a2332;
        padding: 1rem 0;
        display: flex;
        justify-content: space-around;
        border-bottom: 1px solid #2a3442;
        z-index: 1000;
    }
    
    .nav-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        color: #6b7280;
        font-size: 0.85rem;
        cursor: pointer;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .nav-item.active {
        color: white;
    }
    
    .nav-item:hover {
        color: white;
    }
    
    .nav-icon {
        font-size: 1.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
BACKEND_API_URL = "http://localhost:8000"
GEMINI_API_KEY = "AIzaSyAh_aH2sYEI3MBFZ19sKnirwH0-a2hAM9I"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"

# Initialize session state
if 'backend_status' not in st.session_state:
    st.session_state.backend_status = False

# Backend health check
def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Check backend
st.session_state.backend_status = check_backend_health()

# Top Navigation
st.markdown("""
<div class="bottom-nav">
    <div class="nav-item active">
        <div class="nav-icon">🏠</div>
        <div>Home</div>
    </div>
    <div class="nav-item">
        <div class="nav-icon">📊</div>
        <div>ECG</div>
    </div>
    <div class="nav-item">
        <div class="nav-icon">🌙</div>
        <div>Journal</div>
    </div>
    <div class="nav-item">
        <div class="nav-icon">👤</div>
        <div>Profile</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Add spacing for top nav
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Header
st.markdown('<div class="dark-header">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="greeting">Good morning, Alex!</p>', unsafe_allow_html=True)
with col2:
    current_date = datetime.now().strftime("%A, %d %b %Y")
    st.markdown(f'<p style="text-align: right; color: #6b7280; font-size: 0.875rem;">{current_date}</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Navigation
nav_options = ["🏠 Home", "📊 ECG", "👤 Journal", "👤 Profile"]
selected_nav = st.radio("", nav_options, horizontal=True, label_visibility="collapsed", index=0)

# Main Content
if "Home" in selected_nav or "ECG" in selected_nav:
    # Top Row - Device Status and ECG Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="dark-card">
            <p class="card-title">Wearable device</p>
            <p class="card-value">14 hrs, 24 <span style="font-size: 1.5rem; color: #6b7280;">min</span></p>
            <div class="status-badge status-connected">
                <span>●</span> Connected
            </div>
            <div style="margin-top: 1rem;">
                <svg width="60" height="40" viewBox="0 0 60 40">
                    <path d="M5,20 Q10,10 15,20 T25,20 T35,20 T45,20 T55,20" 
                          fill="none" stroke="url(#gradient)" stroke-width="2"/>
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                </svg>
                <p class="card-subtitle">● 96%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="dark-card">
            <p class="card-title">Your ECG</p>
            <p class="card-value">Within Normal Limits</p>
            <p class="card-subtitle">Last recorded: 2:54 pm</p>
            <div style="margin-top: 1rem; height: 40px;">
                <svg width="100%" height="100%" viewBox="0 0 200 40" preserveAspectRatio="none">
                    <path d="M0,20 L40,20 L45,5 L50,35 L55,20 L100,20" 
                          fill="none" stroke="#EF4444" stroke-width="2"/>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stress Monitor Section
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("""
        <div class="dark-card">
            <p class="card-title">Stress Monitor</p>
            <div style="position: relative; width: 200px; height: 200px; margin: 2rem auto;">
                <svg viewBox="0 0 200 200" style="transform: rotate(-90deg);">
                    <circle cx="100" cy="100" r="80" fill="none" stroke="#1a2332" stroke-width="20"/>
                    <circle cx="100" cy="100" r="80" fill="none" 
                            stroke="url(#stress-gradient)" 
                            stroke-width="20"
                            stroke-dasharray="377"
                            stroke-dashoffset="150"
                            stroke-linecap="round"/>
                    <defs>
                        <linearGradient id="stress-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#667eea"/>
                            <stop offset="50%" style="stop-color:#EC4899"/>
                            <stop offset="100%" style="stop-color:#EF4444"/>
                        </linearGradient>
                    </defs>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                    <div style="font-size: 3.5rem; font-weight: 700; color: white;">0.3</div>
                    <div style="font-size: 0.875rem; color: #6b7280;">RMSSD</div>
                </div>
            </div>
            <p class="info-text" style="text-align: center; margin-top: 1rem;">
                Stress level : Very Low
            </p>
            <p class="info-text" style="font-size: 0.75rem; text-align: center;">
                Your stress levels look great! Remember that you're managing stress well and try to maintain this routine, you're doing well. Keep up the good work! Stay healthy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # ECG Chart
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title" style="margin-bottom: 1rem;">Real-time ECG</p>', unsafe_allow_html=True)
        
        # Generate ECG data
        x = np.linspace(0, 4, 500)
        y = []
        for i in x:
            beat_phase = (i % 1.0)
            if 0.0 <= beat_phase < 0.1:
                val = 60 + 10 * np.sin(beat_phase * 10 * np.pi)
            elif 0.2 <= beat_phase < 0.3:
                val = 40 + (beat_phase - 0.2) * 400
            elif 0.3 <= beat_phase < 0.35:
                val = 80 - (beat_phase - 0.3) * 600
            elif 0.45 <= beat_phase < 0.65:
                val = 60 + 15 * np.sin((beat_phase - 0.45) * 5 * np.pi)
            else:
                val = 60
            y.append(val + np.random.random() * 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#EF4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        
        fig.update_layout(
            plot_bgcolor='#141b2d',
            paper_bgcolor='#141b2d',
            xaxis=dict(
                showgrid=False,
                showticklabels=True,
                tickfont=dict(color='#6b7280', size=10),
                title=None
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#1a2332',
                showticklabels=False,
                title=None
            ),
            margin=dict(l=0, r=0, t=0, b=20),
            height=200,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Key Statistics
        st.markdown('<p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 1rem;">Key statistics</p>', unsafe_allow_html=True)
        
        cols = st.columns(4)
        
        metrics = [
            ("💓", "88", "Heart Rate", "bpm"),
            ("🫀", "120", "Blood Oxygen", "SpO₂"),
            ("🌡️", "98", "Body Temp", "°F"),
            ("😴", "20", "Respiration", "breaths/min")
        ]
        
        for col, (icon, value, label, unit) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}<br>{unit}</div>
                </div>
                """, unsafe_allow_html=True)

# Remove bottom navigation section since it's now at top
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
