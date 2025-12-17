import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="EXRT AI Platform Dashboard",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Import Inter Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #F4F5F7;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Top Navigation Bar */
    .top-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #2C3E50;
        color: white;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.25rem;
        font-weight: 700;
    }
    
    .nav-links {
        display: flex;
        gap: 1.5rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .nav-link {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: background-color 0.15s;
        cursor: pointer;
        text-decoration: none;
        color: white;
    }
    
    .nav-link:hover {
        background-color: rgba(255,255,255,0.1);
    }
    
    .nav-link.active {
        background-color: #374151;
        color: #10B981;
    }
    
    /* Main Content Spacing */
    .main-content {
        margin-top: 80px;
        padding: 2rem;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Feature Cards */
    .feature-card {
        text-align: center;
        padding: 1.5rem;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #111827;
    }
    
    .feature-desc {
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-ready {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .badge-caution {
        background-color: #FEF3C7;
        color: #92400E;
    }
    
    .badge-recovery {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    .badge-live {
        background-color: #EF4444;
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Metric Boxes */
    .metric-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    
    /* Readiness Bars */
    .readiness-bar {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: white;
        font-weight: 600;
    }
    
    .bar-ready {
        background-color: #10B981;
    }
    
    .bar-caution {
        background-color: #FBBF24;
        color: #1F2937;
    }
    
    .bar-recovery {
        background-color: #EF4444;
    }
    
    /* Progress Bars */
    .progress-container {
        background: rgba(255,255,255,0.3);
        border-radius: 9999px;
        height: 0.5rem;
        margin-left: 1rem;
        flex: 1;
    }
    
    .progress-bar {
        background: white;
        height: 100%;
        border-radius: 9999px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.15s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(to right, #4F46E5, #7C3AED);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Live Indicator */
    .live-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .live-dot {
        width: 0.75rem;
        height: 0.75rem;
        background-color: #10B981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    /* Table Styles */
    .player-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .player-table th {
        background: #F9FAFB;
        padding: 0.75rem 1.5rem;
        text-align: left;
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .player-table td {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #F3F4F6;
    }
    
    .player-table tr:hover {
        background: #F9FAFB;
        cursor: pointer;
    }
    
    /* Investment Section */
    .investment-card {
        background: white;
        border: 4px solid #C7D2FE;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    
    .impact-metric {
        text-align: center;
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .impact-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .impact-label {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* AI Response Boxes */
    .ai-response {
        background: linear-gradient(to right, #EFF6FF, #E0E7FF);
        border: 1px solid #C7D2FE;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .ai-title {
        font-size: 0.875rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .ai-text {
        font-size: 0.875rem;
        color: #374151;
        line-height: 1.6;
    }
    
    /* Personal Health Section */
    .health-card {
        background: linear-gradient(to right, #DBEAFE, #E0E7FF);
        border: 1px solid #93C5FD;
        border-radius: 1rem;
        padding: 1rem;
    }
    
    .patch-status {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Section Headers */
    .section-header {
        background: #10B981;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 0 0;
        font-weight: 700;
        font-size: 1.125rem;
        text-align: center;
    }
    
    .section-header-blue {
        background: #2563EB;
    }
    
    /* Loading Animation */
    .loading-shimmer {
        background: linear-gradient(to right, #EFF6FF 4%, #DBEAFE 25%, #EFF6FF 36%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite linear;
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
</style>
""", unsafe_allow_html=True)

# Top Navigation
st.markdown("""
<div class="top-nav">
    <div class="nav-logo">
        <span style="font-size: 2rem;">💚</span>
        <span>EXRT AI</span>
    </div>
    <div class="nav-links">
        <a href="#overview" class="nav-link active">Home</a>
        <a href="#live-simulator" class="nav-link">Live</a>
        <a href="#team-dashboard" class="nav-link">Team</a>
        <a href="#personal-health" class="nav-link" style="color: #93C5FD;">Personal Health</a>
        <a href="#business-model" class="nav-link" style="color: #FCD34D;">Business</a>
    </div>
    <div style="background: #374151; padding: 0.5rem; border-radius: 9999px; cursor: pointer;">
        <span style="font-size: 1.5rem;">👤</span>
    </div>
</div>
<div style="height: 80px;"></div>
""", unsafe_allow_html=True)

# Session State Initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'overview'
if 'live_hr' not in st.session_state:
    st.session_state.live_hr = 75
if 'live_fatigue' not in st.session_state:
    st.session_state.live_fatigue = 9.7

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyAh_aH2sYEI3MBFZ19sKnirwH0-a2hAM9I"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"

def call_gemini_api(prompt):
    """Call Gemini API for AI insights"""
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(GEMINI_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Unable to connect to AI service. Please try again later."
    except Exception as e:
        return f"Error: {str(e)}"

# Header
st.markdown("""
<div style="border-bottom: 1px solid #E5E7EB; padding-bottom: 1rem; margin-bottom: 2rem;">
    <h1 style="font-size: 2rem; font-weight: 800; color: #111827; margin: 0;">Platform Overview</h1>
    <p style="color: #6B7280; margin: 0.5rem 0 0 0;">EXRT AI: AI-Driven Health Monitoring and Performance Optimization.</p>
</div>
""", unsafe_allow_html=True)

# Section 1: Overview & Features
st.markdown('<div id="overview"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card feature-card">
        <div class="feature-icon">📤</div>
        <h2 class="feature-title">Upload & Analysis</h2>
        <p class="feature-desc">Securely upload physiological data for in-depth analysis and reporting.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card feature-card">
        <div class="feature-icon">💻</div>
        <h2 class="feature-title">Live Simulator</h2>
        <p class="feature-desc">Real-time monitoring and simulation of biometric signals (e.g., ECG).</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card feature-card">
        <div class="feature-icon">✨</div>
        <h2 class="feature-title">AI-Powered Insights</h2>
        <p class="feature-desc">Predictive readiness levels and tailored recovery/optimization plans.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Readiness Levels
st.markdown("""
<div class="card">
    <h2 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem;">Readiness Levels</h2>
    
    <div class="readiness-bar bar-ready">
        <span style="width: 100px;">Ready (85%)</span>
        <div class="progress-container">
            <div class="progress-bar" style="width: 85%;"></div>
        </div>
    </div>
    
    <div class="readiness-bar bar-caution">
        <span style="width: 100px;">Caution (50%)</span>
        <div class="progress-container">
            <div class="progress-bar" style="width: 50%; background-color: #FBBF24;"></div>
        </div>
        <span style="font-size: 0.875rem; margin-left: 1rem;">Monitor fatigue markers.</span>
    </div>
    
    <div class="readiness-bar bar-recovery">
        <span style="width: 100px;">Recovery (10%)</span>
        <div class="progress-container">
            <div class="progress-bar" style="width: 10%;"></div>
        </div>
        <span style="font-size: 0.875rem; margin-left: 1rem;">Immediate rest required.</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section 2: Live Monitoring Dashboard
st.markdown('<div id="live-simulator"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
    <h2 style="font-size: 1.5rem; font-weight: 700; display: flex; align-items: center; gap: 0.75rem;">
        Live ECG Monitoring Simulator
        <span class="status-badge badge-live">LIVE</span>
    </h2>
</div>
""", unsafe_allow_html=True)

live_col1, live_col2 = st.columns([3, 1])

with live_col2:
    if st.button("⚡ Analyze Live Vitals", use_container_width=True):
        with st.spinner("Connecting to AI engine..."):
            prompt = f"""You are monitoring a high-performance athlete in real-time.
            Current Telemetry:
            - Heart Rate: {st.session_state.live_hr} BPM
            - Fatigue Index: {st.session_state.live_fatigue}%
            - Signal Quality: 98%

            Provide a 1-sentence IMMEDIATE assessment of their physical state and a 1-sentence recommendation.
            Tone: Clinical, urgent if needed."""
            
            result = call_gemini_api(prompt)
            st.session_state.vitals_analysis = result
    
    if st.button("🔮 Predict Fatigue Trend", use_container_width=True):
        with st.spinner("Forecasting metrics..."):
            prompt = f"""Based on recent stress markers showing moderate to high levels.
            Current fatigue is {st.session_state.live_fatigue}%.
            Predict the athlete's fatigue level for the next 30 minutes. 
            Will they experience exhaustion or maintain steady state? 
            Provide a short forecast."""
            
            result = call_gemini_api(prompt)
            st.session_state.fatigue_prediction = result

with live_col1:
    st.markdown("""
    <div class="card">
        <!-- Device Status -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 1.5rem;">
            <div style="background: #10B981; color: white; padding: 1rem; border-radius: 0.75rem; font-weight: 700;">
                <div style="font-size: 1.125rem;">Device Connection Status</div>
                <div style="font-size: 1.875rem; margin-top: 0.25rem;">Connected</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Signal Quality</div>
                <div class="metric-value">98%</div>
                <div style="background: #E5E7EB; border-radius: 9999px; height: 0.25rem; margin-top: 0.5rem;">
                    <div style="background: #10B981; height: 100%; width: 98%; border-radius: 9999px;"></div>
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Battery</div>
                <div class="metric-value">87%</div>
                <div style="background: #E5E7EB; border-radius: 9999px; height: 0.25rem; margin-top: 0.5rem;">
                    <div style="background: #10B981; height: 100%; width: 87%; border-radius: 9999px;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Live Metrics Grid
metric_cols = st.columns(5)

with metric_cols[0]:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Heart Rate (BPM)</div>
        <div class="metric-value">{st.session_state.live_hr}</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Fatigue Score</div>
        <div class="metric-value">{st.session_state.live_fatigue}%</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Signal Integrity</div>
        <div class="metric-value">100.0%</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[3]:
    st.markdown("""
    <div class="metric-box" style="background: #FEF3C7;">
        <div class="metric-label" style="color: #92400E;">Readiness Status</div>
        <div class="metric-value" style="color: #92400E; font-size: 1.25rem;">Fatigued</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[4]:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Recovery Needed</div>
        <div class="metric-value">0%</div>
    </div>
    """, unsafe_allow_html=True)

# AI Analysis Results
if 'vitals_analysis' in st.session_state:
    st.markdown(f"""
    <div class="ai-response">
        <div class="ai-title">⚡ Real-time AI Assessment</div>
        <div class="ai-text">{st.session_state.vitals_analysis}</div>
    </div>
    """, unsafe_allow_html=True)

if 'fatigue_prediction' in st.session_state:
    st.markdown(f"""
    <div class="ai-response">
        <div class="ai-title">🔮 Predictive Fatigue Trajectory</div>
        <div class="ai-text">{st.session_state.fatigue_prediction}</div>
    </div>
    """, unsafe_allow_html=True)

# Stress Chart
st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Meditation Time/Stress Level</h3>", unsafe_allow_html=True)

stress_data = pd.DataFrame({
    'Time': ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'],
    'Stress': [65, 59, 80, 81, 56, 55],
    'Meditation Goal': [40, 40, 40, 40, 40, 40]
})

fig_stress = go.Figure()
fig_stress.add_trace(go.Scatter(x=stress_data['Time'], y=stress_data['Stress'], 
                                mode='lines', name='Stress', line=dict(color='#EF4444', width=3)))
fig_stress.add_trace(go.Scatter(x=stress_data['Time'], y=stress_data['Meditation Goal'], 
                                mode='lines', name='Meditation Goal', 
                                line=dict(color='#10B981', width=2, dash='dash')))
fig_stress.update_layout(
    height=250,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor='#F9FAFB',
    plot_bgcolor='#F9FAFB',
    yaxis=dict(range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_stress, use_container_width=True)

# ECG Signal
st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Realtime ECG Signal</h3>", unsafe_allow_html=True)

# Generate ECG waveform
t = np.linspace(0, 2, 500)
ecg_signal = []
for ti in t:
    phase = (ti % 1.0)
    if 0.1 < phase < 0.15:  # P wave
        val = 0.15 * np.sin((phase - 0.1) * np.pi / 0.05)
    elif 0.2 < phase < 0.25:  # Q wave
        val = -0.2 * np.sin((phase - 0.2) * np.pi / 0.05)
    elif 0.25 < phase < 0.3:  # R wave
        val = 1.5 * np.sin((phase - 0.25) * np.pi / 0.05)
    elif 0.3 < phase < 0.35:  # S wave
        val = -0.5 * np.sin((phase - 0.3) * np.pi / 0.05)
    elif 0.4 < phase < 0.5:  # T wave
        val = 0.3 * np.sin((phase - 0.4) * np.pi / 0.1)
    else:
        val = 0
    ecg_signal.append(val + np.random.normal(0, 0.02))

fig_ecg = go.Figure()
fig_ecg.add_trace(go.Scatter(y=ecg_signal, mode='lines', 
                            line=dict(color='#6366F1', width=2), 
                            fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'))
fig_ecg.update_layout(
    height=250,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor='#F9FAFB',
    plot_bgcolor='#F9FAFB',
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(range=[-1.5, 2.5], showgrid=True, gridcolor='#E5E7EB')
)
st.plotly_chart(fig_ecg, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section 3: Team Dashboard
st.markdown('<div id="team-dashboard"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    Professional Team Monitoring
</div>
<div class="card" style="border-radius: 0 0 1rem 1rem; margin-top: 0;">
    <h2 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem;">Squad Readiness Summary</h2>
</div>
""", unsafe_allow_html=True)

# Summary Metrics
team_cols = st.columns(5)

with team_cols[0]:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Total Players</div>
        <div class="metric-value">25</div>
    </div>
    """, unsafe_allow_html=True)

with team_cols[1]:
    st.markdown("""
    <div class="metric-box" style="background: #D1FAE5; border-color: #6EE7B7;">
        <div class="metric-label" style="color: #065F46;">Ready Players</div>
        <div class="metric-value" style="color: #065F46;">21</div>
    </div>
    """, unsafe_allow_html=True)

with team_cols[2]:
    st.markdown("""
    <div class="metric-box" style="background: #FEF3C7; border-color: #FCD34D;">
        <div class="metric-label" style="color: #92400E;">Caution Players</div>
        <div class="metric-value" style="color: #92400E;">3</div>
    </div>
    """, unsafe_allow_html=True)

with team_cols[3]:
    st.markdown("""
    <div class="metric-box" style="background: #FEE2E2; border-color: #FCA5A5;">
        <div class="metric-label" style="color: #991B1B;">Recovery Players</div>
        <div class="metric-value" style="color: #991B1B;">1</div>
    </div>
    """, unsafe_allow_html=True)

with team_cols[4]:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Avg. Readiness Score</div>
        <div class="metric-value">89.4%</div>
    </div>
    """, unsafe_allow_html=True)

# AI Squad Analysis Buttons
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 2])

with btn_col1:
    if st.button("✨ AI Squad Analyst", use_container_width=True):
        with st.spinner("Analyzing player metrics..."):
            prompt = """You are an elite sports performance analyst. 
            Analyze the following player data: 
            - Player A: Forward, Ready (95%), Fatigue: 3.1%
            - Player B: Midfielder, Caution (45%), Fatigue: 15.5%
            - Player C: Defender, Recovery (10%), Fatigue: 35.2%
            
            Provide a brief, bulleted strategic report. 
            1. Identify the biggest risk.
            2. Suggest a specific training adjustment for the 'Caution' and 'Recovery' players.
            Keep it professional and concise (max 100 words)."""
            
            result = call_gemini_api(prompt)
            st.session_state.squad_analysis = result

with btn_col2:
    if st.button("📋 Match Strategy", use_container_width=True):
        with st.spinner("Formulating match strategy..."):
            prompt = """You are a Head Coach. Based on this squad status:
            - Forwards: High readiness
            - Midfield: Medium-Low (Fatigued)
            - Defense: Critical (Recovery needed)
            
            Suggest a tactical approach for the upcoming match.
            Should we press high or park the bus? Who needs to be subbed early?
            Keep it to 3 bullet points."""
            
            result = call_gemini_api(prompt)
            st.session_state.match_strategy = result

with btn_col3:
    st.button("📄 Generate Full Team Report", use_container_width=True)

# Display AI Analysis
if 'squad_analysis' in st.session_state:
    st.markdown(f"""
    <div class="ai-response" style="background: linear-gradient(to right, #EDE9FE, #DDD6FE); border-color: #C4B5FD;">
        <div class="ai-title">✨ AI Squad Analysis</div>
        <div class="ai-text">{st.session_state.squad_analysis}</div>
    </div>
    """, unsafe_allow_html=True)

if 'match_strategy' in st.session_state:
    st.markdown(f"""
    <div class="ai-response" style="background: linear-gradient(to right, #DBEAFE, #BAE6FD); border-color: #7DD3FC;">
        <div class="ai-title">📋 Match Strategy</div>
        <div class="ai-text">{st.session_state.match_strategy}</div>
    </div>
    """, unsafe_allow_html=True)

# Player Status Table
st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem;'>Player Status Grid</h3>", unsafe_allow_html=True)

player_data = pd.DataFrame({
    'Player': ['Player A', 'Player B', 'Player C'],
    'Position': ['Forward', 'Midfielder', 'Defender'],
    'Score': ['95%', '45%', '10%'],
    'Status': ['Ready', 'Caution', 'Recovery'],
    'Fatigue': ['3.1%', '15.5%', '35.2%']
})

st.dataframe(
    player_data,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Position": st.column_config.TextColumn("Position", width="medium"),
        "Score": st.column_config.TextColumn("Score", width="small"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
        "Fatigue": st.column_config.TextColumn("Fatigue", width="small"),
    }
)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section 4: Personal Health Dashboard
st.markdown('<div id="personal-health"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-header section-header-blue" style="display: flex; justify-content: space-between; align-items: center;">
    <span>Personal Health Dashboard</span>
    <span style="font-size: 0.75rem; background: #3B82F6; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">Consumer Edition</span>
</div>
""", unsafe_allow_html=True)

# Patch Status
st.markdown("""
<div class="patch-status">
    <div class="live-dot"></div>
    <div style="flex: 1;">
        <h3 style="font-weight: 700; color: #1F2937; margin: 0;">ECG Patch Connected</h3>
        <p style="font-size: 0.75rem; color: #6B7280; margin: 0.25rem 0 0 0;">Battery: 92% | Last Sync: Just now</p>
    </div>
    <button style="font-size: 0.875rem; color: #2563EB; font-weight: 600; background: none; border: none; cursor: pointer;">Device Settings</button>
</div>
""", unsafe_allow_html=True)

personal_col1, personal_col2 = st.columns([1, 2])

with personal_col1:
    st.markdown("""
    <div class="metric-box">
        <h4 class="metric-label">Daily HRV</h4>
        <div style="display: flex; align-items: baseline; margin-top: 0.25rem;">
            <span class="metric-value">45</span>
            <span style="font-size: 0.875rem; color: #6B7280; margin-left: 0.25rem;">ms</span>
        </div>
        <p style="font-size: 0.75rem; color: #10B981; margin-top: 0.25rem;">↑ 5% vs yesterday</p>
    </div>
    <br>
    <div class="metric-box">
        <h4 class="metric-label">Stress Balance</h4>
        <div style="display: flex; align-items: baseline; margin-top: 0.25rem;">
            <span class="metric-value" style="color: #D97706;">Medium</span>
        </div>
        <div style="background: #E5E7EB; border-radius: 9999px; height: 0.5rem; margin-top: 0.5rem;">
            <div style="background: #FBBF24; height: 100%; width: 60%; border-radius: 9999px;"></div>
        </div>
    </div>
    <br>
    <div class="metric-box">
        <h4 class="metric-label">Resting Heart Rate</h4>
        <div style="display: flex; align-items: baseline; margin-top: 0.25rem;">
            <span class="metric-value">62</span>
            <span style="font-size: 0.875rem; color: #6B7280; margin-left: 0.25rem;">bpm</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with personal_col2:
    st.markdown("<h4 style='font-weight: 700; color: #374151; margin-bottom: 0.5rem;'>Weekly Wellness Trend</h4>", unsafe_allow_html=True)
    
    wellness_data = pd.DataFrame({
        'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'HRV Score': [42, 45, 40, 48, 52, 45, 47]
    })
    
    fig_wellness = px.bar(wellness_data, x='Day', y='HRV Score', 
                         color_discrete_sequence=['#60A5FA'])
    fig_wellness.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig_wellness, use_container_width=True)
    
    if st.button("✨ Get Daily Wellness Plan", use_container_width=True):
        with st.spinner("Analyzing daily biometrics..."):
            prompt = """You are a personal wellness coach for a regular person using a wearable ECG patch.
            Today's Data:
            - HRV: 45ms (Average for this user)
            - Resting Heart Rate: 62 bpm
            - Stress Balance: Medium
            
            Give a friendly, encouraging 2-sentence advice for today. Suggest a simple activity (like walking or breathing)."""
            
            result = call_gemini_api(prompt)
            st.session_state.wellness_advice = result
    
    if 'wellness_advice' in st.session_state:
        st.markdown(f"""
        <div class="ai-response" style="background: #DBEAFE; border-color: #93C5FD;">
            <div class="ai-text">{st.session_state.wellness_advice}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Section 5: Business Model & Investment
st.markdown('<div id="business-model"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="investment-card">
    <h2 style="font-size: 2rem; font-weight: 800; color: #4338CA; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.75rem;">
        <span style="font-size: 2rem;">🎯</span>
        Investment Thesis & Market Opportunity
    </h2>
</div>
""", unsafe_allow_html=True)

# Projected Impact
impact_cols = st.columns(3)

with impact_cols[0]:
    st.markdown("""
    <div class="impact-metric" style="background: #4F46E5;">
        <div class="impact-value">35%</div>
        <p class="impact-label">Injury Risk Reduction</p>
    </div>
    """, unsafe_allow_html=True)

with impact_cols[1]:
    st.markdown("""
    <div class="impact-metric" style="background: #4F46E5;">
        <div class="impact-value">30%</div>
        <p class="impact-label">Performance Uplift</p>
    </div>
    """, unsafe_allow_html=True)

with impact_cols[2]:
    st.markdown("""
    <div class="impact-metric" style="background: #FBBF24; color: #1F2937;">
        <div class="impact-value">$46M+</div>
        <p class="impact-label">Annual Client Value</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ROI Calculator
st.markdown("<h3 style='font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;'>Investor ROI Modeling (Interactive)</h3>", unsafe_allow_html=True)

roi_col1, roi_col2 = st.columns([1, 2])

with roi_col1:
    st.markdown("""
    <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 1rem;">
        <h4 style="font-weight: 700; color: #374151; margin-bottom: 1rem;">Client Cost Input</h4>
    </div>
    """, unsafe_allow_html=True)
    
    team_size = st.number_input("Team Size (Players)", min_value=1, value=25, step=1)
    avg_salary = st.number_input("Avg. Player Salary ($K)", min_value=1, value=500, step=10)
    
    st.markdown("<p style='font-size: 0.75rem; color: #6B7280; font-style: italic; margin-top: 0.5rem;'>Try changing these values to see real-time projections!</p>", unsafe_allow_html=True)

with roi_col2:
    st.markdown("<h4 style='font-weight: 700; color: #374151; margin-bottom: 1rem;'>Financial Projection: Value Delivered</h4>", unsafe_allow_html=True)
    
    # Calculate ROI
    payroll = team_size * avg_salary * 1000
    injury_savings = payroll * 1.5
    perf_value = payroll * 2.2
    total_value = injury_savings + perf_value
    cost = team_size * 20000
    roi_multiple = total_value / cost if cost > 0 else 0
    
    calc_cols = st.columns(3)
    
    with calc_cols[0]:
        st.markdown(f"""
        <div class="metric-box" style="background: #D1FAE5; border-color: #6EE7B7;">
            <div class="metric-label" style="color: #065F46;">Injury Cost Avoided</div>
            <div class="metric-value" style="color: #065F46; font-size: 1.5rem;">${injury_savings:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with calc_cols[1]:
        st.markdown(f"""
        <div class="metric-box" style="background: #D1FAE5; border-color: #6EE7B7;">
            <div class="metric-label" style="color: #065F46;">Performance Uplift Value</div>
            <div class="metric-value" style="color: #065F46; font-size: 1.5rem;">${perf_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with calc_cols[2]:
        st.markdown(f"""
        <div class="metric-box" style="background: #E0E7FF; border-color: #A5B4FC;">
            <div class="metric-label" style="color: #3730A3;">ROI Multiple (Annual)</div>
            <div class="metric-value" style="color: #3730A3; font-size: 1.5rem;">{roi_multiple:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 0.75rem; padding: 1rem; margin-top: 1rem;">
        <p style="font-size: 0.875rem; font-weight: 600; color: #92400E; margin: 0;">
            <strong>Investment Insight:</strong> Our technology generates a <strong>{roi_multiple:.1f}x</strong> return on investment for an average professional sports client annually.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Investment Memo Generator
st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Generate AI Investment Memo", use_container_width=True, type="primary"):
    with st.spinner("Drafting investment thesis from ROI data..."):
        prompt = f"""Write a professional, persuasive investment memo summary for 'EXRT AI'.
        
        Key Data:
        - Target Client Team Size: {team_size} athletes
        - Projected Annual Injury Savings: ${injury_savings:,.0f}
        - Projected Performance Uplift Value: ${perf_value:,.0f}
        - Calculated ROI Multiple: {roi_multiple:.1f}x

        Structure:
        1. **Executive Summary**: One powerful sentence.
        2. **The Problem**: High cost of injury/fatigue in pro sports.
        3. **The Solution**: EXRT AI's predictive monitoring.
        4. **Financial Thesis**: Highlight the ROI and savings.
        
        Keep it punchy, business-focused, and formatted in Markdown."""
        
        result = call_gemini_api(prompt)
        st.session_state.investment_memo = result

if 'investment_memo' in st.session_state:
    st.markdown(f"""
    <div class="card" style="border: 2px solid #C7D2FE; margin-top: 1.5rem;">
        <h4 style="font-size: 1.5rem; font-weight: 700; color: #4338CA; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #E5E7EB;">
            EXRT AI: Strategic Investment Brief
        </h4>
        <div style="color: #374151; line-height: 1.6;">
            {st.session_state.investment_memo}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<br><br>
<footer style="text-align: center; font-size: 0.75rem; color: #6B7280; border-top: 1px solid #E5E7EB; padding-top: 1rem; margin-top: 3rem;">
    &copy; 2025 EXRT AI. Empowering Next-Level Health Monitoring.
</footer>
""", unsafe_allow_html=True)

# Auto-refresh for live data (optional)
if st.checkbox("Enable Live Data Updates", value=False):
    time.sleep(0.1)
    st.session_state.live_hr = 70 + int(np.random.rand() * 15)
    st.session_state.live_fatigue = round(9 + np.random.rand(), 1)
    st.rerun()
