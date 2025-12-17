import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="EXRT AI - Sports Performance Intelligence",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .ready-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .caution-card {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .recovery-card {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .backend-status {
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 1000;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .backend-online {
        background: #10B981;
        color: white;
    }
    .backend-offline {
        background: #EF4444;
        color: white;
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
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'ecg_data' not in st.session_state:
    st.session_state.ecg_data = None

# Backend health check
def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_API_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get('model_loaded', False)
    except:
        return False

# Generate synthetic ECG
def generate_synthetic_ecg(duration=90, hrv_level='normal'):
    """Generate synthetic ECG data at 700 Hz"""
    fs = 700
    total_samples = duration * fs
    ecg_data = []
    
    for i in range(total_samples):
        t = i / fs
        heart_rate = 72
        beat_period = 60.0 / heart_rate
        phase = (t % beat_period) / beat_period
        
        value = 50.0
        
        # P wave (atrial depolarization)
        if 0.0 <= phase < 0.1:
            value += 5 * np.sin(phase * 10 * np.pi)
        
        # PR segment
        elif 0.1 <= phase < 0.2:
            value += 0
        
        # QRS complex (ventricular depolarization)
        elif 0.2 <= phase < 0.25:
            value -= 2
        elif 0.25 <= phase < 0.30:
            value += 20
        elif 0.30 <= phase < 0.35:
            value -= 5
        
        # ST segment
        elif 0.35 <= phase < 0.45:
            value += 0
        
        # T wave (ventricular repolarization)
        elif 0.45 <= phase < 0.65:
            value += 8 * np.sin((phase - 0.45) * 5 * np.pi)
        
        # Add HRV variability
        if hrv_level == 'high':
            value += np.random.random() * 0.05
        elif hrv_level == 'low':
            value += np.random.random() * 0.01
        else:
            value += np.random.random() * 0.03
        
        ecg_data.append(value)
    
    return ecg_data

# Analyze with backend
def analyze_with_backend(ecg_data):
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/readiness",
            json={
                "ecg": ecg_data,
                "fs": 700,
                "window_sec": 90,
                "step_sec": 15
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Backend API Error: {str(e)}")
        return None

# Call Gemini API
def call_gemini_api(system_prompt, user_prompt):
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Error: Unable to get response from Gemini AI"
    except Exception as e:
        return f"Error: {str(e)}"

# Check backend status
st.session_state.backend_status = check_backend_health()

# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown('<h1 class="main-header">❤️ EXRT AI - Sports Performance Intelligence</h1>', unsafe_allow_html=True)
with col2:
    if st.session_state.backend_status:
        st.markdown('<div class="backend-status backend-online">✓ Backend Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="backend-status backend-offline">✗ Backend Offline</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🏠 Overview", "📊 Live Monitor", "👥 Team Dashboard", "💪 Personal Health", "💼 Business ROI"]
)

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "🏠 Overview":
    st.header("Overview")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Upload & Analysis</h3>
            <p>Upload ECG data and get instant AI-powered readiness assessments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📡 Live Simulator</h3>
            <p>Real-time ECG monitoring with ML backend and Gemini AI insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered Insights</h3>
            <p>Get personalized recommendations powered by Google Gemini AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Readiness levels
    st.subheader("Current Readiness Levels")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="ready-card">
            <h3>🟢 Ready</h3>
            <h1>85%</h1>
            <p>Optimal performance zone</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="caution-card">
            <h3>🟡 Caution</h3>
            <h1>50%</h1>
            <p>Monitor closely</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="recovery-card">
            <h3>🔴 Recovery</h3>
            <h1>15%</h1>
            <p>Rest recommended</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# LIVE MONITOR PAGE
# ============================================================================
elif page == "📊 Live Monitor":
    st.header("Live ECG Simulator")
    
    # Generate and display ECG chart
    if st.session_state.ecg_data is None:
        st.session_state.ecg_data = generate_synthetic_ecg(duration=10, hrv_level='normal')
    
    # Display ECG
    ecg_display = st.session_state.ecg_data[-700:]  # Last 1 second
    time_axis = np.arange(len(ecg_display)) / 700
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ecg_display,
        mode='lines',
        line=dict(color='#10B981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    fig.update_layout(
        title="Real-time ECG Signal",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Live metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("❤️ Heart Rate", "72 BPM", "↑ 2")
    with col2:
        st.metric("😰 Fatigue Index", "12.3%", "↓ 1.2%")
    with col3:
        st.metric("📶 Signal Quality", "98%", "↑ 0.5%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Analysis buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze Live Vitals (Gemini AI)", key="analyze_vitals"):
            with st.spinner("Analyzing with Gemini AI..."):
                system_prompt = "You are an expert sports medicine AI analyzing athlete vital signs."
                user_prompt = "Current vitals: HR 72 BPM, Fatigue 12.3%, Signal Quality 98%. Provide brief analysis and recommendations."
                response = call_gemini_api(system_prompt, user_prompt)
                st.success("Analysis Complete!")
                st.info(response)
    
    with col2:
        if st.button("📈 Predict Fatigue Trend (Gemini AI)", key="predict_fatigue"):
            with st.spinner("Predicting with Gemini AI..."):
                system_prompt = "You are a predictive sports analytics AI."
                user_prompt = "Based on current fatigue at 12.3% and HR at 72 BPM, predict fatigue trend for next 24 hours."
                response = call_gemini_api(system_prompt, user_prompt)
                st.success("Prediction Complete!")
                st.info(response)
    
    with col3:
        if st.button("🤖 ML Readiness Analysis", key="ml_analysis"):
            if not st.session_state.backend_status:
                st.error("❌ Backend Offline: Please start the FastAPI backend on port 8000")
            else:
                with st.spinner("Analyzing with ML Backend..."):
                    # Generate synthetic ECG
                    ecg_data = generate_synthetic_ecg(duration=90, hrv_level='normal')
                    
                    # Analyze with backend
                    results = analyze_with_backend(ecg_data)
                    
                    if results:
                        st.session_state.ml_results = results
                        st.success(f"✓ Analysis Complete: {results['summary']['n_windows']} windows analyzed")
                    else:
                        st.error("Failed to get results from backend")
    
    # Display ML results if available
    if st.session_state.ml_results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🤖 ML Backend Analysis Results")
        
        results = st.session_state.ml_results
        summary = results['summary']
        
        # Overall state badge
        state_colors = {
            'ready': '🟢',
            'neutral': '🟡',
            'fatigued': '🔴'
        }
        overall_state = results.get('overall_state', 'neutral')
        st.markdown(f"### {state_colors.get(overall_state, '🟡')} Overall State: {overall_state.upper()}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Readiness", f"{summary['avg_readiness']:.1f}")
        with col2:
            st.metric("Min Readiness", f"{summary['min_readiness']:.1f}")
        with col3:
            st.metric("Max Readiness", f"{summary['max_readiness']:.1f}")
        with col4:
            st.metric("Windows Analyzed", summary['n_windows'])
        
        # Readiness progression chart
        window_results = results['window_results']
        times = [w['center_time_sec'] for w in window_results]
        readiness = [w['readiness'] for w in window_results]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=readiness,
            mode='lines+markers',
            line=dict(color='#9333EA', width=3),
            fill='tozeroy',
            fillcolor='rgba(147, 51, 234, 0.1)',
            name='Readiness Score'
        ))
        fig.update_layout(
            title="Readiness Progression Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Readiness Score",
            yaxis_range=[0, 100],
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TEAM DASHBOARD PAGE
# ============================================================================
elif page == "👥 Team Dashboard":
    st.header("Team Dashboard")
    
    # Squad summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", "24")
    with col2:
        st.metric("Ready", "18", delta="75%", delta_color="normal")
    with col3:
        st.metric("Caution", "4", delta="17%", delta_color="inverse")
    with col4:
        st.metric("Rest Needed", "2", delta="8%", delta_color="inverse")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Player table
    players_data = {
        "Player": ["Marcus Johnson", "Sarah Chen", "David Park", "James Rodriguez"],
        "Position": ["Forward", "Midfielder", "Defender", "Goalkeeper"],
        "Readiness": [92, 88, 62, 38],
        "Fatigue": ["Low", "Low", "Moderate", "High"],
        "Status": ["🟢 Ready", "🟢 Ready", "🟡 Monitor", "🔴 Rest"]
    }
    
    df = pd.DataFrame(players_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Analysis buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 AI Squad Analyst", key="squad_analyst"):
            with st.spinner("Analyzing squad with Gemini AI..."):
                system_prompt = "You are an expert sports team analyst AI."
                user_prompt = "Analyze squad status: 18 ready, 4 caution, 2 rest. Top performers: Marcus (92%), Sarah (88%). Concerns: James (38%). Provide brief squad analysis."
                response = call_gemini_api(system_prompt, user_prompt)
                st.success("Analysis Complete!")
                st.info(response)
    
    with col2:
        if st.button("⚽ AI Match Strategy", key="match_strategy"):
            with st.spinner("Generating strategy with Gemini AI..."):
                system_prompt = "You are a tactical sports strategy AI."
                user_prompt = "Given squad readiness (75% ready, 17% caution, 8% rest), recommend match strategy and lineup optimization."
                response = call_gemini_api(system_prompt, user_prompt)
                st.success("Strategy Generated!")
                st.info(response)

# ============================================================================
# PERSONAL HEALTH PAGE
# ============================================================================
elif page == "💪 Personal Health":
    st.header("Personal Health Monitoring")
    
    # Connection status
    st.success("✅ Device Connected - Live monitoring active")
    
    # Daily metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("❤️ HRV (Heart Rate Variability)", "62 ms", "↑ 4 ms")
    with col2:
        st.metric("😌 Stress Level", "Low", "↓ 5%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 7-day wellness trend
    wellness_data = {
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Wellness Score": [82, 78, 85, 88, 84, 90, 87]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=wellness_data["Day"],
        y=wellness_data["Wellness Score"],
        marker_color='#6366F1',
        text=wellness_data["Wellness Score"],
        textposition='outside'
    ))
    fig.update_layout(
        title="7-Day Wellness Trend",
        yaxis_title="Wellness Score",
        yaxis_range=[0, 100],
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Wellness Plan
    if st.button("🌟 Get AI Wellness Plan", key="wellness_plan"):
        with st.spinner("Creating personalized plan with Gemini AI..."):
            system_prompt = "You are a holistic wellness AI coach for athletes."
            user_prompt = "Based on HRV 62ms, low stress, and 7-day wellness trend (avg 85), create a personalized wellness plan."
            response = call_gemini_api(system_prompt, user_prompt)
            st.success("Wellness Plan Generated!")
            st.info(response)

# ============================================================================
# BUSINESS ROI PAGE
# ============================================================================
elif page == "💼 Business ROI":
    st.header("Business ROI Calculator")
    
    st.markdown("### Calculate the value of AI-powered athlete monitoring")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        team_size = st.number_input("Team Size", min_value=1, max_value=100, value=25, step=1)
    
    with col2:
        avg_salary = st.number_input("Average Player Salary ($K)", min_value=100, max_value=10000, value=500, step=100)
    
    # Calculate ROI
    annual_payroll = team_size * avg_salary * 1000
    injury_savings = annual_payroll * 0.15  # 15% injury cost reduction
    performance_uplift = annual_payroll * 0.22  # 22% performance improvement value
    platform_cost = 500000
    total_value = injury_savings + performance_uplift
    roi_multiple = total_value / platform_cost
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💰 Annual Payroll", f"${annual_payroll:,.0f}")
        st.metric("🏥 Injury Cost Savings (15%)", f"${injury_savings:,.0f}")
        st.metric("📈 Performance Value Uplift (22%)", f"${performance_uplift:,.0f}")
    
    with col2:
        st.metric("💵 Platform Investment", f"${platform_cost:,.0f}")
        st.metric("✨ Total Annual Value", f"${total_value:,.0f}")
        st.metric("🚀 ROI Multiple", f"{roi_multiple:.1f}x", delta=f"+{(roi_multiple-1)*100:.0f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Investment Memo
    if st.button("📊 Generate AI Investment Memo", key="investment_memo"):
        with st.spinner("Creating investment memo with Gemini AI..."):
            system_prompt = "You are a sports business investment analyst AI."
            user_prompt = f"Create investment memo for AI monitoring platform. Team: {team_size} players, Payroll: ${annual_payroll:,.0f}, ROI: {roi_multiple:.1f}x, Value: ${total_value:,.0f}."
            response = call_gemini_api(system_prompt, user_prompt)
            st.success("Investment Memo Generated!")
            st.info(response)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "EXRT AI © 2025 | Powered by Google Gemini AI & ML Backend"
    "</div>",
    unsafe_allow_html=True
)
