"""
EXRT AI Platform Dashboard - Exact Replica of dashboard.html
Built with Streamlit to match HTML/Tailwind styling pixel-perfect
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import base64

# Page Configuration
st.set_page_config(
    page_title="EXRT AI Platform Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Exact replica of dashboard.html styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
    
    :root {
        --primary-color: #10B981;
        --secondary-color: #6366F1;
        --dark-header: #2C3E50;
        --dark-background: #F4F5F7;
        --ready-color: #10B981;
        --caution-color: #FBBF24;
        --recovery-color: #EF4444;
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Remove Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Body Background */
    .main {
        background-color: var(--dark-background);
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Top Navigation Bar */
    .top-nav {
        background-color: var(--dark-header);
        color: white;
        padding: 0.75rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nav-links {
        display: flex;
        gap: 1.5rem;
        font-size: 0.875rem;
        font-weight: 700;
    }
    
    .nav-link {
        padding: 0.5rem;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: background-color 0.15s;
        text-decoration: none;
        color: white;
    }
    
    .nav-link:hover {
        background-color: #374151;
    }
    
    .nav-link.active {
        background-color: #374151;
        color: #10B981;
    }
    
    /* Main Content Area */
    .main-content {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Section Styling */
    .section-title {
        font-size: 1.875rem;
        font-weight: 800;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* White Cards */
    .white-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: all 0.3s;
    }
    
    .feature-card:hover {
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        transform: translateY(-4px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* Readiness Bars */
    .readiness-bar {
        padding: 0.75rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
    }
    
    .readiness-label {
        width: 120px;
    }
    
    .progress-container {
        flex-grow: 1;
        height: 8px;
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
        margin-left: 1rem;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: white;
        border-radius: 4px;
        box-shadow: 0 0 4px rgba(0,0,0,0.2);
    }
    
    /* Live Indicator */
    .live-badge {
        display: inline-block;
        background: #EF4444;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        animation: pulse-badge 2s infinite;
        margin-left: 0.75rem;
    }
    
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Status Metrics */
    .status-metric {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.15s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    /* Green Header */
    .green-header {
        background: #10B981;
        color: white;
        padding: 1rem;
        border-radius: 0.75rem 0.75rem 0 0;
        font-weight: 700;
        font-size: 1.125rem;
        text-align: center;
    }
    
    /* Blue Header */
    .blue-header {
        background: #3B82F6;
        color: white;
        padding: 1rem;
        border-radius: 0.75rem 0.75rem 0 0;
        font-weight: 700;
        font-size: 1.125rem;
        text-align: center;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    .edition-badge {
        background: #2563EB;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
    }
    
    /* Table Styling */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background: #F9FAFB;
        padding: 0.75rem 1.5rem;
        text-align: left;
        font-size: 0.75rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .dataframe td {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #E5E7EB;
        font-size: 0.875rem;
    }
    
    .dataframe tbody tr {
        background: white;
        cursor: pointer;
        transition: background-color 0.15s;
    }
    
    .dataframe tbody tr:hover {
        background: #F9FAFB;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-ready {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .badge-caution {
        background: #FEF3C7;
        color: #92400E;
    }
    
    .badge-recovery {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    /* Shimmer Loading */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading-shimmer {
        animation: shimmer 2s infinite linear;
        background: linear-gradient(to right, #eff6ff 4%, #dbeafe 25%, #eff6ff 36%);
        background-size: 1000px 100%;
    }
    
    /* Chart Container */
    .chart-container {
        background: #F9FAFB;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        padding: 0.5rem;
    }
    
    /* Streamlit Overrides */
    .stMarkdown {
        max-width: 100%;
    }
    
    div[data-testid="stHorizontalBlock"] {
        gap: 1.5rem;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Investment Section */
    .investment-gradient {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2rem;
        border-radius: 0.75rem;
        color: white;
        margin-bottom: 2rem;
    }
    
    .impact-card {
        padding: 2rem;
        border-radius: 0.75rem;
        text-align: center;
        color: white;
    }
    
    .impact-number {
        font-size: 3.5rem;
        font-weight: 800;
    }
    
    .impact-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'live_hr' not in st.session_state:
    st.session_state.live_hr = 75
if 'live_fatigue' not in st.session_state:
    st.session_state.live_fatigue = 9.7
if 'ecg_data' not in st.session_state:
    st.session_state.ecg_data = []
if 'stress_data' not in st.session_state:
    st.session_state.stress_data = [65, 59, 80, 81, 56, 55]
if 'team_size' not in st.session_state:
    st.session_state.team_size = 25
if 'avg_salary' not in st.session_state:
    st.session_state.avg_salary = 500

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🏥 EXRT AI")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📡 Live Monitoring", "👥 Team Dashboard", 
         "💪 Personal Health", "💰 Business Model"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 👤 Profile")
    if st.button("⚙️ Settings"):
        st.info("Settings page coming soon!")
    if st.button("🚪 Logout"):
        st.warning("Logout functionality not implemented")

# Helper Functions
def generate_dummy_ecg(length=50):
    """Generate realistic ECG pattern"""
    heartbeat_pattern = [0.1, 0.15, 0.1, 0, -0.2, 1.5, -0.5, 0, 0.2, 0.3, 0.2, 0, 0, 0]
    ecg = []
    for i in range(length):
        if i % 14 < len(heartbeat_pattern):
            ecg.append(heartbeat_pattern[i % 14] + np.random.normal(0, 0.05))
        else:
            ecg.append(np.random.normal(0, 0.1))
    return ecg

def calculate_roi(team_size, salary):
    """Calculate ROI metrics"""
    payroll = team_size * salary * 1000
    injury_savings = payroll * 1.5
    perf_value = payroll * 2.2
    total_value = injury_savings + perf_value
    cost = team_size * 20000
    roi = total_value / (cost or 1)
    return injury_savings, perf_value, roi

def get_ai_response(prompt_type):
    """Simulate AI responses"""
    responses = {
        "live_vitals": """**Live Vitals Analysis Report**

Your cardiovascular system is showing optimal stability:

• Heart Rate: 75 bpm - Within healthy resting range
• Fatigue Score: 9.7% - Very low fatigue levels
• Signal Quality: 98% - Outstanding sensor quality

**Recommendations:** Maintain current training intensity and focus on sleep quality.""",
        
        "fatigue_prediction": """**Fatigue Trend Prediction (Next 30 minutes)**

**Current State:** Low Fatigue (9.7%)
**Predicted Peak:** Low-Moderate (18%) at ~22 minutes
**Recovery Time:** 15-20 minutes post-activity

**Action Items:**
- Maintain steady state training
- Consume electrolyte drink within 15 min""",
        
        "team_analysis": """**AI Squad Performance Analysis**

**Player A - Forward (Ready 95%)**
- HRV: 62ms - Elite recovery
- Fatigue: 3.1% - Minimal
- Recommendation: Full training intensity

**Player B - Midfielder (Caution 45%)**
- HRV: 38ms - Below baseline
- Fatigue: 15.5% - Elevated
- Recommendation: 60% intensity

**Player C - Defender (Recovery 10%)**
- HRV: 28ms - Critical recovery needed
- Fatigue: 35.2% - High
- Recommendation: Rest protocol""",
        
        "match_strategy": """**AI Match Strategy Analysis**

**Current Team State:** 84% Ready

**Recommended Formation:**
- Deploy high-readiness forwards
- Rotate cautioned midfielders every 30 minutes
- Bench recovery players

**Predicted Outcome:** 72% win probability with rotation"""
    }
    return responses.get(prompt_type, "AI response not available.")

# ============================================================================
# PAGE: HOME / OVERVIEW
# ============================================================================
if page == "🏠 Home":
    st.markdown('<div class="main-header"><h1>EXRT AI Platform Dashboard</h1><p>AI-Driven Health Monitoring and Performance Optimization</p></div>', unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📤</div>
                <h3>Upload & Analysis</h3>
                <p style="color: #6B7280; font-size: 0.9rem;">
                    Securely upload physiological data for in-depth analysis and reporting.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📡</div>
                <h3>Live Simulator</h3>
                <p style="color: #6B7280; font-size: 0.9rem;">
                    Real-time monitoring and simulation of biometric signals (e.g., ECG).
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">🤖</div>
                <h3>AI-Powered Insights</h3>
                <p style="color: #6B7280; font-size: 0.9rem;">
                    Predictive readiness levels and tailored recovery/optimization plans.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Readiness Levels
    st.markdown("### Readiness Levels")
    
    # Ready
    st.markdown("""
    <div style="background: #10B981; padding: 1rem; border-radius: 8px; color: white; margin-bottom: 0.5rem;">
        <div style="display: flex; align-items: center;">
            <span style="width: 120px; font-weight: 600;">Ready (85%)</span>
            <div style="flex-grow: 1; background: rgba(255,255,255,0.3); height: 8px; border-radius: 4px; margin-left: 1rem;">
                <div style="width: 85%; background: white; height: 8px; border-radius: 4px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Caution
    st.markdown("""
    <div style="background: #FBBF24; padding: 1rem; border-radius: 8px; color: #1F2937; margin-bottom: 0.5rem;">
        <div style="display: flex; align-items: center;">
            <span style="width: 120px; font-weight: 600;">Caution (50%)</span>
            <div style="flex-grow: 1; background: rgba(0,0,0,0.2); height: 8px; border-radius: 4px; margin-left: 1rem;">
                <div style="width: 50%; background: #92400E; height: 8px; border-radius: 4px;"></div>
            </div>
            <span style="margin-left: 1rem; font-size: 0.9rem;">Monitor fatigue markers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recovery
    st.markdown("""
    <div style="background: #EF4444; padding: 1rem; border-radius: 8px; color: white; margin-bottom: 0.5rem;">
        <div style="display: flex; align-items: center;">
            <span style="width: 120px; font-weight: 600;">Recovery (10%)</span>
            <div style="flex-grow: 1; background: rgba(255,255,255,0.3); height: 8px; border-radius: 4px; margin-left: 1rem;">
                <div style="width: 10%; background: white; height: 8px; border-radius: 4px;"></div>
            </div>
            <span style="margin-left: 1rem; font-size: 0.9rem;">Immediate rest required</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: LIVE MONITORING
# ============================================================================
elif page == "📡 Live Monitoring":
    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 1rem;"><h2>Live ECG Monitoring Simulator</h2><span class="live-indicator" style="margin-left: 1rem;"></span><span style="color: #EF4444; font-weight: 600; font-size: 0.8rem;">LIVE</span></div>', unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⚡ Analyze Live Vitals"):
            with st.expander("🤖 AI Analysis", expanded=True):
                st.markdown(get_ai_response("live_vitals"))
    
    with col2:
        if st.button("🔮 Predict Fatigue Trend"):
            with st.expander("🤖 AI Prediction", expanded=True):
                st.markdown(get_ai_response("fatigue_prediction"))
    
    # Status Bar
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #10B981; padding: 1rem; border-radius: 8px; color: white;">
            <div style="font-size: 1.1rem; font-weight: 600;">Device Connection Status</div>
            <div style="font-size: 2rem; font-weight: 700; margin-top: 0.5rem;">Connected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Signal Quality", "98%", delta="2%")
    
    with col3:
        st.metric("Battery", "87%", delta="-5%")
    
    st.markdown("---")
    
    # Live Metrics
    st.markdown("### Live Monitoring Dashboard")
    
    # Update live metrics with slight randomization
    if np.random.random() > 0.5:
        st.session_state.live_hr = 70 + np.random.randint(0, 15)
        st.session_state.live_fatigue = round(9 + np.random.random(), 1)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Heart Rate (BPM)", st.session_state.live_hr)
    with col2:
        st.metric("Fatigue Score", f"{st.session_state.live_fatigue}%")
    with col3:
        st.metric("Signal Integrity", "100.0%")
    with col4:
        st.markdown('<div class="metric-card" style="background: #FEF3C7; border-color: #F59E0B;"><div style="font-size: 0.7rem; color: #92400E;">Readiness Status</div><div style="font-size: 1.5rem; font-weight: 700; color: #92400E;">Fatigued</div></div>', unsafe_allow_html=True)
    with col5:
        st.metric("Recovery Needed", "0%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Meditation Time/Stress Level")
        stress_fig = go.Figure()
        stress_fig.add_trace(go.Scatter(
            x=['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'],
            y=st.session_state.stress_data,
            mode='lines+markers',
            name='Stress',
            line=dict(color='#EF4444', width=3)
        ))
        stress_fig.add_trace(go.Scatter(
            x=['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'],
            y=[40, 40, 40, 40, 40, 40],
            mode='lines',
            name='Meditation Goal',
            line=dict(color='#10B981', width=2, dash='dash')
        ))
        stress_fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Time",
            yaxis_title="Level",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(stress_fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Realtime ECG Signal")
        
        # Generate ECG data
        if len(st.session_state.ecg_data) == 0:
            st.session_state.ecg_data = generate_dummy_ecg(50)
        else:
            # Update with new heartbeat pattern
            st.session_state.ecg_data.pop(0)
            st.session_state.ecg_data.append(generate_dummy_ecg(1)[0])
        
        ecg_fig = go.Figure()
        ecg_fig.add_trace(go.Scatter(
            y=st.session_state.ecg_data,
            mode='lines',
            line=dict(color='#6366F1', width=2),
            showlegend=False
        ))
        ecg_fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(range=[-1.5, 2.5], showgrid=True, gridcolor='#E5E7EB')
        )
        st.plotly_chart(ecg_fig, use_container_width=True)
    
    # Auto-refresh every 2 seconds
    time.sleep(0.05)
    st.rerun()

# ============================================================================
# PAGE: TEAM DASHBOARD
# ============================================================================
elif page == "👥 Team Dashboard":
    st.markdown('<div style="background: #10B981; color: white; padding: 1rem; border-radius: 12px 12px 0 0; text-align: center; font-weight: 700; font-size: 1.2rem;">Professional Team Monitoring</div>', unsafe_allow_html=True)
    
    st.markdown("### Squad Readiness Summary")
    
    # Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Players", "25")
    with col2:
        st.markdown('<div class="metric-card" style="background: #D1FAE5; border-color: #10B981;"><div style="font-size: 0.8rem; color: #065F46;">Ready Players</div><div style="font-size: 2rem; font-weight: 700; color: #065F46;">21</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card" style="background: #FEF3C7; border-color: #FBBF24;"><div style="font-size: 0.8rem; color: #92400E;">Caution Players</div><div style="font-size: 2rem; font-weight: 700; color: #92400E;">3</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card" style="background: #FEE2E2; border-color: #EF4444;"><div style="font-size: 0.8rem; color: #991B1B;">Recovery Players</div><div style="font-size: 2rem; font-weight: 700; color: #991B1B;">1</div></div>', unsafe_allow_html=True)
    with col5:
        st.metric("Avg. Readiness Score", "89.4%")
    
    # AI Analysis Buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📊 Generate Full Team Report"):
            st.info("Full report generation coming soon!")
    
    with col2:
        if st.button("✨ AI Squad Analyst"):
            with st.expander("🤖 AI Squad Analysis", expanded=True):
                st.markdown(get_ai_response("team_analysis"))
    
    with col3:
        if st.button("📋 Match Strategy"):
            with st.expander("🤖 AI Match Strategy", expanded=True):
                st.markdown(get_ai_response("match_strategy"))
    
    st.markdown("---")
    
    # Player Status Grid
    st.markdown("### Player Status Grid")
    
    players_df = pd.DataFrame({
        'Player': ['Player A', 'Player B', 'Player C'],
        'Position': ['Forward', 'Midfielder', 'Defender'],
        'Score': ['95%', '45%', '10%'],
        'Status': ['Ready', 'Caution', 'Recovery'],
        'Fatigue': ['3.1%', '15.5%', '35.2%']
    })
    
    # Custom styling for table
    def color_status(val):
        if val == 'Ready':
            return 'background-color: #D1FAE5; color: #065F46; font-weight: 600;'
        elif val == 'Caution':
            return 'background-color: #FEF3C7; color: #92400E; font-weight: 600;'
        elif val == 'Recovery':
            return 'background-color: #FEE2E2; color: #991B1B; font-weight: 600;'
        return ''
    
    st.dataframe(
        players_df.style.applymap(color_status, subset=['Status']),
        use_container_width=True,
        hide_index=True
    )
    
    # Player Details Modal (using expander)
    st.markdown("---")
    st.markdown("### Player Details")
    
    selected_player = st.selectbox("Select Player", players_df['Player'].tolist())
    
    player_data = players_df[players_df['Player'] == selected_player].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Position:** {player_data['Position']}")
    with col2:
        st.markdown(f"**Status:** {player_data['Status']}")
    with col3:
        st.markdown(f"**Readiness Score:** {player_data['Score']}")
    with col4:
        st.markdown(f"**Fatigue Level:** {player_data['Fatigue']}")
    
    st.info("""
    **Recent Analysis:** HRV metrics indicate optimal recovery. Sleep quality over last 48h has been high.
    
    **Recommendation:** Cleared for full training intensity.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Recovery Coach"):
            st.success("""**Recovery Protocol:**
            
- Light stretching (10 min)
- Hydration: 500ml water + electrolytes
- Foam rolling: 15 minutes
- Sleep: 8+ hours tonight""")
    
    with col2:
        if st.button("🍎 AI Nutritionist"):
            st.success("""**Post-Training Meal:**
            
- 150g grilled salmon (35g protein)
- 200g brown rice (45g carbs)
- Steamed broccoli
- 2-3L water throughout day""")

# ============================================================================
# PAGE: PERSONAL HEALTH
# ============================================================================
elif page == "💪 Personal Health":
    st.markdown('<div style="background: #3B82F6; color: white; padding: 1rem; border-radius: 12px 12px 0 0; display: flex; justify-content: space-between; align-items: center;"><span style="font-weight: 700; font-size: 1.2rem;">Personal Health Dashboard</span><span style="background: #2563EB; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem;">Consumer Edition</span></div>', unsafe_allow_html=True)
    
    # Patch Status
    st.markdown("""
    <div style="background: #DBEAFE; padding: 1rem; border-radius: 8px; border: 1px solid #93C5FD; margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center;">
            <div class="live-indicator"></div>
            <div>
                <div style="font-weight: 700; color: #1F2937;">ECG Patch Connected</div>
                <div style="font-size: 0.8rem; color: #6B7280;">Battery: 92% | Last Sync: Just now</div>
            </div>
        </div>
        <button style="background: none; border: none; color: #3B82F6; font-weight: 600; cursor: pointer;">Device Settings</button>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Daily Metrics")
        
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.7rem; color: #6B7280; text-transform: uppercase;">Daily HRV</div>
            <div style="font-size: 2rem; font-weight: 700; color: #1F2937;">45<span style="font-size: 1rem; color: #6B7280; margin-left: 0.5rem;">ms</span></div>
            <div style="font-size: 0.8rem; color: #10B981; margin-top: 0.5rem;">↑ 5% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="font-size: 0.7rem; color: #6B7280; text-transform: uppercase;">Stress Balance</div>
            <div style="font-size: 2rem; font-weight: 700; color: #F59E0B;">Medium</div>
            <div style="background: #E5E7EB; height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                <div style="width: 60%; background: #F59E0B; height: 8px; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <div style="font-size: 0.7rem; color: #6B7280; text-transform: uppercase;">Resting Heart Rate</div>
            <div style="font-size: 2rem; font-weight: 700; color: #1F2937;">62<span style="font-size: 1rem; color: #6B7280; margin-left: 0.5rem;">bpm</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Weekly Wellness Trend")
        
        wellness_fig = go.Figure()
        wellness_fig.add_trace(go.Bar(
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            y=[42, 45, 40, 48, 52, 45, 47],
            marker_color='#60A5FA',
            name='Daily HRV Score'
        ))
        wellness_fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="HRV Score (ms)",
            showlegend=False
        )
        st.plotly_chart(wellness_fig, use_container_width=True)
        
        if st.button("✨ Get Daily Wellness Plan", use_container_width=True):
            with st.expander("🤖 AI Wellness Coach", expanded=True):
                st.markdown("""**Daily Wellness Plan**

**Today's Metrics:** HRV 45ms (↑ 5%), Stress: Medium, HR: 62 bpm

**Morning Routine (7:00-9:00 AM):**
- Hydration: 500ml water with lemon
- Gentle stretching: 10 minutes
- Box breathing: 5 minutes (4-4-4-4)
- Breakfast: Oats + berries + almond milk

**Midday (1:00-2:00 PM):**
- Balanced lunch with protein + veggies
- 10-minute outdoor walk
- Hydration: 500ml water

**Evening (7:00-9:00 PM):**
- Light dinner 2-3 hours before bed
- Reduce screen time 1 hour before sleep
- Evening stretching: 15 minutes
- Sleep target: 8 hours

**Special Recommendation:** Your HRV is trending up - great sign of recovery!""")

# ============================================================================
# PAGE: BUSINESS MODEL
# ============================================================================
elif page == "💰 Business Model":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; display: flex; align-items: center;">
            <span style="font-size: 2rem; margin-right: 1rem;">🎯</span>
            Investment Thesis & Market Opportunity
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Projected Impact
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #6366F1; padding: 2rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 3.5rem; font-weight: 800;">35%</div>
            <div style="font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">Injury Risk Reduction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #6366F1; padding: 2rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 3.5rem; font-weight: 800;">30%</div>
            <div style="font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">Performance Uplift</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #FBBF24; padding: 2rem; border-radius: 12px; text-align: center; color: #1F2937;">
            <div style="font-size: 3.5rem; font-weight: 800;">$46M+</div>
            <div style="font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">Annual Client Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ROI Calculator
    st.markdown("### Investor ROI Modeling (Interactive)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Client Cost Input")
        
        st.session_state.team_size = st.number_input(
            "Team Size (e.g., Players)",
            min_value=1,
            max_value=100,
            value=st.session_state.team_size,
            step=1
        )
        
        st.session_state.avg_salary = st.number_input(
            "Avg. Player Salary (K)",
            min_value=100,
            max_value=10000,
            value=st.session_state.avg_salary,
            step=50
        )
        
        st.caption("💡 Try changing these values to see real-time projections!")
    
    with col2:
        st.markdown("#### Financial Projection: Value Delivered")
        
        injury_savings, perf_value, roi = calculate_roi(
            st.session_state.team_size,
            st.session_state.avg_salary
        )
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div style="background: #D1FAE5; padding: 1rem; border-radius: 8px; border: 1px solid #10B981; text-align: center;">
                <div style="font-size: 0.8rem; color: #065F46;">Injury Cost Avoided</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #065F46;">${injury_savings:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div style="background: #D1FAE5; padding: 1rem; border-radius: 8px; border: 1px solid #10B981; text-align: center;">
                <div style="font-size: 0.8rem; color: #065F46;">Performance Uplift Value</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #065F46;">${perf_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div style="background: #DBEAFE; padding: 1rem; border-radius: 8px; border: 1px solid #3B82F6; text-align: center;">
                <div style="font-size: 0.8rem; color: #1E40AF;">ROI Multiple (Annual)</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E40AF;">{roi:.1f}x</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"💡 **Investment Insight:** Our technology generates a **{roi:.1f}x** return on investment for an average professional sports client annually.")
    
    st.markdown("---")
    
    # Investment Memo Generator
    if st.button("✨ Generate AI Investment Memo", use_container_width=True):
        with st.expander("📄 EXRT AI: Strategic Investment Brief", expanded=True):
            st.markdown(f"""
**EXRT AI Investment Thesis - Series A Opportunity**

**Company:** EXRT AI Limited  
**Sector:** Digital Health & Sports Performance  
**Round:** Series A ($5M)  
**Market Opportunity:** $4.2B (Global cardiac monitoring TAM)

**The Problem:**
Pro sports teams lose $18M+ annually to preventable injuries and suboptimal performance management.

**The Solution:**
EXRT AI's real-time ECG + ML platform predicts fatigue and injury risk with 94% accuracy.

**Key Metrics:**
• 35% injury reduction
• 30% performance uplift
• {roi:.1f}x ROI for clients
• $46M+ annual client value delivered

**Technology Moat:**
- Proprietary 31-feature HRV extraction
- 8-cluster GMM fatigue model
- Sub-10ms GPU inference
- Patent-pending prediction algorithm

**Traction:**
- 3 pilot clients (pro sports teams)
- $120K ARR, 35% MoM growth
- 92% client retention

**Series A Use of Funds ($5M):**
- Product Development: 40% ($2M)
- Sales & Marketing: 35% ($1.75M)
- Operations & Team: 25% ($1.25M)

**Exit Strategy:** $150M+ acquisition target (Garmin, Whoop, Apple Health)

---
*Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}*
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem; padding: 2rem 0;">
    &copy; 2025 EXRT AI. Empowering Next-Level Health Monitoring.
</div>
""", unsafe_allow_html=True)
