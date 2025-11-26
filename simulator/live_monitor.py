"""Live ECG monitoring Streamlit app with real-time simulation.

This app demonstrates continuous heart health monitoring with simulated ECG data.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
import plotly.express as px
from collections import deque

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from simulator.ecg_generator import ECGGenerator, LiveECGStream
from analysis.models.inference import ReadinessModel


def init_session_state():
    """Initialize session state variables."""
    if 'stream' not in st.session_state:
        st.session_state.stream = LiveECGStream(fs=700, chunk_duration=5.0)
    
    if 'ecg_buffer' not in st.session_state:
        st.session_state.ecg_buffer = deque(maxlen=7000)  # 10 seconds at 700Hz
    
    if 'readiness_history' not in st.session_state:
        st.session_state.readiness_history = deque(maxlen=100)
    
    if 'hr_history' not in st.session_state:
        st.session_state.hr_history = deque(maxlen=100)
    
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = deque(maxlen=100)
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()


@st.cache_resource
def load_model():
    """Load the readiness model."""
    try:
        return ReadinessModel()
    except FileNotFoundError:
        st.error("❌ Model not found. Please train the model first.")
        return None


def plot_live_ecg(ecg_data, fs=700):
    """Plot live ECG waveform."""
    if len(ecg_data) == 0:
        return None
    
    time_axis = np.arange(len(ecg_data)) / fs
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=ecg_data,
            mode='lines',
            name='ECG',
            line=dict(color='#4F86F7', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(79, 134, 247, 0.1)',
        )
    )
    
    fig.update_layout(
        title="<b>Live ECG Signal (Last 10 Seconds)</b>",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified',
    )
    
    return fig


def plot_readiness_trend(timestamps, readiness_values):
    """Plot readiness trend over time."""
    if len(timestamps) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(readiness_values),
            mode='lines+markers',
            name='Readiness',
            line=dict(color='#4F86F7', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(79, 134, 247, 0.2)',
        )
    )
    
    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="green", 
                  annotation_text="Ready", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="orange",
                  annotation_text="Neutral", annotation_position="right")
    
    fig.update_layout(
        title="<b>Readiness Trend</b>",
        xaxis_title="Time Elapsed (seconds)",
        yaxis_title="Readiness Score (%)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis_range=[0, 100],
    )
    
    return fig


def plot_hr_trend(timestamps, hr_values):
    """Plot heart rate trend."""
    if len(timestamps) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(hr_values),
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#e74c3c', width=2.5),
            marker=dict(size=5),
        )
    )
    
    fig.update_layout(
        title="<b>Heart Rate Trend</b>",
        xaxis_title="Time Elapsed (seconds)",
        yaxis_title="Heart Rate (BPM)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="EXRT AI - Live Monitor",
        layout="wide",
        page_icon="❤️",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #4F86F7;
    }
    
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3a6fd9 0%, #2c5bc7 100%);
        box-shadow: 0 6px 16px rgba(79, 134, 247, 0.4);
        transform: translateY(-2px);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    init_session_state()
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>❤️ EXRT AI - Live Heart Health Monitor</h1>
        <p style="font-size: 1.2rem; color: #5a6c7d;">
            Real-time Continuous ECG Monitoring & Readiness Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        
        # State selection
        state = st.selectbox(
            "Physiological State",
            options=['rest', 'active', 'stress', 'recovery'],
            index=0,
            help="Select the simulated physiological state"
        )
        
        state_duration = st.slider(
            "State Duration (sec)",
            min_value=30,
            max_value=300,
            value=60,
            step=10,
        )
        
        if st.button("🔄 Apply State"):
            st.session_state.stream.set_state(state, state_duration)
            st.success(f"State set to: {state.upper()}")
        
        st.markdown("---")
        
        # Scenario presets
        st.subheader("📋 Quick Scenarios")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏃 Workout"):
                st.session_state.stream.set_state('active', 120)
            if st.button("😰 Stress"):
                st.session_state.stream.set_state('stress', 90)
        
        with col2:
            if st.button("🧘 Rest"):
                st.session_state.stream.set_state('rest', 120)
            if st.button("💆 Recovery"):
                st.session_state.stream.set_state('recovery', 90)
        
        st.markdown("---")
        
        # Info
        st.info("""
        **How it works:**
        
        1. Select a physiological state
        2. Start monitoring
        3. Watch real-time ECG and readiness updates
        4. States auto-transition after duration
        """)
    
    # Main controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("▶️ Start Monitoring" if not st.session_state.is_running else "⏸️ Pause Monitoring",
                     use_container_width=True):
            st.session_state.is_running = not st.session_state.is_running
            if st.session_state.is_running:
                st.session_state.start_time = time.time()
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.ecg_buffer.clear()
            st.session_state.readiness_history.clear()
            st.session_state.hr_history.clear()
            st.session_state.timestamps.clear()
            st.session_state.start_time = time.time()
            st.rerun()
    
    # Live monitoring section
    if st.session_state.is_running:
        # Get next chunk
        chunk_data = st.session_state.stream.get_next_chunk()
        
        # Update ECG buffer
        st.session_state.ecg_buffer.extend(chunk_data['signal'])
        
        # Analyze if we have enough data (90 seconds minimum)
        if len(st.session_state.ecg_buffer) >= 63000:  # 90 seconds at 700Hz
            # Take last 90 seconds for analysis
            analysis_window = np.array(list(st.session_state.ecg_buffer)[-63000:])
            
            # Predict readiness
            result = model.predict_from_ecg(analysis_window, fs=700)
            
            if result:
                elapsed = time.time() - st.session_state.start_time
                st.session_state.readiness_history.append(result['score'])
                st.session_state.hr_history.append(chunk_data['actual_hr'])
                st.session_state.timestamps.append(elapsed)
        
        # Auto-refresh
        time.sleep(0.5)
        st.rerun()
    
    # Display current metrics
    st.subheader("📊 Current Status")
    
    if len(st.session_state.readiness_history) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        current_readiness = st.session_state.readiness_history[-1]
        current_hr = st.session_state.hr_history[-1]
        avg_readiness = np.mean(list(st.session_state.readiness_history))
        elapsed = time.time() - st.session_state.start_time
        
        with col1:
            st.metric(
                "🎯 Current Readiness",
                f"{current_readiness:.1f}%",
                delta=f"{current_readiness - avg_readiness:.1f}%"
            )
        
        with col2:
            st.metric(
                "💓 Heart Rate",
                f"{current_hr:.0f} BPM"
            )
        
        with col3:
            st.metric(
                "📈 Avg Readiness",
                f"{avg_readiness:.1f}%"
            )
        
        with col4:
            st.metric(
                "⏱️ Monitoring Time",
                f"{elapsed:.0f}s"
            )
        
        # State indicator
        current_state = st.session_state.stream.current_state
        state_colors = {
            'rest': '#2ecc71',
            'active': '#f39c12',
            'stress': '#e74c3c',
            'recovery': '#3498db',
        }
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: {state_colors.get(current_state, '#95a5a6')}20; 
                    border-radius: 10px; margin: 1rem 0;">
            <span class="status-indicator" style="background-color: {state_colors.get(current_state, '#95a5a6')};"></span>
            <strong>Current State: {current_state.upper()}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📈 Live Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ECG waveform
        ecg_fig = plot_live_ecg(list(st.session_state.ecg_buffer))
        if ecg_fig:
            st.plotly_chart(ecg_fig, use_container_width=True)
    
    with col2:
        # Readiness trend
        readiness_fig = plot_readiness_trend(
            st.session_state.timestamps,
            st.session_state.readiness_history
        )
        if readiness_fig:
            st.plotly_chart(readiness_fig, use_container_width=True)
    
    # HR trend
    hr_fig = plot_hr_trend(
        st.session_state.timestamps,
        st.session_state.hr_history
    )
    if hr_fig:
        st.plotly_chart(hr_fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #7f8c8d; margin-top: 2rem; border-top: 1px solid #e0e7ef;">
        <p style="margin: 0;"><strong>EXRT AI LTD</strong> | Live Heart Health Monitoring Simulator</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">
            Copyright © 2025 EXRT AI LTD | All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
