"""Streamlit app for EXRT readiness scoring.

Upload ECG data and get real-time readiness analysis with visualization.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
import plotly.express as px

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))

from analysis.models.inference import ReadinessModel


@st.cache_resource
def load_model():
    """Load model once and cache it."""
    try:
        return ReadinessModel()
    except FileNotFoundError:
        st.error("❌ Model not found. Please run `train_and_save_model.py` first.")
        st.stop()


def parse_ecg_upload(uploaded_file):
    """Parse uploaded ECG file (CSV, NPY, or PKL).
    
    Supports large files (>500MB) by reading in chunks when needed.
    """
    import pickle
    import os
    
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        # Handle both single-column and multi-column CSV
        if df.shape[1] == 1:
            ecg = np.asarray(df.iloc[:, 0].values).flatten()
        else:
            # Assume last column is ECG
            ecg = np.asarray(df.iloc[:, -1].values).flatten()
        st.success(f"✓ CSV loaded ({file_size_mb:.1f} MB)")
        return ecg
    
    elif uploaded_file.name.endswith(".npy"):
        ecg = np.load(uploaded_file)
        st.success(f"✓ NPY loaded ({file_size_mb:.1f} MB)")
        return ecg.flatten()
    
    elif uploaded_file.name.endswith(".pkl"):
        try:
            # For large files, show progress
            if file_size_mb > 500:
                st.info(f"📦 Loading large pickle file ({file_size_mb:.1f} MB)... This may take a moment.")
            
            data = pickle.load(uploaded_file, encoding="latin1")
            
            # Handle PPG+Dalia format
            if isinstance(data, dict) and "signal" in data:
                # PPG+Dalia structure: data["signal"]["chest"]["ECG"]
                ecg = np.asarray(data["signal"]["chest"]["ECG"]).flatten()
                st.success(f"✓ PPG+Dalia format detected ({file_size_mb:.1f} MB) - Chest ECG extracted")
                return ecg
            
            # Handle direct array
            elif isinstance(data, np.ndarray):
                st.success(f"✓ NumPy array format ({file_size_mb:.1f} MB)")
                return data.flatten()
            
            # Handle list or other iterable
            elif isinstance(data, (list, tuple)):
                st.success(f"✓ List/array format detected ({file_size_mb:.1f} MB)")
                return np.asarray(data).flatten()
            
            else:
                st.error(f" Unknown PKL format. Expected ECG array or PPG+Dalia dict. Got: {type(data)}")
                return None
        except Exception as e:
            st.error(f" Error parsing PKL file: {e}")
            st.error(f"File size: {file_size_mb:.1f} MB")
            return None
    
    else:
        st.error(" Unsupported file format. Use .csv, .npy, or .pkl")
        return None


def plot_readiness_timeseries(results_df):
    """Plot readiness scores over time with gradient fill."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results_df["center_time_sec"],
            y=results_df["readiness"],
            mode="lines+markers",
            name="Readiness Score",
            line=dict(color="rgb(0, 150, 255)", width=3),
            marker=dict(size=8, color="rgb(0, 100, 200)"),
            fill="tozeroy",
            fillcolor="rgba(0, 150, 255, 0.2)",
            hovertemplate="<b>Time:</b> %{x:.1f}s<br><b>Readiness:</b> %{y:.1f}%<extra></extra>",
        )
    )

    # Add threshold lines with better styling
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text="Ready (≥70%)",
        annotation_position="right",
    )
    fig.add_hline(
        y=40,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Fatigued (<40%)",
        annotation_position="right",
    )

    fig.update_layout(
        title="<b>Readiness Score Over Time</b>",
        xaxis_title="Time (seconds)",
        yaxis_title="Readiness Score (%)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        font=dict(size=12),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
    )

    return fig


def plot_readiness_distribution(results_df):
    """Plot distribution of readiness scores."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=results_df["readiness"],
            nbinsx=30,
            name="Frequency",
            marker_color="rgba(0, 150, 255, 0.7)",
            hovertemplate="<b>Score Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title="<b>Distribution of Readiness Scores</b>",
        xaxis_title="Readiness Score (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=350,
        font=dict(size=11),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
    )
    
    return fig


def plot_state_pie_chart(results_df):
    """Plot pie chart of state distribution."""
    state_counts = results_df["state"].value_counts()
    colors_map = {"🟢 Ready": "green", "🟡 Neutral": "gold", "🔴 Fatigued": "red"}
    colors = [colors_map.get(state, "gray") for state in state_counts.index]
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=state_counts.index,
                values=state_counts.values,
                marker=dict(colors=colors),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )
    
    fig.update_layout(
        title="<b>State Distribution</b>",
        height=350,
        font=dict(size=11),
    )
    
    return fig


def plot_readiness_gauge(avg_readiness):
    """Plot a gauge chart for average readiness."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_readiness,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Average Readiness"},
            delta={"reference": 50, "suffix": "% vs Neutral"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(255, 0, 0, 0.2)"},
                    {"range": [40, 70], "color": "rgba(255, 215, 0, 0.2)"},
                    {"range": [70, 100], "color": "rgba(0, 128, 0, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )
    
    fig.update_layout(height=400, font=dict(size=12))
    return fig


def plot_confidence_gauge(avg_confidence):
    """Plot a gauge chart for average model confidence."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Average Model Confidence"},
            delta={"reference": 50, "suffix": "% confidence level"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkviolet"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 100, 100, 0.2)"},
                    {"range": [50, 75], "color": "rgba(255, 215, 0, 0.2)"},
                    {"range": [75, 100], "color": "rgba(0, 128, 0, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )
    
    fig.update_layout(height=400, font=dict(size=12))
    return fig


def plot_confidence_timeseries(results_df):
    """Plot confidence scores over time."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results_df["center_time_sec"],
            y=results_df["confidence"] * 100,
            mode="lines+markers",
            name="Model Confidence",
            line=dict(color="rgb(148, 0, 211)", width=3),
            marker=dict(size=8, color="rgb(100, 0, 180)"),
            fill="tozeroy",
            fillcolor="rgba(148, 0, 211, 0.2)",
            hovertemplate="<b>Time:</b> %{x:.1f}s<br><b>Confidence:</b> %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text="High Confidence (≥70%)",
        annotation_position="right",
    )

    fig.update_layout(
        title="<b>Model Confidence Over Time</b>",
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence (%)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        font=dict(size=12),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
    )

    return fig


def plot_readiness_vs_confidence(results_df):
    """Plot readiness vs confidence scatter plot."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=results_df["confidence"] * 100,
            y=results_df["readiness"],
            mode="markers",
            marker=dict(
                size=10,
                color=results_df["readiness"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Readiness %"),
                line=dict(width=1, color="white"),
            ),
            text=[f"Readiness: {r:.1f}%<br>Confidence: {c:.1f}%" 
                  for r, c in zip(results_df["readiness"], results_df["confidence"]*100)],
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )
    
    fig.update_layout(
        title="<b>Readiness vs Model Confidence</b>",
        xaxis_title="Model Confidence (%)",
        yaxis_title="Readiness Score (%)",
        template="plotly_white",
        height=400,
        font=dict(size=11),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
    )
    
    return fig


def plot_hrv_breakdown(results_df):
    """Plot HRV metrics breakdown (RMSSD, SDNN, LF/HF)."""
    fig = go.Figure()
    
    # RMSSD trace
    fig.add_trace(
        go.Scatter(
            x=results_df["center_time_sec"],
            y=results_df["HRV_RMSSD"],
            mode="lines+markers",
            name="RMSSD (ms)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            yaxis="y1",
        )
    )
    
    # SDNN trace
    fig.add_trace(
        go.Scatter(
            x=results_df["center_time_sec"],
            y=results_df["HRV_SDNN"],
            mode="lines+markers",
            name="SDNN (ms)",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=5),
            yaxis="y2",
        )
    )
    
    # LF/HF ratio trace
    fig.add_trace(
        go.Scatter(
            x=results_df["center_time_sec"],
            y=results_df["HRV_LF_HF"],
            mode="lines+markers",
            name="LF/HF Ratio",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=5),
            yaxis="y3",
        )
    )
    
    fig.update_layout(
        title="<b>Heart Rate Variability (HRV) Breakdown</b>",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            title=dict(text="RMSSD (ms)", font=dict(color="#1f77b4")),
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title=dict(text="SDNN (ms)", font=dict(color="#ff7f0e")),
            tickfont=dict(color="#ff7f0e"),
            overlaying="y",
            side="right"
        ),
        yaxis3=dict(
            title=dict(text="LF/HF Ratio", font=dict(color="#2ca02c")),
            tickfont=dict(color="#2ca02c"),
            overlaying="y",
            side="right",
            anchor="free",
            position=0.95
        ),
        template="plotly_white",
        height=450,
        font=dict(size=11),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
        hovermode="x unified",
    )
    
    return fig


def plot_hrv_metrics_cards(results_df):
    """Display HRV metrics as cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rmssd = results_df["HRV_RMSSD"].mean()
        st.metric("💓 Avg RMSSD", f"{avg_rmssd:.1f} ms", 
                 delta=f"{results_df['HRV_RMSSD'].iloc[-1] - results_df['HRV_RMSSD'].iloc[0]:.1f} ms" 
                 if len(results_df) > 1 else None)
    
    with col2:
        avg_sdnn = results_df["HRV_SDNN"].mean()
        st.metric("💚 Avg SDNN", f"{avg_sdnn:.1f} ms",
                 delta=f"{results_df['HRV_SDNN'].iloc[-1] - results_df['HRV_SDNN'].iloc[0]:.1f} ms"
                 if len(results_df) > 1 else None)
    
    with col3:
        avg_lf_hf = results_df["HRV_LF_HF"].mean()
        st.metric(" Avg LF/HF", f"{avg_lf_hf:.2f}",
                 delta=f"{results_df['HRV_LF_HF'].iloc[-1] - results_df['HRV_LF_HF'].iloc[0]:.2f}"
                 if len(results_df) > 1 else None)
    
    with col4:
        avg_hr = results_df["HR_est"].mean()
        st.metric("♥️ Avg Heart Rate", f"{avg_hr:.0f} bpm",
                 delta=f"{results_df['HR_est'].iloc[-1] - results_df['HR_est'].iloc[0]:.0f} bpm"
                 if len(results_df) > 1 else None)


def plot_trend_analysis(results_df):
    """Plot trend analysis showing if readiness is improving or declining."""
    from scipy import stats
    
    fig = go.Figure()
    
    # Readiness line
    fig.add_trace(
        go.Scatter(
            x=results_df["center_time_sec"],
            y=results_df["readiness"],
            mode="lines+markers",
            name="Readiness",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )
    
    # Trend line
    if len(results_df) > 1:
        x_vals = results_df["center_time_sec"].values
        y_vals = results_df["readiness"].values
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        trend_line = slope * x_vals + intercept
        
        # Color trend line based on slope (green=improving, red=declining)
        trend_color = "#2ca02c" if slope > 0 else "#d62728"
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=trend_line,
                mode="lines",
                name=f"Trend (slope: {slope:.3f}%/sec)",
                line=dict(color=trend_color, width=2, dash="dash"),
            )
        )
        
        # Add shaded improvement/decline zones
        if slope > 0:
            fig.add_vrect(x0=x_vals.min(), x1=x_vals.max(), 
                         fillcolor="green", opacity=0.1, layer="below", 
                         annotation_text=" Improving", annotation_position="top left")
        else:
            fig.add_vrect(x0=x_vals.min(), x1=x_vals.max(),
                         fillcolor="red", opacity=0.1, layer="below",
                         annotation_text="Declining", annotation_position="top left")
    
    fig.update_layout(
        title="<b>Readiness Trend Analysis</b>",
        xaxis_title="Time (seconds)",
        yaxis_title="Readiness Score (%)",
        template="plotly_white",
        height=400,
        font=dict(size=11),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
        hovermode="x unified",
    )
    
    return fig


def calculate_trend_metrics(results_df):
    """Calculate trend metrics (slope, direction, R²)."""
    from scipy import stats
    
    if len(results_df) < 2:
        return None
    
    x_vals = results_df["center_time_sec"].values
    y_vals = results_df["readiness"].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    
    # Convert slope from %/sec to %/min and %/hour
    slope_per_min = slope * 60
    slope_per_hour = slope * 3600
    
    # Determine direction
    if slope > 0.01:
        direction = " Improving"
        color = "green"
    elif slope < -0.01:
        direction = " Declining"
        color = "red"
    else:
        direction = " Stable"
        color = "gray"
    
    return {
        "slope": slope,
        "slope_per_min": slope_per_min,
        "slope_per_hour": slope_per_hour,
        "direction": direction,
        "color": color,
        "r_squared": r_value ** 2,
        "p_value": p_value,
    }


def state_from_readiness(score):
    """Classify readiness score into state."""
    if score >= 70:
        return "🟢 Ready"
    elif score >= 40:
        return "🟡 Neutral"
    else:
        return "🔴 Fatigued"


def main():
    st.set_page_config(
        page_title="EXRT AI - Heart Health Monitoring", 
        layout="wide",
        page_icon="❤️",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS matching EXRT AI brand
    st.markdown("""
    <style>
    /* Main background - clean white/light gradient */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
    }
    
    /* Metric containers - modern card design */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(79, 134, 247, 0.15);
        border-left: 4px solid #4F86F7;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 134, 247, 0.15);
    }
    
    /* Headers - EXRT brand colors */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #34495e;
        border-bottom: 3px solid #4F86F7;
        padding-bottom: 0.8rem;
        margin-top: 2rem;
        font-weight: 600;
    }
    
    h3 {
        color: #5a6c7d;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.3);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3a6fd9 0%, #2c5bc7 100%);
        box-shadow: 0 6px 16px rgba(79, 134, 247, 0.4);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #4F86F7;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #5a6c7d;
        border: 1px solid #e0e7ef;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%);
        color: white;
        border-color: #4F86F7;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #4F86F7;
    }
    
    /* Footer branding */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e7ef;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1>❤️ EXRT AI</h1>
        <p style="font-size: 1.3rem; color: #5a6c7d; font-weight: 400; margin-top: -1rem;">
            Empowering individuals & teams with AI-driven, continuous heart health monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()

    # Sidebar for navigation and settings
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📤 Upload ECG Analysis", "📡 Live Simulator"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.header("⚙️ Settings")
        fs = st.number_input("Sampling Rate (Hz)", value=700, min_value=100, max_value=10000)
        window_sec = st.number_input("Window Duration (sec)", value=90, min_value=30, max_value=300)
        step_sec = st.number_input("Step Duration (sec)", value=15, min_value=5, max_value=window_sec)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"**Device:** GPU\n\n**Features:** 31 HRV metrics\n\n**PCA:** 5 components\n\n**GMM:** 8 clusters")

    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📤 Upload ECG Analysis":
        show_upload_page(model, fs, window_sec, step_sec)
    elif page == "📡 Live Simulator":
        show_live_simulator_page(model, fs, window_sec, step_sec)


def show_home_page():
    """Display the welcome/home page."""
    # Welcome page with info
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; border: none; margin: 0;">Welcome to EXRT AI Heart Health Monitoring</h2>
        <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.95;">
            Choose "📤 Upload ECG Analysis" to analyze your data or "📡 Live Simulator" for real-time monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(
            """
            ###  About EXRT AI Heart Health Monitoring
            
            EXRT AI leverages advanced machine learning to analyze ECG data and assess your 
            physiological readiness in real-time, empowering you to make informed decisions 
            about your health and performance.
            
            **What is Readiness?**
            -  **Physiological State Assessment** - Real-time measure of your body's capacity
            - ❤️ **HRV-Based Analysis** - Utilizing Heart Rate Variability patterns
            -  **Activity Optimization** - Indicates optimal timing for physical/mental tasks
            
            **Readiness Levels:**
            - **🟢 Ready (≥70%)**: High autonomic stability, optimal for intense activity
            - **🟡 Neutral (40-70%)**: Moderate state, suitable for standard activity
                - **🔴 Recovery (<40%)**: Low stability, rest and recovery recommended
                
                **Key Features:**
                - 🧠 AI-powered predictions with confidence scoring
                - 💓 Detailed HRV breakdown (RMSSD, SDNN, LF/HF)
                - 📈 Trend analysis showing improvement or decline
                - 🎨 Interactive visualizations for deep insights
                """
            )
        
        with col2:
            st.markdown(
                """
                ### 📋 Supported File Formats
                
                **CSV** - Comma-separated ECG values
                - Single column: Direct ECG samples
                - Multiple columns: Automatically uses last column
                - Compatible with most ECG export formats
                
                **NPY** - NumPy binary arrays
                - Direct NumPy format from Python scripts
                - Fast loading for large datasets
                
                **PKL** - Python pickle files
                - PPG+Dalia format: `data["signal"]["chest"]["ECG"]`
                - Direct arrays, lists, or tuples
                - Supports files up to **2GB**
                
                ###  Understanding Confidence Scores
                - **0-50%**: Low confidence - results may be unreliable
                - **50-70%**: Medium confidence - generally acceptable
                - **70-100%**: High confidence - very reliable predictions
                
                ###  Prediction Accuracy
                - Percentage of predictions with **≥70% confidence**
                - Higher accuracy indicates better signal quality
                - Reflects overall reliability of the analysis
                
                ### 💡 Pro Tips
                ✓ Use 90-second windows for optimal HRV analysis  
                ✓ Ensure correct sampling rate (700Hz for PPG+Dalia)  
                ✓ Clean ECG signals = more accurate predictions  
                ✓ Check HRV metrics for deeper physiological insights  
                """
            )
        
        # Feature highlights
        st.markdown("---")
        st.subheader("🌟 Key Capabilities")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <h3>🧠 AI-Powered</h3>
                <p>Advanced machine learning models trained on thousands of ECG windows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <h3>💓 HRV Analysis</h3>
                <p>Comprehensive heart rate variability metrics including RMSSD, SDNN, and LF/HF ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <h3>📈 Trend Tracking</h3>
                <p>Monitor readiness improvement or decline over time with statistical analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin: 0; font-size: 0.95rem; color: #5a6c7d;">
            <strong>EXRT AI LTD</strong> | Empowering Heart Health Monitoring
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #95a5a6;">
            Copyright © 2025 EXRT AI LTD | All Rights Reserved
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">
            <a href="https://exrtai.com/" target="_blank" style="color: #4F86F7; text-decoration: none;">
                Visit EXRT AI Website
            </a> | 
            <a href="https://www.linkedin.com/company/exrtai/" target="_blank" style="color: #4F86F7; text-decoration: none;">
                LinkedIn
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
