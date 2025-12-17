"""Multi-page Streamlit app for EXRT AI Heart Health Monitoring.

Pages:
1. Home - Welcome and introduction
2. Upload ECG Analysis - Upload and analyze ECG files
3. Live Simulator - Real-time ECG monitoring with synthetic data
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
import plotly.express as px
from scipy import stats

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent / "simulator"))

from analysis.models.inference import ReadinessModel
from simulator.ecg_generator import generate_ecg_segment


# Import all visualization functions from original app
# (We'll copy them inline for now)

@st.cache_resource
def load_model():
    """Load model once and cache it."""
    try:
        return ReadinessModel()
    except FileNotFoundError:
        st.error(" Model not found. Please run `train_and_save_model.py` first.")
        st.stop()


def parse_ecg_upload(uploaded_file):
    """Parse uploaded ECG file (CSV, NPY, or PKL)."""
    import pickle
    
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            ecg = np.asarray(df.iloc[:, 0].values).flatten()
        else:
            ecg = np.asarray(df.iloc[:, -1].values).flatten()
        st.success(f"✓ CSV loaded ({file_size_mb:.1f} MB)")
        return ecg
    
    elif uploaded_file.name.endswith(".npy"):
        ecg = np.load(uploaded_file)
        st.success(f"✓ NPY loaded ({file_size_mb:.1f} MB)")
        return ecg.flatten()
    
    elif uploaded_file.name.endswith(".pkl"):
        try:
            if file_size_mb > 500:
                st.info(f"Loading large pickle file ({file_size_mb:.1f} MB)...")
            
            data = pickle.load(uploaded_file, encoding="latin1")
            
            if isinstance(data, dict) and "signal" in data:
                ecg = np.asarray(data["signal"]["chest"]["ECG"]).flatten()
                st.success(f"✓ PPG+Dalia format detected ({file_size_mb:.1f} MB)")
                return ecg
            elif isinstance(data, np.ndarray):
                st.success(f"✓ NumPy array format ({file_size_mb:.1f} MB)")
                return data.flatten()
            elif isinstance(data, (list, tuple)):
                st.success(f"✓ List/array format detected ({file_size_mb:.1f} MB)")
                return np.asarray(data).flatten()
            else:
                st.error(f" Unknown PKL format. Got: {type(data)}")
                return None
        except Exception as e:
            st.error(f" Error parsing PKL file: {e}")
            return None
    else:
        st.error(" Unsupported file format. Use .csv, .npy, or .pkl")
        return None


def state_from_readiness(score):
    """Classify readiness score into state."""
    if score >= 70:
        return "● Ready"
    elif score >= 40:
        return "○ Neutral"
    else:
        return "■ Fatigued"


def _classify_physiological_state(readiness, hrv_rmssd, heart_rate):
    """Classify physiological state based on readiness, HRV, and HR."""
    # High readiness + high HRV + low HR = Resting
    if readiness >= 75 and hrv_rmssd > 40 and heart_rate < 70:
        return "Resting/Sleep"
    # High readiness + moderate HRV + moderate HR = Recovery
    elif readiness >= 60 and 60 <= heart_rate <= 80:
        return "Recovery"
    # Moderate readiness + moderate HRV + elevated HR = Active
    elif 50 <= readiness < 75 and 80 < heart_rate < 100:
        return "Light Activity"
    # Low readiness + low HRV + high HR = Stress
    elif readiness < 50 and heart_rate >= 100:
        return "High Stress"
    # Default moderate state
    else:
        return "Normal Active"


def generate_ai_insights(results_df, avg_readiness, avg_confidence):
    """Generate AI-powered insights and recommendations."""
    insights = []
    
    # Overall readiness assessment
    if avg_readiness >= 75:
        insights.append("**Excellent Readiness** - Optimal state for high-intensity activities and performance.")
    elif avg_readiness >= 60:
        insights.append("**Good Readiness** - Suitable for moderate activities. Consider recovery if sustained.")
    elif avg_readiness >= 40:
        insights.append("**Moderate Readiness** - Light activities recommended. Monitor for fatigue.")
    else:
        insights.append("**Low Readiness** - Rest and recovery strongly recommended. Avoid intense activities.")
    
    # Confidence assessment
    if avg_confidence >= 0.75:
        insights.append(f"**High Model Confidence** ({avg_confidence*100:.1f}%) - Results are highly reliable.")
    elif avg_confidence < 0.5:
        insights.append(f"**Lower Confidence** ({avg_confidence*100:.1f}%) - Signal quality may vary. Consider longer recordings.")
    
    # Trend analysis
    trend_metrics = calculate_trend_metrics(results_df)
    if trend_metrics:
        if trend_metrics['direction'] == "Improving":
            insights.append(f"**Positive Trend** - Readiness improving at {abs(trend_metrics['slope_per_hour']):.1f}% per hour.")
        elif trend_metrics['direction'] == "Declining":
            insights.append(f"**Declining Trend** - Readiness decreasing at {abs(trend_metrics['slope_per_hour']):.1f}% per hour. Consider rest.")
        else:
            insights.append("**Stable Trend** - Readiness levels consistent.")
    
    # HRV insights
    if 'HRV_RMSSD' in results_df.columns:
        avg_rmssd = results_df['HRV_RMSSD'].mean()
        if avg_rmssd > 50:
            insights.append(f"**Excellent HRV** (RMSSD: {avg_rmssd:.1f}ms) - Strong recovery capacity.")
        elif avg_rmssd < 20:
            insights.append(f"**Low HRV** (RMSSD: {avg_rmssd:.1f}ms) - May indicate stress or fatigue.")
    
    # State distribution insights
    state_counts = results_df['state'].value_counts()
    total = len(results_df)
    ready_pct = (state_counts.get('● Ready', 0) / total * 100)
    fatigued_pct = (state_counts.get('■ Fatigued', 0) / total * 100)
    
    if ready_pct > 70:
        insights.append(f"**Predominantly Ready** ({ready_pct:.0f}% of time) - Excellent state maintained.")
    elif fatigued_pct > 30:
        insights.append(f"**High Fatigue Time** ({fatigued_pct:.0f}% of time) - Extended recovery needed.")
    
    # Recommendations
    insights.append("\n**Recommendations:**")
    if avg_readiness < 50:
        insights.append("• Focus on sleep quality and stress management")
        insights.append("• Avoid intense training for 24-48 hours")
    elif avg_readiness >= 75:
        insights.append("• Optimal time for peak performance activities")
        insights.append("• Can handle high-intensity training")
    else:
        insights.append("• Moderate activity levels appropriate")
        insights.append("• Monitor for signs of accumulated fatigue")
    
    return "\n\n".join(insights)


def show_analytics_dashboard(model):
    """Display comprehensive analytics dashboard aligned with EXRT AI pitch deck."""
    st.subheader("EXRT AI - Platform Overview")
    
    # Hero Section matching pitch
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                padding: 2.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">EXRT AI</h1>
        <h3 style="color: white; margin: 0.5rem 0; opacity: 0.95;">AI-Driven Heart Health Monitoring for Athletes</h3>
        <p style="font-size: 1.2rem; margin-top: 1rem; font-weight: 500; opacity: 0.9;">
            Continuous Monitoring • Granular Heart Insights • Performance Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Impact Metrics from pitch
    st.markdown("### Proven Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%); 
                    padding: 2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <h1 style="color: white; margin: 0; font-size: 3rem;">93%</h1>
            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.95;">Reduction in Cardiac Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%); 
                    padding: 2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <h1 style="color: white; margin: 0; font-size: 3rem;">30%</h1>
            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.95;">Performance Improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%); 
                    padding: 2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">Real-Time</h1>
            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.95;">Continuous Heart Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mission Statement
    st.markdown("### Our Mission")
    st.info("""
    **Provide real-time insights for proactive heart health management and peak performance**
    
    We empower athletes with continuous, AI-driven heart health monitoring to optimize performance,
    enhance recovery, and prevent cardiac incidents through predictive analytics.
    """)
    
    # What We Are / What We're Not
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### What We Are
        - **AI Software Company** focused on heart health analytics
        - **Continuous ECG Monitoring** platform with granular insights
        - **Performance Optimization** through predictive analytics
        - **Medical-Grade Accuracy** validated technology
        """)
    
    with col2:
        st.info("""
        ### What We're NOT
        - **Hardware Manufacturer** - We integrate with 3rd party devices
        - **Spot-Check Monitor** - We provide continuous 24/7 tracking
        - **Consumer Wearable** - We're purpose-built for athletes
        - **Basic HR Tracker** - We offer granular cardiac insights
        """)
    
    st.markdown("---")
    
    # Technical Capabilities
    st.markdown("### Technical Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### AI/ML Analytics
        - **31 HRV Features** extracted in real-time
        - **PCA Dimensionality Reduction** (5 components)
        - **GMM Clustering** for state classification
        - **GPU-Accelerated** processing (<1s/window)
        - **Confidence Scoring** for prediction reliability
        - **Trend Prediction** with statistical validation
        """)
    
    with col2:
        st.markdown("""
        #### Physiological Metrics
        - **ECG Rhythm** continuous monitoring
        - **Heart Rate & HRV** (RMSSD, SDNN, LF/HF)
        - **Readiness Score** (0-100% scale)
        - **Stress Monitor** real-time tracking
        - **Respiratory Rate** estimation
        - **RR-Interval** analysis
        """)
    
    with col3:
        st.markdown("""
        #### Data Management
        - **Live Streaming** ECG data
        - **Up to 3GB** file processing
        - **Multi-format Support** (CSV, NPY, PKL)
        - **Export to Excel/CSV/JSON**
        - **IoT-Enabled** device integration
        - **Batch Processing** for teams
        """)
    
    st.markdown("---")
    
    # Competitive Differentiation
    st.markdown("### Competitive Advantage")
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4F86F7;">
        <h4 style="margin-top: 0;">Why EXRT AI?</h4>
        <ul style="font-size: 1.05rem;">
            <li><strong>Continuous Vital Signal Assessment</strong> - 24/7 monitoring vs. spot checks</li>
            <li><strong>Granular Heart Insights</strong> - 31 features vs. competitors' 5-10 basic metrics</li>
            <li><strong>AI-Driven Predictions</strong> - ML models detect subtle physiological signatures</li>
            <li><strong>First-to-Market</strong> - AI-driven heart health monitoring specifically for athletes</li>
            <li><strong>Medical-Grade Accuracy</strong> - Validated with professional teams</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Competitor Matrix
    st.markdown("### Competitive Matrix")
    
    competitor_data = pd.DataFrame({
        'Platform': ['EXRT AI', 'WHOOP', 'Oura', 'Fitbit', 'Garmin', 'Fourth Frontier'],
        'Continuous Monitoring': [' Yes', ' Yes', ' No (Sleep only)', 'Limited', ' Limited', ' Yes'],
        'Granular Heart Insights': [' 31 Features', ' Basic HRV', ' Basic HRV', ' HR only', ' Basic HRV', 'Limited'],
        'AI-Driven Analytics': [' Advanced ML', ' Basic', ' Basic', ' No', ' Basic', ' Basic'],
        'ECG Monitoring': [' Continuous', ' No', ' No', ' Spot Check', ' Spot Check', ' Yes'],
        'Athlete-Focused': [' Purpose-Built', 'Yes', ' Wellness', ' Consumer', ' Sports', ' Limited']
    })
    
    st.dataframe(competitor_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Market Opportunity
    st.markdown("### Market Opportunity")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Market size visualization
        market_data = pd.DataFrame({
            'Market': ['TAM', 'SAM', 'Target Market', 'EXRT Market Share'],
            'Value ($B)': [265.4, 18.0, 1.5, 0.55],
            'Description': [
                'Total Addressable Market',
                'Serviceable Addressable Market', 
                'Serviceable Obtainable Market',
                'Initial Target'
            ]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=market_data['Market'],
                y=market_data['Value ($B)'],
                text=[f"${v:.1f}B" if v >= 1 else f"${v*1000:.0f}M" for v in market_data['Value ($B)']],
                textposition='auto',
                marker=dict(
                    color=['#667eea', '#764ba2', '#4F86F7', '#27ae60'],
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>%{x}</b><br>Value: $%{y:.2f}B<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="<b>Market Size Analysis</b>",
            xaxis_title="Market Category",
            yaxis_title="Market Value (Billions USD)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        #### Market Sizing
        
        **TAM: $265.4B**  
        Total global wearable & health monitoring market
        
        **SAM: $18B**  
        Athletic performance & health monitoring segment
        
        **SOM: $1.5B**  
        Professional & semi-pro sports teams globally
        
        **Target: $550M**  
        Initial market penetration focus (Year 1-3)
        """)
    
    st.markdown("---")
    
    # Additional Verticals
    st.markdown("### Additional Vertical Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Healthcare
        - **Remote Patient Monitoring** - Cardiac patients at home
        - **Clinical Trials** - Continuous physiological tracking
        - **Cardiac Rehabilitation** - Recovery monitoring
        - **Preventive Cardiology** - Early risk detection
        - **Emergency Medicine** - Real-time critical care
        """)
    
    with col2:
        st.markdown("""
        #### Business & Insurance
        - **Corporate Wellness** - Employee health programs
        - **Insurance Underwriting** - Risk assessment data
        - **Workplace Safety** - High-risk role monitoring
        - **Life Insurance** - Premium optimization
        - **Health Insurance** - Claims prevention
        """)
    
    st.markdown("---")
    
    # Financial Model Preview
    st.markdown("### Business Model")
    
    st.markdown("""
    **Annual SaaS Revenue + Monthly Device Payment**
    
    | User Tier | Analytics Fee | Device Cost | Annual Revenue per Team |
    |-----------|--------------|-------------|-------------------------|
    | **High-Tier Pro Teams** | £15-30K/year | £300/player | Advanced heart analytics |
    | **Low-Tier Pro Teams** | £5-15K/year | £150/player | Mid-tier heart analytics |
    | **Consumers (Premium)** | £65/month | £250 (one-time) | Analytics add-on |
    """)
    
    st.success("""
    **Revenue Projections (5-Year)**
    - Year 1 (2025/26): £1.2M revenue, £240K net profit
    - Year 3 (2027/28): £11.7M revenue, £2.3M net profit  
    - Year 5 (2029/30): £59.5M revenue, £11.9M net profit
    
    **Plus Consumer Market:** Additional £153M+ revenue potential by 2029
    """)
    
    st.markdown("---")
    
    # Go-to-Market
    st.markdown("### Go-to-Market Strategy")
    
    phases = pd.DataFrame({
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        'Timeline': ['2025 (Current)', '2025 (3 months)', '2026 (8 months)', '2026+ (14 months)'],
        'Focus': [
            'Beta Test & Initial Validation',
            'MVP Development & Iteration',
            'Early Market Penetration & Branding',
            'Scaling Sales & Market Expansion'
        ],
        'Status': ['In Progress', 'Next', 'Planned', 'Planned']
    })
    
    st.dataframe(phases, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Traction & Partnerships
    st.markdown("### Interest & Partnerships")
    
    st.info("""
    **Key Connections & Validation:**
    - ⚽ **Liverpool FC** - Club Doctor engagement
    - ⚽ **Chelsea FC** - Co-Sporting Director interest
    - ⚽ **Premier League** - Chief Medical Officer involvement
    - **Wayne Rooney** - England & Man Utd Legend endorsement
    - **Tom Lockyer** - Cardiac arrest survivor advocate
    """)
    
    st.markdown("---")
    
    # Platform Demo CTA
    st.markdown("### Live Platform Demo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Upload & Analyze**  \nFull analysis with 9 visualization tabs")
    with col2:
        st.info("**Live Simulator**  \nReal-time AI state detection")
    with col3:
        st.info("**Export Capabilities**  \nCSV, Excel, JSON formats")

    st.markdown("---")

    # ==========================================
    # NEW: INVESTOR TOOLS SECTION
    # ==========================================
    
    st.subheader("Investor Tools: ROI Calculator")
    st.markdown("Demonstrate the financial impact of EXRT AI for a professional sports team.")
    
    # ROI Calculator
    roi_col1, roi_col2 = st.columns([1, 2])
    
    with roi_col1:
        st.markdown("#### Input Assumptions")
        avg_salary = st.number_input("Avg Player Salary (£)", value=3000000, step=100000, format="%d")
        squad_size = st.number_input("Squad Size", value=25, step=1)
        injury_rate = st.slider("Current Injury Rate (%)", 10, 50, 25)
        risk_reduction = st.slider("EXRT Risk Reduction (%)", 0, 100, 93, disabled=True, help="Based on validated clinical data")
        
    with roi_col2:
        st.markdown("#### Projected Savings")
        
        # Calculations
        total_payroll = avg_salary * squad_size
        current_loss = total_payroll * (injury_rate / 100)
        projected_loss = total_payroll * ((injury_rate * (1 - risk_reduction/100)) / 100)
        savings = current_loss - projected_loss
        roi_multiple = savings / 30000  # Assuming £30k annual fee
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Payroll Risk", f"£{current_loss:,.0f}")
        metric_col2.metric("Projected Savings", f"£{savings:,.0f}", delta="Money Saved")
        metric_col3.metric("ROI Multiple", f"{roi_multiple:.1f}x", delta="Return on £30k Fee")
        
        st.progress(risk_reduction / 100, text=f"Risk Reduced by {risk_reduction}%")
        st.caption(f"*Based on {risk_reduction}% reduction in cardiac/fatigue-related unavailability.*")

    st.markdown("---")

    # API Showcase
    st.subheader("Technical Scalability: API Architecture")
    st.markdown("EXRT is built as an API-first platform, ready for seamless integration with 3rd party wearables and team management systems.")
    
    api_col1, api_col2 = st.columns(2)
    
    with api_col1:
        st.code("""
# EXRT Readiness API (Production Ready)
POST /readiness
{
  "ecg": [0.12, 0.15, 0.8, ...],
  "fs": 700,
  "window_sec": 90
}

# Response
{
  "readiness": 88.5,
  "state": "ready",
  "confidence": 0.92
}
        """, language="json")
        
    with api_col2:
        st.markdown("""
        **Integration Capabilities:**
        - **RESTful API** (FastAPI)
        - **Real-time Streaming** support
        - **Batch Processing** for historical data
        - **Secure Authentication** (OAuth2)
        - **Device Agnostic** (Polar, Garmin, Apple Watch)
        """)
        st.button("View Full API Documentation")


def show_team_dashboard(model):
    """Display B2B Team/Squad Dashboard with real-time player monitoring."""
    st.subheader("Team Dashboard - Squad Readiness Overview")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; border: none; margin: 0;">Professional Team Monitoring</h2>
        <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.95;">
            Real-time readiness tracking for 25 squad members
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate mock team data
    np.random.seed(42)
    players = []
    positions = ['GK', 'DEF', 'DEF', 'DEF', 'DEF', 'MID', 'MID', 'MID', 'MID', 'MID', 
                 'FWD', 'FWD', 'FWD', 'GK', 'DEF', 'DEF', 'MID', 'MID', 'MID', 'FWD', 
                 'FWD', 'DEF', 'MID', 'FWD', 'MID']
    
    for i in range(25):
        readiness = np.random.uniform(30, 95)
        hr = int(np.random.uniform(50, 85))
        hrv_rmssd = np.random.uniform(20, 80)
        
        if readiness >= 70:
            status = "● Ready"
            risk = "Low"
        elif readiness >= 40:
            status = "○ Monitor"
            risk = "Medium"
        else:
            status = "■ Rest"
            risk = "High"
        
        players.append({
            'Player': f"Player {i+1}",
            'Position': positions[i],
            'Readiness': round(readiness, 1),
            'Status': status,
            'HR': hr,
            'HRV': round(hrv_rmssd, 1),
            'Risk': risk
        })
    
    team_df = pd.DataFrame(players)
    
    # Team Summary Metrics
    st.markdown("###  Squad Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    ready_count = len(team_df[team_df['Status'] == '● Ready'])
    monitor_count = len(team_df[team_df['Status'] == '○ Monitor'])
    rest_count = len(team_df[team_df['Status'] == '■ Rest'])
    avg_readiness = team_df['Readiness'].mean()
    
    col1.metric("Total Squad", "25 Players")
    col2.metric("Ready to Play", f"{ready_count} Players", delta="Available")
    col3.metric("Monitor Closely", f"{monitor_count} Players", delta="Caution")
    col4.metric("Requires Rest", f"{rest_count} Players", delta="High Risk")
    col5.metric("Avg Readiness", f"{avg_readiness:.1f}%")
    
    st.markdown("---")
    
    # Position-based filtering
    st.markdown("###  Filter by Position")
    selected_position = st.selectbox("Select Position", ["All Positions", "GK", "DEF", "MID", "FWD"])
    
    if selected_position != "All Positions":
        filtered_df = team_df[team_df['Position'] == selected_position]
    else:
        filtered_df = team_df
    
    # Player Grid View
    st.markdown("### Player Status Grid")
    
    # Color-coded dataframe
    def color_status(val):
        if '●' in val:
            return 'background-color: #d4edda; color: #155724;'
        elif '○' in val:
            return 'background-color: #fff3cd; color: #856404;'
        elif '■' in val:
            return 'background-color: #f8d7da; color: #721c24;'
        return ''
    
    styled_df = filtered_df.style.map(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
    
    st.markdown("---")
    
    # Readiness Distribution Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Readiness Distribution")
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=team_df['Readiness'],
                nbinsx=15,
                marker_color='rgba(79, 134, 247, 0.7)',
                hovertemplate='<b>Readiness:</b> %{x:.1f}%<br><b>Players:</b> %{y}<extra></extra>'
            )
        ])
        fig_hist.update_layout(
            xaxis_title="Readiness Score (%)",
            yaxis_title="Number of Players",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("#### Status Breakdown")
        status_counts = team_df['Status'].value_counts()
        colors_pie = {'● Ready': '#28a745', '○ Monitor': '#ffc107', '■ Rest': '#dc3545'}
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                marker=dict(colors=[colors_pie.get(s, 'gray') for s in status_counts.index]),
                hovertemplate='<b>%{label}</b><br>Players: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Export Options
    st.markdown("### Export Team Report")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv_data = team_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"team_readiness_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with export_col2:
        # Excel export
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            team_df.to_excel(writer, index=False, sheet_name='Squad Status')
        excel_data = buffer.getvalue()
        
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=f"team_readiness_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with export_col3:
        # Generate PDF Report
        if st.button("Generate PDF Report"):
            st.info("Generating professional PDF report...")
            time.sleep(1.5)  # Simulate generation
            st.success("""
            **PDF Report Ready!**
            
            Report includes:
            - Team summary statistics
            - Individual player profiles
            - Readiness trends
            - AI-generated recommendations
            - Medical-grade certification
            """)
            st.markdown("[Download PDF Report](#)")  # Placeholder link
    
    st.markdown("---")
    
    # Team Insights
    st.markdown("### AI Team Insights")
    
    insights = []
    
    if ready_count >= 18:
        insights.append("**Strong Squad Availability**: 72%+ of players are ready for match day.")
    elif ready_count < 12:
        insights.append("**Squad Depth Concern**: Less than 50% of players at optimal readiness.")
    
    if rest_count >= 5:
        insights.append(f"**High Fatigue Alert**: {rest_count} players require immediate rest and recovery protocols.")
    
    high_risk_players = team_df[team_df['Risk'] == 'High']
    if len(high_risk_players) > 0:
        insights.append(f"**Cardiac Risk**: {len(high_risk_players)} players showing elevated fatigue markers.")
    
    if avg_readiness >= 75:
        insights.append("**Peak Performance State**: Team average readiness is excellent for competition.")
    elif avg_readiness < 55:
        insights.append("**Team Fatigue Detected**: Consider reducing training load this week.")
    
    for insight in insights:
        st.info(insight)
    
    st.markdown("---")
    
    st.success("""
    **B2B SaaS Value Proposition:**
    
    This dashboard demonstrates the core value of EXRT AI for professional teams:
    - Monitor entire squad (25+ players) in real-time
    - Prevent injuries through proactive fatigue detection
    - Optimize training load based on objective cardiac data
    - Export compliance reports for medical staff
    - £30,000/year subscription unlocks unlimited monitoring
    """)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

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
    
    fig.add_hline(y=70, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="Ready (≥70%)", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="Fatigued (<40%)", annotation_position="right")
    
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
    
    fig.add_hline(y=70, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="High Confidence (≥70%)", annotation_position="right")
    
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
    colors_map = {"● Ready": "green", "○ Neutral": "gold", "■ Fatigued": "red"}
    colors = [colors_map.get(state, "gray") for state in state_counts.index]
    
    fig = go.Figure(
        data=[go.Pie(
            labels=state_counts.index,
            values=state_counts.values,
            marker=dict(colors=colors),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        )]
    )
    
    fig.update_layout(title="<b>State Distribution</b>", height=350, font=dict(size=11))
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
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
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
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
        )
    )
    fig.update_layout(height=400, font=dict(size=12))
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
        delta_rmssd = (results_df["HRV_RMSSD"].iloc[-1] - results_df["HRV_RMSSD"].iloc[0]) if len(results_df) > 1 else None
        st.metric("Avg RMSSD", f"{avg_rmssd:.1f} ms", 
                 delta=f"{delta_rmssd:.1f} ms" if delta_rmssd is not None else None)
    
    with col2:
        avg_sdnn = results_df["HRV_SDNN"].mean()
        delta_sdnn = (results_df["HRV_SDNN"].iloc[-1] - results_df["HRV_SDNN"].iloc[0]) if len(results_df) > 1 else None
        st.metric("Avg SDNN", f"{avg_sdnn:.1f} ms",
                 delta=f"{delta_sdnn:.1f} ms" if delta_sdnn is not None else None)
    
    with col3:
        avg_lf_hf = results_df["HRV_LF_HF"].mean()
        delta_lf_hf = (results_df["HRV_LF_HF"].iloc[-1] - results_df["HRV_LF_HF"].iloc[0]) if len(results_df) > 1 else None
        st.metric("Avg LF/HF", f"{avg_lf_hf:.2f}",
                 delta=f"{delta_lf_hf:.2f}" if delta_lf_hf is not None else None)
    
    with col4:
        avg_hr = results_df["HR_est"].mean()
        delta_hr = (results_df["HR_est"].iloc[-1] - results_df["HR_est"].iloc[0]) if len(results_df) > 1 else None
        st.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm",
                 delta=f"{delta_hr:.0f} bpm" if delta_hr is not None else None)


def plot_trend_analysis(results_df):
    """Plot trend analysis showing if readiness is improving or declining."""
    fig = go.Figure()
    
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
    
    if len(results_df) > 1:
        x_vals = results_df["center_time_sec"].values
        y_vals = results_df["readiness"].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        trend_line = slope * x_vals + intercept
        
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
        
        if slope > 0:
            fig.add_vrect(x0=x_vals.min(), x1=x_vals.max(), 
                         fillcolor="green", opacity=0.1, layer="below", 
                         annotation_text="Improving", annotation_position="top left")
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
    if len(results_df) < 2:
        return None
    
    x_vals = results_df["center_time_sec"].values
    y_vals = results_df["readiness"].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    
    slope_per_min = slope * 60
    slope_per_hour = slope * 3600
    
    if slope > 0.01:
        direction = "Improving"
        color = "green"
    elif slope < -0.01:
        direction = "Declining"
        color = "red"
    else:
        direction = "Stable"
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


def plot_raw_ecg_sample(ecg, fs, max_duration=10):
    """Plot a sample of the raw ECG signal."""
    max_samples = int(max_duration * fs)
    ecg_sample = ecg[:min(len(ecg), max_samples)]
    time_axis = np.arange(len(ecg_sample)) / fs
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=ecg_sample,
            mode="lines",
            line=dict(color="#4F86F7", width=1),
            name="ECG"
        )
    )
    
    fig.update_layout(
        title=f"<b>Raw ECG Signal (First {max_duration}s)</b>",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=350,
        font=dict(size=11),
        plot_bgcolor="rgba(240, 248, 255, 0.5)",
    )
    return fig


def main():
    st.set_page_config(
        page_title="EXRT AI - Heart Health Monitoring",
        layout="wide",
        page_icon="❤",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "EXRT AI - Advanced Heart Health Monitoring Platform"
        }
    )    # Custom CSS matching EXRT AI brand
    st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%); }
    
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
    
    h2 { color: #34495e; border-bottom: 3px solid #4F86F7; padding-bottom: 0.8rem; margin-top: 2rem; font-weight: 600; }
    h3 { color: #5a6c7d; font-weight: 500; }
    
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #ffffff !important; }
    
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
        <h1>EXRT AI</h1>
        <p style="font-size: 1.3rem; color: #5a6c7d; font-weight: 400; margin-top: -1rem;">
            Empowering individuals & teams with AI-driven, continuous heart health monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()

    # Sidebar for navigation and settings
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page:",
            ["Home", "Upload & Analyze", "Live Simulator", "Team Dashboard", "Analytics Dashboard"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.header("Analysis Settings")
        fs = st.number_input("Sampling Rate (Hz)", value=700, min_value=100, max_value=10000)
        window_sec = st.number_input("Window Duration (sec)", value=90, min_value=30, max_value=300)
        step_sec = st.number_input("Step Duration (sec)", value=15, min_value=5, max_value=window_sec)
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info(f"**Device:** GPU\n\n**Features:** 31 HRV metrics\n\n**PCA:** 5 components\n\n**GMM:** 8 clusters")

    # Route to different pages
    if page == "Home":
        show_home_page()
    elif page == "Upload & Analyze":
        show_upload_page(model, fs, window_sec, step_sec)
    elif page == "Live Simulator":
        show_live_simulator_page(model, fs, window_sec, step_sec)
    elif page == "Team Dashboard":
        show_team_dashboard(model)
    elif page == "Analytics Dashboard":
        show_analytics_dashboard(model)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin: 0; font-size: 0.95rem; color: #5a6c7d;">
            <strong>EXRT AI LTD</strong> | Empowering Heart Health Monitoring
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #95a5a6;">
            Copyright © 2025 EXRT AI LTD | All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_home_page():
    """Display the welcome/home page matching exrtai.com design."""
    
    # Hero Banner
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem 2rem 1rem;">
        <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #4F86F7 0%, #3a6fd9 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            EXRT AI
        </h1>
        <p style="font-size: 1.4rem; color: #5a6c7d; margin-top: 1rem; font-weight: 400;">
            Empowering Heart Health Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Three main feature cards matching website
    st.markdown("### Platform Features")
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 2.5rem 1.5rem; border-radius: 12px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #e8ecf1;
                    text-align: center; min-height: 320px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">■</div>
            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;">Upload & Analyze</h3>
            <p style="color: #5a6c7d; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">
                Upload your ECG data files (CSV, NPY, PKL) and get comprehensive readiness analysis with:
            </p>
            <ul style="text-align: left; color: #5a6c7d; font-size: 0.9rem; padding-left: 1.5rem; line-height: 1.8;">
                <li>HRV metrics breakdown</li>
                <li>Readiness scoring</li>
                <li>Trend analysis</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2.5rem 1.5rem; border-radius: 12px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #e8ecf1;
                    text-align: center; min-height: 320px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">▲</div>
            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;">Live Simulator</h3>
            <p style="color: #5a6c7d; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">
                Experience real-time ECG monitoring with synthetic data generation featuring:
            </p>
            <ul style="text-align: left; color: #5a6c7d; font-size: 0.9rem; padding-left: 1.5rem; line-height: 1.8;">
                <li>Continuous streaming</li>
                <li>Live readiness updates</li>
                <li>Real-time HRV calculation</li>
                <li>Simulated scenarios</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 2.5rem 1.5rem; border-radius: 12px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #e8ecf1;
                    text-align: center; min-height: 320px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">●</div>
            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;">AI-Powered Insights</h3>
            <p style="color: #5a6c7d; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">
                Advanced machine learning provides:
            </p>
            <ul style="text-align: left; color: #5a6c7d; font-size: 0.9rem; padding-left: 1.5rem; line-height: 1.8;">
                <li>Confidence scoring</li>
                <li>State classification</li>
                <li>Physiological readiness</li>
                <li>Predictive analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # About and Readiness section
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style="padding: 1.5rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">●</span>
                <h3 style="color: #2c3e50; margin: 0;">About EXRT AI</h3>
            </div>
            <p style="color: #5a6c7d; font-size: 1rem; line-height: 1.8;">
                EXRT AI leverages cutting-edge machine learning to analyze ECG data and assess 
                physiological readiness in real-time. Our platform empowers individuals and teams 
                to make data-driven decisions about health, performance, and recovery.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fc 0%, #e8ecf1 100%); 
                    padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4F86F7;">
            <h4 style="color: #2c3e50; margin-top: 0;">Key Features:</h4>
            <ul style="color: #5a6c7d; font-size: 0.95rem; line-height: 1.8; margin-bottom: 0;">
                <li>AI-powered predictions with confidence scoring</li>
                <li>Detailed HRV breakdown (RMSSD, SDNN, LF/HF)</li>
                <li>Trend analysis showing improvement or decline</li>
                <li>Interactive visualizations for deep insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1.5rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <h3 style="color: #2c3e50; margin: 0;">Readiness Levels</h3>
            </div>
            <p style="color: #5a6c7d; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1.5rem;">
                Our system classifies your physiological state into three levels:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Readiness level cards
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                    padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #28a745;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.8rem;">●</span>
                <div>
                    <h4 style="margin: 0; color: #155724;">Ready (≥70%)</h4>
                    <p style="margin: 0.3rem 0 0 0; color: #155724; font-size: 0.85rem;">
                        High autonomic stability - optimal for intense physical/mental activity
                    </p>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                    padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #ffc107;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.8rem;">○</span>
                <div>
                    <h4 style="margin: 0; color: #856404;">Neutral (40-70%)</h4>
                    <p style="margin: 0.3rem 0 0 0; color: #856404; font-size: 0.85rem;">
                        Moderate state - suitable for standard activities
                    </p>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                    padding: 1.2rem; border-radius: 8px; border-left: 4px solid #dc3545;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.8rem;">■</span>
                <div>
                    <h4 style="margin: 0; color: #721c24;">Recovery (<40%)</h4>
                    <p style="margin: 0.3rem 0 0 0; color: #721c24; font-size: 0.85rem;">
                        Low stability - rest and recovery recommended
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def show_upload_page(model, fs, window_sec, step_sec):
    """Display upload and analysis page with comprehensive visualizations."""
    st.subheader("Upload ECG Data for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an ECG file (.csv, .npy, or .pkl)",
        type=["csv", "npy", "pkl"],
        help="Upload your ECG data file - supports files up to 3GB"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading ECG data..."):
            ecg = parse_ecg_upload(uploaded_file)
            if ecg is None:
                st.stop()
        
        st.success(f"✓ Loaded {len(ecg):,} samples ({len(ecg)/fs:.1f} seconds)")
        
        with st.spinner("Computing readiness analysis..."):
            results_df = model.batch_predict_ecg(ecg, fs=fs, window_sec=window_sec, step_sec=step_sec)
        
        if results_df.empty:
            st.error(" No valid windows. Check ECG quality or adjust window parameters.")
            st.stop()
        
        # Add state classification
        results_df["state"] = results_df["readiness"].apply(state_from_readiness)
        
        st.success(f"✓ Analyzed {len(results_df)} windows")
        
        # ========== SUMMARY METRICS ==========
        st.subheader(" Analysis Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        avg_readiness = results_df["readiness"].mean()
        avg_confidence = results_df["confidence"].mean()
        high_conf_pct = (results_df["confidence"] >= 0.7).sum() / len(results_df) * 100
        
        with col1:
            st.metric("Avg Readiness", f"{avg_readiness:.1f}%")
        with col2:
            st.metric("Max", f"{results_df['readiness'].max():.1f}%")
        with col3:
            st.metric("Min", f"{results_df['readiness'].min():.1f}%")
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        with col5:
            st.metric("High Confidence", f"{high_conf_pct:.0f}%")
        
        # ========== TABBED VISUALIZATIONS ==========
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "Time-Series",
            "Confidence",
            "Distribution",
            "State Dist.",
            "Gauge",
            "HRV Metrics",
            "Trend",
            "Correlation",
            "Raw ECG"
        ])
        
        with tab1:
            st.plotly_chart(plot_readiness_timeseries(results_df), use_container_width=True)
            
        with tab2:
            st.plotly_chart(plot_confidence_timeseries(results_df), use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_readiness_distribution(results_df), use_container_width=True)
            with col2:
                st.markdown("### Distribution Stats")
                st.write(f"**Mean:** {avg_readiness:.1f}%")
                st.write(f"**Median:** {results_df['readiness'].median():.1f}%")
                st.write(f"**Std Dev:** {results_df['readiness'].std():.1f}%")
                st.write(f"**Range:** {results_df['readiness'].min():.1f}% - {results_df['readiness'].max():.1f}%")
                st.write(f"**Windows:** {len(results_df)}")
        
        with tab4:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(plot_state_pie_chart(results_df), use_container_width=True)
            with col2:
                st.markdown("### State Counts")
                state_counts = results_df["state"].value_counts()
                for state, count in state_counts.items():
                    pct = count / len(results_df) * 100
                    st.write(f"{state}: **{count}** windows ({pct:.1f}%)")
        
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_readiness_gauge(avg_readiness), use_container_width=True)
            with col2:
                st.plotly_chart(plot_confidence_gauge(avg_confidence), use_container_width=True)
        
        with tab6:
            st.markdown("### Heart Rate Variability Analysis")
            plot_hrv_metrics_cards(results_df)
            st.plotly_chart(plot_hrv_breakdown(results_df), use_container_width=True)
        
        with tab7:
            st.plotly_chart(plot_trend_analysis(results_df), use_container_width=True)
            
            trend_metrics = calculate_trend_metrics(results_df)
            if trend_metrics:
                st.markdown("### Trend Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Direction", trend_metrics["direction"])
                with col2:
                    st.metric("Slope (per min)", f"{trend_metrics['slope_per_min']:.3f}%")
                with col3:
                    st.metric("Slope (per hour)", f"{trend_metrics['slope_per_hour']:.2f}%")
                with col4:
                    st.metric("R² (fit quality)", f"{trend_metrics['r_squared']:.3f}")
        
        with tab8:
            st.plotly_chart(plot_readiness_vs_confidence(results_df), use_container_width=True)
            st.markdown("""
            **Interpretation:**
            - Points in **upper-right**: High readiness + high confidence (ideal)
            - Points in **lower-left**: Low readiness + low confidence (uncertain)
            - Scattered points suggest variable signal quality
            """)
        
        with tab9:
            st.plotly_chart(plot_raw_ecg_sample(ecg, fs, max_duration=10), use_container_width=True)
            st.markdown(f"**Signal Info:** {len(ecg):,} samples @ {fs} Hz = {len(ecg)/fs:.1f} seconds total")
        
        # ========== EXPORT & AI INSIGHTS ==========
        st.markdown("---")
        st.subheader("AI-Powered Insights & Recommendations")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            insights = generate_ai_insights(results_df, avg_readiness, avg_confidence)
            st.markdown(insights)
        
        with col2:
            st.markdown("### Export Options")
            
            # CSV Export
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"ecg_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Excel Export
            from io import BytesIO
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Analysis Results')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"ecg_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # JSON Export
            json_data = results_df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"ecg_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # ========== DATA TABLE ==========
        st.markdown("---")
        st.subheader("Detailed Results Table")
        st.dataframe(results_df.round(2), use_container_width=True, hide_index=True)
    
    else:
        st.info("Upload an ECG file to begin analysis")


def show_live_simulator_page(model, fs, window_sec, step_sec):
    """Display live ECG simulator page with comprehensive visualizations."""
    st.subheader("Live ECG Monitoring Simulator")
    
    st.markdown("""
    <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
        <strong>⚡ Live Mode:</strong> This simulator generates synthetic ECG data in real-time 
        and the AI model automatically detects your physiological state (resting, active, stressed, etc.).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Device Connection Simulation
    st.markdown("### Device Connection Status")
    
    device_col1, device_col2, device_col3 = st.columns(3)
    
    with device_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3 style="color: white; margin: 0;">Connected</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">EXRT Patch #47A2B</p>
        </div>
        """, unsafe_allow_html=True)
    
    with device_col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 2px solid #e0e7ef; text-align: center;">
            <h4 style="margin: 0; color: #4F86F7;">Signal Quality</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #28a745;">98%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with device_col3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 2px solid #e0e7ef; text-align: center;">
            <h4 style="margin: 0; color: #4F86F7;">Battery</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #28a745;">87%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("""
    **Device Integration Demo:**
    - **Device:** EXRT Patch (3rd-party ECG monitor)
    - **Connection:** Bluetooth Low Energy (BLE)
    - **Sampling Rate:** 700 Hz
    - **Status:** Live streaming active
    """)
    
    st.markdown("---")
    
    # Simulator controls (no scenario selection - model detects automatically)
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        update_interval = st.slider("Update Interval (seconds)", 1, 10, 3)
    
    with col2:
        iterations = st.number_input("Iterations", 5, 50, 20)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        start_button = st.button("Start Auto-Detection", use_container_width=True)
    
    # Show history if exists
    if 'history' in st.session_state and len(st.session_state['history']) > 0:
        history_df = pd.DataFrame(st.session_state['history'])
        
        st.markdown("---")
        st.subheader("Live Monitoring Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Readings", len(history_df))
        with col2:
            st.metric("Avg Readiness", f"{history_df['score'].mean():.1f}%")
        with col3:
            st.metric("Avg Confidence", f"{history_df['confidence'].mean()*100:.1f}%")
        with col4:
            latest_state = state_from_readiness(history_df['score'].iloc[-1])
            st.metric("Current State", latest_state)
        with col5:
            ready_pct = (history_df['score'] >= 70).sum() / len(history_df) * 100
            st.metric("Ready %", f"{ready_pct:.0f}%")
        
        # Tabbed visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Time-Series",
            "HRV Metrics",
            "Distribution",
            "Trend",
            "Data Table"
        ])
        
        with tab1:
            # Create readiness time-series
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['score'],
                    mode='lines+markers',
                    name='Readiness',
                    line=dict(color='#4F86F7', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(79, 134, 247, 0.2)'
                )
            )
            fig.add_hline(y=70, line_dash="dash", line_color="green", line_width=2,
                         annotation_text="Ready (≥70%)")
            fig.add_hline(y=40, line_dash="dash", line_color="red", line_width=2,
                         annotation_text="Fatigued (<40%)")
            fig.update_layout(
                title="<b>Readiness Over Time</b>",
                xaxis_title="Reading #",
                yaxis_title="Readiness Score (%)",
                template="plotly_white",
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence time-series
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['confidence'] * 100,
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='#9B59B6', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(155, 89, 182, 0.2)'
                )
            )
            fig2.update_layout(
                title="<b>Model Confidence Over Time</b>",
                xaxis_title="Reading #",
                yaxis_title="Confidence (%)",
                template="plotly_white",
                height=350,
                hovermode="x unified"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # HRV metrics if available
            if 'HRV_RMSSD' in history_df.columns:
                st.markdown("### Heart Rate Variability Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg RMSSD", f"{history_df['HRV_RMSSD'].mean():.1f} ms")
                with col2:
                    st.metric("Avg SDNN", f"{history_df['HRV_SDNN'].mean():.1f} ms")
                with col3:
                    st.metric("Avg LF/HF", f"{history_df['HRV_LF_HF'].mean():.2f}")
                with col4:
                    st.metric("Avg HR", f"{history_df['HR_est'].mean():.0f} bpm")
                
                # HRV breakdown plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['HRV_RMSSD'],
                    mode='lines+markers',
                    name='RMSSD (ms)',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=5)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['HRV_SDNN'],
                    mode='lines+markers',
                    name='SDNN (ms)',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=5),
                    yaxis='y2'
                ))
                fig.update_layout(
                    title="<b>HRV Metrics Over Time</b>",
                    xaxis_title="Reading #",
                    yaxis=dict(title="RMSSD (ms)", titlefont=dict(color="#1f77b4")),
                    yaxis2=dict(title="SDNN (ms)", titlefont=dict(color="#ff7f0e"),
                               overlaying='y', side='right'),
                    template="plotly_white",
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("HRV metrics not available in monitoring mode. Try upload analysis for detailed HRV breakdown.")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Readiness distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=history_df['score'],
                    nbinsx=20,
                    marker_color='rgba(79, 134, 247, 0.7)',
                    name='Readiness'
                ))
                fig.update_layout(
                    title="<b>Readiness Distribution</b>",
                    xaxis_title="Readiness Score (%)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # State distribution pie chart
                history_df_copy = history_df.copy()
                history_df_copy['state'] = history_df_copy['score'].apply(state_from_readiness)
                state_counts = history_df_copy['state'].value_counts()
                
                colors_map = {"● Ready": "green", "○ Neutral": "gold", "■ Fatigued": "red"}
                colors = [colors_map.get(state, "gray") for state in state_counts.index]
                
                fig = go.Figure(data=[go.Pie(
                    labels=state_counts.index,
                    values=state_counts.values,
                    marker=dict(colors=colors)
                )])
                fig.update_layout(
                    title="<b>State Distribution</b>",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Trend analysis
            if len(history_df) >= 2:
                x_vals = np.arange(len(history_df))
                y_vals = history_df['score'].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                trend_line = slope * x_vals + intercept
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name='Readiness',
                    line=dict(color='#4F86F7', width=3),
                    marker=dict(size=7)
                ))
                
                trend_color = "#2ca02c" if slope > 0 else "#d62728"
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=trend_line,
                    mode='lines',
                    name=f'Trend (slope: {slope:.3f})',
                    line=dict(color=trend_color, width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="<b>Readiness Trend Analysis</b>",
                    xaxis_title="Reading #",
                    yaxis_title="Readiness Score (%)",
                    template="plotly_white",
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    direction = "Improving" if slope > 0.01 else ("Declining" if slope < -0.01 else "Stable")
                    st.metric("Trend Direction", direction)
                with col2:
                    st.metric("Slope (per reading)", f"{slope:.3f}%")
                with col3:
                    st.metric("R² (fit quality)", f"{r_value**2:.3f}")
            else:
                st.info("Need at least 2 readings for trend analysis")
        
        with tab5:
            st.markdown("### Complete Monitoring History")
            display_cols = ['detected_state', 'score', 'confidence', 'prob_ready']
            if 'actual_state' in history_df.columns:
                display_cols.insert(0, 'actual_state')
            if 'HRV_RMSSD' in history_df.columns:
                display_cols.extend(['HRV_RMSSD', 'HRV_SDNN', 'HR_est'])
            available_cols = [col for col in display_cols if col in history_df.columns]
            st.dataframe(history_df[available_cols].round(2), use_container_width=True, hide_index=True)
            
            # Show detection accuracy if both actual and detected states exist
            if 'actual_state' in history_df.columns and 'detected_state' in history_df.columns:
                st.markdown("### AI Detection Performance")
                st.info("**Note:** The model detects physiological state purely from ECG analysis, without prior knowledge of the actual scenario.")
        
        # Clear history button
        if st.button("Clear History", use_container_width=False):
            st.session_state['history'] = []
            st.rerun()
    
    # Start monitoring
    if start_button:
        st.session_state['monitoring'] = True
        
        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        # Create placeholders for live updates
        progress_bar = st.progress(0)
        live_ecg_plot = st.empty()
        live_metrics = st.empty()
        
        # Simulation loop - randomly mix different physiological states
        # Model will detect the state automatically
        import random
        available_states = ['rest', 'active', 'stress', 'recovery', 'rest']  # Rest twice for higher probability
        
        for i in range(int(iterations)):
            progress_bar.progress((i + 1) / iterations)
            
            # Randomly select state for realistic mixed scenarios
            current_state = random.choice(available_states)
            ecg_segment = generate_ecg_segment(
                duration_sec=window_sec,
                fs=fs,
                scenario=current_state
            )
            
            # Analyze - model detects the state from ECG
            result = model.predict_from_ecg(ecg_segment, fs=fs)
            
            if result:
                timestamp = time.time()
                result['timestamp'] = timestamp
                # Determine detected state based on readiness score and confidence
                detected_state = _classify_physiological_state(result['score'], 
                                                                result.get('HRV_RMSSD', 0),
                                                                result.get('HR_est', 70))
                result['detected_state'] = detected_state
                result['actual_state'] = current_state.title()
                st.session_state['history'].append(result)
                
                # Update live ECG plot
                with live_ecg_plot.container():
                    fig = go.Figure()
                    time_axis = np.arange(len(ecg_segment)) / fs
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=ecg_segment,
                        mode='lines',
                        name='ECG',
                        line=dict(color='#4F86F7', width=1)
                    ))
                    fig.update_layout(
                        title=f"<b>Live ECG Signal - AI Detecting State... (Reading {i+1}/{iterations})</b>",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude",
                        height=300,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Update live metrics
                with live_metrics.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Readiness", f"{result['score']:.1f}%")
                    with col2:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    with col3:
                        st.metric("Ready Prob", f"{result['prob_ready']*100:.1f}%")
                    with col4:
                        st.metric("Detected State", detected_state)
                    with col5:
                        st.metric("Progress", f"{i+1}/{iterations}")
            
            time.sleep(update_interval)
        
        st.success(f"Monitoring session complete! Collected {iterations} readings.")
        st.info("Scroll up to view comprehensive analysis dashboard")
        st.rerun()


if __name__ == "__main__":
    main()
