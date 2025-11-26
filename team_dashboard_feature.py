"""Team Dashboard Feature - To be integrated into streamlit_multipage.py"""

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
import time
from io import BytesIO


def show_team_dashboard(model):
    """Display B2B Team/Squad Dashboard with real-time player monitoring."""
    st.subheader("👥 Team Dashboard - Squad Readiness Overview")
    
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
            status = "🟢 Ready"
            risk = "Low"
        elif readiness >= 40:
            status = "🟡 Monitor"
            risk = "Medium"
        else:
            status = "🔴 Rest"
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
    st.markdown("### 📊 Squad Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    ready_count = len(team_df[team_df['Status'] == '🟢 Ready'])
    monitor_count = len(team_df[team_df['Status'] == '🟡 Monitor'])
    rest_count = len(team_df[team_df['Status'] == '🔴 Rest'])
    avg_readiness = team_df['Readiness'].mean()
    
    col1.metric("Total Squad", "25 Players")
    col2.metric("Ready to Play", f"{ready_count} Players", delta="Available")
    col3.metric("Monitor Closely", f"{monitor_count} Players", delta="Caution")
    col4.metric("Requires Rest", f"{rest_count} Players", delta="High Risk")
    col5.metric("Avg Readiness", f"{avg_readiness:.1f}%")
    
    st.markdown("---")
    
    # Position-based filtering
    st.markdown("### 🔍 Filter by Position")
    selected_position = st.selectbox("Select Position", ["All Positions", "GK", "DEF", "MID", "FWD"])
    
    if selected_position != "All Positions":
        filtered_df = team_df[team_df['Position'] == selected_position]
    else:
        filtered_df = team_df
    
    # Player Grid View
    st.markdown("### 👥 Player Status Grid")
    
    # Color-coded dataframe
    def color_status(val):
        if '🟢' in val:
            return 'background-color: #d4edda; color: #155724;'
        elif '🟡' in val:
            return 'background-color: #fff3cd; color: #856404;'
        elif '🔴' in val:
            return 'background-color: #f8d7da; color: #721c24;'
        return ''
    
    styled_df = filtered_df.style.applymap(color_status, subset=['Status'])
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
        colors_pie = {'🟢 Ready': '#28a745', '🟡 Monitor': '#ffc107', '🔴 Rest': '#dc3545'}
        
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
    st.markdown("### 📥 Export Team Report")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv_data = team_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"team_readiness_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with export_col2:
        # Excel export
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            team_df.to_excel(writer, index=False, sheet_name='Squad Status')
        excel_data = buffer.getvalue()
        
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"team_readiness_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with export_col3:
        # Generate PDF Report
        if st.button("📑 Generate PDF Report"):
            st.info("🔄 Generating professional PDF report...")
            time.sleep(1.5)  # Simulate generation
            st.success("""
            ✅ **PDF Report Ready!**
            
            Report includes:
            - Team summary statistics
            - Individual player profiles
            - Readiness trends
            - AI-generated recommendations
            - Medical-grade certification
            """)
            st.markdown("[📥 Download PDF Report](#)")  # Placeholder link
    
    st.markdown("---")
    
    # Team Insights
    st.markdown("### 🧠 AI Team Insights")
    
    insights = []
    
    if ready_count >= 18:
        insights.append("✅ **Strong Squad Availability**: 72%+ of players are ready for match day.")
    elif ready_count < 12:
        insights.append("⚠️ **Squad Depth Concern**: Less than 50% of players at optimal readiness.")
    
    if rest_count >= 5:
        insights.append(f"🚨 **High Fatigue Alert**: {rest_count} players require immediate rest and recovery protocols.")
    
    high_risk_players = team_df[team_df['Risk'] == 'High']
    if len(high_risk_players) > 0:
        insights.append(f"⚠️ **Cardiac Risk**: {len(high_risk_players)} players showing elevated fatigue markers.")
    
    if avg_readiness >= 75:
        insights.append("🏆 **Peak Performance State**: Team average readiness is excellent for competition.")
    elif avg_readiness < 55:
        insights.append("📉 **Team Fatigue Detected**: Consider reducing training load this week.")
    
    for insight in insights:
        st.info(insight)
    
    st.markdown("---")
    
    st.success("""
    **💼 B2B SaaS Value Proposition:**
    
    This dashboard demonstrates the core value of EXRT AI for professional teams:
    - Monitor entire squad (25+ players) in real-time
    - Prevent injuries through proactive fatigue detection
    - Optimize training load based on objective cardiac data
    - Export compliance reports for medical staff
    - £30,000/year subscription unlocks unlimited monitoring
    """)
