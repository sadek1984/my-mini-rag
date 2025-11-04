"""
Predictive Analytics Dashboard with Enhanced Visual Style
REORGANIZED: Improved tab structure and merged related features
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime
import re
from scipy.stats import f_oneway
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
from pathlib import Path
import os

os.environ['GEMINI_API_KEY'] = 'AIzaSyBH6QPeFgtZ1B26rt-p2GHaiD55IHEP6s0'

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from hybrid_query_system import HybridQuerySystem
    HYBRID_AVAILABLE = True
    print("✅ Hybrid system imported!")
except ImportError as e:
    HYBRID_AVAILABLE = False
    print(f"⚠️ Import failed: {e}")


st.set_page_config(
    page_title="🔮 LARS Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "intro_shown" not in st.session_state:
        st.session_state.intro_shown = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard Overview"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = None
    if "project_id" not in st.session_state:
        st.session_state.project_id = "TrhBVe_m5gg2002_E5VVqS"
    if "indexed_years" not in st.session_state:
        st.session_state.indexed_years = set()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

# [KEEP ALL YOUR EXISTING IMPORTS AND HELPER FUNCTIONS HERE]
# Including: APIClient, CSS styling, load_data functions, etc.

# ============================================================================
# SIDEBAR - UPDATED NAVIGATION
# ============================================================================

def render_sidebar(api_client):
    """Render sidebar with REORGANIZED navigation"""
    
    with st.sidebar:
        st.markdown("## 🔗 API Configuration")
        current_url = st.text_input("API Base URL", value="http://127.0.0.1:8000", key="api_url")
        
        st.markdown("---")
        
        if st.button("🔄 Check API Connection", key="health_check", use_container_width=True, type="primary"):
            with st.spinner("Checking API connection..."):
                health_status = api_client.health_check()
                if health_status:
                    st.success("✅ API Connected")
                else:
                    st.error("❌ API Disconnected")
        
        st.markdown("---")
        
        # UPDATED NAVIGATION with new structure
        st.markdown("## 📍 Navigate to:")
        
        page_options = [
            "🏠 Dashboard Overview",      # Merged Home + key metrics
            "📊 Risk Analysis",           # Merged High-Risk Combinations + Risk Heatmaps
            "📈 Trend Analysis",          # Time trends + seasonal patterns
            "📋 Statistical Insights",    # Merged ANOVA + Statistical tests
            "🔮 Predictions",            # Merged Predictions + Time Series Forecasting
            "🤖 AI Assistant",           # Keep as is
            "⚙️ Advanced Tools"          # Debug + technical features
        ]
        
        current_page = st.session_state.get('current_page', '🏠 Dashboard Overview')
        
        try:
            default_index = page_options.index(current_page)
        except ValueError:
            default_index = 0
        
        page = st.selectbox(
            "Select page:",
            page_options,
            index=default_index,
            key="current_page"
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    if 'df' not in st.session_state:
        data_path = '/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx'
        if os.path.exists(data_path):
            st.session_state.df = pd.read_excel(data_path)
            st.session_state.data_loaded = True
            
    initialize_session_state()
    
    try:
        if not st.session_state.intro_shown:
            # [KEEP YOUR EXISTING render_animated_header() function]
            st.session_state.intro_shown = True
        
        # [KEEP YOUR EXISTING APIClient initialization]
        api_client = APIClient()
        
        st.markdown("---")
        st.subheader("Dashboard Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", "1,234", "12%")
        with col2:
            st.metric("Risk Level", "Medium", "-3%")
        with col3:
            st.metric("Completed Analysis", "89%", "4%")

        st.markdown("---")
        
        render_sidebar(api_client)

        page = st.session_state.get('current_page', '🏠 Dashboard Overview')
        
        # ✅ UPDATED PAGE ROUTER with new structure
        pages = {
            "🏠 Dashboard Overview": show_dashboard_overview,     # NEW: Merged Home + metrics
            "📊 Risk Analysis": show_risk_analysis_page,         # NEW: Merged combinations + heatmaps
            "📈 Trend Analysis": show_trend_analysis_page,       # NEW: Merged trends + seasonality
            "📋 Statistical Insights": show_statistical_insights_page,  # Merged ANOVA + stats
            "🔮 Predictions": show_predictions_page,             # Merged predictions + forecasting
            "🤖 AI Assistant": show_chat_page,                   # Keep as is
            "⚙️ Advanced Tools": show_advanced_tools_page,      # NEW: Debug + technical
        }
        
        page_function = pages.get(page)
        if page_function:
            page_function(api_client)
        else:
            st.error(f"Unknown page: {page}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW (Merged Home + Key Metrics)
# ============================================================================

def show_dashboard_overview(api_client):
    """Enhanced dashboard overview with key metrics"""
    
    st.title("🏠 Dashboard Overview")
    
    # Enhanced metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #4caf50;">
            <div style="font-size: 40px; margin-bottom: 10px;">📊</div>
            <h3 style="color: #2e7d32;">1,234</h3>
            <p style="color: #666;">Total Samples</p>
            <p style="color: #4caf50; font-weight: 600;">↑ 12% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #ff9800;">
            <div style="font-size: 40px; margin-bottom: 10px;">⚠️</div>
            <h3 style="color: #f57c00;">Medium</h3>
            <p style="color: #666;">Risk Level</p>
            <p style="color: #f57c00; font-weight: 600;">↓ 3% improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #2196f3;">
            <div style="font-size: 40px; margin-bottom: 10px;">✅</div>
            <h3 style="color: #1976d2;">89%</h3>
            <p style="color: #666;">Completed Analysis</p>
            <p style="color: #2196f3; font-weight: 600;">↑ 4% progress</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick insights
    st.subheader("📌 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Priority Actions:**
        - Review high-risk vegetable-pesticide combinations
        - Monitor seasonal contamination patterns
        - Update forecasting models with latest data
        """)
    
    with col2:
        st.success("""
        **✅ Recent Achievements:**
        - 15% reduction in violations this quarter
        - 3 new vegetables added to monitoring program
        - AI assistant accuracy improved to 94%
        """)

# ============================================================================
# PAGE 2: RISK ANALYSIS (Merged High-Risk Combinations + Heatmaps)
# ============================================================================

def show_risk_analysis_page(api_client):
    """Merged: High-Risk Combinations + Risk Heatmaps"""
    
    st.title("📊 Risk Analysis")
    
    st.info("""
    **Comprehensive Risk Assessment:**
    - Identify highest-risk vegetable-pesticide combinations
    - Visualize risk patterns with interactive heatmaps
    - Prioritize testing resources efficiently
    """)
    
    # [INSERT YOUR EXISTING show_advanced_insights_page CODE HERE]
    # Specifically TAB 1 (High-Risk Combinations) and TAB 3 (Risk Heatmaps)
    # Combine them into a single view with sections instead of tabs
    
    st.header("🔥 High-Risk Combinations")
    # [Your existing high-risk combinations code]
    
    st.markdown("---")
    
    st.header("🗺️ Risk Heatmaps")
    # [Your existing heatmap code]

# ============================================================================
# PAGE 3: TREND ANALYSIS (Merged Trends + Seasonal Patterns)
# ============================================================================

def show_trend_analysis_page(api_client):
    """Merged: Time Trends + Seasonal Analysis"""
    
    st.title("📈 Trend Analysis")
    
    st.info("""
    **Track contamination patterns over time:**
    - Identify long-term trends (improving/worsening)
    - Analyze seasonal variations
    - Monitor intervention effectiveness
    """)
    
    # Section 1: Time Series Trends
    st.header("📊 Time Series Trends")
    # [Your existing time series trend code from show_advanced_insights_page]
    
    st.markdown("---")
    
    # Section 2: Seasonal Analysis (from ANOVA page)
    st.header("🌡️ Seasonal Contamination Patterns")
    # [Move the seasonal analysis portion from ANOVA page here]

# ============================================================================
# PAGE 4: STATISTICAL INSIGHTS (Merged ANOVA + Statistical Tests)
# ============================================================================

def show_statistical_insights_page(api_client):
    """Merged: ANOVA Analysis + Statistical Tests"""
    
    st.title("📋 Statistical Insights")
    
    st.info("""
    **Statistical validation of patterns:**
    - ANOVA testing for seasonal/vegetable differences
    - Trend detection and significance testing
    - Statistical summaries and comparisons
    """)
    
    # [COMBINE both show_anova_analysis_page and original show_statistical_insights_page]
    # Keep ANOVA as the main feature, add original statistical insights as secondary

# ============================================================================
# PAGE 5: PREDICTIONS (Merged Predictions + Time Series Forecasting)
# ============================================================================

def show_predictions_page(api_client):
    """Merged: Predictions + Time Series Forecasting"""
    
    st.title("🔮 Predictions & Forecasting")
    
    # Create tabs for different prediction types
    tab1, tab2 = st.tabs(["🎯 Risk Assessment", "📈 Time Series Forecasting"])
    
    with tab1:
        st.header("Enhanced Risk Assessment")
        # [Your existing show_enhanced_risk_form code]
    
    with tab2:
        st.header("Time Series Forecasting")
        # [Your existing show_timeseries_page code]

# ============================================================================
# PAGE 6: AI ASSISTANT (Keep as is)
# ============================================================================

def show_chat_page(api_client):
    """AI Assistant - unchanged"""
    # [Keep your existing show_chat_page code exactly as is]
    pass

# ============================================================================
# PAGE 7: ADVANCED TOOLS (New: Debug + Technical Features)
# ============================================================================

def show_advanced_tools_page(api_client):
    """Advanced tools and debugging features"""
    
    st.title("⚙️ Advanced Tools")
    
    st.warning("🔧 This section contains technical debugging and advanced configuration options.")
    
    # Your debug panel code here
    if st.checkbox("Show Debug Panel", value=False):
        st.markdown("## 🐛 DEBUG PANEL")
        # [Your existing debug panel code]

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()