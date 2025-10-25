"""
Predictive Analytics Dashboard with Enhanced Visual Style
Includes Statistical Insights and Enhanced Risk Assessment
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
# Import your existing NLP controller
import os

os.environ['GEMINI_API_KEY'] = 'AIzaSyBH6QPeFgtZ1B26rt-p2GHaiD55IHEP6s0'  # Add your actual key

# Add parent directory to path
# project_root = Path(__file__).parent.parent  # Go from LARS/ to src/
# sys.path.insert(0, str(project_root))

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from hybrid_query_system import HybridQuerySystem  # No lars_simple prefix!
    HYBRID_AVAILABLE = True
    print("✅ Hybrid system imported!")
except ImportError as e:
    HYBRID_AVAILABLE = False
    print(f"⚠️ Import failed: {e}")


# ============================================================================
# SESSION STATE INITIALIZATION - Add this at the TOP of your file
# ============================================================================

def initialize_session_state():
    """
    Initialize all session state variables to prevent AttributeError.
    Call this at the very beginning of main() function.
    """
    
    # Header display control
    if "intro_shown" not in st.session_state:
        st.session_state.intro_shown = False
    
    # Page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    # AI Assistant chat states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Year selection for filtering
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = None  # or set a default year like 2024
    
    # Project ID (if you're using projects)
    if "project_id" not in st.session_state:
        st.session_state.project_id = None
    
    # Sample/data selection states
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None
    
    if "selected_samples" not in st.session_state:
        st.session_state.selected_samples = []
    
    # Analysis states
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    
    # Filter states
    if "filter_status" not in st.session_state:
        st.session_state.filter_status = "all"
    
    if "filter_date_range" not in st.session_state:
        st.session_state.filter_date_range = None
    
    # Data loading states
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
    
    # User preferences
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    
    if "language" not in st.session_state:
        st.session_state.language = "en"
    
    # Add any other session state variables your app uses
    # Format: if "variable_name" not in st.session_state:
    #             st.session_state.variable_name = default_value

@st.cache_resource
def get_hybrid_system_optional():
    """
    Initialize hybrid query system
    Works WITHOUT NLPController (no ISO docs, but pesticide queries work!)
    """
    
    # Check if hybrid module available
    if not HYBRID_AVAILABLE:
        print("⚠️ HYBRID_AVAILABLE is False")
        return None
    
    # Check if API key is set
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_key:
        print("⚠️ GEMINI_API_KEY not set")
        return None
    
    try:
        print("🔄 Initializing hybrid system...")
        
        # Initialize WITHOUT NLP controller (works for pesticide queries!)
        hybrid_system = HybridQuerySystem(
            excel_path='/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx',
            iso_rag_client=None,  # ⭐ Set to None - no ISO docs but pesticide works!
            gemini_api_key=gemini_key
        )
        
        print("✅ Hybrid system initialized successfully!")
        print("📊 Pesticide queries enabled (ISO docs disabled)")
        return hybrid_system
    
    except FileNotFoundError as e:
        print(f"⚠️ Data file not found: {e}")
        return None
    
    except Exception as e:
        print(f"⚠️ Hybrid initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

@st.cache_resource
def get_hybrid_system():
    """
    Initialize hybrid query system
    Only for AI Assistant - keeps existing features unchanged
    """
    
    if not HYBRID_AVAILABLE:
        return None
    
    gemini_key = os.environ.get('GEMINI_API_KEY')
    
    if not gemini_key:
        st.sidebar.warning("⚠️ Set GEMINI_API_KEY for hybrid features")
        return None
    
    try:
        # Your existing NLP controller
        from controllers.NLPController import EnhancedNLPController
        nlp_controller = EnhancedNLPController(gemini_api_key=gemini_key)
        
        # Initialize hybrid
        hybrid = HybridQuerySystem(
            excel_path='/Users/a12/mini-rag/src/data/processed_data_output.xlsx',  # Adjust path
            iso_rag_client=nlp_controller,
            gemini_api_key=gemini_key
        )
        
        return hybrid
    
    except Exception as e:
        st.sidebar.error(f"Hybrid init error: {e}")
        return None

def show_lars_intro():
    """Show LARS animated intro for 1 second"""
    
    # Create a container for the intro
    intro_container = st.empty()
    
    # CSS for animation
    intro_css = """
    <style>
    @keyframes slideUp {
        0% {
            transform: translateY(100vh);
            opacity: 0;
        }
        20% {
            transform: translateY(0);
            opacity: 1;
        }
        80% {
            transform: translateY(0);
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh);
            opacity: 0;
        }
    }
    
    .lars-intro {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        animation: slideUp 1.2s ease-in-out forwards;
    }
    
    .lars-title {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        letter-spacing: 4px;
    }
    
    .lars-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 1rem;
    }
    </style>
    """
    
    # HTML for intro
    intro_html = f"""
    {intro_css}
    <div class="lars-intro">
        <div style="text-align: center;">
            <div class="lars-title">LARS</div>
            <div class="lars-subtitle">Laboratory Analytics & Risk System</div>
        </div>
    </div>
    """
    
    # Show the intro
    intro_container.markdown(intro_html, unsafe_allow_html=True)
    
    # Wait for 1.2 seconds (animation duration)
    time.sleep(1.2)
    
    # Clear the intro
    intro_container.empty()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

def perform_anova_analysis(df, group_by, target_col='reading'):
    """
    Perform ANOVA analysis grouped by specified column
    
    Parameters:
    -----------
    df : DataFrame
        Your pesticide data
    group_by : str
        'season' or 'vegetable_english'
    target_col : str
        Column to analyze (default: 'reading')
    
    Returns:
    --------
    dict with f_stat, p_value, groups_data, summary
    """
    
    # Group data
    groups = df.groupby(group_by)[target_col].apply(list)
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups.values)
    
    # Calculate group statistics
    summary = df.groupby(group_by)[target_col].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(4)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'groups_data': groups,
        'summary': summary,
        'significant': p_value < 0.05
    }

def create_anova_visualization(df: pd.DataFrame, group_by: str, target_col: str = 'reading'):
    """Create box plot visualization for ANOVA results with custom colors"""
    
    # ✅ Define color mapping for vegetables
    vegetable_colors = {
        'tomato': '#FF6B6B',
        'tomatoes': '#FF6B6B',
        'cucumber': '#51CF66',
        'cucumbers': '#51CF66',
        'pepper': '#FFA94D',
        'peppers': '#FFA94D',
        'lettuce': '#94D82D',
        'parsley': '#2F9E44',
        'carrot': '#FF922B',
        'carrots': '#FF922B',
        'potato': '#A0785A',
        'potatoes': '#A0785A',
        'eggplant': '#7950F2',
        'eggplants': '#7950F2',
        'spinach': '#37B24D',
        'cabbage': '#74C0FC',
        'onion': '#E7F5FF',
        'onions': '#E7F5FF',
        'garlic': '#F8F9FA',
        'strawberry': '#FF6B9D',
        'strawberries': '#FF6B9D',
        'grape': '#9775FA',
        'grapes': '#9775FA',
        'apple': '#FA5252',
        'apples': '#FA5252',
        'orange': '#FF922B',
        'oranges': '#FF922B'
    }
    
    # Season colors
    season_colors = {
        'Winter': '#339AF0',
        'Spring': '#51CF66',
        'Summer': '#FF922B',
        'Fall': '#FA5252',
        'Autumn': '#FA5252'
    }
    
    # Determine color map based on grouping
    if group_by == 'vegetable_english':
        # Create color map for vegetables
        unique_veggies = df[group_by].unique()
        color_map = {}
        for veg in unique_veggies:
            veg_lower = str(veg).lower()
            color_map[veg] = vegetable_colors.get(veg_lower, '#868E96')  # Default gray
    elif group_by == 'season':
        # Use season colors
        color_map = season_colors
    else:
        color_map = None
    
    # Create figure with colors
    fig = px.box(
        df, 
        x=group_by, 
        y=target_col,
        color=group_by,
        color_discrete_map=color_map,
        title=f'{target_col.replace("_", " ").title()} Distribution by {group_by.replace("_", " ").title()}',
        labels={
            group_by: group_by.replace('_', ' ').title(), 
            target_col: target_col.replace('_', ' ').title()
        },
        points="outliers"  # Show outlier points
    )
    
    fig.update_layout(
        showlegend=False,
        height=500,
        hovermode='x unified',
        xaxis_title=group_by.replace('_', ' ').title(),
        yaxis_title=target_col.replace('_', ' ').title()
    )
    
    return fig

def create_means_comparison_plot(summary_df: pd.DataFrame, group_by: str):
    """Create bar plot comparing means with error bars and colors"""
    
    # Same color mappings as above
    vegetable_colors = {
        'tomato': '#FF6B6B', 'cucumber': '#51CF66', 'pepper': '#FFA94D',
        'lettuce': '#94D82D', 'parsley': '#2F9E44', 'carrot': '#FF922B',
        'potato': '#A0785A', 'eggplant': '#7950F2', 'spinach': '#37B24D',
        'cabbage': '#74C0FC', 'strawberry': '#FF6B9D', 'grape': '#9775FA',
        'apple': '#FA5252', 'orange': '#FF922B'
    }
    
    season_colors = {
        'Winter': '#339AF0',
        'Spring': '#51CF66',
        'Summer': '#FF922B',
        'Fall': '#FA5252',
        'Autumn': '#FA5252'
    }
    
    # Determine colors
    if group_by == 'vegetable_english':
        colors = [vegetable_colors.get(str(idx).lower(), '#868E96') for idx in summary_df.index]
    elif group_by == 'season':
        colors = [season_colors.get(str(idx), '#868E96') for idx in summary_df.index]
    else:
        colors = 'lightblue'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=summary_df.index,
        y=summary_df['mean'],
        error_y=dict(
            type='data',
            array=summary_df['std'],
            visible=True
        ),
        marker_color=colors,
        name='Mean ± Std Dev',
        text=summary_df['mean'].round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Mean {group_by.replace("_", " ").title()} Readings',
        xaxis_title=group_by.replace('_', ' ').title(),
        yaxis_title='Mean Reading',
        height=400,
        showlegend=False
    )
    
    return fig

st.set_page_config(
    page_title="🔮 LARS Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with LARS theme
st.markdown("""
<style>
    /* ===============================================
       GLOBAL ENHANCEMENTS
       =============================================== */
    
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    
    /* Global font */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main app background */
    .main {
        background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
    }
    
    /* ===============================================
       HEADER STYLING
       =============================================== */
    .main h1 {
        color: #2e7d32;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #1b5e20, #4caf50);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main h2 {
        color: #2e7d32;
        font-weight: 600;
    }
    
    .main h3 {
        color: #388e3c;
        font-weight: 600;
    }
    
    /* ===============================================
       SIDEBAR ENHANCEMENTS
       =============================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5e8 0%, #c8e6c9 100%);
        border-right: 3px solid #81c784;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1b5e20;
        font-weight: 700;
    }
    
    /* ===============================================
       BUTTON STYLING
       =============================================== */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.4);
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Primary button variant */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #1b5e20 0%, #388e3c 100%);
    }
    
    /* Secondary button variant */
    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%);
        color: #1b5e20;
    }
    
    /* ===============================================
       METRIC CARDS
       =============================================== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1b5e20, #4caf50);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Metric container enhancement */
    [data-testid="metric-container"] {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #4caf50;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.2);
    }
    
    /* ===============================================
       INFO BOXES & ALERTS
       =============================================== */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #4caf50;
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-color: #4caf50;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-color: #2196f3;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-color: #ff9800;
    }
    
    .stError {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-color: #f44336;
    }
    
    /* ===============================================
       EXPANDER STYLING
       =============================================== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        border-radius: 10px;
        font-weight: 600;
        border-left: 4px solid #4caf50;
        padding: 15px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #dcedc8 0%, #c5e1a5 100%);
        transform: translateX(5px);
    }
    
    /* ===============================================
       DATAFRAME STYLING
       =============================================== */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 2px solid #c8e6c9;
    }
    
    /* ===============================================
       TAB STYLING
       =============================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #c8e6c9;
    }
    
    /* ===============================================
       INPUT FIELDS
       =============================================== */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #c8e6c9;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #4caf50;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
    }
    
    /* ===============================================
       SLIDER STYLING
       =============================================== */
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, #c8e6c9 0%, #4caf50 100%);
    }
    
    /* ===============================================
       CHAT MESSAGE STYLING
       =============================================== */
    .stChatMessage {
        background: white;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #4caf50;
    }
    
    /* ===============================================
       DIVIDER
       =============================================== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4caf50, transparent);
        margin: 30px 0;
    }
    
    /* ===============================================
       SCROLLBAR
       =============================================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f8e9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #81c784, #4caf50);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #66bb6a, #388e3c);
    }
    
    /* ===============================================
       LOADING SPINNER
       =============================================== */
    .stSpinner > div {
        border-top-color: #4caf50 !important;
    }
    
    /* ===============================================
       DOWNLOAD BUTTON
       =============================================== */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

API_BASE_URL = "http://127.0.0.1:8000"
API_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

ENDPOINTS = {
    "health": "api/v1/dashboards/health",
    "predict": "api/v1/predict",
    "local_index": "api/v1/nlp/local/index",
    "local_answer": "api/v1/nlp/local/answer",
    "statistical_insights": "api/v1/analytics/statistical-insights",
    "risk_assess": "api/v1/risk/assess",
    "timeseries_dashboard": "api/v1/timeseries/interactive-dashboard"
}

AVAILABLE_YEARS = ["2022", "2023", "2024", "2025"]
DEFAULT_YEAR = "2022"

# ============================================================================
# API CLIENT CLASS
# ============================================================================

class APIClient:
    """Handles all API communication"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.headers = API_HEADERS.copy()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Bypass proxy for localhost/127.0.0.1
        self.session.trust_env = False
        self.session.proxies = {
            'http': None,
            'https': None
        }

    def _make_request(self, method: str, endpoint: str, 
                     data: Any = None, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=120)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, params=params, timeout=120)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection Error: Could not connect to {url}")
            st.error("Please check if your API server is running and the URL is correct.")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The server might be overloaded.")
            return None
        except requests.exceptions.HTTPError as e:
            if hasattr(e.response, 'text'):
                status_code = e.response.status_code
                if status_code == 404:
                    st.error(f"🔍 Endpoint Not Found: {url}")
                elif status_code == 500:
                    st.error("🔥 Server Error: Something went wrong on the server side")
                else:
                    st.error(f"❌ HTTP Error {status_code}: {e}")
                with st.expander("Error Details"):
                    st.text(e.response.text)
            return None
        except Exception as e:
            st.error(f"💥 Unexpected error: {str(e)}")
            return None
    
    def health_check(self) -> Optional[Dict]:
        """Check API health"""
        return self._make_request("GET", ENDPOINTS["health"])
    
    def predict_single(self, prediction_data: Dict) -> Optional[Dict]:
        """Run single prediction"""
        return self._make_request("POST", ENDPOINTS["predict"], data=prediction_data)
    
    def local_index(self, year: str, project_id: int = 1, do_reset: bool = False) -> Optional[Dict]:
        """Index data for a specific year"""
        endpoint = f"{ENDPOINTS['local_index']}/{project_id}"
        data = {
            "year": year,
            "data_folder_path": "/app/data",
            "months": [],
            "do_reset": do_reset
        }
        return self._make_request("POST", endpoint, data=data)
    
    def local_answer(self, query: str, year: str, project_id: int = 1,
                    n_results: int = 15, include_risk: bool = True) -> Optional[Dict]:
        """Query indexed data"""
        endpoint = f"{ENDPOINTS['local_answer']}/{project_id}"
        data = {
            "query": query,
            "n_results": n_results,
            "include_risk_assessment": include_risk,
            "population_type": "adult_average"
        }
        params = {"year": year}
        return self._make_request("POST", endpoint, data=data, params=params)
    
    def get_statistical_insights(self) -> Optional[Dict]:
        """Get statistical insights"""
        return self._make_request("GET", ENDPOINTS["statistical_insights"])
    
    def assess_risk(self, risk_data: Dict) -> Optional[Dict]:
        """Enhanced risk assessment"""
        return self._make_request("POST", ENDPOINTS["risk_assess"], data=risk_data)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_year_from_text(text: str) -> Optional[str]:
    """Extract year from user query"""
    year_match = re.search(r'\b(202[2-5])\b', text)
    if year_match:
        return year_match.group(1)
    
    for year in AVAILABLE_YEARS:
        if year in text:
            return year
    
    text_lower = text.lower()
    current_year = datetime.now().year
    
    if "last year" in text_lower or "previous year" in text_lower:
        year = str(current_year - 1)
        return year if year in AVAILABLE_YEARS else None
    
    if "this year" in text_lower or "current year" in text_lower:
        year = str(current_year)
        return year if year in AVAILABLE_YEARS else None
    
    return None
def map_season_names(df):
    """Convert numeric seasons to text names"""
    season_mapping = {
        0: 'Winter',
        1: 'Spring', 
        2: 'Summer',
        3: 'Fall',
        '0': 'Winter',
        '1': 'Spring',
        '2': 'Summer', 
        '3': 'Fall',
        'winter': 'Winter',
        'spring': 'Spring',
        'summer': 'Summer',
        'fall': 'Fall',
        'autumn': 'Fall'
    }
    
    if 'season' in df.columns:
        # Create new column with mapped names
        df['season_name'] = df['season'].map(season_mapping)
        
        # If mapping failed (some values not in dict), keep original
        df['season_name'] = df['season_name'].fillna(df['season'])
        
        # Use season_name for analysis
        df['season'] = df['season_name']
        
    return df
# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": """👋 Hello! I'm your pesticide analysis AI assistant.

**What I can help you with:**
- 📊 Query pesticide data by year
- 🔍 Find samples exceeding limits
- 🎯 Perform risk assessments
- 📈 Analyze trends across years
- 📊 View statistical insights

**Example queries:**
- "Find pyridaben exceeding limits in 2023"
- "Show me tomato samples from 2024"
- "Assess health risk for carbamates in 2022"

Ask your question to get started! 🚀"""
        }]
    
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = DEFAULT_YEAR
    
    if 'indexed_years' not in st.session_state:
        st.session_state.indexed_years = set()
    
    if 'project_id' not in st.session_state:
        st.session_state.project_id = "TrhBVe_m5gg2002_E5VVqS"
    # Track if intro has been shown
    if 'intro_shown' not in st.session_state:
        st.session_state.intro_shown = False

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def get_api_client():
    """Get API client instance (not cached to avoid issues)"""
    return APIClient()

import textwrap

def render_animated_header():
    """Render the animated LARS header exactly as shown in the screenshot"""
    
    html_content = (
        '<div class="lars-hero-container">'
        '<div class="lars-content">'
        '<h1 class="lars-main-title">'
        '<span class="letter" style="animation-delay: 0s">L</span>'
        '<span class="letter" style="animation-delay: 0.1s">A</span>'
        '<span class="letter" style="animation-delay: 0.2s">R</span>'
        '<span class="letter" style="animation-delay: 0.3s">S</span>'
        '</h1>'
        '<h2 class="lars-subtitle-enhanced">'
        '<span class="highlight">Laboratory</span> <span class="accent-text">Analytics</span> & <span class="accent-text">Risk</span> <span class="highlight">System</span>'
        '</h2>'
        '<p class="lars-tagline">'
        'Powered by AI • Trusted by Scientists • Protecting Public Health'
        '</p>'
        '<div class="feature-badges">'
        '<div class="badge">Real-time Analysis</div>'
        '<div class="badge">Advanced Analytics</div>'
        '<div class="badge">Secure & Compliant</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    
    st.markdown(html_content, unsafe_allow_html=True)
    
    css_styles = textwrap.dedent("""
    <style>
    .lars-hero-container {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 25%, #a5d6a7 50%, #81c784 75%, #66bb6a 100%);
        padding: 60px 20px;
        margin: -20px -60px 40px -60px;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 8px 30px rgba(46, 125, 50, 0.25);
        text-align: center;
    }
    
    .lars-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .lars-main-title {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 20px 0;
        letter-spacing: 15px;
        color: #1b5e20;
    }
    
    .letter {
        display: inline-block;
        opacity: 0;
        transform: translateY(-30px);
        animation: letterPop 0.8s ease-out forwards;
        background: linear-gradient(135deg, #1b5e20, #2e7d32, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    @keyframes letterPop {
        0% { opacity: 0; transform: translateY(-30px) scale(0.8); }
        50% { transform: translateY(5px) scale(1.05); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    .lars-subtitle-enhanced {
        font-size: 1.8rem;
        margin: 25px 0;
        font-weight: 500;
        color: #1b5e20;
        animation: fadeInUp 1s ease 0.5s both;
    }
    
    .highlight {
        font-weight: 700;
        color: #2e7d32;
    }
    
    .accent-text {
        font-weight: 700;
        color: #1b5e20;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .lars-tagline {
        font-size: 1.1rem;
        color: #1b5e20;
        margin: 30px 0;
        font-weight: 500;
        opacity: 0.9;
        animation: fadeInUp 1s ease 0.7s both;
    }
    
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 35px;
        animation: fadeInUp 1s ease 0.9s both;
    }
    
    .badge {
        background: rgba(255, 255, 255, 0.9);
        color: #2e7d32;
        padding: 12px 25px;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        border: 1px solid rgba(46, 125, 50, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.2);
        background: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .lars-hero-container { padding: 40px 15px; margin: -20px -20px 30px -20px; }
        .lars-main-title { font-size: 3rem; letter-spacing: 10px; }
        .lars-subtitle-enhanced { font-size: 1.4rem; }
        .lars-tagline { font-size: 1rem; }
        .feature-badges { gap: 15px; }
        .badge { padding: 10px 20px; font-size: 0.9rem; }
    }
    
    @media (max-width: 480px) {
        .lars-main-title { font-size: 2.5rem; letter-spacing: 8px; }
        .lars-subtitle-enhanced { font-size: 1.2rem; }
        .feature-badges { flex-direction: column; align-items: center; }
    }
    </style>
    """)
    
    st.markdown(css_styles, unsafe_allow_html=True)

# The main function remains the same, as the issue was in the header rendering
def main():

    if 'df' not in st.session_state:
        data_path = '/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx'
        if os.path.exists(data_path):
            st.session_state.df = pd.read_excel(data_path)
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded {len(st.session_state.df):,} rows")
            
    """Main application entry point"""
    initialize_session_state()
    if 'intro_shown' not in st.session_state:
        st.session_state.intro_shown = False

    try:
        # Show LARS animated header only once
        if not st.session_state.intro_shown:
            render_animated_header()
            st.session_state.intro_shown = True
        
        # Rest of your existing code...
        api_client = get_api_client()
        
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
        
        # Your existing sidebar and page routing code...
        render_sidebar(api_client)

        page = st.session_state.get('current_page', '🏠 Home')
        
        # Dictionary-based router for page navigation
        pages = {
            "🏠 Home": show_home_page,
            "🔮 Predictions": show_predictions_page,
            "💬 AI Assistant": show_chat_page,
            "📊 Statistical Insights": show_statistical_insights_page,
            "📈 ANOVA Analysis": show_anova_analysis_page,
            "🎯 Advanced Insights": show_advanced_insights_page,
            "🚀 Time Series Forecasting": show_timeseries_page,
        }
        
        page_function = pages.get(page)
        if page_function:
            page_function(api_client)
        else:
            st.error(f"Unknown page: {page}")

    except Exception as e:
        st.error(f"An error occurred in the main application: {e}")
        st.exception(e)
# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(api_client):
    """Render sidebar with navigation and settings"""
    
    with st.sidebar:
        
        st.markdown("## 🔗 API Configuration")
        
        current_url = st.text_input("API Base URL", value=API_BASE_URL, key="api_url")
        
        if current_url != API_BASE_URL:
            api_client = APIClient(current_url)
        
        st.markdown("---")
        
        # API Status Check
        if st.button("🔄 Check API Connection", key="health_check", use_container_width=True, type="primary"):
            with st.spinner("Checking API connection..."):
                health_status = api_client.health_check()
                if health_status:
                    st.success("✅ API Connected")
                    with st.expander("API Response"):
                        st.json(health_status)
                else:
                    st.error("❌ API Disconnected")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 📍 Navigate to:")
        
        # ✅ FIX: Use index to set default page
        page_options = [
            "🏠 Home", 
            "💬 AI Assistant", 
            "🔮 Predictions", 
            "📊 Statistical Insights",
            "📈 ANOVA Analysis",
            "🎯 Advanced Insights",
            "🚀 Time Series Forecasting"
        ]
        
        # Get current page or default to Home
        current_page = st.session_state.get('current_page', '🏠 Home')
        
        # Find index of current page
        try:
            default_index = page_options.index(current_page)
        except ValueError:
            default_index = 0  # Default to Home if not found
        
        page = st.selectbox(
            "Select page:",
            page_options,
            index=default_index,
            key="current_page"
        )

# ============================================================================
# PAGE: HOME
# ============================================================================
def show_timeseries_page(api_client):
    """Time Series Forecasting page with embedded dashboard"""
    
    st.title("🚀 Time Series Forecasting Dashboard")
    
    st.markdown("""
    **Interactive Pesticide Forecasting**
    - 📊 View historical trends and future predictions
    - 📈 Multiple forecasting models (Prophet, ARIMA)
    - 🔮 Confidence intervals and uncertainty estimates
    - 📅 Different time horizons: Daily, Weekly, Monthly, Yearly
    """)
    
    # Dashboard URL
    dashboard_url = f"{api_client.base_url}/api/v1/timeseries/interactive-dashboard"
    
    # Display options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📡 **Dashboard URL:** `{dashboard_url}`")
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_timeseries", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Access Dashboard Section
    st.subheader("🌐 Access Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <a href="{dashboard_url}" target="_blank">
            <button style="
                background: linear-gradient(135deg, #51CF66 0%, #40C057 100%);
                color: white;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                width: 100%;
                box-shadow: 0 4px 12px rgba(81, 207, 102, 0.3);
            ">
                🚀 Open Interactive Dashboard
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.link_button(
            "📱 Open in New Tab",
            dashboard_url,
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Embedded Dashboard
    st.subheader("📺 Embedded Dashboard")
    
    iframe_height = st.slider(
        "Adjust dashboard height:",
        min_value=400,
        max_value=1200,
        value=800,
        step=50
    )
    
    st.markdown(f"""
    <iframe 
        src="{dashboard_url}" 
        width="100%" 
        height="{iframe_height}"
        style="border: 2px solid #51CF66; border-radius: 10px;"
        sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
    ></iframe>
    """, unsafe_allow_html=True)


def show_home_page(api_client):
    """Home page with overview"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #4caf50;
            text-align: center;
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-10px)'; this.style.boxShadow='0 10px 30px rgba(76,175,80,0.3)'"
           onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.1)'">
            <div style="font-size: 40px; margin-bottom: 10px;">📊</div>
            <h3 style="color: #2e7d32; margin: 10px 0;">1,234</h3>
            <p style="color: #666; font-size: 14px;">Total Samples</p>
            <p style="color: #4caf50; font-weight: 600;">↑ 12% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #ff9800;
            text-align: center;
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-10px)'; this.style.boxShadow='0 10px 30px rgba(255,152,0,0.3)'"
           onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.1)'">
            <div style="font-size: 40px; margin-bottom: 10px;">⚠️</div>
            <h3 style="color: #f57c00; margin: 10px 0;">Medium</h3>
            <p style="color: #666; font-size: 14px;">Risk Level</p>
            <p style="color: #f57c00; font-weight: 600;">↓ 3% improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #2196f3;
            text-align: center;
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-10px)'; this.style.boxShadow='0 10px 30px rgba(33,150,243,0.3)'"
           onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.1)'">
            <div style="font-size: 40px; margin-bottom: 10px;">✅</div>
            <h3 style="color: #1976d2; margin: 10px 0;">89%</h3>
            <p style="color: #666; font-size: 14px;">Completed Analysis</p>
            <p style="color: #2196f3; font-weight: 600;">↑ 4% progress</p>
        </div>
        """, unsafe_allow_html=True)

    
# ============================================================================
# PAGE: STATISTICAL INSIGHTS
# ============================================================================

def show_statistical_insights_page(api_client):
    """Statistical Insights page with Mann-Kendall and ANOVA"""
    
    st.title("📊 Statistical Insights")
    
    st.write("""
    Advanced statistical analysis of pesticide contamination patterns:
    - **Mann-Kendall Test**: Detects temporal trends (increasing/decreasing contamination)
    - **ANOVA**: Compares readings across seasons and vegetable categories
    """)
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_insights"):
            st.rerun()
    
    st.divider()
    
    try:
        with st.spinner("🔍 Analyzing data..."):
            response = api_client.get_statistical_insights()
            
            if not response:
                st.error("❌ Failed to fetch insights from API")
                return
            
            # Check if insights exist
            if 'insights' not in response or not response['insights']:
                st.info("""
                ℹ️ No significant statistical patterns detected. This could mean:
                - Data is stable (good news!)
                - Insufficient data for trend detection
                - Need more time periods for analysis
                """)
                return
            
            insights = response['insights']
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Insights", response.get('total_insights', 0))
            with col2:
                high_count = response.get('high_severity_count', 0)
                st.metric("High Priority", high_count, 
                         delta="⚠️ Action Required" if high_count > 0 else "✅ All Good")
            with col3:
                st.metric("Analysis Time", datetime.now().strftime("%H:%M:%S"))
            
            st.divider()
            
            # Display insights by type
            severity_icons = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }
            
            # Group insights by type
            trend_insights = [i for i in insights if i.get('type') == 'trend']
            seasonal_insights = [i for i in insights if i.get('type') == 'seasonal']
            category_insights = [i for i in insights if i.get('type') == 'category']
            
            # Trend Insights
            if trend_insights:
                st.subheader("📈 Temporal Trends (Mann-Kendall Test)")
                st.write("Statistical test to detect if contamination is increasing or decreasing over time")
                
                for insight in trend_insights:
                    severity = insight.get('severity', 'low')
                    icon = severity_icons.get(severity, '⚪')
                    
                    st.info(f"{icon} {insight.get('message', 'No message')}")
                    
                    if severity == 'high':
                        st.warning("⚠️ **Recommended Action**: Investigate contamination source and review farming practices")
            
            # Seasonal Insights
            if seasonal_insights:
                st.subheader("🌍 Seasonal Analysis (ANOVA)")
                st.write("Comparing average readings across different seasons")
                
                for insight in seasonal_insights:
                    severity = insight.get('severity', 'low')
                    icon = severity_icons.get(severity, '⚪')
                    
                    st.info(f"{icon} {insight.get('message', 'No message')}")
                    
                    # Show seasonal data chart if available
                    if 'season_data' in insight:
                        st.bar_chart(insight['season_data'])
                        st.caption("Higher bars indicate seasons with elevated contamination levels")
            
            # Category Insights
            if category_insights:
                st.subheader("🥬 Vegetable Category Analysis")
                st.write("Comparing contamination levels across different vegetable types")
                
                for insight in category_insights:
                    severity = insight.get('severity', 'low')
                    icon = severity_icons.get(severity, '⚪')
                    
                    st.info(f"{icon} {insight.get('message', 'No message')}")
                    
                    # Show category data chart if available
                    if 'category_data' in insight:
                        st.bar_chart(insight['category_data'])
                        st.caption("Comparison of average contamination by vegetable category")
            
            # Export insights
            st.divider()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if st.download_button(
                label="📥 Export Insights (JSON)",
                data=json.dumps(response, indent=2, ensure_ascii=False),
                file_name=f"statistical_insights_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            ):
                st.success("✅ Insights exported successfully!")
    
    except Exception as e:
        st.error(f"❌ Error loading insights: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

def show_predictions_page(api_client):
    """Predictions page with enhanced risk assessment only"""
    
    st.title("🎯 Enhanced Risk Assessment")
    
    # Remove the tabs and directly show enhanced risk assessment
    show_enhanced_risk_form(api_client)

def show_enhanced_risk_form(api_client):
    """Enhanced risk assessment form"""
    
    st.markdown("### Comprehensive Risk Analysis")
    st.write("""
    Uses advanced statistical methods:
    - **Monte Carlo Simulation**: Probabilistic exceedance estimation
    - **Bayesian Analysis**: Prior knowledge from pesticide group
    - **Hazard Modeling**: Vegetable-specific exposure multipliers
    - **Confidence Intervals**: Statistical uncertainty quantification
    """)
    
    # Create form
    with st.form("enhanced_risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vegetable = st.text_input("Vegetable", value="parsley", help="Name of the vegetable/fruit being tested")
            veg_category = st.selectbox("Vegetable Category",
                ["leafy_greens", "peppers", "herbs", "root_vegetables", "fruits", "berries", "cruciferous"],
                help="Category affects exposure multipliers")
            pesticide_group = st.selectbox("Pesticide Group",
                ["organophosphorus", "pyrethroid", "chitin_synthesis_inhibitor",
                 "neonicotinoid", "phenylpyrazoles", "carbamates", "triazoles"],
                help="Chemical group affects toxicity profile")
        
        with col2:
            reading = st.number_input("Reading (mg/kg)", value=50.0, min_value=0.0, step=0.1, 
                                    help="Measured pesticide concentration")
            limit = st.number_input("Limit (mg/kg)", value=20.0, min_value=0.0, step=0.1,
                                  help="Maximum residue limit (MRL)")
            
            # Real-time compliance check
            exceedance_ratio = reading / limit if limit > 0 else 0
            is_compliant = reading <= limit
            
            # Display compliance status immediately
            if is_compliant:
                st.success(f"✅ Compliant (Exceedance: {exceedance_ratio:.2f}x)")
            else:
                st.error(f"❌ Non-Compliant (Exceedance: {exceedance_ratio:.2f}x)")
        
        # Additional risk parameters
        st.markdown("---")
        st.markdown("#### 🔧 Advanced Parameters")
        
        col3, col4 = st.columns(2)
        
        with col3:
            population_type = st.selectbox(
                "Population Type",
                ["general", "vulnerable", "children", "pregnant_women"],
                help="Target population for risk assessment"
            )
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95,
                help="Statistical confidence level for intervals"
            )
        
        with col4:
            simulation_runs = st.selectbox(
                "Monte Carlo Simulations",
                [1000, 5000, 10000, 50000],
                index=1,
                help="Number of simulations for probabilistic analysis"
            )
            include_uncertainty = st.checkbox(
                "Include Measurement Uncertainty",
                value=True,
                help="Account for analytical measurement errors"
            )
        
        # Submit button
        submitted = st.form_submit_button("🎯 Run Comprehensive Risk Assessment", 
                                        type="primary", 
                                        use_container_width=True)
    
    # Handle form submission
    if submitted:
        risk_data = {
            "vegetable": vegetable,
            "vegetable_category": veg_category,
            "pesticide_group": pesticide_group,
            "reading": reading,
            "limit": limit,
            "is_compliant": is_compliant,
            "exceedance_ratio": exceedance_ratio,
            "population_type": population_type,
            "confidence_level": confidence_level,
            "simulation_runs": simulation_runs,
            "include_uncertainty": include_uncertainty
        }
        
        with st.spinner("🔄 Performing comprehensive risk assessment..."):
            try:
                result = api_client.assess_risk(risk_data)
                
                if result and 'risk_score' in result:
                    st.success("✅ Risk assessment completed!")
                    display_enhanced_risk_result(result)
                elif result and result.get('success'):
                    st.success("✅ Risk assessment completed!")
                    display_enhanced_risk_result(result.get('data', {}))
                elif result:
                    st.error(f"❌ Assessment failed: {result.get('error', 'Unknown error')}")
                    with st.expander("API Response"):
                        st.json(result)
                else:
                    st.error("❌ No response from API. Check if the endpoint is available.")
                    st.info("💡 Make sure the API server is running and the risk assessment endpoint is configured.")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)

def display_enhanced_risk_result(risk_result: Dict):
    """Display enhanced risk assessment results"""
    
    st.markdown("## 📊 Risk Assessment Results")
    
    # Risk Level Banner
    risk_level = risk_result.get('risk_level', 'UNKNOWN')
    risk_score = risk_result.get('risk_score', 0)
    exceedance_prob = risk_result.get('exceedance_probability', 0) * 100
    
    if risk_level == 'CRITICAL':
        st.error(f"""
        🔴 **CRITICAL RISK** 
        
        **Risk Score:** {risk_score}/5.0 | **Exceedance Probability:** {exceedance_prob:.1f}%
        
        ⚠️ **Immediate action required**: Do not distribute. Investigate contamination source immediately.
        """)
    elif risk_level == 'HIGH':
        st.warning(f"""
        🟠 **HIGH RISK**
        
        **Risk Score:** {risk_score}/5.0 | **Exceedance Probability:** {exceedance_prob:.1f}%
        
        ℹ️ **Recommendation**: Enhanced monitoring required. Consider retesting before distribution.
        """)
    elif risk_level == 'MEDIUM':
        st.info(f"""
        🟡 **MEDIUM RISK**
        
        **Risk Score:** {risk_score}/5.0 | **Exceedance Probability:** {exceedance_prob:.1f}%
        
        📋 **Recommendation**: Standard monitoring. Review sourcing practices.
        """)
    else:
        st.success(f"""
        🟢 **LOW RISK**
        
        **Risk Score:** {risk_score}/5.0 | **Exceedance Probability:** {exceedance_prob:.1f}%
        
        ✅ **Status**: Acceptable for distribution with routine monitoring.
        """)
    
    # Key Metrics
    st.markdown("### 📈 Key Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bayesian_risk = risk_result.get('bayesian_risk', 0)
        bayesian_confidence = risk_result.get('bayesian_confidence', (0, 0))
        st.metric(
            "Bayesian Risk Score",
            f"{bayesian_risk:.2f}",
            delta=f"×{risk_result.get('vegetable_multiplier', 1.0)} exposure"
        )
        st.write(f"**95% Credible Interval:** [{bayesian_confidence[0]:.4f}, {bayesian_confidence[1]:.4f}]")

    with col2:
        critical_threshold = risk_result.get('critical_threshold', 0)
        st.metric("Critical Threshold", f"{critical_threshold:.1f}")
    
    with col3:
        monte_carlo_est = risk_result.get('exceedance_probability', 0)
        st.metric("Monte Carlo Estimate", f"{monte_carlo_est:.3f}")
    
    with col4:
        confidence = risk_result.get('confidence_interval', 95)
        st.metric("Confidence Level", f"{confidence}%")
    
    # Confidence Interval
    ci = risk_result.get('confidence_interval', (0, 0))
    st.write(f"**{confidence}% Confidence Interval:** [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Detailed Breakdown
    with st.expander("🔍 Detailed Risk Components"):
        components = risk_result.get('components', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Risk Score Components")
            st.write(f"- **Base Risk (Exceedance):** {components.get('base_risk', 0):.3f}")
            st.write(f"- **Probabilistic Risk (Monte Carlo):** {components.get('probabilistic_risk', 0):.3f}")
            st.write(f"- **Bayesian Component:** {components.get('bayesian_component', 0):.3f}")
            st.write(f"- **Uncertainty Adjustment:** {components.get('uncertainty_adjustment', 0):.3f}")
        
        with col2:
            st.markdown("#### ⚖️ Risk Modifiers")
            st.write(f"- **Vegetable Hazard Multiplier:** {risk_result.get('vegetable_multiplier', 1.0)}×")
            st.write(f"- **Pesticide Group Factor:** {risk_result.get('pesticide_group_factor', 1.0)}×")
            st.write(f"- **Population Sensitivity:** {risk_result.get('population_sensitivity', 1.0)}×")
            st.write(f"- **Analytical Method:** {risk_result.get('method', 'LC-MS/MS')}")
    
    # Recommendations
    with st.expander("💡 Actionable Recommendations"):
        recommendations = risk_result.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write("1. **Immediate Testing**: Conduct confirmatory analysis")
            st.write("2. **Source Investigation**: Review supply chain and growing practices")
            st.write("3. **Enhanced Monitoring**: Increase sampling frequency")
            st.write("4. **Documentation**: Maintain detailed records for regulatory compliance")
    
    # Export Options
    st.markdown("---")
    st.markdown("### 📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        json_data = json.dumps(risk_result, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_data,
            file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV export for key metrics
        key_metrics = {
            'Risk_Level': risk_level,
            'Risk_Score': risk_score,
            'Exceedance_Probability': exceedance_prob,
            'Bayesian_Risk': bayesian_risk,
            'Critical_Threshold': critical_threshold
        }
        csv_data = pd.DataFrame([key_metrics]).to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Summary",
            data=csv_data,
            file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )



# ============================================================================
# PAGE: AI ASSISTANT (CHAT)
# ============================================================================

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import numpy as np


# def show_chat_page(api_client):
#     """
#     Fixed chat page that works with simple query system
#     Just copy-paste this to replace your existing show_chat_page function
    
#     Args:
#         api_client: Your existing API client (or None)
#     """
    
#     st.header("💬 AI Assistant 🌱")
#     st.caption("Powered by Gemini - Ask questions in natural language")
    
#     # ISO Document Management in Sidebar
#     with st.sidebar:
#         with st.expander("📄 ISO Document Upload"):
#             uploaded_file = st.file_uploader(
#                 "Upload ISO Document",
#                 type=['pdf', 'docx', 'xlsx'],
#                 help="Limit 200MB per file"
#             )
            
#             doc_type = st.selectbox(
#                 "Document Type",
#                 ["Quality Manual", "Procedure", "Work Instruction", "Form"]
#             )
            
#         if uploaded_file is not None:
#             # Show what will be uploaded
#             st.caption(f"Will upload to project: {st.session_state.project_id}")
            
#             if st.button("📤 Upload ISO Document", use_container_width=True, type="primary"):
#                 # Pass the project ID from session state
#                 upload_iso_document_ui(
#                     api_client, 
#                     uploaded_file, 
#                     doc_type, 
#                     st.session_state.project_id  # ✅ Use actual project ID
#                 )
            
#         # Settings
#         st.markdown("---")
#         st.markdown("### ⚙️ Settings")
        
#         risk_mode = st.radio(
#             "Risk Assessment Mode",
#             ["🔵 Auto-detect", "✅ Always On", "❌ Always Off"]
#         )
        
#         # Apply risk mode to detection
#         if risk_mode == "✅ Always On":
#             has_risk = True  # Force risk assessment
#         elif risk_mode == "❌ Always Off":
#             has_risk = False  # Disable risk assessment
    
#     # ========================================================================
#     # STEP 1: Get DataFrame (adjust based on your app)
#     # ========================================================================
    
#     try:
#         # Try multiple ways to get the dataframe
#         if api_client and hasattr(api_client, 'get_data'):
#             df = api_client.get_data()
#         elif api_client and hasattr(api_client, 'df'):
#             df = api_client.df
#         elif 'current_data' in st.session_state:
#             df = st.session_state.current_data
#         elif 'df' in st.session_state:
#             df = st.session_state.df
#         else:
#             st.warning("⚠️ No data loaded. Please load data first.")
#             st.info("💡 Go to the data loading page or upload your Excel file.")
#             return
    
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return
    
#     # ========================================================================
#     # STEP 2: Initialize Gemini
#     # ========================================================================
    
#     api_key = os.environ.get('GEMINI_API_KEY')
    
#     if not api_key:
#         st.error("⚠️ GEMINI_API_KEY not set")
#         st.info("""
#         **How to fix:**
#         1. Get your API key from: https://makersuite.google.com/app/apikey
#         2. Set it in your code: `os.environ['GEMINI_API_KEY'] = "your-key"`
#         3. Or set as environment variable: `export GEMINI_API_KEY="your-key"`
#         """)
#         return
    
#     # Configure Gemini
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
#     # ========================================================================
#     # STEP 3: Show Instructions
#     # ========================================================================
    
#     st.info("""
#     **🌱 How to use:**
#     1. Type your question about the pesticide data
#     2. Press Enter or click "Ask"
#     3. View the AI-generated code and results
    
#     **Example Questions:**
#     - "What are the top 5 vegetables with highest violation rates?"
#     - "Show me compliance trends by year"
#     - "Which pesticides have the most violations?"
#     - "Calculate average exceedance ratio by vegetable"
#     - "Find vegetables with more than 10 violations"
#     """)
    
#     # ========================================================================
#     # STEP 4: Get User Input (THIS IS THE KEY PART!)
#     # ========================================================================
    
#     # Method 1: Chat input (modern Streamlit way)
#     chat_question = st.chat_input("Ask a question about the data...")
    
#     # Method 2: Text area + button (alternative)
#     st.markdown("---")
#     col1, col2 = st.columns([4, 1])
    
#     with col1:
#         text_question = st.text_area(
#             "Or type your question here:",
#             height=100,
#             placeholder="Example: What are the top 10 vegetables by violation count?"
#         )
    
#     with col2:
#         st.write("")  # Spacing
#         st.write("")  # Spacing
#         ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
#     # Determine which input to use
#     user_question = chat_question or (text_question if ask_button else None)
    
#     # ========================================================================
#     # STEP 5: Process Question
#     # ========================================================================
    
#     if user_question:
#         # DETECT QUERY TYPE
#         iso_keywords = ['iso', '17025', 'standard', 'procedure', 'manual', 
#                         'quality', 'calibration', 'validation', 'document', 'sop']
#         risk_keywords = ['risk', 'assessment', 'health', 'safety', 'adi', 'arfd']
        
#         query_lower = user_question.lower()
#         is_iso = any(kw in query_lower for kw in iso_keywords)
#         has_risk = any(kw in query_lower for kw in risk_keywords)
        
#         # HANDLE ISO QUERIES WITH RAG
#         if is_iso:
#             with st.spinner("🔍 Searching ISO documents..."):
#                 if hasattr(api_client, 'nlp_controller'):
#                     try:
#                         result = api_client.nlp_controller.query(user_question)
#                         st.success("✅ Found ISO information")
#                         st.markdown("### 📋 Answer:")
#                         st.write(result.get('answer', 'No information found'))
                        
#                         if result.get('sources'):
#                             with st.expander("📚 Sources"):
#                                 for src in result['sources']:
#                                     st.write(f"• {src}")
#                     except Exception as e:
#                         st.error(f"ISO search error: {e}")
#                 else:
#                     st.info("ISO documents not configured. Falling back to general query.")
#                     is_iso = False  # Fall back to Gemini
        
#         # HANDLE PESTICIDE QUERIES WITH GEMINI (your existing code)
#         if not is_iso:  # Only run if not an ISO query
#             with st.spinner("🔄 Generating query and analyzing data..."):
                
#                 risk_instruction = ""
#                 if has_risk:
#                     risk_instruction = """
                    
# Also calculate risk assessment:
# - Add 'Risk_Level' column: High (>2x MRL), Medium (1-2x), Low (<1x)
# - Calculate exceedance ratio (Result/MRL) if possible
# - Include health risk indicators
# """
#                 prompt = f"""Generate pandas code to answer this question about pesticide testing data.

# Dataset information:
# - Columns: {list(df.columns)}
# - Shape: {df.shape}
# - Sample data (first 2 rows):
# {df.head(2).to_dict('records')}

# Question: {user_question}
# {risk_instruction}

# Requirements:
# 1. Write Python code using pandas (imported as 'pd')
# 2. The DataFrame is available as variable 'df'
# 3. Store the final result in a variable called 'result'
# 4. Handle any errors gracefully
# 5. Return clean, formatted output (DataFrame or simple value)
# 6. If aggregating, include meaningful column names

# Generate only the Python code, wrapped in ```python``` markers.
# """
                
#                 try:
#                     # Generate code with Gemini
#                     response = model.generate_content(prompt)
                    
#                     # Extract code from response
#                     code_match = re.search(r'```python\n(.*?)```', response.text, re.DOTALL)
                    
#                     if code_match:
#                         generated_code = code_match.group(1).strip()
#                     else:
#                         # Fallback: try without python specifier
#                         code_match = re.search(r'```\n(.*?)```', response.text, re.DOTALL)
#                         if code_match:
#                             generated_code = code_match.group(1).strip()
#                         else:
#                             # Last resort: use entire response
#                             generated_code = response.text.strip()
                    
#                     # Execute the generated code
#                     local_vars = {
#                         'df': df.copy(),
#                         'pd': pd,
#                         'np': np,
#                         'result': None
#                     }
                    
#                     exec(generated_code, {}, local_vars)
                    
#                     result = local_vars.get('result')
                    
#                     # ========================================================
#                     # STEP 6: Display Results
#                     # ========================================================
                    
#                     st.success("✅ Query executed successfully!")
                    
#                     # Show the result
#                     st.subheader("📊 Result:")
                    
#                     if isinstance(result, pd.DataFrame):
#                         st.dataframe(result, use_container_width=True)

#                         # Show risk assessment summary if applicable
#                         if has_risk and isinstance(result, pd.DataFrame):
#                             if 'Risk_Level' in result.columns:
#                                 with st.expander("⚠️ Risk Assessment Summary"):
#                                     risk_counts = result['Risk_Level'].value_counts()
                                    
#                                     col1, col2, col3 = st.columns(3)
#                                     with col1:
#                                         st.metric("🔴 High Risk", risk_counts.get('High', 0))
#                                     with col2:
#                                         st.metric("🟡 Medium", risk_counts.get('Medium', 0))
#                                     with col3:
#                                         st.metric("🟢 Low", risk_counts.get('Low', 0))
                                    
#                                     if 'Exceedance_Ratio' in result.columns:
#                                         max_exc = result['Exceedance_Ratio'].max()
#                                         st.write(f"**Max Exceedance:** {max_exc:.2f}x")
                        
#                         # Add download button for DataFrames
#                         csv = result.to_csv(index=False)
#                         st.download_button(
#                             label="📥 Download as CSV",
#                             data=csv,
#                             file_name="query_result.csv",
#                             mime="text/csv"
#                         )
                    
#                     elif isinstance(result, (pd.Series, list, dict)):
#                         st.dataframe(pd.DataFrame(result), use_container_width=True)
                    
#                     else:
#                         # For simple values
#                         st.write(result)
                    
#                     # Show generated code
#                     with st.expander("👨‍💻 View Generated Code"):
#                         st.code(generated_code, language='python')
#                         st.caption("This code was AI-generated and executed on your data")
                    
#                     # Add to history
#                     if 'chat_history' not in st.session_state:
#                         st.session_state.chat_history = []
                    
#                     st.session_state.chat_history.append({
#                         'question': user_question,
#                         'result': result,
#                         'code': generated_code,
#                         'timestamp': pd.Timestamp.now()
#                     })
                
#                 except Exception as e:  # ✅ FIXED: Proper exception handling
#                     st.error(f"❌ Error executing query: {str(e)}")
                    
#                     # Show helpful error information
#                     with st.expander("🔍 Error Details"):
#                         st.code(str(e))
#                         if 'generated_code' in locals():
#                             st.markdown("**Generated Code:**")
#                             st.code(generated_code, language='python')
                    
#                     st.info("""
#                     **Common issues:**
#                     - Column names might not match exactly
#                     - Data type mismatches
#                     - Try rephrasing your question
#                     - Check the generated code for syntax errors
#                     """)


#     # ========================================================================
#     # STEP 7: Show Chat History (Optional)
#     # ========================================================================
    
#     if 'chat_history' in st.session_state and st.session_state.chat_history:
#         st.markdown("---")
#         st.subheader("📜 Recent Queries")
        
#         # Show last 5 queries
#         for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
#             with st.expander(f"Q{i+1}: {item['question'][:60]}..."):
#                 st.markdown(f"**Question:** {item['question']}")
#                 st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
#                 if isinstance(item['result'], pd.DataFrame):
#                     st.dataframe(item['result'].head(10))
#                 else:
#                     st.write(item['result'])
                
#                 show_code = st.checkbox("👨‍💻 Show generated code", value=True, key=f"show_code_hist_{i}")
#                 if show_code:
#                     st.code('code', language='python')
                        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.chat_history = []
#             st.rerun()
    
#     # ========================================================================
#     # STEP 8: Quick Query Buttons (Optional)
#     # ========================================================================
    
#     st.markdown("---")
#     st.subheader("⚡ Quick Queries")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("📊 Top Vegetables"):
#             # Trigger quick query
#             st.session_state.quick_query = "What are the top 10 vegetables by violation count?"
#             st.rerun()
    
#     with col2:
#         if st.button("🧪 Top Pesticides"):
#             st.session_state.quick_query = "What are the top 10 pesticides by violation count?"
#             st.rerun()
    
#     with col3:
#         if st.button("📈 Yearly Trends"):
#             st.session_state.quick_query = "Show me compliance rate by year"
#             st.rerun()
    
#     # Handle quick query
#     if 'quick_query' in st.session_state and st.session_state.quick_query:
#         user_question = st.session_state.quick_query
#         st.session_state.quick_query = None
#         st.rerun()

#     # Call the chat handler function
#     handle_chat_input(api_client)

# # ============================================================================
# # TESTING
# # ============================================================================

# if __name__ == "__main__":
#     print("""
#     ╔══════════════════════════════════════════════════════════╗
#     ║  DROP-IN REPLACEMENT FOR show_chat_page()                ║
#     ╠══════════════════════════════════════════════════════════╣
#     ║                                                          ║
#     ║  ✅ Handles question input internally                   ║
#     ║  ✅ Works with Gemini AI                                ║
#     ║  ✅ No 'question' parameter needed                      ║
#     ║  ✅ Shows generated code                                ║
#     ║  ✅ Includes chat history                               ║
#     ║  ✅ Has quick query buttons                             ║
#     ║  ✅ Download results as CSV                             ║
#     ║                                                          ║
#     ║  Just copy this function to your app.py!                ║
#     ║                                                          ║
#     ╚══════════════════════════════════════════════════════════╝
#     """)

    


def show_chat_page(api_client):
    """
    Enhanced chat page with both RAG (ISO documents) and Gemini (data analysis) capabilities
    Includes complete sidebar with upload and risk assessment options
    """
    
    st.header("💬 AI Assistant 🌱")
    st.caption("Powered by Gemini + RAG - Ask about ISO documents or analyze pesticide data")
    
    # ========================================================================
    # SIDEBAR COMPONENTS
    # ========================================================================
    
    with st.sidebar:
        st.markdown("---")
        
        # Project Configuration
        st.markdown("## 🏢 Project Configuration")
        
        # Initialize project ID in session state if not exists
        if 'project_id' not in st.session_state:
            st.session_state.project_id = "TrhBVe_m5gg2002_E5VVqS"
        
        project_id = st.text_input(
            "Project ID",
            value=st.session_state.project_id,
            help="Enter your project identifier",
            key="project_id_input"
        )
        
        if project_id != st.session_state.project_id:
            st.session_state.project_id = project_id
        
        st.info(f"📁 Active Project: `{st.session_state.project_id}`")
        
        st.markdown("---")
        
        # ISO Document Upload
        st.markdown("## 📄 ISO Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload ISO Document",
            type=['pdf', 'docx', 'xlsx'],
            help="Upload Quality Manual, Procedures, Work Instructions, or Forms"
        )
        
        doc_type = st.selectbox(
            "Document Type",
            [
                "Quality Manual",
                "Procedure", 
                "Work Instruction",
                "Form/Record",
                "Unknown"
            ]
        )
        
        if uploaded_file is not None:
            st.caption(f"Will upload to project: {st.session_state.project_id}")
            
            if st.button("📤 Upload ISO Document", use_container_width=True, type="primary"):
                upload_iso_document_ui(
                    api_client, 
                    uploaded_file, 
                    doc_type, 
                    st.session_state.project_id
                )
        
        st.markdown("---")
        
        # Data Context
        st.markdown("## 📅 Data Context")
        
        # Initialize selected_year if not exists
        if 'selected_year' not in st.session_state:
            st.session_state.selected_year = "2022"
        
        st.info(f"**Active Year:** {st.session_state.selected_year}")
        
        selected_year = st.selectbox(
            "Default Year",
            AVAILABLE_YEARS,
            index=AVAILABLE_YEARS.index(st.session_state.selected_year)
        )
        
        if selected_year != st.session_state.selected_year:
            st.session_state.selected_year = selected_year
        
        st.markdown("---")
        
        # Data Indexing
        st.markdown("## 🗄️ Data Indexing")
        
        year_to_index = st.selectbox("Select Year to Index", AVAILABLE_YEARS)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Index", use_container_width=True):
                index_year_data(api_client, year_to_index, do_reset=False)
        
        with col2:
            if st.button("🗑️ Reset", use_container_width=True):
                index_year_data(api_client, year_to_index, do_reset=True)
        
        # Initialize indexed_years if not exists
        if 'indexed_years' not in st.session_state:
            st.session_state.indexed_years = set()
        
        if st.session_state.indexed_years:
            st.markdown("**✅ Indexed Years:**")
            for year in sorted(st.session_state.indexed_years):
                st.markdown(f"- {year}")

        st.markdown("---")
        
        # Risk Assessment Settings
        st.markdown("## ⚙️ Risk Assessment")
        
        risk_mode = st.radio(
            "Risk Assessment Mode",
            options=["🤖 Auto-detect", "✅ Always On", "❌ Always Off"],
            index=0,
            help="Auto-detect: Enables when you ask about safety/risk\nAlways On: Include for every query\nAlways Off: Never include"
        )
        
        st.session_state.risk_mode = risk_mode
        
        # Additional risk settings
        with st.expander("🔧 Advanced Risk Settings"):
            population_type = st.selectbox(
                "Population Type",
                ["adult_average", "adult_high_consumption", "child_average", "child_high_consumption"],
                index=0,
                help="Select population for risk calculations"
            )
            st.session_state.population_type = population_type
            
            risk_threshold = st.slider(
                "Risk Threshold (%)",
                min_value=50,
                max_value=200,
                value=100,
                help="Percentage of ADI considered as risk threshold"
            )
            st.session_state.risk_threshold = risk_threshold

    # ========================================================================
    # MAIN CHAT INTERFACE
    # ========================================================================
    
    # STEP 1: Get DataFrame (for Gemini analysis)
    try:
        if api_client and hasattr(api_client, 'get_data'):
            df = api_client.get_data()
        elif api_client and hasattr(api_client, 'df'):
            df = api_client.df
        elif 'current_data' in st.session_state:
            df = st.session_state.current_data
        elif 'df' in st.session_state:
            df = st.session_state.df
        else:
            df = None
            st.info("💡 No dataset loaded. You can still ask about ISO documents!")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = None

    # STEP 2: Initialize Gemini
    api_key = os.environ.get('GEMINI_API_KEY')
    gemini_available = bool(api_key)
    
    if gemini_available:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        except Exception as e:
            st.warning(f"⚠️ Gemini configuration issue: {e}")
            gemini_available = False
    else:
        st.info("🔑 Set GEMINI_API_KEY for data analysis features")

    # STEP 3: Show Instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📄 ISO Document Queries:**
        - "Show me calibration procedures"
        - "What are the GC-MS requirements?"
        - "Find sample handling procedures"
        - "Quality manual requirements"
        """)
    
    with col2:
        if df is not None and gemini_available:
            st.info("""
            **📊 Data Analysis Queries:**
            - "Top 5 vegetables with violations"
            - "Compliance trends by year" 
            - "Pesticides with most exceedances"
            - "Calculate average readings by crop"
            """)
        else:
            st.info("""
            **📊 Data Analysis:**
            Load dataset and set GEMINI_API_KEY
            to enable data analysis features
            """)

    # STEP 4: Dual Input System
    st.markdown("---")
    st.subheader("🔍 Ask Detailed Questions")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        text_question = st.text_area(
            "Type your question here:",
            height=100,
            placeholder="Example: What are the top 10 vegetables by violation count? Or ask about ISO procedures...",
            key="text_question"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True, key="ask_button")
    
    # Simple chat input
    st.markdown("---")
    st.subheader("💬 Quick Chat")
    chat_question = st.chat_input("Ask a quick question...")
    
    # Determine which input to use
    user_question = None
    if ask_button and text_question:
        user_question = text_question
    elif chat_question:
        user_question = chat_question

    # STEP 5: Process Question
    if user_question:
        # Enhanced query type detection
        iso_keywords = [
            'iso', '17025', 'clause', 'requirement', 'accreditation',
            'calibration', 'validation', 'verification', 'maintenance',
            'procedure', 'manual', 'quality manual', 'work instruction', 
            'form', 'record', 'document', 'policy', 'sop',
            'gc', 'gc-ms', 'gcms', 'hplc', 'instrument', 'equipment',
            'laboratory', 'lab', 'sample handling', 'traceability',
            'معايرة', 'إجراء', 'دليل الجودة', 'متطلب', 'اعتماد'
        ]
        
        data_keywords = [
            'pesticide', 'vegetable', 'violation', 'compliance', 'exceed',
            'limit', 'reading', 'sample', 'data', 'analysis', 'trend',
            'statistic', 'count', 'average', 'maximum', 'minimum',
            'طماطم', 'فلفل', 'خيار', 'مبيد', 'عينة'
        ]
        
        query_lower = user_question.lower()
        
        iso_score = sum(1 for kw in iso_keywords if kw in query_lower)
        data_score = sum(1 for kw in data_keywords if kw in query_lower)
        
        has_year = any(year in query_lower for year in ['2022', '2023', '2024', '2025'])
        has_data_terms = any(term in query_lower for term in ['top', 'trend', 'average', 'count', 'sum', 'max', 'min'])
        
        if iso_score > data_score:
            query_type = "iso"
        elif data_score > iso_score or has_year or has_data_terms:
            query_type = "data"
        else:
            query_type = "iso"
        
        # Display detection info
        if query_type == "iso":
            st.info("🔍 Detected ISO/document query - searching uploaded documents...")
            process_iso_query(api_client, user_question)
        else:
            if df is None:
                st.error("📊 No dataset available for analysis. Please load your Excel file first.")
                st.info("💡 You can still ask ISO/document questions without a dataset!")
                return
            
            if not gemini_available:
                st.error("🔑 Gemini API key not configured for data analysis.")
                st.info("💡 Set GEMINI_API_KEY environment variable to enable data analysis.")
                return
            
            st.info("🔍 Detected data analysis query - using AI to analyze your dataset...")
            
            # Apply risk assessment settings
            risk_mode = st.session_state.get('risk_mode', '🤖 Auto-detect')
            if risk_mode == "✅ Always On":
                include_risk = True
            elif risk_mode == "❌ Always Off":
                include_risk = False
            else:  # Auto-detect
                risk_keywords = ['risk', 'safety', 'safe', 'health', 'assess', 'eat', 'consume', 'adult', 'child', 'population']
                include_risk = any(kw in query_lower for kw in risk_keywords)
            
            if include_risk:
                st.success("🎯 Risk assessment enabled")
            
            process_data_query(user_question, df, model, include_risk)

    # STEP 6: Quick Action Buttons
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 ISO Procedures", use_container_width=True):
            st.session_state.quick_query = "Show me all calibration procedures"
            st.rerun()
    
    with col2:
        if st.button("🔬 Lab Equipment", use_container_width=True):
            st.session_state.quick_query = "What are the GC-MS maintenance requirements?"
            st.rerun()
    
    with col3:
        if df is not None and st.button("📊 Top Vegetables", use_container_width=True):
            st.session_state.quick_query = "What are the top 10 vegetables by violation count?"
            st.rerun()
    
    with col4:
        if df is not None and st.button("🧪 Risk Analysis", use_container_width=True):
            st.session_state.quick_query = "Show me samples with health risk concerns"
            st.rerun()

    # Handle quick queries
    if 'quick_query' in st.session_state and st.session_state.quick_query:
        user_question = st.session_state.quick_query
        st.session_state.quick_query = None
        st.rerun()

    # STEP 7: Show Chat History
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Recent Queries")
        
        for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
            # Remove the 'key' parameter from expander
            with st.expander(f"Q{i+1}: {item['question'][:50]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Type:** {item.get('query_type', 'Unknown')}")
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                # Display results
                if isinstance(item['result'], pd.DataFrame):
                    st.dataframe(item['result'].head(5))
                elif isinstance(item['result'], dict):
                    st.json(item['result'])
                else:
                    st.write(item['result'])
                
                # Show code in a separate section (NOT nested expander)
                if item.get('code'):
                    st.markdown("---")
                    st.markdown("#### 👨‍💻 Generated Code")
                    st.code(item['code'], language='python')
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
# ========================================================================
# SUPPORTING FUNCTIONS
# ========================================================================

def upload_iso_document_ui(api_client, uploaded_file, doc_type: str, project_id: str):
    """Handle ISO document upload from Streamlit UI"""
    
    with st.spinner(f"📤 Uploading {uploaded_file.name}..."):
        try:
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            data = {
                'document_type': doc_type
            }
            
            url = f"{api_client.base_url}/api/v1/iso/upload/{project_id}"

            st.info(f"🔗 Uploading to: {url}")
            
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
            
            with st.expander("📋 Upload Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Project ID:** {project_id}")
                    st.write(f"**Document ID:** {result.get('document_id', 'N/A')}")
                    st.write(f"**Document Type:** {result.get('document_type', 'N/A')}")
                    st.write(f"**Chunks Indexed:** {result.get('chunks_indexed', 0)}")
                
                with col2:
                    iso_clauses = result.get('iso_clauses', [])
                    if iso_clauses:
                        st.write("**ISO Clauses Found:**")
                        for clause in iso_clauses[:5]:
                            st.write(f"- {clause}")
            
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []
                
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"✅ ISO document '{uploaded_file.name}' has been uploaded and indexed for project {project_id}. You can now ask questions about it!"
            })
            
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Upload failed: {e}")
            if hasattr(e.response, 'text'):
                with st.expander("Error Details"):
                    st.text(e.response.text)
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def index_year_data(api_client, year: str, do_reset: bool = False):
    """Index data for a specific year"""
    action = "Resetting and re-indexing" if do_reset else "Indexing"
    
    with st.spinner(f"{action} {year} data..."):
        result = api_client.local_index(year, do_reset=do_reset)
        
        if result and result.get('signal') == 'LOCAL_INDEX_SUCCESS':
            count = result.get('indexed_documents', 0)
            if count > 0:
                st.session_state.indexed_years.add(year)
                st.success(f"✅ {year} indexed: {count} documents")
            else:
                st.error(f"⚠️ No documents found for {year}")
        else:
            st.error(f"❌ Failed to index {year}")

def process_iso_query(api_client, query: str):
    """Process ISO document queries using RAG"""
    
    project_id = st.session_state.get('project_id', 'default_project')
    
    with st.spinner("🔍 Searching ISO documents..."):
        try:
            url = f"{api_client.base_url}/api/v1/iso/query/{project_id}"
            
            response = requests.post(
                url,
                json={"text": query},
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            st.markdown("### 📋 Answer:")
            st.markdown(result.get('answer', 'No information found in ISO documents.'))
            
            sources = result.get('sources', [])
            if sources:
                with st.expander("📚 Source Documents"):
                    for idx, source in enumerate(sources, 1):
                        st.write(f"**{idx}. {source.get('file', 'Unknown')}**")
                        st.write(f"Type: {source.get('type', 'Unknown')}")
                        if source.get('clauses'):
                            st.write(f"ISO Clauses: {', '.join(source.get('clauses', []))}")
                        st.markdown("---")
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({
                'question': query,
                'result': result,
                'query_type': 'ISO/RAG',
                'timestamp': pd.Timestamp.now()
            })
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                st.error("📄 No ISO documents found. Please upload documents first!")
                st.info("💡 Use the ISO Document Upload in the sidebar to add your documents.")
            else:
                st.error(f"❌ Query failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def process_data_query(query: str, df: pd.DataFrame, model, include_risk: bool = False):
    """Process data analysis queries using Gemini with risk assessment option"""
    
    with st.spinner("🔄 Generating analysis code..."):
        try:
            # Enhanced prompt with risk assessment
            risk_instruction = ""
            if include_risk:
                risk_instruction = """
                
Also include risk assessment calculations:
- Calculate exceedance ratio (Result/MRL) 
- Add 'Risk_Level' column: High (>2x MRL), Medium (1-2x MRL), Low (<1x MRL)
- Include health risk indicators if possible
- Show risk summary statistics
"""
            
            prompt = f"""Generate pandas code to answer this question about the dataset.

Dataset Information:
- Columns: {list(df.columns)}
- Shape: {df.shape}
- First 3 rows:
{df.head(3).to_string()}

Question: {query}
{risk_instruction}

Requirements:
1. Use pandas (imported as pd) for all operations
2. The DataFrame variable is 'df'
3. Store final result in variable 'result'
4. Handle missing values appropriately
5. Return clean, readable output
6. Include proper error handling
7. Add comments for clarity

Generate ONLY the Python code wrapped in ```python``` markers:"""

            response = model.generate_content(prompt)
            
            code_match = re.search(r'```python\n(.*?)```', response.text, re.DOTALL)
            
            if code_match:
                generated_code = code_match.group(1).strip()
            else:
                code_match = re.search(r'```\n(.*?)```', response.text, re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1).strip()
                else:
                    generated_code = response.text.strip()
            
            local_vars = {
                'df': df.copy(),
                'pd': pd,
                'np': np,
                'result': None
            }
            
            exec(generated_code, {}, local_vars)
            
            result = local_vars.get('result')
            
            st.success("✅ Analysis completed!")
            st.markdown("### 📊 Results:")
            
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
                
                # Show risk assessment summary if applicable
                if include_risk and isinstance(result, pd.DataFrame):
                    if 'Risk_Level' in result.columns:
                        with st.expander("⚠️ Risk Assessment Summary"):
                            risk_counts = result['Risk_Level'].value_counts()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🔴 High Risk", risk_counts.get('High', 0))
                            with col2:
                                st.metric("🟡 Medium Risk", risk_counts.get('Medium', 0))
                            with col3:
                                st.metric("🟢 Low Risk", risk_counts.get('Low', 0))
                            
                            if 'Exceedance_Ratio' in result.columns:
                                max_exc = result['Exceedance_Ratio'].max()
                                avg_exc = result['Exceedance_Ratio'].mean()
                                st.write(f"**Max Exceedance:** {max_exc:.2f}x MRL")
                                st.write(f"**Average Exceedance:** {avg_exc:.2f}x MRL")
                
                csv = result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="analysis_results.csv",
                    mime="text/csv"
                )
                
            elif isinstance(result, (pd.Series, list, dict)):
                st.dataframe(pd.DataFrame(result), use_container_width=True)
            else:
                st.write(result)
            
            with st.expander("👨‍💻 View Generated Code"):
                st.code(generated_code, language='python')
                st.caption("This code was AI-generated and executed on your dataset")
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({
                'question': query,
                'result': result,
                'code': generated_code,
                'query_type': 'Data Analysis',
                'timestamp': pd.Timestamp.now(),
                'risk_assessment': include_risk
            })
            
        except Exception as e:
            st.error(f"❌ Error in analysis: {str(e)}")
            
            with st.expander("🔍 Debug Details"):
                st.code(f"Error: {str(e)}", language='python')
                if 'generated_code' in locals():
                    st.markdown("**Generated Code:**")
                    st.code(generated_code, language='python')





def render_chat_sidebar(api_client):
    """Render chat-specific sidebar controls"""
    
    with st.sidebar:
        st.markdown("---")
        
        # Project Configuration
        st.markdown("## 🏢 Project Configuration")
        # if "project_id" not in st.session_state:
        #     st.session_state.project_id = "default_project" 
        project_id = st.text_input(
            "Project ID",
            value=st.session_state.project_id,
            help="Enter your project identifier",
            key="project_id_input"
        )
        
        if project_id != st.session_state.project_id:
            st.session_state.project_id = project_id
        
        st.info(f"📁 Active Project: `{st.session_state.project_id}`")
        
        st.markdown("---")
        
        # ==================== ISO DOCUMENT UPLOAD ====================
        st.markdown("## 📄 ISO Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload ISO Document",
            type=['pdf', 'docx', 'xlsx'],
            help="Upload Quality Manual, Procedures, Work Instructions, or Forms"
        )
        
        doc_type = st.selectbox(
            "Document Type",
            [
                "Quality Manual",
                "Procedure", 
                "Work Instruction",
                "Form/Record",
                "Unknown"
            ]
        )
        
        if uploaded_file is not None:
            # Show what will be uploaded
            st.caption(f"Will upload to project: {st.session_state.project_id}")
            
            if st.button("📤 Upload ISO Document", use_container_width=True, type="primary"):
                # Pass the project ID from session state
                upload_iso_document_ui(
                    api_client, 
                    uploaded_file, 
                    doc_type, 
                    st.session_state.project_id  # ✅ Use actual project ID
                )


        st.markdown("---")
        
        # Data Context
        st.markdown("## 📅 Data Context")
        
        st.info(f"**Active Year:** {st.session_state.selected_year}")
        
        selected_year = st.selectbox(
            "Default Year",
            AVAILABLE_YEARS,
            index=AVAILABLE_YEARS.index(st.session_state.selected_year)
        )
        
        if selected_year != st.session_state.selected_year:
            st.session_state.selected_year = selected_year
        
        st.markdown("---")
        
        # Data Indexing
        st.markdown("## 🗄️ Data Indexing")
        
        year_to_index = st.selectbox("Select Year to Index", AVAILABLE_YEARS)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Index", use_container_width=True):
                index_year_data(api_client, year_to_index, do_reset=False)
        
        with col2:
            if st.button("🗑️ Reset", use_container_width=True):
                index_year_data(api_client, year_to_index, do_reset=True)
        
        if st.session_state.indexed_years:
            st.markdown("**✅ Indexed Years:**")
            for year in sorted(st.session_state.indexed_years):
                st.markdown(f"- {year}")

                # ==================== SETTINGS ====================
        st.markdown("## ⚙️ Settings")
        
        risk_mode = st.radio(
            "Risk Assessment Mode",
            options=["🤖 Auto-detect", "✅ Always On", "❌ Always Off"],
            index=0,
            help="Auto-detect: Enables when you ask about safety/risk\nAlways On: Include for every query\nAlways Off: Never include"
        )
        
        st.session_state.risk_mode = risk_mode


def upload_iso_document_ui(api_client, uploaded_file, doc_type: str, project_id: str):
    """Handle ISO document upload from Streamlit UI"""
    
    with st.spinner(f"📤 Uploading {uploaded_file.name}..."):
        try:
            # Prepare the file for upload
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            data = {
                'document_type': doc_type
            }
            
            # Use the provided project_id
            url = f"{api_client.base_url}/api/v1/iso/upload/{project_id}"

            st.info(f"🔗 Uploading to: {url}")  # Debug info
            
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Display success
            st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
            
            # Show details in expander
            with st.expander("📋 Upload Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Project ID:** {project_id}")
                    st.write(f"**Document ID:** {result.get('document_id', 'N/A')}")
                    st.write(f"**Document Type:** {result.get('document_type', 'N/A')}")
                    st.write(f"**Chunks Indexed:** {result.get('chunks_indexed', 0)}")
                
                with col2:
                    iso_clauses = result.get('iso_clauses', [])
                    if iso_clauses:
                        st.write("**ISO Clauses Found:**")
                        for clause in iso_clauses[:5]:
                            st.write(f"- {clause}")
                
                ref_docs = result.get('referenced_documents', [])
                if ref_docs:
                    st.write(f"**Referenced Documents:** {', '.join(ref_docs)}")
                
                sections = result.get('main_sections', [])
                if sections:
                    st.write("**Main Sections:**")
                    for section in sections[:3]:
                        st.write(f"- {section}")
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"✅ ISO document '{uploaded_file.name}' has been uploaded and indexed for project {project_id}. You can now ask questions about it!"
            })
            
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Upload failed: {e}")
            if hasattr(e.response, 'text'):
                with st.expander("Error Details"):
                    st.text(e.response.text)
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")


def index_year_data(api_client, year: str, do_reset: bool = False):
    """Index data for a specific year"""
    action = "Resetting and re-indexing" if do_reset else "Indexing"
    
    with st.spinner(f"{action} {year} data..."):
        result = api_client.local_index(year, do_reset=do_reset)
        
        if result and result.get('signal') == 'LOCAL_INDEX_SUCCESS':
            count = result.get('indexed_documents', 0)
            if count > 0:
                st.session_state.indexed_years.add(year)
                st.success(f"✅ {year} indexed: {count} documents")
            else:
                st.error(f"⚠️ No documents found for {year}")
                st.json(result)
        else:
            st.error(f"❌ Failed to index {year}")
            if result:
                st.json(result)


def display_chat_messages():
    """Display all chat messages"""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        


def handle_chat_input(api_client):
    """Handle user chat input"""
    
    if prompt := st.chat_input("Ask about pesticide data..."):
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            process_query(api_client, prompt)



def process_query(api_client, query: str):
    """Process user query - now checks for ISO queries too"""
    
    with st.spinner("🔍 Analyzing query..."):
        
        # Enhanced ISO keyword detection
        iso_keywords = [
            # ISO standards
            'iso', '17025', 'clause', 'requirement', 'accreditation',
            # Lab equipment and procedures
            'gc', 'gc-ms', 'gcms', 'hplc', 'column', 'instrument', 'equipment',
            'calibration', 'maintenance', 'validation', 'verification',
            # Documents
            'procedure', 'manual', 'quality manual', 'work instruction', 
            'form', 'record', 'document', 'policy',
            # Lab terms
            'method', 'protocol', 'sop', 'standard operating',
            'laboratory', 'lab', 'sample handling', 'traceability',
            # Arabic terms
            'معايرة', 'إجراء', 'دليل الجودة', 'متطلب', 'اعتماد',
            'مختبر', 'جهاز', 'معدات'
        ]
        
        # Pesticide-specific keywords (more specific)
        pesticide_keywords = [
            'pesticide', 'bifenthrin', 'pyridaben', 'mبيد',
            'tomato', 'pepper', 'cucumber', 'طماطم', 'فلفل', 'خيار',
            'exceeding', 'limit', 'violation', 'غير مطابق', 'مطابق',
            'sample code', 'reading', 'ppm', 'ppb'
        ]
        
        query_lower = query.lower()
        
        # Count keyword matches
        iso_score = sum(1 for kw in iso_keywords if kw in query_lower)
        pesticide_score = sum(1 for kw in pesticide_keywords if kw in query_lower)
        
        # Check for year in query (strong indicator of pesticide query)
        has_year = any(year in query for year in AVAILABLE_YEARS)
        
        # Decision logic
        if iso_score > pesticide_score:
            is_iso_query = True
        elif pesticide_score > iso_score:
            is_iso_query = False
        elif has_year:
            is_iso_query = False  # Year mentioned = pesticide query
        else:
            is_iso_query = True  # Default to ISO if ambiguous
        
        if is_iso_query:
            st.info("🔍 Detected ISO 17025 query - searching ISO documents...")
            process_iso_query(api_client, query)
        else:
            st.info("🔍 Detected pesticide query - searching pesticide data...")
            # Original pesticide query logic
            query_year = extract_year_from_text(query)
            
            if not query_year:
                query_year = st.session_state.selected_year
                st.info(f"⚙️ Using default year: {query_year}")
            else:
                st.success(f"✅ Using year from query: {query_year}")
            
            if query_year not in st.session_state.indexed_years:
                response = f"⚠️ Data for {query_year} is not indexed yet. Please index it first."
                st.warning(response)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response
                })
                return
            
            # Query the API
            try:
                result = api_client.local_answer(
                    query=query,
                    year=query_year,
                    project_id=1,
                    n_results=5
                )
                
                # Check if data is indexed
                if result.get('error') and 'not indexed' in str(result.get('error')).lower():
                    st.error(f"⚠️ Data for {query_year} is not indexed in RAG system")
                    st.info("""
                    **Solutions:**
                    1. Enable Hybrid System (no indexing needed!) - Set GEMINI_API_KEY
                    2. Or index the data first using the indexing button in sidebar
                    3. Or try a different year that's already indexed
                    """)
                    return
                
                # Display results
                display_query_results(result, query_year, False)
            
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Try enabling Hybrid System for direct data access (no indexing needed)")

            risk_mode = st.session_state.get('risk_mode', '🤖 Auto-detect')
            
            if risk_mode == "✅ Always On":
                include_risk = True
            elif risk_mode == "❌ Always Off":
                include_risk = False
            else:  # Auto-detect
                risk_keywords = ['risk', 'safety', 'safe', 'health', 'assess', 'eat', 'consume']
                include_risk = any(kw in query.lower() for kw in risk_keywords)
            
            if include_risk:
                st.success("🎯 Risk assessment enabled")
            
            result = api_client.local_answer(query=query, year=query_year, include_risk=include_risk)

def process_iso_query(api_client, query: str):
    """Process ISO document query"""
    
    project_id = st.session_state.project_id  # ✅ Use actual project ID, not year
    
    try:
        url = f"{api_client.base_url}/api/v1/iso/query/{project_id}"

        
        response = requests.post(
            url,
            json={"text": query},
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Display answer
        st.markdown(result.get('answer', 'No answer found'))
        
        # Display sources
        sources = result.get('sources', [])
        if sources:
            st.markdown("---")
            st.markdown("### 📚 Sources")
            
            for idx, source in enumerate(sources, 1):
                with st.expander(f"📄 Source {idx}: {source.get('file', 'Unknown')}"):
                    st.write(f"**Type:** {source.get('type', 'Unknown')}")
                    st.write(f"**Document ID:** {source.get('doc_id', 'Unknown')}")
                    
                    clauses = source.get('clauses', [])
                    if clauses:
                        st.write(f"**Related ISO Clauses:** {', '.join(clauses)}")
        
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result.get('answer', 'No answer found')
        })
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"📄 No ISO documents found for project {project_id}. Please upload ISO documents first!")
        else:
            st.error(f"❌ Query failed: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def display_query_results(result: Dict, year: str, include_risk: bool):
    """Display query results in a structured format"""
    
    # Display main answer
    answer = result.get('answer', 'No results found')
    st.markdown(answer)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Retrieved Documents", result.get('retrieved_docs_count', 0))
    with col2:
        st.metric("📚 Collection Total", result.get('collection_total', 0))
    
    # Display data table
    data_frame = result.get('data_frame', [])
    if data_frame:
        st.markdown("---")
        st.markdown("### 📊 Interactive Data Table")
        df = pd.DataFrame(data_frame)
        st.dataframe(df, use_container_width=True, height=300)
        
        st.markdown("### 📋 Individual Results")
        display_individual_results(data_frame)
    
    # Display risk assessment
    risk_assessment = result.get('risk_assessment')
    if risk_assessment:
        display_risk_assessment_results(risk_assessment)
    
    st.markdown("---")
    display_download_options(result, year, data_frame)
    
    # Store in session
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": f"✅ Analysis completed for {year}",
        "data": result
    })

def display_individual_results(data_frame: list):
    """Display individual result cards"""
    for idx, record in enumerate(data_frame, 1):
        with st.expander(f"📝 Result {idx}: {record.get('Sample Code', 'N/A')} - {record.get('Pesticide Name', 'N/A')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Sample Code:** {record.get('Sample Code', 'N/A')}")
                st.write(f"**Sample Name:** {record.get('Sample Name', 'N/A')}")
                st.write(f"**Pesticide:** {record.get('Pesticide Name', 'N/A')}")
            
            with col2:
                st.write(f"**Reading:** {record.get('Reading Value', 'N/A')}")
                st.write(f"**Limit:** {record.get('Limit', 'N/A')}")
                
                result_status = record.get('Result', 'N/A')
                if 'مطابق' in str(result_status) or 'Safe' in str(result_status):
                    st.success(f"**Result:** ✅ {result_status}")
                else:
                    st.error(f"**Result:** ⚠️ {result_status}")

def display_risk_assessment_results(risk_assessment: Dict):
    """Display detailed risk assessment"""
    
    st.markdown("---")
    st.markdown("## 🎯 Risk Assessment Summary")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", risk_assessment.get('total_samples_assessed', 0))
    with col2:
        st.metric("Population Type", risk_assessment.get('population_type', 'N/A'))
    with col3:
        dist = risk_assessment.get('overall_risk_distribution', {})
        acceptable = dist.get('acceptable', 0)
        st.metric("✅ Safe Samples", acceptable)
    
    # Individual sample assessments
    assessment_results = risk_assessment.get('assessment_results', [])
    if assessment_results:
        for idx, assessment in enumerate(assessment_results, 1):
            display_sample_assessment(assessment, idx)

def display_sample_assessment(assessment: Dict, idx: int):
    """Display individual sample assessment"""
    
    sample_code = assessment.get('sample_code', 'N/A')
    pesticide_name = assessment.get('pesticide_name', 'N/A')
    risk_level = assessment.get('risk_level', 'unknown')
    
    # Determine status
    status_icon = "✅" if risk_level == 'acceptable' else "⚠️"
    status_text = "Safe" if risk_level == 'acceptable' else "Warning"
    
    with st.expander(f"{status_icon} Sample {idx}: {sample_code} - {pesticide_name} - {status_text}"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Sample Information")
            st.write(f"**Sample Code:** {assessment.get('sample_code', 'N/A')}")
            st.write(f"**Pesticide:** {assessment.get('pesticide_name', 'N/A')}")
            st.write(f"**Food Item:** {assessment.get('food_item', 'unknown')}")
            st.write(f"**Concentration (ppb):** {assessment.get('concentration_ppb', 0):.2f}")
            st.write(f"**Concentration (mg/kg):** {assessment.get('concentration_mg_kg', 0):.3f}")
        
        with col2:
            st.markdown("#### 🔬 Risk Metrics")
            st.write(f"**EDI Value:** {assessment.get('edi_value', 0):.6f} mg/kg/day")
            st.write(f"**ADI Value:** {assessment.get('adi_value', 0):.2f} mg/kg/day")
            st.write(f"**ADI Source:** {assessment.get('adi_source', 'N/A')}")
            
            risk_pct = assessment.get('risk_percentage', 0)
            if risk_pct < 50:
                st.success(f"**Risk Percentage:** {risk_pct:.2f}% of ADI")
            elif risk_pct < 100:
                st.warning(f"**Risk Percentage:** {risk_pct:.2f}% of ADI")
            else:
                st.error(f"**Risk Percentage:** {risk_pct:.2f}% of ADI")
        
        # Status and recommendation
        st.markdown("#### 📋 Assessment")
        risk_status = assessment.get('risk_status', 'N/A')
        recommendation = assessment.get('recommendation', 'N/A')
        
        if risk_level == 'acceptable':
            st.success(f"**Status:** {risk_status}")
        else:
            st.error(f"**Status:** {risk_status}")
        
        st.info(f"**Recommendation:** {recommendation}")

def display_download_options(result: Dict, year: str, data_frame: list):
    """Display download buttons for results"""
    
    col1, col2 = st.columns(2)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"query_{year}_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        if data_frame:
            csv_data = pd.DataFrame(data_frame).to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"query_{year}_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
def show_anova_analysis_page(api_client):
    """ANOVA Analysis page with dynamic switching and Excel support"""
    
    st.title("📈 ANOVA Analysis - Statistical Testing")
    
    st.info("""
    **What is ANOVA?**
    Analysis of Variance (ANOVA) tests if there are statistically significant differences 
    between groups (seasons or vegetables). 
    
    - **F-statistic**: Measures variance between groups vs within groups
    - **P-value < 0.05**: Groups ARE significantly different
    - **P-value ≥ 0.05**: Groups are NOT significantly different
    """)
    
    # Load your data
    @st.cache_data
    def load_data_from_path():
        """Load pesticide data from predefined paths"""
        try:
            # Try multiple possible paths
            possible_paths = [
                'data/cleaned_data.csv',
                '../data/cleaned_data.csv',
                '../../data/cleaned_data.csv',
                '/app/data/cleaned_data.csv',
                'data/processed_data_output.xlsx',
                '../data/processed_data_output.xlsx',
                '../../data/processed_data_output.xlsx'
            ]
            
            for path in possible_paths:
                try:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                    elif path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(path)
                    else:
                        continue
                    
                    st.success(f"✅ Data loaded from: {path}")
                    return df
                except FileNotFoundError:
                    continue
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            return None
    
    # Try to load from predefined paths first
    df = load_data_from_path()
    
    if df is None:
        st.markdown("### 📤 Upload Your Data")
        
        # File uploader with multiple file types
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Drag and drop file here (Limit 200MB per file)"
        )
        
        if uploaded_file:
            try:
                # Determine file type and read accordingly
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                with st.spinner(f"📂 Loading {file_extension.upper()} file..."):
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension in ['xlsx', 'xls']:
                        # For Excel files, allow sheet selection
                        excel_file = pd.ExcelFile(uploaded_file)
                        
                        if len(excel_file.sheet_names) > 1:
                            st.info(f"📋 Found {len(excel_file.sheet_names)} sheets in the Excel file")
                            selected_sheet = st.selectbox(
                                "Select sheet to analyze:",
                                options=excel_file.sheet_names,
                                index=0
                            )
                        else:
                            selected_sheet = excel_file.sheet_names[0]
                        
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    else:
                        st.error(f"❌ Unsupported file type: {file_extension}")
                        return
                
                st.success(f"✅ Data uploaded successfully from {uploaded_file.name}!")
                
                # Show data preview
                with st.expander("👀 Preview Data"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Total Columns", len(df.columns))
                    with col3:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.write("**Column Names:**")
                    st.write(", ".join(df.columns.tolist()))
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)
                return
        else:
            # No file uploaded
            st.info("Expected columns: vegetable_english, season, pesticide, reading, limits")
            return
    
    # Map seasons
    df = map_season_names(df)

    # ===== DATA VALIDATION =====

    # Show data preview
    with st.expander("🔍 Data Preview & Validation"):
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.write("**Column names:**")
        st.write(df.columns.tolist())
        
        st.write("**Data types:**")
        st.write(df.dtypes)

    # Ensure required columns exist
    required_cols = ['vegetable_english', 'season', 'pesticide', 'reading', 'limits']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        
        with st.expander("🔍 Column Mapping Helper"):
            st.write("**Required columns:**", required_cols)
            st.write("**Available columns:**", df.columns.tolist())
            
            st.markdown("**Suggestions:**")
            st.markdown("- `vegetable_english` → vegetable name column")
            st.markdown("- `season` → season/quarter/period column")
            st.markdown("- `pesticide` → pesticide name column")
            st.markdown("- `reading` → measured concentration value")
            
            # Allow column renaming
            st.markdown("#### 🔄 Quick Column Rename")
            
            col_mapping = {}
            for required_col in required_cols:
                if required_col not in df.columns.tolist():
                    selected_col = st.selectbox(
                        f"Map '{required_col}' to:",
                        options=['None'] + df.columns.tolist(),
                        key=f"map_{required_col}"
                    )
                    if selected_col != 'None':
                        col_mapping[selected_col] = required_col
            
            if st.button("Apply Column Mapping"):
                df = df.rename(columns=col_mapping)
                st.success("✅ Columns renamed successfully!")
                st.rerun()
        
        return

    # ===== VIOLATION CALCULATION =====

    st.info("🔧 Analyzing existing is_compliant column...")

    # ✅ Check what's actually in is_compliant
    if 'is_compliant' in df.columns:
        with st.expander("📊 Current is_compliant Analysis"):
            st.write("**Data type:**", df['is_compliant'].dtype)
            st.write("**Unique values:**", df['is_compliant'].unique())
            st.write("**Value counts:**")
            st.write(df['is_compliant'].value_counts())
            
            # Show sample rows
            sample = df[['vegetable_english', 'pesticide', 'reading', 'limits', 'is_compliant']].head(10)
            st.dataframe(sample, use_container_width=True)

    # Clean the data
    df = df.dropna(subset=['reading', 'limits'])

    # Ensure numeric types for reading and limits
    df['reading'] = pd.to_numeric(df['reading'], errors='coerce')
    df['limits'] = pd.to_numeric(df['limits'], errors='coerce')

    # Remove invalid values
    df = df.dropna(subset=['reading', 'limits'])
    df = df[(df['reading'] >= 0) & (df['limits'] > 0)]

    # ✅ RECALCULATE violations regardless of existing column
    st.info("🔧 Recalculating violations from scratch...")

    # Drop existing violation columns
    for col in ['violation', 'violation_numeric', 'is_violation']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ✅ CORRECT LOGIC: A violation occurs when reading EXCEEDS limits
    # is_compliant should be TRUE (1) when reading <= limits
    # is_compliant should be FALSE (0) when reading > limits

    df['violation_numeric'] = (df['reading'] > df['limits']).astype(int)  # 1 = violation
    df['is_compliant_NEW'] = (df['reading'] <= df['limits']).astype(int)  # 1 = compliant
    df['violation'] = (df['reading'] > df['limits'])  # Boolean

    # ✅ Compare with existing column if it exists
    if 'is_compliant' in df.columns:
        with st.expander("🔍 Comparison: Old vs New Calculation", expanded=True):
            comparison = df[['vegetable_english', 'pesticide', 'reading', 'limits', 
                            'is_compliant', 'is_compliant_NEW', 'violation_numeric']].head(20).copy()
            
            # Ensure old column is numeric for comparison
            comparison['is_compliant_old'] = pd.to_numeric(comparison['is_compliant'], errors='coerce')
            
            comparison['Match'] = comparison['is_compliant_old'] == comparison['is_compliant_NEW']
            
            st.dataframe(comparison, use_container_width=True)
            
            matches = comparison['Match'].sum()
            total = len(comparison)
            st.write(f"**Matching rows: {matches}/{total}**")
            
            if matches < total:
                st.warning(f"⚠️ Found {total - matches} mismatches! Using new calculation.")

    # Use the new calculation
    df['is_compliant'] = df['is_compliant_NEW']
    df = df.drop(columns=['is_compliant_NEW'])

    # ✅ Verify results
    total_violations = int(df['violation_numeric'].sum())
    total_samples = len(df)
    violation_rate = (total_violations / total_samples * 100) if total_samples > 0 else 0

    with st.expander("✅ Final Calculation Results", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", total_samples)
        with col2:
            st.metric("Violations", total_violations)
        with col3:
            st.metric("Violation Rate", f"{violation_rate:.1f}%")
        
        # Show logic verification
        st.write("**Logic Verification (first 10 rows):**")
        verify = df[['vegetable_english', 'pesticide', 'reading', 'limits', 'violation_numeric']].head(10).copy()
        verify['Check'] = verify.apply(
            lambda x: '✅' if (x['violation_numeric'] == 1 and x['reading'] > x['limits']) or 
                            (x['violation_numeric'] == 0 and x['reading'] <= x['limits']) else '❌',
            axis=1
        )
        st.dataframe(verify, use_container_width=True)

    # Final validation
    if total_violations <= 0:
        st.error("❌ ERROR: No violations found or negative count!")
        st.info("Check if all your samples are truly compliant, or if there's a data issue.")
    else:
        st.success(f"✅ Successfully calculated {total_violations} violations ({violation_rate:.1f}% rate)")

    # Show final metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", total_samples)
    with col2:
        st.metric("Total Violations", total_violations)
    with col3:
        st.metric("Violation Rate", f"{violation_rate:.1f}%")
    with col4:
        compliant = total_samples - total_violations
        st.metric("Compliant Samples", compliant)

    # ===== DYNAMIC SWITCHING CONTROLS =====
    st.sidebar.header("🔧 Analysis Controls")

    # Main switch: Vegetables vs Seasons
    analysis_mode = st.sidebar.radio(
        "Group Analysis By:",
        options=['Season', 'Vegetable'],
        index=0,
        help="Choose whether to compare across seasons or vegetables"
    )

    # Map selection to column name
    group_column = 'season' if analysis_mode == 'Season' else 'vegetable_english'

    # Optional filters
    st.sidebar.subheader("🔍 Filters")

    if analysis_mode == 'Season':
        # When analyzing by season, allow vegetable selection
        available_vegetables = sorted(df['vegetable_english'].dropna().unique())
        selected_vegetables = st.sidebar.multiselect(
            "Filter by Vegetables:",
            options=available_vegetables,
            default=[]
        )
        
        if selected_vegetables:
            df_filtered = df[df['vegetable_english'].isin(selected_vegetables)].copy()
        else:
            df_filtered = df.copy()
            
    else:  # Vegetable mode
        # When analyzing by vegetable, allow season selection
        available_seasons = sorted(df['season'].dropna().unique())
        selected_seasons = st.sidebar.multiselect(
            "Filter by Seasons:",
            options=available_seasons,
            default=[]
        )
        
        if selected_seasons:
            df_filtered = df[df['season'].isin(selected_seasons)].copy()
        else:
            df_filtered = df.copy()

    # Additional filter: Pesticide type (optional)
    available_pesticides = sorted(df['pesticide'].dropna().unique())
    selected_pesticides = st.sidebar.multiselect(
        "Filter by Pesticide:",
        options=available_pesticides,
        default=[],
        help="Leave empty to analyze all pesticides"
    )

    if selected_pesticides:
        df_filtered = df_filtered[df_filtered['pesticide'].isin(selected_pesticides)].copy()

    # Target variable selection
    available_targets = ['reading']
    if 'exceedance_ratio' in df.columns:
        available_targets.append('exceedance_ratio')
    if 'limits' in df.columns:
        available_targets.append('limits')
    if 'violation_numeric' in df.columns:
        available_targets.append('violation_numeric')

    target_col = st.sidebar.selectbox(
        "Analyze:",
        options=available_targets,
        index=0
    )

    # Remove rows with NaN in target column
    df_filtered = df_filtered.dropna(subset=[target_col, group_column])

    # ===== PERFORM ANALYSIS =====
    st.header(f"Analysis: {target_col.replace('_', ' ').title()} by {analysis_mode}")

    # Show current filter status
    with st.expander("📊 Current Filters Applied"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Samples:** {len(df):,}")
            st.write(f"**Filtered Samples:** {len(df_filtered):,}")
        with col2:
            st.write(f"**Analysis Mode:** {analysis_mode}")
            st.write(f"**Target Variable:** {target_col}")
            if selected_pesticides:
                st.write(f"**Pesticide Filter:** {', '.join(selected_pesticides)}")

    if len(df_filtered) < 10:
        st.warning("⚠️ Not enough data for analysis. Please adjust filters.")
        st.info(f"""
        **Current situation:**
        - Filtered samples: {len(df_filtered)}
        - Minimum required: 10 samples
        
        **Suggestions:**
        - Remove some filters (especially pesticide filter)
        - Select more vegetables or seasons
        - Try analyzing all data without filters
        """)
        return

    # Check if we have multiple groups
    n_groups = df_filtered[group_column].nunique()
    if n_groups < 2:
        st.warning(f"⚠️ Need at least 2 {analysis_mode.lower()}s for comparison.")
        unique_groups = df_filtered[group_column].unique().tolist()
        st.info(f"Current {analysis_mode.lower()}(s) in filtered data: {', '.join(map(str, unique_groups))}")
        st.info("💡 Tip: Remove some filters to get more groups")
        return

    # Perform ANOVA
    with st.spinner("🔄 Calculating ANOVA..."):
        results = perform_anova_analysis(df_filtered, group_column, target_col)

    # ===== DISPLAY RESULTS =====

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("F-Statistic", f"{results['f_statistic']:.4f}")

    with col2:
        st.metric("P-Value", f"{results['p_value']:.6f}")

    with col3:
        significance = "✅ Yes" if results['significant'] else "❌ No"
        st.metric("Significant?", significance)

    with col4:
        st.metric("Groups", n_groups)

    # Interpretation
    st.subheader("📋 Interpretation")

    if results['significant']:
        st.success(f"""
        **Statistically Significant Difference Found!**
        
        - The {analysis_mode.lower()}s have significantly different {target_col} values
        - P-value ({results['p_value']:.6f}) < 0.05
        - F-statistic: {results['f_statistic']:.4f}
        
        **Conclusion:** {analysis_mode} DOES affect {target_col}!
        """)
    else:
        st.info(f"""
        **No Significant Difference Found**
        
        - The {analysis_mode.lower()}s have similar {target_col} values
        - P-value ({results['p_value']:.6f}) ≥ 0.05
        - F-statistic: {results['f_statistic']:.4f}
        
        **Conclusion:** {analysis_mode} does NOT significantly affect {target_col}
        """)

    # Summary statistics table
    st.subheader("📊 Group Statistics")
    st.dataframe(
        results['summary'].style.highlight_max(axis=0, color='lightgreen')
                                .highlight_min(axis=0, color='lightcoral'),
        use_container_width=True
    )

    # Visualizations
    st.subheader("📈 Visualizations")

    tab1, tab2 = st.tabs(["Box Plot", "Means Comparison"])

    with tab1:
        fig_box = create_anova_visualization(df_filtered, group_column, target_col)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        fig_means = create_means_comparison_plot(results['summary'], group_column)
        st.plotly_chart(fig_means, use_container_width=True)

    # Post-hoc analysis (if significant)
    if results['significant'] and n_groups > 2:
        st.subheader("🔬 Post-Hoc Analysis")
        
        with st.expander("View Pairwise Comparisons (Tukey HSD)"):
            from scipy.stats import tukey_hsd
            
            groups_list = [group.values for name, group in df_filtered.groupby(group_column)[target_col]]
            
            try:
                res = tukey_hsd(*groups_list)
                
                # Create comparison matrix
                group_names = sorted(df_filtered[group_column].unique())
                comparison_df = pd.DataFrame(
                    res.pvalue,
                    index=group_names,
                    columns=group_names
                )
                
                st.write("**P-values for pairwise comparisons:**")
                st.dataframe(
                    comparison_df.style.background_gradient(cmap='RdYlGn_r', vmin=0, vmax=0.1),
                    use_container_width=True
                )
                
                st.caption("Green = significant difference (p < 0.05), Red = no significant difference")
                
            except Exception as e:
                st.warning(f"Could not perform Tukey HSD: {e}")

    # Export results
    st.subheader("💾 Export Results")

    # Prepare export data
    export_data = {
        'Analysis Type': f'ANOVA - {analysis_mode}',
        'F-Statistic': results['f_statistic'],
        'P-Value': results['p_value'],
        'Significant': results['significant'],
        'Sample Size': len(df_filtered),
        'Number of Groups': n_groups
    }

    export_df = pd.DataFrame([export_data])

    col1, col2 = st.columns(2)

    with col1:
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=csv,
            file_name=f"anova_{analysis_mode.lower()}_{target_col}.csv",
            mime="text/csv"
        )

    with col2:
        csv_detailed = results['summary'].to_csv()
        st.download_button(
            label="📥 Download Detailed Stats (CSV)",
            data=csv_detailed,
            file_name=f"anova_detailed_{analysis_mode.lower()}_{target_col}.csv",
            mime="text/csv"
        )
def show_advanced_insights_page(api_client):
    """Advanced insights with risk combinations and forecasting"""
    
    st.title("🎯 Advanced Risk Intelligence")
    
    st.markdown("""
    **Strategic Analytics for Lab Operations**
    - Identify high-risk vegetable-pesticide combinations
    - Forecast seasonal contamination risks
    - Optimize testing schedules and resource allocation
    """)
    
    # Load data
    @st.cache_data
    def load_data_for_insights():
        """Load pesticide data"""
        try:
            possible_paths = [
                'data/cleaned_data.csv',
                'data/processed_data_output.xlsx',
                '../data/processed_data_output.xlsx',
            ]
            
            for path in possible_paths:
                try:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                    elif path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(path)
                    else:
                        continue
                    return df
                except:
                    continue
            return None
        except:
            return None
    
    df = load_data_for_insights()
    
    if df is None:
        st.warning("📤 Please upload your data first")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
                return
        else:
            return
    
    # ===== DEBUG PANEL =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🐛 Debug Tools")
    
    if st.sidebar.checkbox("Show Debug Panel", value=False):
        st.markdown("---")
        st.markdown("## 🐛 DEBUG PANEL")
        
        with st.expander("📊 Raw Data Inspection", expanded=True):
            st.write("**DataFrame Shape:**", df.shape)
            st.write("**Columns:**")
            st.json(df.columns.tolist())
            
            debug_cols = ['vegetable_english', 'pesticide', 'reading', 'limits']
            if 'is_compliant' in df.columns:
                debug_cols.append('is_compliant')
            
            available_debug_cols = [col for col in debug_cols if col in df.columns]
            st.dataframe(df[available_debug_cols].head(20), use_container_width=True)
        
        with st.expander("🔍 Violation Column Analysis", expanded=True):
            st.write("**Checking existing columns:**")
            
            violation_related = ['is_compliant', 'violation', 'violation_numeric', 
                                'compliance', 'compliant', 'result', 'is_violation']
            
            for col in violation_related:
                if col in df.columns:
                    st.write(f"✅ Found column: `{col}`")
                    st.write(f"   - Type: {df[col].dtype}")
                    st.write(f"   - Unique values: {df[col].nunique()}")
                    st.write(f"   - Value counts:")
                    st.write(df[col].value_counts())
                else:
                    st.write(f"❌ Missing column: `{col}`")
        
        with st.expander("🧮 Manual Violation Calculation Test", expanded=True):
            st.write("**Testing: reading > limits = violation**")
            
            test_df = df[['vegetable_english', 'pesticide', 'reading', 'limits']].head(10).copy()
            test_df['my_calculation'] = test_df['reading'] > test_df['limits']
            test_df['my_violation'] = test_df['my_calculation'].astype(int)
            
            if 'is_compliant' in df.columns:
                test_df['file_is_compliant'] = df['is_compliant'].head(10)
            
            st.dataframe(test_df, use_container_width=True)
            
            st.write("**Expected logic:**")
            st.code("""
reading=100, limits=50  → reading > limits = TRUE  → VIOLATION
reading=30,  limits=50  → reading > limits = FALSE → COMPLIANT
            """)
        
        with st.expander("📈 Statistics Summary", expanded=True):
            st.write("**Reading vs Limits:**")
            st.write(f"- Min reading: {df['reading'].min()}")
            st.write(f"- Max reading: {df['reading'].max()}")
            st.write(f"- Mean reading: {df['reading'].mean():.2f}")
            st.write(f"- Min limits: {df['limits'].min()}")
            st.write(f"- Max limits: {df['limits'].max()}")
            st.write(f"- Mean limits: {df['limits'].mean():.2f}")
            
            exceeds = (df['reading'] > df['limits']).sum()
            total = len(df)
            st.write(f"- Readings > Limits: {exceeds} / {total} ({exceeds/total*100:.1f}%)")
        
        with st.expander("💾 Export Debug Data", expanded=False):
            debug_df = df[available_debug_cols].head(100)
            csv = debug_df.to_csv(index=False)
            st.download_button(
                label="📥 Download First 100 Rows (CSV)",
                data=csv,
                file_name="debug_data.csv",
                mime="text/csv"
            )
    
    # ===== DATA CLEANING & VALIDATION =====
    
    # Map seasons
    df = map_season_names(df)
    
    # Show data preview
    with st.expander("🔍 Data Preview & Validation"):
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.write("**Column names:**")
        st.write(df.columns.tolist())
        
        st.write("**Data types:**")
        st.write(df.dtypes)
    
    # Ensure required columns exist
    required_cols = ['vegetable_english', 'season', 'pesticide', 'reading', 'limits']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.info("Required columns: vegetable_english, season, pesticide, reading, limits")
        return
    
    # ✅ CRITICAL FIX: Recalculate violations correctly
    st.info("🔧 Recalculating violations from scratch...")
    
    # Clean the data
    df = df.dropna(subset=['reading', 'limits'])
    
    # Ensure numeric types
    df['reading'] = pd.to_numeric(df['reading'], errors='coerce')
    df['limits'] = pd.to_numeric(df['limits'], errors='coerce')
    
    # Remove invalid values
    df = df.dropna(subset=['reading', 'limits'])
    df = df[(df['reading'] >= 0) & (df['limits'] > 0)]
    
    # ✅ CORRECT CALCULATION (matches debug panel)
    # Drop ALL existing violation columns first
    for col in ['violation', 'violation_numeric', 'is_violation', 'is_compliant']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Calculate fresh: reading > limits = VIOLATION
    df['violation_numeric'] = (df['reading'] > df['limits']).astype(int)
    df['violation'] = (df['reading'] > df['limits'])
    df['is_compliant'] = (df['reading'] <= df['limits'])
    
    # ✅ Verify calculation
    total_violations = int(df['violation_numeric'].sum())
    total_samples = len(df)
    violation_rate = (total_violations / total_samples * 100) if total_samples > 0 else 0
    
    # Show verification
    with st.expander("✅ Violation Calculation Verification", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", total_samples)
        with col2:
            st.metric("Total Violations", total_violations)
        with col3:
            st.metric("Violation Rate", f"{violation_rate:.1f}%")
        
        st.write("**Sample verification (first 10 rows):**")
        verify = df[['vegetable_english', 'pesticide', 'reading', 'limits', 'violation_numeric']].head(10).copy()
        verify['Logic Check'] = verify.apply(
            lambda x: '✅ Correct' if (
                (x['reading'] > x['limits'] and x['violation_numeric'] == 1) or
                (x['reading'] <= x['limits'] and x['violation_numeric'] == 0)
            ) else '❌ Wrong',
            axis=1
        )
        st.dataframe(verify, use_container_width=True)
    
    # Final validation
    if total_violations <= 0:
        st.error("❌ ERROR: No violations found!")
        st.info("This could mean all samples are compliant, or there's a data issue.")
        if total_violations < 0:
            st.error("Negative violations detected! Check your data source.")
            st.stop()
    else:
        st.success(f"✅ Successfully calculated {total_violations} violations ({violation_rate:.1f}% rate)")
    
    # Show final metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", total_samples)
    with col2:
        st.metric("Total Violations", total_violations)
    with col3:
        st.metric("Violation Rate", f"{violation_rate:.1f}%")
    with col4:
        compliant = total_samples - total_violations
        st.metric("Compliant Samples", compliant)
    
    # ============================================================================
    # CREATE TABS FOR ANALYSIS
    # ============================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 High-Risk Combinations",
        "📅 Seasonal Forecasting", 
        "🗺️ Risk Heatmaps",
        "📈 Trend Analysis"
    ])
    
    # ============================================================================
    # TAB 1: HIGH-RISK COMBINATIONS
    # ============================================================================
    
    with tab1:
        st.header("🔥 High-Risk Vegetable-Pesticide Combinations")
        
        st.info("""
        **Business Value:** Focus your testing resources on combinations that cause 80% of violations.
        """)
        
        # Calculate violation rates by combination
        combinations = df.groupby(['vegetable_english', 'pesticide']).agg({
            'violation_numeric': ['sum', 'count', 'mean'],
            'reading': 'mean',
            'limits': 'mean'
        }).reset_index()
        
        combinations.columns = ['Vegetable', 'Pesticide', 'Violations', 'Total_Tests', 'Violation_Rate', 'Avg_Reading', 'Avg_Limit']
        
        # Calculate risk score
        combinations['Risk_Score'] = (
            combinations['Violation_Rate'] * 0.5 + 
            (combinations['Violations'] / combinations['Violations'].max()) * 0.3 +
            np.minimum(combinations['Avg_Reading'] / combinations['Avg_Limit'], 5) * 0.04
        )
        
        # Sort by risk score
        high_risk = combinations.nlargest(20, 'Risk_Score')
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_violations_display = int(df['violation_numeric'].sum())
            st.metric("Total Violations", f"{total_violations_display:,}")
        
        with col2:
            top_10_violations = high_risk.head(10)['Violations'].sum()
            coverage = (top_10_violations / total_violations_display * 100) if total_violations_display > 0 else 0
            st.metric("Top 10 Coverage", f"{coverage:.1f}%")
        
        with col3:
            avg_risk = high_risk.head(10)['Violation_Rate'].mean() * 100
            st.metric("Avg Violation Rate", f"{avg_risk:.1f}%")
        
        # Display table with proper types
        st.subheader("🎯 Top 10 High-Risk Combinations")
        
        display_cols = ['Vegetable', 'Pesticide', 'Violations', 'Total_Tests', 'Violation_Rate', 'Risk_Score']
        display_df = high_risk.head(10)[display_cols].copy()
        
        # Ensure proper data types
        display_df['Vegetable'] = display_df['Vegetable'].astype(str)
        display_df['Pesticide'] = display_df['Pesticide'].astype(str)
        display_df['Violations'] = display_df['Violations'].astype(float)
        display_df['Total_Tests'] = display_df['Total_Tests'].astype(float)
        display_df['Violation_Rate'] = display_df['Violation_Rate'].astype(float)
        display_df['Risk_Score'] = display_df['Risk_Score'].astype(float)
        
        st.dataframe(
            display_df.style.background_gradient(subset=['Risk_Score'], cmap='Reds')
            .format({
                'Violation_Rate': '{:.1%}',
                'Risk_Score': '{:.3f}',
                'Violations': '{:.0f}',
                'Total_Tests': '{:.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Pareto Chart
        st.subheader("📊 Pareto Analysis: Cumulative Violation Coverage")
        
        # Prepare data for Pareto chart
        pareto_data = high_risk.copy()
        pareto_data = pareto_data.sort_values('Violations', ascending=False)
        pareto_data['Cumulative_Violations'] = pareto_data['Violations'].cumsum()
        pareto_data['Cumulative_Percentage'] = (pareto_data['Cumulative_Violations'] / pareto_data['Violations'].sum()) * 100
        pareto_data['Combination'] = pareto_data['Vegetable'] + ' + ' + pareto_data['Pesticide']
        
        # Create Pareto chart
        fig_pareto = go.Figure()
        
        # Bar chart for violations
        fig_pareto.add_trace(go.Bar(
            x=pareto_data['Combination'],
            y=pareto_data['Violations'],
            name='Violations',
            marker_color='#FF6B6B',
            yaxis='y'
        ))
        
        # Line chart for cumulative percentage
        fig_pareto.add_trace(go.Scatter(
            x=pareto_data['Combination'],
            y=pareto_data['Cumulative_Percentage'],
            name='Cumulative %',
            mode='lines+markers',
            marker=dict(size=8, color='#4ECDC4'),
            line=dict(width=3),
            yaxis='y2'
        ))
        
        # Add 80% reference line
        fig_pareto.add_hline(
            y=80, 
            line_dash="dash", 
            line_color="green", 
            annotation_text="80% Goal",
            yref='y2'
        )
        
        fig_pareto.update_layout(
            title='Top 20 High-Risk Combinations (80/20 Rule)',
            xaxis_title='Vegetable-Pesticide Combination',
            yaxis=dict(title='Number of Violations', side='left'),
            yaxis2=dict(title='Cumulative Percentage (%)', side='right', overlaying='y', range=[0, 110]),
            hovermode='x unified',
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_pareto, use_container_width=True)
    
    # ============================================================================
    # TAB 2: SEASONAL FORECASTING
    # ============================================================================
    
    with tab2:
        st.header("📅 Seasonal Risk Forecasting")
        
        # Calculate seasonal violation rates
        seasonal_stats = df.groupby('season').agg({
            'violation_numeric': ['sum', 'count', 'mean'],
            'reading': 'mean',
            'vegetable_english': 'nunique',
            'pesticide': 'nunique'
        }).reset_index()
        
        seasonal_stats.columns = ['Season', 'Violations', 'Total_Tests', 'Violation_Rate', 
                                   'Avg_Reading', 'Unique_Vegetables', 'Unique_Pesticides']
        
        seasonal_stats = seasonal_stats.sort_values('Violation_Rate', ascending=False)
        
        # Display seasonal metrics
        st.subheader("🌍 Seasonal Risk Overview")
        
        cols = st.columns(len(seasonal_stats))
        
        season_colors = {
            'Winter': '🔵',
            'Spring': '🟢',
            'Summer': '🟠',
            'Fall': '🔴'
        }
        
        for idx, (col, row) in enumerate(zip(cols, seasonal_stats.itertuples())):
            with col:
                icon = season_colors.get(row.Season, '⚪')
                st.metric(
                    f"{icon} {row.Season}",
                    f"{row.Violation_Rate:.1%}",
                    delta=f"{int(row.Violations)} violations"
                )
                st.caption(f"{int(row.Total_Tests)} tests")
        
        # Visualization
        st.subheader("📊 Seasonal Comparison")
        
        fig_seasonal = go.Figure()
        
        fig_seasonal.add_trace(go.Bar(
            x=seasonal_stats['Season'],
            y=seasonal_stats['Violation_Rate'] * 100,
            marker_color=['#339AF0' if s == 'Winter' else '#51CF66' if s == 'Spring' 
                         else '#FF922B' if s == 'Summer' else '#FA5252' 
                         for s in seasonal_stats['Season']],
            text=seasonal_stats['Violation_Rate'].apply(lambda x: f"{x:.1%}"),
            textposition='outside',
            name='Violation Rate'
        ))
        
        fig_seasonal.update_layout(
            title='Violation Rate by Season',
            xaxis_title='Season',
            yaxis_title='Violation Rate (%)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Display table
        st.subheader("📋 Detailed Seasonal Statistics")
        
        display_seasonal = seasonal_stats[['Season', 'Violations', 'Total_Tests', 'Violation_Rate']].copy()
        display_seasonal['Season'] = display_seasonal['Season'].astype(str)
        display_seasonal['Violations'] = display_seasonal['Violations'].astype(float)
        display_seasonal['Total_Tests'] = display_seasonal['Total_Tests'].astype(float)
        display_seasonal['Violation_Rate'] = display_seasonal['Violation_Rate'].astype(float)
        
        st.dataframe(
            display_seasonal.style.background_gradient(subset=['Violation_Rate'], cmap='Reds')
            .format({
                'Violation_Rate': '{:.1%}',
                'Violations': '{:.0f}',
                'Total_Tests': '{:.0f}'
            }),
            use_container_width=True
        )


    # # ============================================================================
    # # TAB 3: RISK HEATMAPS - CUSTOM COLOR MAPPING APPROACH
    # # ============================================================================

    with tab3:
        st.header("🗺️ Risk Heatmaps")
        
        # Prepare data for heatmaps
        heatmap_data = df.groupby(['vegetable_english', 'pesticide']).agg({
            'violation_numeric': ['sum', 'count', 'mean']
        }).reset_index()
        heatmap_data.columns = ['Vegetable', 'Pesticide', 'Violations', 'Tests', 'Violation_Rate']
        heatmap_data['Violations'] = pd.to_numeric(heatmap_data['Violations'], errors='coerce')
        heatmap_data['Tests'] = pd.to_numeric(heatmap_data['Tests'], errors='coerce')
        heatmap_data['Violation_Rate'] = pd.to_numeric(heatmap_data['Violation_Rate'], errors='coerce')
        heatmap_data = heatmap_data.dropna()
        heatmap_data['Violations'] = heatmap_data['Violations'].astype(int)
        
        # Filter by minimum tests
        min_tests = st.slider("Minimum tests required:", 1, 10, 3)
        heatmap_filtered = heatmap_data[heatmap_data['Tests'] >= min_tests].copy()
        
        if len(heatmap_filtered) < 3:
            st.warning("⚠️ Not enough data with the current filter. Try reducing minimum tests.")
        else:
            top_combos = heatmap_filtered.nlargest(30, 'Violations').copy()
            
            # ============================================================================
            # VISUALIZATION 1: BUBBLE CHART
            # ============================================================================
            
            st.subheader("🔥 Violation Rate Bubble Chart")
            
            # Manual color mapping for bubble chart
            def get_color_for_rate(rate):
                if rate <= 0.33:
                    return 'green'    # Low risk
                elif rate <= 0.66:
                    return 'yellow'   # Medium risk  
                else:
                    return 'red'      # High risk
            
            # Create color array manually
            colors_manual = [get_color_for_rate(rate) for rate in top_combos['Violation_Rate']]
            
            # Size calculation
            min_violations = top_combos['Violations'].min()
            max_violations = top_combos['Violations'].max()
            bubble_sizes = [10 + (v - min_violations) / (max_violations - min_violations) * 40 
                        for v in top_combos['Violations']]
            
            # Create scatter plot with manual colors
            fig_bubble = go.Figure()
            
            # Group by color and create separate traces
            for color_name in ['green', 'yellow', 'red']:
                mask = [c == color_name for c in colors_manual]
                if any(mask):
                    filtered_data = top_combos[mask]
                    filtered_sizes = [bubble_sizes[i] for i, m in enumerate(mask) if m]
                    
                    fig_bubble.add_trace(go.Scatter(
                        x=filtered_data['Pesticide'],
                        y=filtered_data['Vegetable'],
                        mode='markers',
                        marker=dict(
                            size=filtered_sizes,
                            color=color_name,
                            line=dict(width=1.5, color='black'),
                            opacity=0.85
                        ),
                        name=f'{color_name.title()} Risk',
                        text=[f"{row['Vegetable']} + {row['Pesticide']}<br>"
                            f"Violations: {row['Violations']}<br>"
                            f"Rate: {row['Violation_Rate']:.1%}" 
                            for _, row in filtered_data.iterrows()],
                        hovertemplate='%{text}<extra></extra>'
                    ))
            
            fig_bubble.update_layout(
                title=f'Top {len(top_combos)} Combinations by Violation Count',
                xaxis_title='Pesticide',
                yaxis_title='Vegetable',
                height=600,
                xaxis_tickangle=-45,
                showlegend=True
            )
            
            st.plotly_chart(fig_bubble, use_container_width=True)
            
            # ============================================================================
            # VISUALIZATION 2: CLEAN TREEMAP WITH LIGHT GREEN COLOR SCHEME
            # ============================================================================
            
            st.subheader("📊 Clean Treemap - Actionable Insights")
            
            try:
                # Use only top 15 combinations to avoid clutter
                clean_treemap_data = heatmap_filtered.nlargest(15, 'Violations').copy()
                
                # CRITICAL FIX: Ensure Violation_Rate is numeric
                clean_treemap_data['Violation_Rate'] = pd.to_numeric(
                    clean_treemap_data['Violation_Rate'], 
                    errors='coerce'
                )
                
                # Create clear labels
                clean_treemap_data['Display_Label'] = clean_treemap_data.apply(
                    lambda row: f"{row['Pesticide']}<br>{row['Violations']} violations<br>{row['Violation_Rate']:.0%} rate", 
                    axis=1
                )
                
                # ✅ UPDATED: Create treemap with LIGHT GREEN color scheme
                fig_clean = px.treemap(
                    clean_treemap_data,
                    path=['Vegetable', 'Display_Label'],
                    values='Violations',
                    color='Violation_Rate',
                    # ✅ LIGHT GREEN COLOR SCHEME: Light green to dark green gradient
                    color_continuous_scale=['#E8F5E8', '#2E8B57', '#1A5D1A'],
                    range_color=[0, 1],
                    title='Top 15 Highest-Risk Combinations (Light Green Theme)'
                )
                
                # Clean up the layout
                fig_clean.update_layout(
                    height=550,
                    margin=dict(t=60, l=10, r=10, b=10)
                )
                
                # Enhanced traces
                fig_clean.update_traces(
                    textinfo="label",
                    textfont=dict(size=13, color='white'),
                    textposition="middle center",
                    hovertemplate=(
                        '<b>%{label}</b><br>' +
                        'Violations: %{value}<br>' +
                        'Risk Level: %{color:.1%}<br>' +
                        '<extra></extra>'
                    ),
                    marker=dict(
                        line=dict(width=2.5, color='white'),
                        cornerradius=4
                    )
                )
                
                # ✅ UPDATED: Configure colorbar for green theme
                fig_clean.update_coloraxes(
                    colorbar=dict(
                        title=dict(
                            text="<b>Risk Level</b>",
                            side="right"
                        ),
                        tickmode='array',
                        tickvals=[0, 0.33, 0.66, 1.0],
                        ticktext=[
                            '🌱 Low (0-33%)',
                            '🌿 Medium (33-66%)',
                            '🍃 High (66-85%)',
                            '🌲 Critical (85%+)'
                        ],
                        len=0.75,
                        thickness=22,
                        x=1.02
                    ),
                    cmin=0,
                    cmax=1
                )
                
                st.plotly_chart(fig_clean, use_container_width=True)
                
            except Exception as e:
                st.error(f"Treemap visualization failed: {e}")
                
                # FALLBACK: Manual color assignment with green theme
                try:
                    def get_risk_color(rate):
                        """Assign green-based colors based on risk level"""
                        if rate < 0.33:
                            return '#E8F5E8'  # Very light green
                        elif rate < 0.66:
                            return '#90EE90'  # Light green
                        elif rate < 0.85:
                            return '#32CD32'  # Medium green
                        else:
                            return '#2E8B57'  # Sea green
                    
                    # Assign colors manually
                    clean_treemap_data['Manual_Color'] = clean_treemap_data['Violation_Rate'].apply(get_risk_color)
                    clean_treemap_data['Risk_Text'] = clean_treemap_data['Violation_Rate'].apply(
                        lambda x: '🌱 Low' if x < 0.33 else '🌿 Medium' if x < 0.66 else '🍃 High' if x < 0.85 else '🌲 Critical'
                    )
                    
                    # Create treemap with manual colors
                    fig_fallback = go.Figure()
                    
                    # Group by vegetable
                    for veg in clean_treemap_data['Vegetable'].unique():
                        veg_data = clean_treemap_data[clean_treemap_data['Vegetable'] == veg]
                        
                        # Add each pesticide combination
                        labels = [veg]
                        parents = ['']
                        values = [0]
                        colors = [get_risk_color(veg_data['Violation_Rate'].mean())]
                        
                        for _, row in veg_data.iterrows():
                            labels.append(f"{row['Pesticide']}<br>{row['Violations']} violations")
                            parents.append(veg)
                            values.append(row['Violations'])
                            colors.append(row['Manual_Color'])
                        
                        fig_fallback.add_trace(go.Treemap(
                            labels=labels,
                            parents=parents,
                            values=values,
                            marker=dict(
                                colors=colors,
                                line=dict(width=2.5, color='white'),
                                cornerradius=4
                            ),
                            textfont=dict(size=12, color='white'),
                            hovertemplate='<b>%{label}</b><br>Violations: %{value}<extra></extra>',
                            showlegend=False
                        ))
                    
                    fig_fallback.update_layout(
                        title='Top 15 Risk Combinations - Green Theme',
                        height=550,
                        margin=dict(t=60, l=10, r=10, b=10)
                    )
                    
                    st.plotly_chart(fig_fallback, use_container_width=True)
                    
                    # Show color legend
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("🌱 **Low**: 0-33%")
                    with col2:
                        st.markdown("🌿 **Medium**: 33-66%")
                    with col3:
                        st.markdown("🍃 **High**: 66-85%")
                    with col4:
                        st.markdown("🌲 **Critical**: 85%+")
                        
                except Exception as e2:
                    st.error(f"Fallback also failed: {e2}")
            
            
            # ============================================================================
            # VISUALIZATION 3: PRIORITY MATRIX TABLE
            # ============================================================================
            
            st.subheader("🎯 Priority Matrix - For Immediate Action")
            
            # Create a simple priority matrix
            priority_data = heatmap_filtered.nlargest(10, 'Violations').copy()
            
            # Categorize by risk and impact
            def get_priority(row):
                if row['Violation_Rate'] > 0.66 and row['Violations'] > 5:
                    return "🚨 CRITICAL"
                elif row['Violation_Rate'] > 0.66:
                    return "🔴 High Risk"
                elif row['Violations'] > 5:
                    return "🟡 High Impact"
                else:
                    return "🟢 Monitor"
            
            priority_data['Priority'] = priority_data.apply(get_priority, axis=1)
            
            # Display as a clean table
            st.dataframe(
                priority_data[['Vegetable', 'Pesticide', 'Violations', 'Violation_Rate', 'Priority']]
                .style.format({'Violation_Rate': '{:.1%}'})
                .background_gradient(subset=['Violations'], cmap='Reds'),
                use_container_width=True
            )
            
            # ============================================================================
            # VISUALIZATION 4: RISK MATRIX TABLE
            # ============================================================================
            
            st.subheader("📋 Risk Matrix Table")
            
            # Create a pivot table for heatmap
            pivot_data = top_combos.pivot_table(
                index='Vegetable', 
                columns='Pesticide', 
                values='Violation_Rate',
                aggfunc='first'
            ).fillna(0)
            
            # Create styled dataframe
            def style_risk_cell(val):
                if val <= 0.33:
                    color = 'background-color: #90EE90'  # Light green
                elif val <= 0.66:
                    color = 'background-color: #FFFFCC'  # Light yellow
                else:
                    color = 'background-color: #FFB6C1'  # Light red
                return color
            
            styled_df = pivot_data.style.format('{:.1%}').applymap(style_risk_cell)
            
            st.dataframe(styled_df, use_container_width=True)
            
            st.info("""
            **🎨 Color Guide:**
            - **🟢 Green:** Low risk (0-33% violation rate)
            - **🟡 Yellow:** Medium risk (33-66% violation rate) 
            - **🔴 Red:** High risk (66-100% violation rate)
            """)
            
            # Optional: Data verification expander
            with st.expander("📊 Verify Data Quality"):
                st.write("**Violation Rate Distribution:**")
                verify_df = clean_treemap_data[['Vegetable', 'Pesticide', 'Violations', 'Violation_Rate']].copy()
                verify_df['Violation_Rate_Display'] = verify_df['Violation_Rate'].apply(lambda x: f"{x:.1%}")
                verify_df['Risk_Level'] = clean_treemap_data['Violation_Rate'].apply(
                    lambda x: '🟢 Low' if x < 0.33 else '🟡 Medium' if x < 0.66 else '🟠 High' if x < 0.85 else '🔴 Critical'
                )
                
                # Display the dataframe
                display_cols = ['Vegetable', 'Pesticide', 'Violations', 'Violation_Rate_Display', 'Risk_Level']
                st.dataframe(verify_df[display_cols], use_container_width=True)

    # ============================================================================
    # END OF TAB 3 CODE
    # ============================================================================

    # Notes:
    # 1. This code should ONLY appear in Tab 3 (Risk Heatmaps)
    # 2. Remove all duplicate treemap code from other tabs
    # 3. Make sure heatmap_filtered variable is available in Tab 3 scope
    # 4. The two duplicate visualizations you showed are now removed
    # 5. Only clean, essential visualizations remain

    # ============================================================================
    # TAB 4: TREND ANALYSIS
    # ============================================================================

    with tab4:
        st.header("📈 Contamination Trend Analysis")
        
        st.info("""
        **Business Value:** Track how contamination patterns change over time.
        - Identify if contamination is improving or worsening
        - Monitor effectiveness of interventions
        - Predict future contamination risks
        """)
        
        # Check if we have temporal data
        if 'year' in df.columns or 'month' in df.columns or 'date' in df.columns:
            
            # Time series by pesticide group
            if 'pesticide_group' in df.columns:
                st.subheader("📊 Violation Trends by Pesticide Group")
                
                # Assuming you have year and month
                if 'year' in df.columns and 'month' in df.columns:
                    df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
                    
                    trend_data = df.groupby(['period', 'pesticide_group']).agg({
                        'violation': ['sum', 'count', 'mean']
                    }).reset_index()
                    
                    trend_data.columns = ['Period', 'Pesticide_Group', 'Violations', 'Tests', 'Violation_Rate']
                    
                    fig_trend = px.line(
                        trend_data,
                        x='Period',
                        y='Violation_Rate',
                        color='Pesticide_Group',
                        title='Violation Rate Trends by Pesticide Group Over Time',
                        markers=True
                    )
                    
                    fig_trend.update_layout(
                        xaxis_title='Time Period',
                        yaxis_title='Violation Rate',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                
            # Seasonal trend
            st.subheader("🌍 Seasonal Contamination Trends")

        
        # Seasonal trend
        seasonal_trend = df.groupby('season').agg({
            'violation_numeric': 'mean',
            'reading': 'mean'
        }).reset_index()
        
        # Order seasons properly
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_trend['season'] = pd.Categorical(seasonal_trend['season'], categories=season_order, ordered=True)
        seasonal_trend = seasonal_trend.sort_values('season')
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=seasonal_trend['season'],
            y=seasonal_trend['violation_numeric'] * 100,
            mode='lines+markers',
            name='Violation Rate (%)',
            marker=dict(size=12, color='#FF6B6B'),
            line=dict(width=3)
        ))
        
        fig_trend.update_layout(
            title='Seasonal Violation Rate Pattern',
            xaxis_title='Season',
            yaxis_title='Violation Rate (%)',
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Top vegetables by violations
        st.subheader("🥬 Top 10 Vegetables by Violation Count")
        
        veg_violations = df.groupby('vegetable_english')['violation_numeric'].sum().nlargest(10).reset_index()
        veg_violations.columns = ['Vegetable', 'Total_Violations']
        
        fig_veg = px.bar(
            veg_violations,
            x='Vegetable',
            y='Total_Violations',
            color='Total_Violations',
            color_continuous_scale='Reds',
            title='Vegetables with Most Violations'
        )
        
        fig_veg.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_veg, use_container_width=True)
        
        # Top pesticides by violations
        st.subheader("🧪 Top 10 Pesticides by Violation Count")
        
        pest_violations = df.groupby('pesticide')['violation_numeric'].sum().nlargest(10).reset_index()
        pest_violations.columns = ['Pesticide', 'Total_Violations']
        
        fig_pest = px.bar(
            pest_violations,
            x='Pesticide',
            y='Total_Violations',
            color='Total_Violations',
            color_continuous_scale='Oranges',
            title='Pesticides with Most Violations'
        )
        
        fig_pest.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_pest, use_container_width=True)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()

