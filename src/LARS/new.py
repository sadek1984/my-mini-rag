"""
Predictive Analytics Dashboard with Enhanced Visual Style
REORGANIZED VERSION - Applied all structural improvements
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
import os

os.environ['GEMINI_API_KEY'] = 'AIzaSyBH6QPeFgtZ1B26rt-p2GHaiD55IHEP6s0'  # Add your actual key

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
    """
    Initialize all session state variables to prevent AttributeError.
    Call this at the very beginning of main() function.
    """
    
    # Header display control
    if "intro_shown" not in st.session_state:
        st.session_state.intro_shown = False
    
    # Page navigation - UPDATED default page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard Overview"
    
    # AI Assistant chat states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Year selection for filtering
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = None
    
    # Project ID
    if "project_id" not in st.session_state:
        st.session_state.project_id = "TrhBVe_m5gg2002_E5VVqS"
    
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
    
    # Chat messages
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Indexed years
    if 'indexed_years' not in st.session_state:
        st.session_state.indexed_years = set()

@st.cache_resource
def get_hybrid_system_optional():
    """
    Initialize hybrid query system
    Works WITHOUT NLPController (no ISO docs, but pesticide queries work!)
    """
    
    if not HYBRID_AVAILABLE:
        print("⚠️ HYBRID_AVAILABLE is False")
        return None
    
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_key:
        print("⚠️ GEMINI_API_KEY not set")
        return None
    
    try:
        print("🔄 Initializing hybrid system...")
        
        hybrid_system = HybridQuerySystem(
            excel_path='/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx',
            iso_rag_client=None,
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
        from controllers.NLPController import EnhancedNLPController
        nlp_controller = EnhancedNLPController(gemini_api_key=gemini_key)
        
        hybrid = HybridQuerySystem(
            excel_path='/Users/a12/mini-rag/src/data/processed_data_output.xlsx',
            iso_rag_client=nlp_controller,
            gemini_api_key=gemini_key
        )
        
        return hybrid
    
    except Exception as e:
        st.sidebar.error(f"Hybrid init error: {e}")
        return None

def show_lars_intro():
    """Show LARS animated intro for 1 second"""
    
    intro_container = st.empty()
    
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
    
    intro_html = f"""
    {intro_css}
    <div class="lars-intro">
        <div style="text-align: center;">
            <div class="lars-title">LARS</div>
            <div class="lars-subtitle">Laboratory Analytics & Risk System</div>
        </div>
    </div>
    """
    
    intro_container.markdown(intro_html, unsafe_allow_html=True)
    time.sleep(1.2)
    intro_container.empty()

# ============================================================================
# HELPER FUNCTIONS (Keep all existing)
# ============================================================================

def perform_anova_analysis(df, group_by, target_col='reading'):
    """Perform ANOVA analysis grouped by specified column"""
    
    groups = df.groupby(group_by)[target_col].apply(list)
    f_stat, p_value = f_oneway(*groups.values)
    
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

def remove_outliers(df, column, threshold=3):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_clean

def create_anova_visualization(df: pd.DataFrame, group_by: str, target_col: str = 'reading'):
    """Bar chart showing MEAN values"""
    
    season_colors = {
        'Winter': '#339AF0',
        'Spring': '#51CF66',
        'Summer': '#FF922B',
        'Fall': '#FA5252'
    }
    
    stats = df.groupby(group_by, observed=True)[target_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'count')
    ]).reset_index()
    
    if group_by == 'season':
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        stats[group_by] = pd.Categorical(stats[group_by], categories=season_order, ordered=True)
        stats = stats.sort_values(group_by)
        colors = [season_colors[s] for s in stats[group_by]]
    else:
        colors = '#4CAF50'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats[group_by],
        y=stats['mean'],
        error_y=dict(
            type='data',
            array=stats['std'],
            visible=True,
            color='rgba(0,0,0,0.4)',
            thickness=2,
            width=10
        ),
        marker_color=colors,
        text=stats['mean'].round(1),
        textposition='outside',
        textfont=dict(size=18, color='black', family='Arial Black'),
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'Mean: %{y:.1f}<br>' +
            'Std Dev: %{error_y.array:.1f}<br>' +
            '<extra></extra>'
        ),
        name=f'Mean {target_col}'
    ))
    
    for idx, season in enumerate(stats[group_by]):
        season_data = df[df[group_by] == season][target_col]
        fig.add_trace(go.Scatter(
            x=[season] * len(season_data),
            y=season_data,
            mode='markers',
            marker=dict(
                size=6,
                color=colors[idx] if isinstance(colors, list) else colors,
                opacity=0.3,
                line=dict(width=1, color='white')
            ),
            showlegend=False,
            hovertemplate='%{x}<br>Reading: %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Mean {target_col.replace("_", " ").title()} by {group_by.replace("_", " ").title()}</b>',
            font=dict(size=22, color='#2e7d32', family='Arial Black')
        ),
        xaxis_title=group_by.replace('_', ' ').title(),
        yaxis_title=f'{target_col.replace("_", " ").title()} (ppb)',
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(248,255,248,0.5)',
        bargap=0.3
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=True
    )
    
    return fig

def create_means_comparison_plot(summary_df: pd.DataFrame, group_by: str):
    """Create bar plot comparing means with error bars and colors"""
    
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
        df['season_name'] = df['season'].map(season_mapping)
        df['season_name'] = df['season_name'].fillna(df['season'])
        df['season'] = df['season_name']
        
    return df

# Custom CSS (keep all existing)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
    }
    
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
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5e8 0%, #c8e6c9 100%);
        border-right: 3px solid #81c784;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1b5e20;
        font-weight: 700;
    }
    
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
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1b5e20, #4caf50);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
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
    
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #4caf50;
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4caf50, transparent);
        margin: 30px 0;
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

def get_api_client():
    """Get API client instance"""
    return APIClient()

import textwrap

def render_animated_header():
    """Render the animated LARS header"""
    
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
            render_animated_header()
            st.session_state.intro_shown = True
        
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
        
        render_sidebar(api_client)

        page = st.session_state.get('current_page', '🏠 Dashboard Overview')
        
        # ✅ UPDATED PAGE ROUTER with new structure
        pages = {
            "🏠 Dashboard Overview": show_dashboard_overview,
            "📊 Risk Analysis": show_risk_analysis_page,
            "📈 Trend Analysis": show_trend_analysis_page,
            "📋 Statistical Insights": show_statistical_insights_page,
            "🔮 Predictions": show_predictions_page,
            "🤖 AI Assistant": show_chat_page,
            "⚙️ Advanced Tools": show_advanced_tools_page,
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
# SIDEBAR - UPDATED NAVIGATION
# ============================================================================

def render_sidebar(api_client):
    """Render sidebar with REORGANIZED navigation"""
    
    with st.sidebar:
        
        st.markdown("## 🔗 API Configuration")
        
        current_url = st.text_input("API Base URL", value=API_BASE_URL, key="api_url")
        
        if current_url != API_BASE_URL:
            api_client = APIClient(current_url)
        
        st.markdown("---")
        
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
        
        # ✅ UPDATED NAVIGATION with new structure
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
# PAGE 1: DASHBOARD OVERVIEW (Enhanced Home)
# ============================================================================

def show_dashboard_overview(api_client):
    """Dashboard Overview - Enhanced home page"""
    
    st.title("🏠 Dashboard Overview")
    
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
    
    st.markdown("---")
    
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
    """Risk Analysis - Merged high-risk combinations and heatmaps"""
    
    st.title("📊 Risk Analysis")
    
    st.info("""
    **Comprehensive Risk Assessment:**
    - Identify highest-risk vegetable-pesticide combinations
    - Visualize risk patterns with interactive heatmaps
    - Prioritize testing resources efficiently
    """)
    
    # Load data
    @st.cache_data
    def load_data_for_risk():
        """Load pesticide data"""
        try:
            possible_paths = [
                'data/cleaned_data.csv',
                'data/processed_data_output.xlsx',
                '../data/processed_data_output.xlsx',
                '/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx'
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
    
    df = load_data_for_risk()
    
    if df is None:
        st.warning("📤 Please upload your data first")
        return
    
    # Clean data
    df = map_season_names(df)
    df = df.dropna(subset=['reading', 'limits'])
    df['reading'] = pd.to_numeric(df['reading'], errors='coerce')
    df['limits'] = pd.to_numeric(df['limits'], errors='coerce')
    df = df.dropna(subset=['reading', 'limits'])
    df = df[(df['reading'] >= 0) & (df['limits'] > 0)]
    
    # Calculate violations
    df['violation_numeric'] = (df['reading'] > df['limits']).astype(int)
    df['violation'] = (df['reading'] > df['limits'])
    
    # ========== SECTION 1: High-Risk Combinations ==========
    st.header("🔥 High-Risk Combinations")
    
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
    
    high_risk = combinations.nlargest(20, 'Risk_Score')
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_violations = int(df['violation_numeric'].sum())
        st.metric("Total Violations", f"{total_violations:,}")
    
    with col2:
        top_10_violations = high_risk.head(10)['Violations'].sum()
        coverage = (top_10_violations / total_violations * 100) if total_violations > 0 else 0
        st.metric("Top 10 Coverage", f"{coverage:.1f}%")
    
    with col3:
        avg_risk = high_risk.head(10)['Violation_Rate'].mean() * 100
        st.metric("Avg Violation Rate", f"{avg_risk:.1f}%")
    
    # Display table
    st.subheader("🎯 Top 10 High-Risk Combinations")
    
    display_cols = ['Vegetable', 'Pesticide', 'Violations', 'Total_Tests', 'Violation_Rate', 'Risk_Score']
    display_df = high_risk.head(10)[display_cols].copy()
    
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
    
    st.markdown("---")
    
    # ========== SECTION 2: Risk Heatmaps ==========
    st.header("🗺️ Risk Heatmaps")
    
    # Prepare heatmap data
    heatmap_data = df.groupby(['vegetable_english', 'pesticide']).agg({
        'violation_numeric': ['sum', 'count', 'mean']
    }).reset_index()
    heatmap_data.columns = ['Vegetable', 'Pesticide', 'Violations', 'Tests', 'Violation_Rate']
    heatmap_data['Violations'] = pd.to_numeric(heatmap_data['Violations'], errors='coerce')
    heatmap_data = heatmap_data.dropna()
    
    min_tests = st.slider("Minimum tests required:", 1, 10, 3)
    heatmap_filtered = heatmap_data[heatmap_data['Tests'] >= min_tests].copy()
    
    if len(heatmap_filtered) >= 3:
        top_combos = heatmap_filtered.nlargest(30, 'Violations').copy()
        
        # Bubble chart
        st.subheader("🔥 Violation Rate Bubble Chart")
        
        def get_color_for_rate(rate):
            if rate <= 0.33:
                return 'green'
            elif rate <= 0.66:
                return 'yellow'
            else:
                return 'red'
        
        colors_manual = [get_color_for_rate(rate) for rate in top_combos['Violation_Rate']]
        
        min_violations = top_combos['Violations'].min()
        max_violations = top_combos['Violations'].max()
        bubble_sizes = [10 + (v - min_violations) / (max_violations - min_violations) * 40 
                    for v in top_combos['Violations']]
        
        fig_bubble = go.Figure()
        
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
        
        # Treemap
        st.subheader("📊 Risk Treemap")
        
        try:
            clean_treemap_data = heatmap_filtered.nlargest(15, 'Violations').copy()
            clean_treemap_data['Violation_Rate'] = pd.to_numeric(
                clean_treemap_data['Violation_Rate'], 
                errors='coerce'
            )
            
            def get_risk_category(rate):
                if rate < 0.5:
                    return 'Low (<50%)'
                elif rate < 0.75:
                    return 'Medium (50-75%)'
                elif rate < 0.9:
                    return 'High (75-90%)'
                else:
                    return 'Critical (90%+)'

            clean_treemap_data['Risk_Category'] = clean_treemap_data['Violation_Rate'].apply(get_risk_category)
            
            clean_treemap_data['Display_Label'] = clean_treemap_data.apply(
                lambda row: f"{row['Pesticide']}<br>{row['Violations']} violations<br>{row['Violation_Rate']:.0%} rate", 
                axis=1
            )

            fig_clean = px.treemap(
                clean_treemap_data,
                path=['Vegetable', 'Display_Label'],
                values='Violations',
                color='Risk_Category',
                color_discrete_map={
                    'Low (<50%)': '#C8E6C9',
                    'Medium (50-75%)': '#81C784',
                    'High (75-90%)': '#4CAF50',
                    'Critical (90%+)': '#2E7D32'
                },
                title='Top 15 Highest-Risk Combinations'
            )
            
            fig_clean.update_layout(
                height=550,
                margin=dict(t=60, l=10, r=10, b=10)
            )
            
            fig_clean.update_traces(
                textinfo="label",
                textfont=dict(size=13, color='white'),
                textposition="middle center",
                hovertemplate=(
                    '<b>%{label}</b><br>' +
                    'Violations: %{value}<br>' +
                    '<extra></extra>'
                ),
                marker=dict(
                    line=dict(width=2.5, color='white'),
                    cornerradius=4
                )
            )
            
            st.plotly_chart(fig_clean, use_container_width=True)
            
        except Exception as e:
            st.error(f"Treemap visualization failed: {e}")
    
    else:
        st.warning("⚠️ Not enough data with current filter settings")

# ============================================================================
# PAGE 3: TREND ANALYSIS (Merged Time Trends + Seasonal Patterns)
# ============================================================================

def show_trend_analysis_page(api_client):
    """Trend Analysis - Merged time trends and seasonal patterns"""
    
    st.title("📈 Trend Analysis")
    
    st.info("""
    **Track contamination patterns over time:**
    - Identify long-term trends (improving/worsening)
    - Analyze seasonal variations
    - Monitor intervention effectiveness
    """)
    
    # Load data
    @st.cache_data
    def load_data_for_trends():
        """Load pesticide data"""
        try:
            possible_paths = [
                'data/cleaned_data.csv',
                'data/processed_data_output.xlsx',
                '../data/processed_data_output.xlsx',
                '/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx'
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
    
    df = load_data_for_trends()
    
    if df is None:
        st.warning("📤 Please upload your data first")
        return
    
    # Clean data
    df = map_season_names(df)
    
    # ========== SECTION 1: Time Series Trends ==========
    st.header("📊 Time Series Trends")
    
    st.info("Track how violations change over time")
    
    # Calculate violations
    if 'violation_numeric' not in df.columns:
        df['violation_numeric'] = (df['reading'] > df['limits']).astype(int)
    
    # Top vegetables and pesticides
    veg_violations = df.groupby('vegetable_english')['violation_numeric'].sum().nlargest(10).reset_index()
    veg_violations.columns = ['Vegetable', 'Total_Violations']
    
    pest_violations = df.groupby('pesticide')['violation_numeric'].sum().nlargest(10).reset_index()
    pest_violations.columns = ['Pesticide', 'Total_Violations']
    
    col1, col2 = st.columns(2)
    
    with col1:
        top5_veg = veg_violations.head(5)
        fig_veg = go.Figure()
        
        fig_veg.add_trace(go.Scatterpolar(
            r=top5_veg['Total_Violations'].tolist() + [top5_veg['Total_Violations'].iloc[0]],
            theta=top5_veg['Vegetable'].tolist() + [top5_veg['Vegetable'].iloc[0]],
            fill='toself',
            fillcolor='rgba(255, 99, 71, 0.3)',
            line=dict(color='tomato', width=2),
            name='Vegetables'
        ))
        
        fig_veg.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, top5_veg['Total_Violations'].max() * 1.2])
            ),
            title='Top 5 Vegetables by Violations',
            height=400
        )
        st.plotly_chart(fig_veg, use_container_width=True)
    
    with col2:
        top5_pest = pest_violations.head(5)
        fig_pest = go.Figure()
        
        fig_pest.add_trace(go.Scatterpolar(
            r=top5_pest['Total_Violations'].tolist() + [top5_pest['Total_Violations'].iloc[0]],
            theta=top5_pest['Pesticide'].tolist() + [top5_pest['Pesticide'].iloc[0]],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.3)',
            line=dict(color='orange', width=2),
            name='Pesticides'
        ))
        
        fig_pest.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, top5_pest['Total_Violations'].max() * 1.2])
            ),
            title='Top 5 Pesticides by Violations',
            height=400
        )
        st.plotly_chart(fig_pest, use_container_width=True)
    
    st.markdown("---")
    
    # ========== SECTION 2: Seasonal Patterns ==========
    st.header("🌡️ Seasonal Contamination Patterns")
    
    st.info("Understand how contamination varies across different seasons")
    
    if 'season' in df.columns:
        # Create seasonal visualization
        fig = create_anova_visualization(df, 'season', 'reading')
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal statistics
        seasonal_stats = df.groupby('season')['reading'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std Dev', 'std'),
            ('Count', 'count')
        ]).round(2)
        
        st.subheader("📊 Seasonal Statistics")
        st.dataframe(seasonal_stats, use_container_width=True)
    else:
        st.warning("No seasonal data available in dataset")

# ============================================================================
# PAGE 4: STATISTICAL INSIGHTS (Merged ANOVA + Statistical Tests)
# ============================================================================

def show_statistical_insights_page(api_client):
    """Statistical Insights - Merged ANOVA and statistical tests"""
    
    st.title("📋 Statistical Insights")
    
    st.info("""
    **Statistical validation of patterns:**
    - ANOVA testing for seasonal/vegetable differences
    - Trend detection and significance testing
    - Statistical summaries and comparisons
    """)
    
    # Load data
    @st.cache_data
    def load_data_for_stats():
        """Load pesticide data"""
        try:
            possible_paths = [
                'data/cleaned_data.csv',
                'data/processed_data_output.xlsx',
                '../data/processed_data_output.xlsx',
                '/Users/a12/mini-rag/src/LARS/processed_data_output.xlsx'
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
    
    df = load_data_for_stats()
    
    if df is None:
        st.warning("📤 Please upload your data first")
        return
    
    # Clean data
    df = map_season_names(df)
    df = df.dropna(subset=['reading'])
    
    # ========== SECTION 1: ANOVA Testing ==========
    st.header("📊 ANOVA Testing")
    
    st.sidebar.header("🔧 Analysis Controls")
    
    analysis_mode = st.sidebar.radio(
        "Group Analysis By:",
        options=['Season', 'Vegetable'],
        index=0
    )
    
    group_column = 'season' if analysis_mode == 'Season' else 'vegetable_english'
    
    if group_column in df.columns:
        df_filtered = df.dropna(subset=[group_column, 'reading'])
        
        if len(df_filtered) >= 10:
            n_groups = df_filtered[group_column].nunique()
            
            if n_groups >= 2:
                # Perform ANOVA
                with st.spinner("🔄 Calculating ANOVA..."):
                    results = perform_anova_analysis(df_filtered, group_column, 'reading')
                
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
                    
                    - The {analysis_mode.lower()}s have significantly different reading values
                    - P-value ({results['p_value']:.6f}) < 0.05
                    - F-statistic: {results['f_statistic']:.4f}
                    
                    **Conclusion:** {analysis_mode} DOES affect contamination levels!
                    """)
                else:
                    st.info(f"""
                    **No Significant Difference Found**
                    
                    - The {analysis_mode.lower()}s have similar reading values
                    - P-value ({results['p_value']:.6f}) ≥ 0.05
                    - F-statistic: {results['f_statistic']:.4f}
                    
                    **Conclusion:** {analysis_mode} does NOT significantly affect contamination levels
                    """)
                
                # Summary statistics
                st.subheader("📊 Group Statistics")
                st.dataframe(
                    results['summary'].style.highlight_max(axis=0, color='lightgreen')
                                            .highlight_min(axis=0, color='lightcoral'),
                    use_container_width=True
                )
            else:
                st.warning(f"Need at least 2 {analysis_mode.lower()}s for comparison")
        else:
            st.warning("Not enough data for analysis (minimum 10 samples required)")
    else:
        st.error(f"Column '{group_column}' not found in dataset")
    
    st.markdown("---")
    
    # ========== SECTION 2: Statistical Summaries ==========
    st.header("📈 Statistical Summaries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Get Statistical Insights", use_container_width=True):
            with st.spinner("Analyzing contamination data..."):
                response = api_client.get_statistical_insights()
                
                if response and 'insights' in response:
                    insights = response['insights']
                    
                    if insights:
                        for insight in insights:
                            if insight.get('severity') == 'high':
                                st.error(f"⚠️ {insight.get('message', '')}")
                            elif insight.get('severity') == 'medium':
                                st.warning(f"📊 {insight.get('message', '')}")
                            else:
                                st.info(f"✅ {insight.get('message', '')}")
                    else:
                        st.success("✅ No concerning patterns detected - all stable!")
                else:
                    st.info("Unable to retrieve statistical insights")

# ============================================================================
# PAGE 5: PREDICTIONS (Merged Predictions + Time Series Forecasting)
# ============================================================================

def show_predictions_page(api_client):
    """Predictions - Merged predictions and time series forecasting"""
    
    st.title("🔮 Predictions & Forecasting")
    
    # Create tabs for different prediction types
    tab1, tab2 = st.tabs(["🎯 Risk Assessment", "📈 Time Series Forecasting"])
    
    with tab1:
        st.header("Enhanced Risk Assessment")
        show_enhanced_risk_form(api_client)
    
    with tab2:
        st.header("Time Series Forecasting")
        show_timeseries_dashboard(api_client)

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
    
    with st.form("enhanced_risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vegetable = st.text_input("Vegetable", value="parsley")
            pesticide_group = st.selectbox("Pesticide Group",
                ["organophosphorus", "pyrethroid", "chitin_synthesis_inhibitor",
                 "neonicotinoid", "phenylpyrazoles", "carbamates", "triazoles"])
        
        with col2:
            reading = st.number_input("Reading (mg/kg)", value=50.0, min_value=0.0, step=0.1)
            limit = st.number_input("Limit (mg/kg)", value=20.0, min_value=0.0, step=0.1)
            
            exceedance_ratio = reading / limit if limit > 0 else 0
            is_compliant = reading <= limit
            
            if is_compliant:
                st.success(f"✅ Compliant (Exceedance: {exceedance_ratio:.2f}x)")
            else:
                st.error(f"❌ Non-Compliant (Exceedance: {exceedance_ratio:.2f}x)")
        
        st.markdown("---")
        st.markdown("#### 🔧 Advanced Parameters")
        
        col3, col4 = st.columns(2)
        
        with col3:
            population_type = st.selectbox(
                "Population Type",
                ["general", "vulnerable", "children", "pregnant_women"]
            )
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95
            )
        
        with col4:
            simulation_runs = st.selectbox(
                "Monte Carlo Simulations",
                [1000, 5000, 10000, 50000],
                index=1
            )
            include_uncertainty = st.checkbox(
                "Include Measurement Uncertainty",
                value=True
            )
        
        submitted = st.form_submit_button("🎯 Run Comprehensive Risk Assessment", 
                                        type="primary", 
                                        use_container_width=True)
    
    if submitted:
        risk_data = {
            "vegetable": vegetable,
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
                else:
                    st.error("❌ No response from API")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def display_enhanced_risk_result(risk_result: Dict):
    """Display enhanced risk assessment results"""
    
    st.markdown("## 📊 Risk Assessment Results")
    
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
        st.metric("Bayesian Risk Score", f"{bayesian_risk:.2f}")
    
    with col2:
        critical_threshold = risk_result.get('critical_threshold', 0)
        st.metric("Critical Threshold", f"{critical_threshold:.1f}")
    
    with col3:
        monte_carlo_est = risk_result.get('exceedance_probability', 0)
        st.metric("Monte Carlo Estimate", f"{monte_carlo_est:.3f}")
    
    with col4:
        confidence = risk_result.get('confidence_interval', 95)
        st.metric("Confidence Level", f"{confidence}%")

def show_timeseries_dashboard(api_client):
    """Time Series Forecasting dashboard"""
    
    st.markdown("""
    **Interactive Pesticide Forecasting**
    - 📊 View historical trends and future predictions
    - 📈 Multiple forecasting models (Prophet, ARIMA)
    - 🔮 Confidence intervals and uncertainty estimates
    - 📅 Different time horizons: Daily, Weekly, Monthly, Yearly
    """)
    
    dashboard_url = f"{api_client.base_url}/api/v1/timeseries/interactive-dashboard"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📡 **Dashboard URL:** `{dashboard_url}`")
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_timeseries", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
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

# ============================================================================
# PAGE 6: AI ASSISTANT (Keep as is - UNCHANGED)
# ============================================================================


import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import numpy as np


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
        
        # # Data Context
        # st.markdown("## 📅 Data Context")
        
        # # Initialize selected_year if not exists
        # if 'selected_year' not in st.session_state:
        #     st.session_state.selected_year = "2022"
        
        # st.info(f"**Active Year:** {st.session_state.selected_year}")
        
        # selected_year = st.selectbox(
        #     "Default Year",
        #     AVAILABLE_YEARS,
        #     index=AVAILABLE_YEARS.index(st.session_state.selected_year)
        # )
        
        # if selected_year != st.session_state.selected_year:
        #     st.session_state.selected_year = selected_year
        
        # st.markdown("---")
        
        # # Data Indexing
        # st.markdown("## 🗄️ Data Indexing")
        
        # year_to_index = st.selectbox("Select Year to Index", AVAILABLE_YEARS)
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     if st.button("📥 Index", use_container_width=True):
        #         index_year_data(api_client, year_to_index, do_reset=False)
        
        # with col2:
        #     if st.button("🗑️ Reset", use_container_width=True):
        #         index_year_data(api_client, year_to_index, do_reset=True)
        
        # # Initialize indexed_years if not exists
        # if 'indexed_years' not in st.session_state:
        #     st.session_state.indexed_years = set()
        
        # if st.session_state.indexed_years:
        #     st.markdown("**✅ Indexed Years:**")
        #     for year in sorted(st.session_state.indexed_years):
        #         st.markdown(f"- {year}")

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
            
            # 🔒 SCHEMA-ONLY PROMPTING - Privacy Protection
            # Generate DataFrame metadata WITHOUT exposing actual sensitive data
            import io
            
            # Get column types as markdown table
            dtypes_markdown = df.dtypes.to_frame('Type').to_markdown()
            
            # Get structure info (column names, non-null counts, memory usage)
            buffer = io.StringIO()
            df.info(buf=buffer)
            df_info_str = buffer.getvalue()
            
            # Get sample statistics (no actual values, just ranges/stats)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            sample_stats = ""
            if numeric_cols:
                stats_df = df[numeric_cols].describe()
                # Get min, max, mean from the describe output (these are rows, not columns)
                stats_summary = stats_df.loc[['min', 'max', 'mean']]
                sample_stats = f"\nNumeric column ranges:\n{stats_summary.to_markdown()}"
            
            # 🆕 Add categorical column info (unique counts, NO actual values for sensitive columns)
            categorical_info = "\n**Categorical Columns Info:**\n"
            for col in df.select_dtypes(include=['object']).columns:
                unique_count = df[col].nunique()
                # Don't expose sensitive data like client names or sample IDs
                if col in ['client_name', 'sample_id', 'sample_code']:
                    categorical_info += f"- `{col}`: {unique_count} unique values (sensitive, not shown)\n"
                else:
                    categorical_info += f"- `{col}`: {unique_count} unique values\n"
            
            # 🆕 Add example value patterns (generic, non-sensitive)
            value_patterns = """
**Common Value Patterns (Generic Examples):**
- `vegetable_english`: English vegetable names (e.g., 'Tomato', 'Cucumber', 'Pepper')
- `vegetable_arabic`: Arabic vegetable names (e.g., 'طماطم', 'خيار', 'فلفل')
- `pesticide_standardized`: English pesticide names (use for filtering)
- `pesticide_arabic`: Arabic pesticide names (translate using pesticide_mapping)
- `neighborhood_arabic`: Arabic neighborhood names (e.g., 'الإسكان', 'الصفراء', 'الموطأ')
- `is_compliant`: 1=compliant (below limit), 0=violation (above limit)
- `result`: Arabic text 'مطابق' (compliant) or 'غير مطابق' (non-compliant)
"""
            
            prompt = f"""Generate pandas code to answer this question about the dataset.

🔒 Dataset Schema (Metadata Only - No Actual Data Exposed):

**Column Information:**
{dtypes_markdown}

**DataFrame Structure:**
{df_info_str}
{sample_stats}
{categorical_info}
{value_patterns}

**Available Columns:** {list(df.columns)}
**Total Records:** {len(df)} samples
**Shape:** {df.shape}

⚠️ PRIVACY NOTE: You are receiving ONLY schema information (column names and types).
DO NOT assume specific values exist. Write generic filtering/aggregation code based on column names only.

Question: {query}
{risk_instruction}

Requirements:
1. Use pandas (imported as pd) for all operations
2. The DataFrame variable is 'df'
3. Store final result in variable 'result'
4. Handle missing values appropriately (use na=False with .str.contains())
5. Return clean, readable output
6. Include proper error handling
7. Add comments for clarity
8. For Arabic queries, use Arabic column names in output

**Example Patterns:**

For neighborhood + vegetable analysis:
```python
# Filter by neighborhood
neighborhood_data = df[df['neighborhood_arabic'].str.contains('الإسكان', case=False, na=False)]
# Group by vegetable
result = neighborhood_data.groupby('vegetable_arabic').agg(
    عدد=('vegetable_arabic', 'count'),
    فوق_الحد=('is_compliant', lambda x: (x == 0).sum()),
    تحت_الحد=('is_compliant', lambda x: (x == 1).sum())
).reset_index()
result.columns = ['الخضار', 'عدد العينات', 'فوق الحد', 'تحت الحد']
```

For pesticide + vegetable analysis:
```python
# Use pesticide_mapping for translation (available in scope)
pesticide_name = pesticide_mapping.get('البايفنثرن', 'Bifenthrin')
pesticide_data = df[df['pesticide_standardized'].str.contains(pesticide_name, case=False, na=False)]
result = pesticide_data.groupby('vegetable_arabic').agg(
    عدد=('vegetable_arabic', 'count'),
    تحت_الحد=('is_compliant', lambda x: (x == 1).sum()),
    فوق_الحد=('is_compliant', lambda x: (x == 0).sum())
).reset_index()
result.columns = ['الخضار', 'عدد العينات', 'تحت الحد', 'فوق الحد']
```

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


# ============================================================================
# PAGE 7: ADVANCED TOOLS (New - Debug + Technical Features)
# ============================================================================

def show_advanced_tools_page(api_client):
    """Advanced tools and debugging features"""
    
    st.title("⚙️ Advanced Tools")
    
    st.warning("🔧 This section contains technical debugging and advanced configuration options.")
    
    if st.checkbox("Show Debug Panel", value=False):
        st.markdown("## 🐛 DEBUG PANEL")
        
        # Load data for debugging
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            
            with st.expander("📊 Raw Data Inspection", expanded=True):
                st.write("**DataFrame Shape:**", df.shape)
                st.write("**Columns:**")
                st.json(df.columns.tolist())
                
                st.dataframe(df.head(20), use_container_width=True)
            
            with st.expander("🔍 Data Quality Checks", expanded=False):
                st.write("**Missing Values:**")
                st.write(df.isnull().sum())
                
                st.write("**Data Types:**")
                st.write(df.dtypes)
                
                st.write("**Basic Statistics:**")
                st.write(df.describe())
            
            with st.expander("💾 Export Debug Data", expanded=False):
                csv = df.head(100).to_csv(index=False)
                st.download_button(
                    label="📥 Download First 100 Rows (CSV)",
                    data=csv,
                    file_name="debug_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No data loaded yet")
    
    st.markdown("---")
    
    st.subheader("🔧 System Diagnostics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}")
    
    with col2:
        st.metric("Streamlit Version", st.__version__)
    
    with col3:
        if 'df' in st.session_state and st.session_state.df is not None:
            memory_mb = st.session_state.df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Data Memory", f"{memory_mb:.2f} MB")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()