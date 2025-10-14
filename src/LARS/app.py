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
    page_title="🔮 Predictive Analytics Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main h1 {
        color: #FF4B4B;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2f6 0%, #e8eaf0 100%);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        border-top: 2px solid #f0f2f6;
        padding-top: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
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

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def get_api_client():
    """Get API client instance (not cached to avoid issues)"""
    return APIClient()

def main():
    """Main application entry point"""
    
    try:
        initialize_session_state()
        api_client = get_api_client()
        
        # App header with emoji
        st.title("🔮 Predictive Analytics Dashboard")
        st.markdown("---")
        
        if 'page_nav' in st.session_state:
            st.session_state.current_page = st.session_state.page_nav
            del st.session_state.page_nav
        
        # Sidebar
        render_sidebar(api_client)
        
        # Get selected page
        page = st.session_state.get('current_page', '💬 AI Assistant')
        
        # Debug: Show current page
        # st.sidebar.write(f"DEBUG: Current page = {page}")  # Uncomment for debugging
        
        # Route to pages
        try:
            if page == "🏠 Home":
                show_home_page(api_client)
            elif page == "🔮 Predictions":
                show_predictions_page(api_client)
            elif page == "💬 AI Assistant":
                show_chat_page(api_client)
            elif page == "📊 Statistical Insights":
                show_statistical_insights_page(api_client)
            elif page == "📈 ANOVA Analysis":
                show_anova_analysis_page(api_client)
            elif page == "🎯 Advanced Insights":
                show_advanced_insights_page(api_client)
            else:
                st.error(f"Unknown page: {page}")
                st.info("Available pages: Home, AI Assistant, Predictions, Statistical Insights, ANOVA Analysis, Advanced Insights")
        
        except Exception as page_error:
            st.error(f"❌ Error loading page: {str(page_error)}")
            st.exception(page_error)
            
            # Show fallback
            st.markdown("---")
            st.markdown("### 🏠 Welcome")
            st.info("There was an error loading the selected page. Please select another page from the sidebar.")
    
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        st.exception(e)
        st.info("Please check the terminal for detailed error messages.")

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
            "🎯 Advanced Insights"
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

def show_home_page(api_client):
    """Home page with overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Your Predictive Analytics Dashboard! 
        
        This application provides:
        
        - 📊 **Statistical Insights** - Mann-Kendall trends, ANOVA analysis
        - 📈 **ANOVA Analysis** - Compare readings by seasons/vegetables
        - 🔮 **Enhanced Risk Assessment** - Monte Carlo + Bayesian analysis
        - 💬 **AI Assistant** - Natural language queries
        - 📈 **Real-time Dashboards** - Executive, Laboratory views
        
        ### 🚀 Getting Started:
        1. Check API connection in sidebar
        2. Index your data by year
        3. Start analyzing!
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Actions")
        
        # ✅ Use unique keys for buttons and set state differently
        if st.button("📊 Statistical Insights", use_container_width=True, type="primary", key="home_btn_insights"):
            st.session_state.page_nav = "📊 Statistical Insights"
            st.rerun()

        if st.button("📈 ANOVA Analysis", use_container_width=True, type="primary", key="home_btn_anova"):
            st.session_state.page_nav = "📈 ANOVA Analysis"
            st.rerun()
        
        if st.button("💬 AI Assistant", use_container_width=True, type="secondary", key="home_btn_assistant"):
            st.session_state.page_nav = "💬 AI Assistant"
            st.rerun()
        
        if st.button("🔮 Predictions", use_container_width=True, key="home_btn_predictions"):
            st.session_state.page_nav = "🔮 Predictions"
            st.rerun()

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
    """Predictions page with enhanced risk assessment"""
    
    st.title("🔮 Predictions with Enhanced Risk Assessment")
    
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "🎯 Enhanced Risk Assessment"])
    
    with tab1:
        show_single_prediction_form(api_client)
    
    with tab2:
        show_enhanced_risk_form(api_client)

def show_single_prediction_form(api_client):
    """Single prediction form"""
    
    st.markdown("### Basic Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vegetable = st.text_input("Vegetable", value="parsley")
            pesticide = st.text_input("Pesticide Name", value="Quinalphos")
            pesticide_group = st.selectbox("Pesticide Group", 
                ["organophosphorus", "pyrethroid", "chitin_synthesis_inhibitor", 
                 "neonicotinoid", "phenylpyrazoles"])
            reading = st.number_input("Reading", value=347.0, min_value=0.0)
        
        with col2:
            limits = st.number_input("Limits", value=20.0, min_value=0.0)
            sample_code = st.text_input("Sample Code", value="TEST_001")
            collection_date = st.date_input("Collection Date")
        
        if st.form_submit_button("🚀 Run Prediction", type="primary"):
            data = {
                "vegetable": vegetable,
                "pesticide_name": pesticide,
                "pesticide_group": pesticide_group,
                "reading": reading,
                "limits": limits,
                "sample_code": sample_code,
                "collection_date": str(collection_date)
            }
            
            with st.spinner("Running prediction..."):
                result = api_client.predict_single(data)
                
                if result:
                    display_prediction_result(result)
def show_enhanced_risk_form(api_client):
    """Enhanced risk assessment form"""
    
    st.markdown("### Enhanced Risk Assessment")
    st.write("""
    Uses advanced statistical methods:
    - **Monte Carlo Simulation**: Probabilistic exceedance estimation
    - **Bayesian Analysis**: Prior knowledge from pesticide group
    - **Hazard Modeling**: Vegetable-specific exposure multipliers
    """)
    
    # Create form
    with st.form("enhanced_risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vegetable = st.text_input("Vegetable", value="parsley")
            veg_category = st.selectbox("Vegetable Category",
                ["leafy_greens", "peppers", "herbs", "root_vegetables", "fruits"])
            pesticide_group = st.selectbox("Pesticide Group",
                ["organophosphorus", "pyrethroid", "chitin_synthesis_inhibitor",
                 "neonicotinoid", "phenylpyrazoles"])
        
        with col2:
            reading = st.number_input("Reading", value=50.0, min_value=0.0, step=0.1)
            limit = st.number_input("Limit", value=20.0, min_value=0.0, step=0.1)
            is_compliant = reading <= limit
            
            # Display compliance status
            if is_compliant:
                st.success("Compliance: ✅ Compliant")
            else:
                st.error("Compliance: ❌ Non-Compliant")
        
        # Submit button (INSIDE FORM)
        submitted = st.form_submit_button("🎯 Assess Risk", type="primary", use_container_width=True)
    
    # ✅ EVERYTHING BELOW IS OUTSIDE THE FORM
    
    # Handle form submission
    if submitted:
        risk_data = {
            "vegetable": vegetable,
            "vegetable_category": veg_category,
            "pesticide_group": pesticide_group,
            "reading": reading,
            "limit": limit,
            "is_compliant": is_compliant
        }
        
        with st.spinner("🔄 Performing enhanced risk assessment..."):
            try:
                result = api_client.assess_risk(risk_data)
                
                # ✅ FIX: Check if result exists and has risk_score (not 'success' field)
                if result and 'risk_score' in result:
                    st.success("✅ Risk assessment completed!")
                    display_enhanced_risk_result(result)  # Pass result directly, not result['data']
                elif result and result.get('success'):
                    # Handle wrapped response format
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
    
    # Debug section (OUTSIDE FORM)
    st.markdown("---")
    st.markdown("### 🐛 Debug")
    
    if st.button("Test API Connection"):
        with st.spinner("Testing..."):
            try:
                # Test health endpoint first
                health = api_client.health_check()
                if health:
                    st.success("✅ API is reachable")
                    st.json(health)
                else:
                    st.error("❌ API not responding")
                
                # Test risk endpoint with sample data
                test_data = {
                    "vegetable": "parsley",
                    "vegetable_category": "leafy_greens",
                    "pesticide_group": "organophosphorus",
                    "reading": 50.0,
                    "limit": 20.0,
                    "is_compliant": False
                }
                
                st.write("Testing risk assessment endpoint...")
                result = api_client.assess_risk(test_data)
                
                if result:
                    st.success("✅ Risk endpoint working")
                    st.json(result)
                else:
                    st.error("❌ Risk endpoint not responding")
                    st.warning("Make sure you have the risk_assessment route registered in main.py")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

def display_prediction_result(result: Dict):
    """Display prediction result"""
    
    st.success("✅ Prediction completed!")
    
    # Check if enhanced risk assessment is available
    if 'enhanced_risk_assessment' in result and result['enhanced_risk_assessment'].get('enabled'):
        display_enhanced_risk_result(result['enhanced_risk_assessment'])
    else:
        # Display basic results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = result.get('risk_assessment', {}).get('risk_level', 'Unknown')
            st.metric("Risk Level", risk_level)
        
        with col2:
            exceedance = result.get('sample_info', {}).get('exceedance_ratio', 0)
            st.metric("Exceedance Ratio", f"{exceedance:.2f}")
        
        with col3:
            compliance = result.get('predictions', {}).get('compliance', {}).get('prediction', 'Unknown')
            st.metric("Compliance", compliance)
        
        with st.expander("📋 Full API Response"):
            st.json(result)

def display_enhanced_risk_result(risk_result: Dict):
    """Display enhanced risk assessment results"""
    
    st.markdown("### 🎯 Enhanced Risk Assessment Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_score = risk_result.get('risk_score', 0)
        risk_level = risk_result.get('risk_level', 'Unknown')
        st.metric("Risk Score", f"{risk_score}/5.0", delta=risk_level)
    
    with col2:
        prob = risk_result.get('exceedance_probability', 0)
        st.metric("Exceedance Probability", f"{prob*100:.1f}%",
                 delta="High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low")
    
    with col3:
        bayesian = risk_result.get('bayesian_risk', 0)
        st.metric("Bayesian Risk", f"{bayesian:.2f}",
                 delta=f"×{risk_result.get('vegetable_multiplier', 1.0)} exposure")
    
    with col4:
        threshold = risk_result.get('critical_threshold', 0)
        st.metric("Critical Threshold", f"{threshold:.1f}")
    
    # Risk level banner
    risk_level = risk_result.get('risk_level', 'Unknown')
    
    if risk_level == 'CRITICAL':
        st.error(f"🔴 CRITICAL RISK: {prob*100:.1f}% probability of exceeding limits")
        st.warning("⚠️ **Immediate action required**: Do not distribute. Investigate contamination source.")
    elif risk_level == 'HIGH':
        st.warning(f"🟠 HIGH RISK: {prob*100:.1f}% probability of exceeding limits")
        st.info("ℹ️ **Recommendation**: Enhanced monitoring. Consider retesting before distribution.")
    elif risk_level == 'MEDIUM':
        st.info(f"🟡 MEDIUM RISK: {prob*100:.1f}% probability of exceeding limits")
    else:
        st.success(f"🟢 LOW RISK: {prob*100:.1f}% probability of exceeding limits")
    
    # Confidence interval
    ci = risk_result.get('confidence_interval', (0, 0))
    st.write(f"**95% Confidence Interval:** [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # Risk components breakdown
    with st.expander("📊 Risk Components Breakdown"):
        components = risk_result.get('components', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Risk Score Components:**")
            st.write(f"- Base Risk (Exceedance): {components.get('base_risk', 0):.2f}")
            st.write(f"- Probabilistic Risk (Monte Carlo): {components.get('probabilistic_risk', 0):.2f}")
            st.write(f"- Bayesian Component: {components.get('bayesian_component', 0):.2f}")
        
        with col2:
            st.write("**Modifiers:**")
            st.write(f"- Vegetable Hazard Multiplier: {risk_result.get('vegetable_multiplier', 1.0)}×")
            st.write(f"- Critical Threshold: {risk_result.get('critical_threshold', 4.0)}")
            st.write(f"- Method: {risk_result.get('method', 'N/A')}")

# ============================================================================
# PAGE: AI ASSISTANT (CHAT)
# ============================================================================

def show_chat_page(api_client):
    """AI Assistant chat page"""
    
    st.title("💬 AI Assistant")
    
    render_chat_sidebar(api_client)
    display_chat_messages()
    handle_chat_input(api_client)

def render_chat_sidebar(api_client):
    """Render chat-specific sidebar controls"""
    
    with st.sidebar:
        st.markdown("---")
        
        # Project Configuration
        st.markdown("## 🏢 Project Configuration")
        
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
    """Process user query"""
    
    with st.spinner("🔍 Analyzing query..."):
        
        # Extract year from query
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
        
        # Detect if risk assessment needed
        risk_keywords = ['risk', 'safety', 'safe', 'health', 'assess', 'eat', 'consume']
        include_risk = any(kw in query.lower() for kw in risk_keywords)
        
        if include_risk:
            st.success("🎯 Risk assessment enabled")
        
        result = api_client.local_answer(query=query, year=query_year, include_risk=include_risk)
        
        if result:
            display_query_results(result, query_year, include_risk)

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
    
    # ============================================================================
    # TAB 3: RISK HEATMAPS
    # ============================================================================
    
    with tab3:
        st.header("🗺️ Risk Heatmaps")
        
        st.info("""
        **Business Value:** Visual pattern recognition for contamination hotspots.
        """)
        
        # Calculate violation rates by combination
        heatmap_data = df.groupby(['vegetable_english', 'pesticide']).agg({
            'violation_numeric': ['sum', 'count', 'mean']
        }).reset_index()
        
        heatmap_data.columns = ['Vegetable', 'Pesticide', 'Violations', 'Tests', 'Violation_Rate']
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Combinations", len(heatmap_data))
        with col2:
            st.metric("Unique Vegetables", df['vegetable_english'].nunique())
        with col3:
            st.metric("Unique Pesticides", df['pesticide'].nunique())
        
        # Filter
        min_tests = st.slider("Minimum tests required:", 1, 10, 3)
        heatmap_filtered = heatmap_data[heatmap_data['Tests'] >= min_tests].copy()
        
        st.info(f"📊 Showing {len(heatmap_filtered)} combinations with at least {min_tests} tests")
        
        if len(heatmap_filtered) < 5:
            st.warning("⚠️ Not enough data. Lower the minimum tests requirement.")
            return
        
        # Get top items
        top_n = min(15, heatmap_filtered['Vegetable'].nunique())
        top_vegetables = heatmap_filtered.groupby('Vegetable')['Violations'].sum().nlargest(top_n).index
        top_pesticides = heatmap_filtered.groupby('Pesticide')['Violations'].sum().nlargest(top_n).index
        
        heatmap_top = heatmap_filtered[
            heatmap_filtered['Vegetable'].isin(top_vegetables) & 
            heatmap_filtered['Pesticide'].isin(top_pesticides)
        ]
        
        st.write(f"**Displaying top {len(top_vegetables)} vegetables × {len(top_pesticides)} pesticides**")
        
        # Create pivot
        pivot_violations = heatmap_top.pivot(
            index='Vegetable',
            columns='Pesticide',
            values='Violation_Rate'
        ).fillna(0)
        
        if not pivot_violations.empty:
            # Heatmap 1: Violation Rate
            st.subheader("🔥 Violation Rate Heatmap")
            
            fig_heatmap1 = px.imshow(
                pivot_violations,
                labels=dict(x="Pesticide", y="Vegetable", color="Violation Rate"),
                color_continuous_scale='Reds',
                aspect='auto',
                title=f'Violation Rate ({pivot_violations.shape[0]}×{pivot_violations.shape[1]})',
                zmin=0,
                zmax=1
            )
            
            fig_heatmap1.update_layout(
                height=max(400, 30 * len(pivot_violations.index)),
                margin=dict(l=150, r=50, t=80, b=150)
            )
            
            fig_heatmap1.update_xaxes(tickangle=45)
            st.plotly_chart(fig_heatmap1, use_container_width=True)
            
            # Heatmap 2: Total Violations
            st.subheader("📊 Total Violations Heatmap")
            
            pivot_total = heatmap_top.pivot(
                index='Vegetable',
                columns='Pesticide',
                values='Violations'
            ).fillna(0)
            
            fig_heatmap2 = px.imshow(
                pivot_total,
                labels=dict(x="Pesticide", y="Vegetable", color="Violations"),
                color_continuous_scale='YlOrRd',
                aspect='auto',
                title=f'Total Violations ({pivot_total.shape[0]}×{pivot_total.shape[1]})'
            )
            
            fig_heatmap2.update_layout(
                height=max(400, 30 * len(pivot_total.index)),
                margin=dict(l=150, r=50, t=80, b=150)
            )
            
            fig_heatmap2.update_xaxes(tickangle=45)
            st.plotly_chart(fig_heatmap2, use_container_width=True)
            
            # Top 5 table
            st.subheader("🎯 Top 5 Worst Combinations")
            top_5 = heatmap_top.nlargest(5, 'Violations').copy()
            top_5['Vegetable'] = top_5['Vegetable'].astype(str)
            top_5['Pesticide'] = top_5['Pesticide'].astype(str)
            top_5['Violations'] = top_5['Violations'].astype(float)
            top_5['Tests'] = top_5['Tests'].astype(float)
            top_5['Violation_Rate'] = top_5['Violation_Rate'].astype(float)
            
            st.dataframe(
                top_5[['Vegetable', 'Pesticide', 'Violations', 'Tests', 'Violation_Rate']]
                .style.format({
                    'Violation_Rate': '{:.1%}', 
                    'Violations': '{:.0f}', 
                    'Tests': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        # Seasonal heatmap
        st.divider()
        st.subheader("🌍 Seasonal Violation Patterns")
        
        seasonal_pivot = df.groupby(['season', 'vegetable_english'])['violation_numeric'].mean().reset_index()
        seasonal_pivot_wide = seasonal_pivot.pivot(
            index='vegetable_english',
            columns='season',
            values='violation_numeric'
        ).fillna(0)
        
        top_20_veg = df.groupby('vegetable_english')['violation_numeric'].sum().nlargest(20).index
        seasonal_pivot_wide = seasonal_pivot_wide.loc[seasonal_pivot_wide.index.isin(top_20_veg)]
        
        if not seasonal_pivot_wide.empty:
            fig_seasonal_heat = px.imshow(
                seasonal_pivot_wide,
                labels=dict(x="Season", y="Vegetable", color="Violation Rate"),
                color_continuous_scale='RdYlGn_r',
                aspect='auto',
                title=f'Seasonal Patterns (Top {len(seasonal_pivot_wide)} Vegetables)',
                zmin=0,
                zmax=1
            )
            
            fig_seasonal_heat.update_layout(
                height=max(400, 25 * len(seasonal_pivot_wide.index)),
                margin=dict(l=150, r=50, t=80, b=100)
            )
            
            st.plotly_chart(fig_seasonal_heat, use_container_width=True)
    
    # ============================================================================
    # TAB 4: TREND ANALYSIS
    # ============================================================================
    
    with tab4:
        st.header("📈 Trend Analysis")
        
        st.info("Temporal analysis of contamination patterns")
        
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