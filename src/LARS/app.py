"""
Predictive Analytics Dashboard with Enhanced Visual Style
Includes Statistical Insights and Enhanced Risk Assessment
"""

import streamlit as st
import pandas as pd
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime
import re

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

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

def main():
    """Main application entry point"""
    
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
    
    # Route to pages
    if page == "🏠 Home":
        show_home_page(api_client)
    elif page == "🔮 Predictions":
        show_predictions_page(api_client)
    elif page == "💬 AI Assistant":
        show_chat_page(api_client)
    elif page == "📊 Statistical Insights":
        show_statistical_insights_page(api_client)

@st.cache_resource
def get_api_client():
    """Get cached API client instance"""
    return APIClient()

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
        page = st.selectbox(
            "Select page:",
            ["🏠 Home", "💬 AI Assistant", "🔮 Predictions", "📊 Statistical Insights"],
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

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()