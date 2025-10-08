# # ============================================================================
# # FIXED STREAMLIT APP - app.py (Pydantic Compatible)
# # ============================================================================

# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# import requests
# import json
# import os
# from typing import Dict, Any, Optional, ClassVar

# # ============================================================================
# # ALTERNATIVE: Simple Configuration (Recommended for Streamlit)
# # ============================================================================

# # Even simpler approach - just use constants
# API_BASE_URL = "http://127.0.0.1"  # Fixed: removed port 8000
# API_HEADERS = {
#     "Content-Type": "application/json",
#     "Accept": "application/json",
# }

# ENDPOINTS = {
#     "health": "api/v1/dashboards/health",  # Fixed: correct health endpoint
#     "predict": "api/v1/predict",  # Updated with proper API path
#     "predict_batch": "api/v1/predict_batch", 
#     "predict_and_index": "api/v1/prediction/predict-and-index/1",
#     "dashboards_executive": "api/v1/dashboards/executive",  # Fixed path
#     "dashboards_laboratory": "api/v1/dashboards/laboratory",  # Fixed path
#     "dashboards_regulatory": "api/v1/dashboards/regulatory",  # Fixed path
#     "iso_upload_document": "api/v1/iso_upload_document",
#     "iso_upload_hierarchy": "api/v1/iso_upload_hierarchy",
#     "query": "api/v1/query",
#     "query_export": "api/v1/query_export",
#     "prediction_status": "api/v1/prediction_status",
#     "time_series_predict": "api/v1/time_series/predict",
#     "local_index": "api/v1/l/local/index",
#     "local_answer": "api/v1/local/answer",
# }

# # ============================================================================
# # API CLIENT (Fixed for Pydantic compatibility)
# # ============================================================================

# class APIClient:
#     """API Client for making requests to your prediction API"""
    
#     def __init__(self, base_url: str = None):
#         self.base_url = (base_url or API_BASE_URL).rstrip('/')
#         self.headers = API_HEADERS.copy()
#         self.session = requests.Session()
#         self.session.headers.update(self.headers)
    
#     def _make_request(self, method: str, endpoint: str, data: Any = None, 
#                      files: Dict = None, params: Dict = None) -> Optional[Dict]:
#         """Make HTTP request with error handling"""
#         url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
#         try:
#             if method.upper() == "GET":
#                 response = self.session.get(url, params=params, timeout=30)
#             elif method.upper() == "POST":
#                 if files:
#                     # For file uploads, don't set Content-Type header
#                     headers = {k: v for k, v in self.headers.items() 
#                               if k.lower() != 'content-type'}
#                     response = requests.post(url, data=data, files=files, 
#                                            headers=headers, timeout=30)
#                 else:
#                     response = self.session.post(url, json=data, timeout=30)
#             else:
#                 raise ValueError(f"Unsupported HTTP method: {method}")
            
#             response.raise_for_status()
            
#             # Try to return JSON, fallback to text
#             try:
#                 return response.json()
#             except json.JSONDecodeError:
#                 return {"response": response.text, "status_code": response.status_code}
                
#         except requests.exceptions.ConnectionError:
#             st.error(f"❌ Connection Error: Could not connect to {url}")
#             st.error("Please check if your API server is running and the URL is correct.")
#             return None
#         except requests.exceptions.Timeout:
#             st.error("⏱️ Request timed out. The server might be overloaded.")
#             return None
#         except requests.exceptions.HTTPError as e:
#             if hasattr(e, 'response'):
#                 status_code = e.response.status_code
#                 if status_code == 401:
#                     st.error("🔐 Authentication Error: Invalid API credentials")
#                 elif status_code == 403:
#                     st.error("🚫 Access Forbidden: You don't have permission for this endpoint")
#                 elif status_code == 404:
#                     st.error(f"🔍 Endpoint Not Found: {url}")
#                     st.error("Please check if the endpoint URL is correct.")
#                 elif status_code == 422:
#                     st.error("📋 Validation Error: Check your request data format")
#                     try:
#                         error_detail = e.response.json()
#                         st.json(error_detail)
#                     except:
#                         st.text(e.response.text)
#                 elif status_code == 500:
#                     st.error("🔥 Server Error: Something went wrong on the server side")
#                 else:
#                     st.error(f"❌ HTTP Error {status_code}: {e}")
#             return None
#         except Exception as e:
#             st.error(f"💥 Unexpected error: {str(e)}")
#             return None
    
#     # API Methods
#     def health_check(self) -> Optional[Dict]:
#         """Check API health status"""
#         return self._make_request("GET", ENDPOINTS["health"])
    
#     def get_prediction_status(self) -> Optional[Dict]:
#         """Get prediction system status"""
#         return self._make_request("GET", ENDPOINTS["prediction_status"])
    
#     def get_dashboard_executive(self) -> Optional[Dict]:
#         """Get executive dashboard data"""
#         return self._make_request("GET", ENDPOINTS["dashboards_executive"])
    
#     def get_dashboard_laboratory(self) -> Optional[Dict]:
#         """Get laboratory dashboard data"""
#         return self._make_request("GET", ENDPOINTS["dashboards_laboratory"])
    
#     def get_dashboard_regulatory(self) -> Optional[Dict]:
#         """Get regulatory dashboard data"""
#         return self._make_request("GET", ENDPOINTS["dashboards_regulatory"])
    
#     def predict_single(self, prediction_data: Dict) -> Optional[Dict]:
#         """Run single prediction"""
#         return self._make_request("POST", ENDPOINTS["predict"], data=prediction_data)
    
#     def predict_and_index(self, data: Dict) -> Optional[Dict]:
#         """Run prediction and index results"""
#         return self._make_request("POST", ENDPOINTS["predict_and_index"], data=data)
    
#     def predict_batch(self, batch_data: Dict) -> Optional[Dict]:
#         """Run batch predictions"""
#         return self._make_request("POST", ENDPOINTS["predict_batch"], data=batch_data)
    
#     def upload_document(self, file_obj, filename: str = None) -> Optional[Dict]:
#         """Upload document file"""
#         if filename is None:
#             filename = getattr(file_obj, 'name', 'uploaded_file')
        
#         files = {'file': (filename, file_obj, 'application/octet-stream')}
#         return self._make_request("POST", ENDPOINTS["iso_upload_document"], files=files)
    
#     def upload_hierarchy(self, file_obj, filename: str = None) -> Optional[Dict]:
#         """Upload hierarchy file"""
#         if filename is None:
#             filename = getattr(file_obj, 'name', 'uploaded_file')
            
#         files = {'file': (filename, file_obj, 'application/octet-stream')}
#         return self._make_request("POST", ENDPOINTS["iso_upload_hierarchy"], files=files)
    
#     def execute_query(self, query_data: Dict) -> Optional[Dict]:
#         """Execute data query"""
#         return self._make_request("POST", ENDPOINTS["query"], data=query_data)
    
#     def export_query(self, export_params: Dict = None) -> Optional[Dict]:
#         """Export query results"""
#         data = export_params or {}
#         return self._make_request("POST", ENDPOINTS["query_export"], data=data)

# # ============================================================================
# # STREAMLIT APP
# # ============================================================================

# # Page configuration
# st.set_page_config(
#     page_title="🔮 Predictive Analytics Dashboard",
#     page_icon="🔮",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ============================================================================
# # MAIN APP
# # ============================================================================

# def main():
#     """Main Streamlit application"""
    
#     # Initialize API client
#     @st.cache_resource
#     def get_api_client():
#         return APIClient()
    
#     api_client = get_api_client()
    
#     # App header
#     st.title("🔮 Predictive Analytics Dashboard")
#     st.markdown("---")
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("## 🔗 API Configuration")
        
#         # Show current API URL with option to change it
#         current_url = st.text_input("API Base URL", value=API_BASE_URL, key="api_url")
        
#         if current_url != API_BASE_URL:
#             # Create new client with updated URL
#             api_client = APIClient(current_url)
        
#         st.markdown("---")
        
#         # API Status Check
#         if st.button("🔄 Check API Connection", key="health_check"):
#             with st.spinner("Checking API connection..."):
#                 health_status = api_client.health_check()
#                 if health_status:
#                     st.success("✅ API Connected")
#                     with st.expander("API Response"):
#                         st.json(health_status)
#                 else:
#                     st.error("❌ API Disconnected")
#                     st.markdown("""
#                     **Troubleshooting:**
#                     1. Make sure `docker compose up` is running
#                     2. Check if API is on port 8000: `curl http://localhost:8000/health`
#                     3. Verify API endpoints are available
#                     4. Check Docker container logs: `docker compose logs`
#                     """)
        
#         st.markdown("---")
        
#         # Navigation
#         page = st.selectbox(
#             "📍 Navigate to:",
#             ["🏠 Home", "📊 Dashboard", "🔮 Predictions", "📁 Data Management", "💬 AI Assistant"]
#         )
    
#     # ========================================================================
#     # PAGE ROUTING
#     # ========================================================================
    
#     if page == "🏠 Home":
#         show_home_page(api_client)
#     elif page == "📊 Dashboard":
#         show_dashboard_page(api_client)
#     elif page == "🔮 Predictions":
#         show_predictions_page(api_client)
#     elif page == "📁 Data Management":
#         show_data_management_page(api_client)
#     elif page == "💬 AI Assistant":
#         show_chat_page()

# # ============================================================================
# # PAGE FUNCTIONS
# # ============================================================================

# def show_home_page(api_client):
#     """Home page with overview and quick actions"""
    
#     # Welcome section
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("""
#         ## Welcome to Your Predictive Analytics Dashboard! 
        
#         This application connects to your prediction API endpoints and provides:
        
#         - 📊 **Real-time Dashboards** - Executive, Laboratory, and Regulatory views
#         - 🔮 **Prediction Engine** - Single predictions, batch processing, and time series
#         - 📁 **Data Management** - File uploads and query processing
#         - 💬 **AI Assistant** - Get help with your data and predictions
        
#         ### 🚀 Getting Started:
#         1. Make sure your API server is running: `docker compose up`
#         2. Click "Check API Connection" in the sidebar
#         3. Start making predictions!
#         """)
    
#     with col2:
#         st.markdown("### 🚀 Quick Actions")
        
#         if st.button("🔮 Quick Prediction", use_container_width=True):
#             st.session_state.quick_predict = True
        
#         if st.button("📊 View Dashboard", use_container_width=True):
#             st.session_state.page = "📊 Dashboard"
#             st.rerun()
        
#         if st.button("📤 Upload Data", use_container_width=True):
#             st.session_state.page = "📁 Data Management"
#             st.rerun()
    
#     # Quick prediction form
#     if st.session_state.get('quick_predict', False):
#         st.markdown("---")
#         st.markdown("### 🔮 Quick Prediction")
        
#         with st.form("quick_prediction"):
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 vegetable = st.text_input("Vegetable", value="pepper")
#                 pesticide_group = st.selectbox("Pesticide Group", 
#                     ["organophosphorus", "carbamates", "pyrethroids"])
            
#             with col2:
#                 reading = st.number_input("Reading", value=45.8, min_value=0.0)
#                 limits = st.number_input("Limits", value=10.0, min_value=0.0)
            
#             with col3:
#                 sample_code = st.text_input("Sample Code", value="TEST002")
            
#             if st.form_submit_button("🚀 Run Prediction", type="primary"):
#                 prediction_data = {
#                     "vegetable": vegetable,
#                     "pesticide_group": pesticide_group,
#                     "reading": reading,
#                     "limits": limits,
#                     "sample_code": sample_code
#                 }
                
#                 with st.spinner("Running prediction..."):
#                     result = api_client.predict_single(prediction_data)
                    
#                     if result:
#                         st.success("✅ Prediction completed!")
                        
#                         # Show result in a nice format
#                         col1, col2 = st.columns([1, 2])
                        
#                         with col1:
#                             # Simulate result analysis
#                             risk_ratio = reading / limits
#                             if risk_ratio <= 1:
#                                 st.success("🟢 **SAFE**")
#                                 risk_level = "Low Risk"
#                             else:
#                                 st.error("🔴 **UNSAFE**")
#                                 risk_level = "High Risk"
                            
#                             st.metric("Risk Ratio", f"{risk_ratio:.2f}")
#                             st.metric("Risk Level", risk_level)
                        
#                         with col2:
#                             with st.expander("📋 Full API Response"):
#                                 st.json(result)
    
#     # System status
#     st.markdown("---")
#     st.markdown("### 📊 System Overview")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         # Test connection status
#         try:
#             status = api_client.health_check()
#             if status:
#                 st.metric("API Status", "🟢 Online", delta="Healthy")
#             else:
#                 st.metric("API Status", "🔴 Offline", delta="Check Connection")
#         except:
#             st.metric("API Status", "⚠️ Unknown", delta="Test Connection")
    
#     with col2:
#         st.metric("Total Endpoints", "10+", delta="Available")
    
#     with col3:
#         st.metric("Last Check", "Just now", delta="Active")

# def show_dashboard_page(api_client):
#     """Dashboard page with executive, laboratory, and regulatory views"""
    
#     st.title("📊 Dashboard")
    
#     # Dashboard type selector
#     dashboard_type = st.selectbox(
#         "Select Dashboard Type",
#         ["Executive", "Laboratory", "Regulatory"]
#     )
    
#     col1, col2 = st.columns([3, 1])
    
#     with col2:
#         if st.button("🔄 Refresh Dashboard", type="primary"):
#             with st.spinner(f"Loading {dashboard_type.lower()} dashboard..."):
#                 if dashboard_type == "Executive":
#                     dashboard_data = api_client.get_dashboard_executive()
#                 elif dashboard_type == "Laboratory":
#                     dashboard_data = api_client.get_dashboard_laboratory()
#                 else:
#                     dashboard_data = api_client.get_dashboard_regulatory()
                
#                 if dashboard_data:
#                     st.session_state.dashboard_data = dashboard_data
#                     st.success("Dashboard updated!")
    
#     # Show dashboard content
#     if 'dashboard_data' in st.session_state:
#         data = st.session_state.dashboard_data
        
#         # Metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Predictions", data.get('total_predictions', 'N/A'))
#         with col2:
#             st.metric("Success Rate", f"{data.get('success_rate', 0)}%")
#         with col3:
#             st.metric("Active Models", data.get('active_models', 'N/A'))
#         with col4:
#             st.metric("Queue Size", data.get('queue_size', 'N/A'))
        
#         # Raw data
#         with st.expander("🔍 Raw Dashboard Data"):
#             st.json(data)
#     else:
#         st.info("👆 Click 'Refresh Dashboard' to load data from your API")

# def show_predictions_page(api_client):
#     """Predictions page with various prediction types"""
    
#     st.title("🔮 Predictions")
    
#     # Prediction tabs
#     tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Predict & Index"])
    
#     with tab1:
#         st.markdown("### Single Prediction")
        
#         with st.form("single_prediction"):
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 vegetable = st.text_input("Vegetable", value="pepper")
#                 pesticide_group = st.selectbox("Pesticide Group", 
#                     ["organophosphorus", "carbamates", "pyrethroids", "neonicotinoids"])
#                 reading = st.number_input("Reading", value=45.8, min_value=0.0)
            
#             with col2:
#                 limits = st.number_input("Limits", value=10.0, min_value=0.0)
#                 sample_code = st.text_input("Sample Code", value="TEST002")
            
#             if st.form_submit_button("🚀 Run Prediction", type="primary"):
#                 prediction_data = {
#                     "vegetable": vegetable,
#                     "pesticide_group": pesticide_group,
#                     "reading": reading,
#                     "limits": limits,
#                     "sample_code": sample_code
#                 }
                
#                 with st.spinner("Running prediction..."):
#                     result = api_client.predict_single(prediction_data)
                    
#                     if result:
#                         st.success("✅ Prediction completed!")
#                         st.json(result)
    
#     with tab2:
#         st.markdown("### Batch Prediction")
        
#         uploaded_file = st.file_uploader(
#             "Upload batch prediction file", 
#             type=['csv', 'json', 'xlsx'],
#             help="Upload a file containing multiple samples for batch prediction"
#         )
        
#         if uploaded_file:
#             if st.button("🚀 Run Batch Prediction", type="primary"):
#                 with st.spinner("Processing batch predictions..."):
#                     # You might need to convert the file format
#                     batch_data = {"file_name": uploaded_file.name}
#                     result = api_client.predict_batch(batch_data)
                    
#                     if result:
#                         st.success("✅ Batch prediction completed!")
#                         st.json(result)
    
#     with tab3:
#         st.markdown("### Predict & Index")
#         st.info("This uses your `predict-and-index` endpoint")
        
#         with st.form("predict_index_form"):
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 vegetable = st.text_input("Vegetable", value="pepper", key="pi_veg")
#                 pesticide_group = st.selectbox("Pesticide Group", 
#                     ["organophosphorus", "carbamates", "pyrethroids"], key="pi_pest")
#                 reading = st.number_input("Reading", value=45.8, key="pi_reading")
            
#             with col2:
#                 limits = st.number_input("Limits", value=10.0, key="pi_limits")
#                 sample_code = st.text_input("Sample Code", value="TEST002", key="pi_sample")
            
#             if st.form_submit_button("🚀 Predict & Index", type="primary"):
#                 data = {
#                     "vegetable": vegetable,
#                     "pesticide_group": pesticide_group,
#                     "reading": reading,
#                     "limits": limits,
#                     "sample_code": sample_code
#                 }
                
#                 with st.spinner("Running prediction and indexing..."):
#                     result = api_client.predict_and_index(data)
                    
#                     if result:
#                         st.success("✅ Prediction and indexing completed!")
#                         st.json(result)

# def show_data_management_page(api_client):
#     """Data management page for file uploads and queries"""
    
#     st.title("📁 Data Management")
    
#     # File upload section
#     st.markdown("### 📤 Upload Files")
    
#     upload_type = st.selectbox(
#         "Select Upload Type",
#         ["Document", "Hierarchy"]
#     )
    
#     uploaded_file = st.file_uploader(
#         "Choose a file",
#         type=['json', 'csv', 'xlsx', 'txt', 'pdf']
#     )
    
#     if uploaded_file and st.button("📤 Upload File", type="primary"):
#         with st.spinner(f"Uploading {upload_type.lower()}..."):
#             if upload_type == "Document":
#                 result = api_client.upload_document(uploaded_file)
#             else:  # Hierarchy
#                 result = api_client.upload_hierarchy(uploaded_file)
            
#             if result:
#                 st.success("✅ File uploaded successfully!")
#                 st.json(result)
    
#     st.markdown("---")
    
#     # Query section
#     st.markdown("### 🔍 Data Query")
    
#     query_text = st.text_area("Enter your query", height=100)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         if st.button("▶️ Execute Query", type="primary"):
#             if query_text:
#                 query_data = {"query": query_text}
                
#                 with st.spinner("Executing query..."):
#                     result = api_client.execute_query(query_data)
                    
#                     if result:
#                         st.success("✅ Query executed!")
#                         st.session_state.query_result = result
#                         st.json(result)
    
#     with col2:
#         if st.button("📥 Export Results") and 'query_result' in st.session_state:
#             result = api_client.export_query()
#             if result:
#                 st.download_button(
#                     label="⬇️ Download Export",
#                     data=json.dumps(result, indent=2),
#                     file_name="query_export.json",
#                     mime="application/json"
#                 )

# def show_chat_page():
#     """AI Assistant chat page"""
    
#     st.title("💬 AI Assistant")
    
#     # Initialize chat history
#     if 'chat_messages' not in st.session_state:
#         st.session_state.chat_messages = [
#             {"role": "assistant", "content": "Hello! I'm your AI assistant. I can help you with predictions, data analysis, and API troubleshooting. What would you like to know?"}
#         ]
    
#     # Display chat messages
#     for message in st.session_state.chat_messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about predictions, data, or analysis..."):
#         st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.write(prompt)
        
#         # AI response (basic implementation)
#         with st.chat_message("assistant"):
#             response = f"I understand you're asking about: '{prompt}'. This is a basic response - you can integrate with OpenAI API or other AI services here."
#             st.write(response)
#             st.session_state.chat_messages.append({"role": "assistant", "content": response})

# # ============================================================================
# # RUN THE APP
# # ============================================================================

# if __name__ == "__main__":
#     main()



#### working code below ---



# """
# Predictive Analytics Dashboard with Local NLP Chatbot
# Clean and organized structure
# """

# import streamlit as st
# import pandas as pd
# import requests
# import json
# from typing import Dict, Any, Optional
# from datetime import datetime
# import re

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================

# st.set_page_config(
#     page_title="Predictive Analytics Dashboard",
#     page_icon="🔮",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ============================================================================
# # CONSTANTS
# # ============================================================================

# API_BASE_URL = "http://127.0.0.1:8000"
# API_HEADERS = {
#     "Content-Type": "application/json",
#     "Accept": "application/json",
# }

# ENDPOINTS = {
#     "health": "api/v1/dashboards/health",
#     "predict": "api/v1/predict",
#     "local_index": "api/v1/nlp/local/index",
#     "local_answer": "api/v1/nlp/local/answer",
# }

# AVAILABLE_YEARS = ["2022", "2023", "2024", "2025"]
# DEFAULT_YEAR = "2022"  # Changed from 2025 to 2022

# # ============================================================================
# # API CLIENT CLASS
# # ============================================================================

# class APIClient:
#     """Handles all API communication"""
    
#     def __init__(self, base_url: str = API_BASE_URL):
#         self.base_url = base_url.rstrip('/')
#         self.headers = API_HEADERS.copy()
#         self.session = requests.Session()
#         self.session.headers.update(self.headers)
    
#     def _make_request(self, method: str, endpoint: str, 
#                      data: Any = None, params: Dict = None) -> Optional[Dict]:
#         """Make HTTP request with error handling"""
#         url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
#         try:
#             if method.upper() == "GET":
#                 response = self.session.get(url, params=params, timeout=120)
#             elif method.upper() == "POST":
#                 response = self.session.post(url, json=data, params=params, timeout=120)  # Added params=params
#             else:
#                 raise ValueError(f"Unsupported method: {method}")
            
#             response.raise_for_status()
#             return response.json()
            
#         except requests.exceptions.ConnectionError:
#             st.error(f"Connection Error: Could not connect to {url}")
#             return None
#         except requests.exceptions.HTTPError as e:
#             st.error(f"HTTP Error: {e}")
#             if hasattr(e.response, 'text'):
#                 with st.expander("Error Details"):
#                     st.text(e.response.text)
#             return None
#         except Exception as e:
#             st.error(f"Unexpected error: {str(e)}")
#             return None
    
#     def health_check(self) -> Optional[Dict]:
#         """Check API health"""
#         return self._make_request("GET", ENDPOINTS["health"])
    
#     def predict_single(self, prediction_data: Dict) -> Optional[Dict]:
#         """Run single prediction"""
#         return self._make_request("POST", ENDPOINTS["predict"], data=prediction_data)
    
#     def local_index(self, year: str, project_id: int = 1) -> Optional[Dict]:
#         """Index data for a specific year"""
#         endpoint = f"{ENDPOINTS['local_index']}/{project_id}"
#         data = {
#             "year": year,
#             "data_folder_path": "/app/data",
#             "months": [],
#             "do_reset": False
#         }
#         return self._make_request("POST", endpoint, data=data)
    
#     def local_answer(self, query: str, year: str, project_id: int = 1,
#                     n_results: int = 15, include_risk: bool = True) -> Optional[Dict]:
#         """Query indexed data"""
#         endpoint = f"{ENDPOINTS['local_answer']}/{project_id}"
#         data = {
#             "query": query,
#             "n_results": n_results,
#             "include_risk_assessment": include_risk,
#             "population_type": "adult_average"
#             # Don't include year in body - only in params
#         }
#         params = {"year": year}
#         return self._make_request("POST", endpoint, data=data, params=params)

# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def extract_year_from_text(text: str) -> Optional[str]:
#     """Extract year from user query"""
#     # First, look for 4-digit years
#     import re
#     year_match = re.search(r'\b(202[2-5])\b', text)
#     if year_match:
#         return year_match.group(1)
    
#     # Then check if year is mentioned as word
#     for year in AVAILABLE_YEARS:
#         if year in text:
#             return year
    
#     text_lower = text.lower()
#     current_year = datetime.now().year
    
#     if "last year" in text_lower or "previous year" in text_lower:
#         year = str(current_year - 1)
#         return year if year in AVAILABLE_YEARS else None
    
#     if "this year" in text_lower or "current year" in text_lower:
#         year = str(current_year)
#         return year if year in AVAILABLE_YEARS else None
    
#     return None

# def format_risk_assessment(risk_data: Dict) -> str:
#     """Format risk assessment data as markdown"""
#     if not risk_data:
#         return ""
    
#     output = "\n\n### Risk Assessment Summary\n\n"
#     output += f"**Total Samples:** {risk_data.get('total_samples_assessed', 0)}\n"
#     output += f"**Population Type:** {risk_data.get('population_type', 'N/A')}\n\n"
    
#     results = risk_data.get('assessment_results', [])
#     for idx, result in enumerate(results, 1):
#         status = "Safe" if result.get('risk_level') == 'acceptable' else "Unsafe"
#         output += f"#### Sample {idx}: {result.get('sample_code', 'N/A')} - {status}\n"
#         output += f"- **Pesticide:** {result.get('pesticide_name', 'N/A')}\n"
#         output += f"- **Food Item:** {result.get('food_item', 'N/A')}\n"
#         output += f"- **Risk:** {result.get('risk_percentage', 0):.2f}%\n\n"
    
#     return output

# # ============================================================================
# # SESSION STATE INITIALIZATION
# # ============================================================================

# def initialize_session_state():
#     """Initialize all session state variables"""
#     if 'chat_messages' not in st.session_state:
#         st.session_state.chat_messages = [{
#             "role": "assistant",
#             "content": """Hello! I'm your pesticide analysis AI assistant.

# **What I can help you with:**
# - Query pesticide data by year
# - Find samples exceeding limits
# - Perform risk assessments
# - Analyze trends across years

# **Example queries:**
# - "Find pyridaben exceeding limits in 2023"
# - "Show me tomato samples from 2024"
# - "Assess health risk for carbamates in 2022"

# Ask your question to get started!"""
#         }]
    
#     if 'selected_year' not in st.session_state:
#         st.session_state.selected_year = DEFAULT_YEAR
    
#     if 'indexed_years' not in st.session_state:
#         st.session_state.indexed_years = set()

# # ============================================================================
# # MAIN APPLICATION
# # ============================================================================

# def main():
#     """Main application entry point"""
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Get API client
#     api_client = get_api_client()
    
#     # App header
#     st.title("Predictive Analytics Dashboard")
#     st.markdown("---")
    
#     # Sidebar
#     render_sidebar(api_client)
    
#     # Get selected page from sidebar
#     page = st.session_state.get('current_page', 'AI Assistant')
    
#     # Route to appropriate page
#     if page == "Home":
#         show_home_page(api_client)
#     elif page == "Predictions":
#         show_predictions_page(api_client)
#     elif page == "AI Assistant":
#         show_chat_page(api_client)

# @st.cache_resource
# def get_api_client():
#     """Get cached API client instance"""
#     return APIClient()

# # ============================================================================
# # SIDEBAR
# # ============================================================================

# def render_sidebar(api_client):
#     """Render sidebar with navigation and settings"""
    
#     with st.sidebar:
#         st.markdown("## Configuration")
        
#         # API connection check
#         if st.button("Check API Connection", use_container_width=True):
#             with st.spinner("Checking..."):
#                 health = api_client.health_check()
#                 if health:
#                     st.success("API Connected")
#                 else:
#                     st.error("API Disconnected")
        
#         st.markdown("---")
        
#         # Navigation
#         st.markdown("## Navigation")
#         page = st.selectbox(
#             "Go to:",
#             ["AI Assistant", "Predictions", "Home"],
#             key="current_page"
#         )

# # ============================================================================
# # PAGE: HOME
# # ============================================================================

# def show_home_page(api_client):
#     """Home page with overview"""
    
#     st.markdown("""
#     ## Welcome to Predictive Analytics Dashboard
    
#     This application provides:
    
#     **Features:**
#     - Real-time pesticide predictions
#     - Year-based data querying
#     - AI-powered natural language queries
#     - Health risk assessments
    
#     **Getting Started:**
#     1. Ensure API server is running
#     2. Navigate to "AI Assistant"
#     3. Index a year's data
#     4. Start asking questions
#     """)

# # ============================================================================
# # PAGE: PREDICTIONS
# # ============================================================================

# def show_predictions_page(api_client):
#     """Predictions page"""
    
#     st.title("Single Prediction")
    
#     with st.form("prediction_form"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             vegetable = st.text_input("Vegetable", value="pepper")
#             pesticide = st.selectbox("Pesticide Group", 
#                 ["organophosphorus", "carbamates", "pyrethroids"])
#             reading = st.number_input("Reading", value=45.8, min_value=0.0)
        
#         with col2:
#             limits = st.number_input("Limits", value=10.0, min_value=0.0)
#             sample_code = st.text_input("Sample Code", value="TEST002")
        
#         if st.form_submit_button("Run Prediction", type="primary"):
#             data = {
#                 "vegetable": vegetable,
#                 "pesticide_group": pesticide,
#                 "reading": reading,
#                 "limits": limits,
#                 "sample_code": sample_code
#             }
            
#             with st.spinner("Running prediction..."):
#                 result = api_client.predict_single(data)
#                 if result:
#                     st.success("Prediction completed")
#                     st.json(result)

# # ============================================================================
# # PAGE: AI ASSISTANT (CHAT)
# # ============================================================================

# def show_chat_page(api_client):
#     """AI Assistant chat page"""
    
#     st.title("AI Assistant")
    
#     # Sidebar for chat settings
#     render_chat_sidebar(api_client)
    
#     # Display chat history
#     display_chat_messages()
    
#     # Chat input
#     handle_chat_input(api_client)

# def render_chat_sidebar(api_client):
#     """Render chat-specific sidebar controls"""
    
#     with st.sidebar:
#         st.markdown("---")
#         st.markdown("## Data Context")
        
#         # Current year display
#         st.info(f"**Active Year:** {st.session_state.selected_year}")
        
#         # Year selector
#         selected_year = st.selectbox(
#             "Default Year",
#             AVAILABLE_YEARS,
#             index=AVAILABLE_YEARS.index(st.session_state.selected_year)
#         )
        
#         if selected_year != st.session_state.selected_year:
#             st.session_state.selected_year = selected_year
        
#         st.markdown("---")
#         st.markdown("## Data Indexing")
        
#         # Year to index selector
#         year_to_index = st.selectbox("Select Year to Index", AVAILABLE_YEARS)
        
#         # Index button
#         # Add reset button
#         if st.button("🗑️ Delete & Re-index", use_container_width=True, type="secondary"):
#             with st.spinner(f"Deleting and re-indexing {year_to_index}..."):
#                 # Call index with do_reset=True to delete existing collection
#                 result = api_client.local_index(year_to_index, project_id=1)
                
#                 # Modify the request to include do_reset
#                 endpoint = f"{api_client.base_url}/{ENDPOINTS['local_index']}/1"
#                 data = {
#                     "year": year_to_index,
#                     "data_folder_path": "/app/data",
#                     "months": [],
#                     "do_reset": True  # Force delete and recreate
#                 }
                
#                 import requests
#                 result = requests.post(endpoint, json=data, timeout=120)
                
#                 if result.ok:
#                     response = result.json()
#                     if response.get('signal') == 'LOCAL_INDEX_SUCCESS':
#                         count = response.get('indexed_documents', 0)
#                         if count > 0:
#                             st.session_state.indexed_years.add(year_to_index)
#                             st.success(f"✅ Deleted and re-indexed {year_to_index}: {count} documents")
#                             st.json(response)
#                         else:
#                             st.error(f"⚠️ No documents found for {year_to_index}")
#                             st.json(response)
#                     else:
#                         st.error("Indexing failed")
#                         st.json(response)
#                 else:
#                     st.error(f"Request failed: {result.status_code}")
#                     st.text(result.text)
        
#         # Add reset button
#         if st.button("🗑️ Reset & Re-index", use_container_width=True, type="secondary"):
#             with st.spinner(f"Resetting and re-indexing {year_to_index}..."):
#                 # Call with do_reset=True
#                 endpoint = f"{api_client.base_url}/{ENDPOINTS['local_index']}/{1}"
#                 data = {
#                     "year": year_to_index,
#                     "data_folder_path": "/app/data",
#                     "months": [],
#                     "do_reset": True  # This deletes existing collection
#                 }
#                 result = requests.post(endpoint, json=data, timeout=60)
                
#                 if result.ok and result.json().get('signal') == 'LOCAL_INDEX_SUCCESS':
#                     count = result.json().get('indexed_documents', 0)
#                     if count > 0:
#                         st.session_state.indexed_years.add(year_to_index)
#                         st.success(f"✅ {year_to_index} reset and re-indexed: {count} documents")
#                     else:
#                         st.error(f"⚠️ No documents found for {year_to_index}")
#                 else:
#                     st.error(f"Failed to reset {year_to_index}")
#                     if result.ok:
#                         st.json(result.json())
        
#         # Show indexed years
#         if st.session_state.indexed_years:
#             st.markdown("**Indexed Years:**")
#             for year in sorted(st.session_state.indexed_years):
#                 st.markdown(f"- {year}")

# def index_year_data(api_client, year: str):
#     """Index data for a specific year"""
#     with st.spinner(f"Indexing {year} data..."):
#         result = api_client.local_index(year)
        
#         if result and result.get('signal') == 'LOCAL_INDEX_SUCCESS':
#             count = result.get('indexed_documents', 0)
#             if count > 0:
#                 st.session_state.indexed_years.add(year)
#                 st.success(f"{year} indexed: {count} documents")
#             else:
#                 st.error(f"No documents found for {year}")
#                 st.json(result)
#         else:
#             st.error(f"Failed to index {year}")
#             if result:
#                 st.json(result)

# def display_chat_messages():
#     """Display all chat messages"""
#     for message in st.session_state.chat_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# def handle_chat_input(api_client):
#     """Handle user chat input"""
    
#     if prompt := st.chat_input("Ask about pesticide data..."):
#         # Add user message
#         st.session_state.chat_messages.append({
#             "role": "user",
#             "content": prompt
#         })
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Process query
#         with st.chat_message("assistant"):
#             process_query(api_client, prompt)

# def process_query(api_client, query: str):
#     """Process user query and display results"""
    
#     with st.spinner("Analyzing data..."):
#         # Extract or use default year
#         query_year = extract_year_from_text(query)
        
#         # DEBUG: Show what year was extracted
#         st.info(f"🔍 Extracted year from query: {query_year}")
        
#         if not query_year:
#             query_year = st.session_state.selected_year
#             st.info(f"⚙️ Using default year: {query_year}")
#         else:
#             st.success(f"✅ Using year from query: {query_year}")
        
#         # Check if year is indexed
#         if query_year not in st.session_state.indexed_years:
#             response = f"Data for {query_year} is not indexed yet. Please index it first."
#             st.warning(response)
#             st.session_state.chat_messages.append({
#                 "role": "assistant",
#                 "content": response
#             })
#             return
        
#         # Determine if risk assessment needed
#         include_risk = any(word in query.lower() 
#                           for word in ["risk", "safety", "health"])
        
#         # Query the API
#         result = api_client.local_answer(
#             query=query,
#             year=query_year,
#             n_results=1000,
#             include_risk=include_risk
#         )
        
#         if not result:
#             st.error("Failed to get results from API")
#             return
        
#         # Display results
#         display_query_results(result, query_year, include_risk)

# def display_query_results(result: Dict, year: str, include_risk: bool):
#     """Display query results in a structured format"""
    
#     # Main answer
#     answer = result.get('answer', 'No results found')
#     st.markdown(answer)
    
#     # Metrics
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Retrieved Documents", result.get('retrieved_docs_count', 0))
#     with col2:
#         st.metric("Collection Total", result.get('collection_total', 0))
    
#     # DataFrame results
#     data_frame = result.get('data_frame', [])
#     if data_frame:
#         st.markdown("---")
#         st.markdown("### Interactive Data Table")
#         df = pd.DataFrame(data_frame)
#         st.dataframe(df, use_container_width=True)
        
#         # Individual results
#         st.markdown("### Individual Results")
#         display_individual_results(data_frame)
    
#     # Risk assessment
#     if include_risk and result.get('risk_assessment'):
#         st.markdown("---")
#         risk_text = format_risk_assessment(result['risk_assessment'])
#         st.markdown(risk_text)
    
#     # Download options
#     st.markdown("---")
#     display_download_options(result, year, data_frame)
    
#     # Save to chat history
#     st.session_state.chat_messages.append({
#         "role": "assistant",
#         "content": f"Analysis completed for {year}",
#         "data": result
#     })

# def display_individual_results(data_frame: list):
#     """Display individual result cards"""
#     for idx, record in enumerate(data_frame, 1):
#         with st.expander(f"Result {idx}: {record.get('Sample Code', 'N/A')} - {record.get('Pesticide Name', 'N/A')}"):
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.write(f"**Sample Code:** {record.get('Sample Code', 'N/A')}")
#                 st.write(f"**Sample Name:** {record.get('Sample Name', 'N/A')}")
#                 st.write(f"**Pesticide:** {record.get('Pesticide Name', 'N/A')}")
            
#             with col2:
#                 st.write(f"**Reading:** {record.get('Reading Value', 'N/A')}")
#                 st.write(f"**Limit:** {record.get('Limit', 'N/A')}")
                
#                 result_status = record.get('Result', 'N/A')
#                 if 'مطابق' in str(result_status) or 'Safe' in str(result_status):
#                     st.success(f"**Result:** {result_status}")
#                 else:
#                     st.error(f"**Result:** {result_status}")


# def display_download_options(result: Dict, year: str, data_frame: list):
#     """Display download buttons for results"""
#     col1, col2 = st.columns(2)
    
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
#     with col1:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(result, indent=2, ensure_ascii=False),
#             file_name=f"query_{year}_{timestamp}.json",
#             mime="application/json",
#             use_container_width=True
#         )
    
#     with col2:
#         if data_frame:
#             csv_data = pd.DataFrame(data_frame).to_csv(index=False)
#             st.download_button(
#                 label="Download CSV",
#                 data=csv_data,
#                 file_name=f"query_{year}_{timestamp}.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )


# # ============================================================================
# # RUN APPLICATION
# # ============================================================================

# if __name__ == "__main__":
#     main()





"""
Predictive Analytics Dashboard with Enhanced Visual Style
Combines clean structure with colorful, emoji-rich interface
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

def format_risk_assessment(risk_data: Dict) -> str:
    """Format risk assessment data as markdown"""
    if not risk_data:
        return ""
    
    output = "\n\n### 🎯 Risk Assessment Summary\n\n"
    output += f"**Total Samples:** {risk_data.get('total_samples_assessed', 0)}\n"
    output += f"**Population Type:** {risk_data.get('population_type', 'N/A')}\n\n"
    
    results = risk_data.get('assessment_results', [])
    for idx, result in enumerate(results, 1):
        status = "✅ Safe" if result.get('risk_level') == 'acceptable' else "⚠️ Unsafe"
        output += f"#### Sample {idx}: {result.get('sample_code', 'N/A')} - {status}\n"
        output += f"- **Pesticide:** {result.get('pesticide_name', 'N/A')}\n"
        output += f"- **Food Item:** {result.get('food_item', 'N/A')}\n"
        output += f"- **Risk:** {result.get('risk_percentage', 0):.2f}%\n\n"
    
    return output

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
                    st.markdown("""
                    **Troubleshooting:**
                    1. Make sure your API server is running
                    2. Check if API is accessible
                    3. Verify API endpoints are available
                    4. Check server logs
                    """)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 📍 Navigate to:")
        page = st.selectbox(
            "Select page:",
            ["🏠 Home", "💬 AI Assistant", "🔮 Predictions"],
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
        
        This application connects to your prediction API endpoints and provides:
        
        - 📊 **Real-time Dashboards** - Executive, Laboratory, and Regulatory views
        - 🔮 **Prediction Engine** - Single predictions, batch processing, and time series
        - 📁 **Data Management** - File uploads and query processing
        - 💬 **AI Assistant** - Get help with your data and predictions
        
        ### 🚀 Getting Started:
        1. Make sure your API server is running
        2. Click "Check API Connection" in the sidebar
        3. Index your data by year
        4. Start making predictions!
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("🔮 Quick Prediction", use_container_width=True, type="primary"):
            st.session_state.quick_predict = True
        
        if st.button("💬 AI Assistant", use_container_width=True, type="secondary"):
            st.session_state.current_page = "💬 AI Assistant"
            st.rerun()
        
        if st.button("📊 View Status", use_container_width=True):
            st.session_state.show_status = True
    
    # Quick prediction form
    if st.session_state.get('quick_predict', False):
        st.markdown("---")
        st.markdown("### 🔮 Quick Prediction")
        
        with st.form("quick_prediction"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                vegetable = st.text_input("Vegetable", value="pepper")
                pesticide_group = st.selectbox("Pesticide Group", 
                    ["organophosphorus", "carbamates", "pyrethroids"])
            
            with col2:
                reading = st.number_input("Reading", value=45.8, min_value=0.0)
                limits = st.number_input("Limits", value=10.0, min_value=0.0)
            
            with col3:
                sample_code = st.text_input("Sample Code", value="TEST002")
            
            if st.form_submit_button("🚀 Run Prediction", type="primary"):
                prediction_data = {
                    "vegetable": vegetable,
                    "pesticide_group": pesticide_group,
                    "reading": reading,
                    "limits": limits,
                    "sample_code": sample_code
                }
                
                with st.spinner("Running prediction..."):
                    result = api_client.predict_single(prediction_data)
                    
                    if result:
                        st.success("✅ Prediction completed!")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            risk_ratio = reading / limits
                            if risk_ratio <= 1:
                                st.success("🟢 **SAFE**")
                                risk_level = "Low Risk"
                            else:
                                st.error("🔴 **UNSAFE**")
                                risk_level = "High Risk"
                            
                            st.metric("Risk Ratio", f"{risk_ratio:.2f}")
                            st.metric("Risk Level", risk_level)
                        
                        with col2:
                            with st.expander("📋 Full API Response"):
                                st.json(result)
    
    # System status
    st.markdown("---")
    st.markdown("### 📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            status = api_client.health_check()
            if status:
                st.metric("API Status", "🟢 Online", delta="✅ Healthy")
            else:
                st.metric("API Status", "🔴 Offline", delta="❌ Check Connection")
        except:
            st.metric("API Status", "⚠️ Unknown", delta="Test Connection")
    
    with col2:
        st.metric("Total Endpoints", "10+", delta="⬆️ Available")
    
    with col3:
        st.metric("Last Check", "Just now", delta="⬆️ Active")

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

def show_predictions_page(api_client):
    """Predictions page"""
    
    st.title("🔮 Predictions")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vegetable = st.text_input("Vegetable", value="pepper")
            pesticide = st.selectbox("Pesticide Group", 
                ["organophosphorus", "carbamates", "pyrethroids"])
            reading = st.number_input("Reading", value=45.8, min_value=0.0)
        
        with col2:
            limits = st.number_input("Limits", value=10.0, min_value=0.0)
            sample_code = st.text_input("Sample Code", value="TEST002")
        
        if st.form_submit_button("🚀 Run Prediction", type="primary"):
            data = {
                "vegetable": vegetable,
                "pesticide_group": pesticide,
                "reading": reading,
                "limits": limits,
                "sample_code": sample_code
            }
            
            with st.spinner("Running prediction..."):
                result = api_client.predict_single(data)
                if result:
                    st.success("✅ Prediction completed")
                    st.json(result)

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

        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        risk_mode = st.radio(
            "Risk Assessment Mode",
            options=["🤖 Auto-detect", "✅ Always On", "❌ Always Off"],
            index=0,
            help="Auto-detect: Enables when you ask about safety/risk\nAlways On: Include for every query\nAlways Off: Never include"
        )
        
        st.session_state.risk_mode = risk_mode

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
    
    with st.spinner("🔍 Analyzing data..."):
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
        
        include_risk = any(word in query.lower() 
                          for word in ["risk", "safety", "health"])
        
        result = api_client.local_answer(
            query=query,
            year=query_year,
            n_results=1000,
            include_risk=include_risk
        )
        
        if not result:
            st.error("❌ Failed to get results from API")
            return
        """Process user query and display results"""
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
    
    # Display detailed risk assessment
    risk_assessment = result.get('risk_assessment')
    if risk_assessment:
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
                sample_code = assessment.get('sample_code', 'N/A')
                pesticide_name = assessment.get('pesticide_name', 'N/A')
                risk_level = assessment.get('risk_level', 'unknown')
                risk_percentage = assessment.get('risk_percentage', 0)
                
                # Determine status
                if risk_level == 'acceptable':
                    status_icon = "✅"
                    status_text = "Safe"
                else:
                    status_icon = "⚠️"
                    status_text = "Warning"
                
                # Main expander for each sample
                with st.expander(f"{status_icon} Sample {idx}: {sample_code} - {pesticide_name} - {status_text}"):
                    
                    # Two columns
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
                        st.write(f"**Food Intake Source:** {assessment.get('food_intake_source', 'N/A')}")
                        
                        # Risk percentage with color
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
                    
                    # FIXED: Show calculation details WITHOUT nested expander
                    calc_details = assessment.get('calculation_details', {})
                    if calc_details:
                        st.markdown("---")
                        st.markdown("#### 🧮 Calculation Details")
                        
                        st.markdown("**Formula:**")
                        st.code(calc_details.get('formula', 'N/A'), language="text")
                        
                        st.markdown("**Substitution:**")
                        st.code(calc_details.get('substitution', 'N/A'), language="text")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Population Weight:** {calc_details.get('population_weight', 'N/A')}")
                        with col2:
                            st.write(f"**Daily Intake:** {calc_details.get('daily_intake', 'N/A')}")
    
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