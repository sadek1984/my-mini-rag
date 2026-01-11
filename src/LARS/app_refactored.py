"""
Predictive Analytics Dashboard with Enhanced Visual Style
Laboratory Analytics & Risk System (LARS)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path

# Add current directory to path to allow importing from modules
sys.path.insert(0, str(Path(__file__).parent))

# Import the new modules
from modules.trend_analysis import show_trend_analysis_page
from modules.statistical_insights import show_statistical_insights_page
from modules.predictions import show_predictions_page
from modules.ai_assistant import show_chat_page
from modules.dashboard import show_dashboard_overview
from modules.risk_analysis import show_risk_analysis_page
from modules.api_service import get_api_client, API_BASE_URL, DEFAULT_YEAR, AVAILABLE_YEARS
from modules.utils import map_season_names

# Securely load API Key from docker/env/.env.app
try:
    # Go up 3 levels from src/LARS/app.py to project root
    project_root = Path(__file__).parent.parent.parent 
    env_path = project_root / 'docker' / 'env' / '.env.app'
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('OPENAI_API_KEY='):
                    key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                    if key:
                        os.environ['OPENAI_API_KEY'] = key
                        print(f"✅ Loaded OPENAI_API_KEY")
                        break
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Page Configuration
st.set_page_config(
    page_title="🔮 LARS Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": """👋 Hello! I'm your **pesticide analysis AI assistant** — ready to help you explore, assess, and analyze pesticide data with ease.

---

#### 🧠 **What I can help you with**
- 📊 Query pesticide data by year  
- 🔍 Find samples exceeding or below limits  
- 🎯 Perform detailed **risk assessments**  
- 📈 Compare **trends across years**  
- 🧾 Generate summarized **compliance reports**

---

#### 💡 **Example Queries**
✅ **Basic queries**
- “Show all **Buprofezin** samples from **2024**”  
- “Find **bifenazate** exceeding limits in **2022**”  

⚠️ **Risk & safety checks**
- “Assess **health risk** for **Buprofezin** in **tomatoes 2023**”  
- “Find samples with **risk percentage above 100%** in **2024**”  

---

💬 **Now it’s your turn!**  
Type your question below and I’ll analyze the data for you 🚀
"""
        }]
    
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = DEFAULT_YEAR
    
    if 'indexed_years' not in st.session_state:
        st.session_state.indexed_years = set()
    
    if 'project_id' not in st.session_state:
        st.session_state.project_id = "default_project"
        
    if 'intro_shown' not in st.session_state:
        st.session_state.intro_shown = False
        
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard Overview"
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

def render_animated_header():
    """Render the animated LARS header"""
    html_content = textwrap.dedent("""
        <div class="lars-hero-container">
            <div class="lars-content">
                <h1 class="lars-main-title">
                    <span class="letter" style="animation-delay: 0s">L</span>
                    <span class="letter" style="animation-delay: 0.1s">A</span>
                    <span class="letter" style="animation-delay: 0.2s">R</span>
                    <span class="letter" style="animation-delay: 0.3s">S</span>
                </h1>
                <h2 class="lars-subtitle-enhanced">
                    <span class="highlight">Laboratory</span> <span class="accent-text">Analytics</span> & <span class="accent-text">Risk</span> <span class="highlight">System</span>
                </h2>
                <p class="lars-tagline">
                    Powered by AI • Trusted by Scientists • Protecting Public Health
                </p>
                <div class="feature-badges">
                    <div class="badge">Real-time Analysis</div>
                    <div class="badge">Advanced Analytics</div>
                    <div class="badge">Secure & Compliant</div>
                </div>
            </div>
        </div>
        <style>
            .lars-hero-container {
                background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 25%, #a5d6a7 50%, #81c784 75%, #66bb6a 100%);
                padding: 60px 20px;
                border-radius: 0 0 30px 30px;
                text-align: center;
                margin-bottom: 40px;
                margin-top: -30px;
                box-shadow: 0 8px 30px rgba(46, 125, 50, 0.15);
            }
            .lars-main-title { 
                font-size: 4.5rem; 
                font-weight: 900; 
                letter-spacing: 15px; 
                color: #1b5e20; 
                margin-bottom: 20px;
            }
            .letter { 
                display: inline-block; 
                animation: letterPop 0.8s ease-out forwards; 
            }
            @keyframes letterPop { 
                0% { opacity: 0; transform: translateY(-30px); } 
                100% { opacity: 1; transform: translateY(0); } 
            }
            .lars-subtitle-enhanced { font-size: 1.8rem; color: #1b5e20; margin-bottom: 15px; }
            .highlight { font-weight: 700; color: #2e7d32; }
            .accent-text { color: #1b5e20; font-weight: 700; }
            .lars-tagline { font-style: italic; color: #388e3c; margin-bottom: 25px; }
            .feature-badges { display: flex; justify-content: center; gap: 15px; }
            .badge { 
                background: white; 
                padding: 5px 15px; 
                border-radius: 20px; 
                font-size: 0.9rem; 
                color: #2e7d32; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
            }
        </style>
    """)
    st.markdown(html_content, unsafe_allow_html=True)

def render_sidebar(api_client):
    """Render sidebar with navigation"""
    with st.sidebar:
        st.markdown("# 📍 LARS Menu")
        
        if st.button("🔄 Check API Status", use_container_width=True, type="primary"):
            with st.spinner("Checking..."):
                health = api_client.health_check()
                if health: st.success("✅ Connected")
                else: st.error("❌ Disconnected")
            
        st.markdown("---")
        
        page_options = [
            "🏠 Dashboard Overview",
            "🤖 AI Assistant",
            "📊 Risk Analysis",
            "📈 Trend Analysis",
            "📋 Statistical Insights",
            "🔮 Predictions"
        ]
        
        current_selection = st.session_state.get('current_page', "🏠 Dashboard Overview")
        if current_selection not in page_options:
            current_selection = "🏠 Dashboard Overview"
            
        st.session_state.current_page = st.selectbox(
            "Navigate to:",
            page_options,
            index=page_options.index(current_selection),
            key="page_selector"
        )
        
        st.markdown("---")
        st.info(f"API Base: `{API_BASE_URL}`")
        st.caption(f"LARS v2.0 - Refactored Edition")

def main():
    initialize_session_state()
    api_client = get_api_client()
    
    # Load initial data if needed
    if not st.session_state.data_loaded:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed_data_with_neighborhood.xlsx')
        if os.path.exists(data_path):
            try:
                st.session_state.df = pd.read_excel(data_path)
                st.session_state.data_loaded = True
            except Exception as e:
                print(f"Error loading initial data: {e}")

    # Show LARS header
    if not st.session_state.intro_shown:
        render_animated_header()
        st.session_state.intro_shown = True
        
    render_sidebar(api_client)
    
    pages = {
        "🏠 Dashboard Overview": show_dashboard_overview,
        "📊 Risk Analysis": show_risk_analysis_page,
        "📈 Trend Analysis": show_trend_analysis_page,
        "📋 Statistical Insights": show_statistical_insights_page,
        "🔮 Predictions": show_predictions_page,
        "🤖 AI Assistant": show_chat_page,
    }
    
    page_func = pages.get(st.session_state.current_page)
    if page_func:
        page_func(api_client)
    else:
        st.error(f"Page '{st.session_state.current_page}' is not implemented yet.")

if __name__ == "__main__":
    main()
