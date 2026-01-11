import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import map_season_names, calculate_violations, load_data_from_path

def show_risk_analysis_page(api_client):
    """Advanced insights with risk combinations and forecasting"""
    
    st.title("🎯 Advanced Risk Intelligence")
    
    st.markdown("""
    **Strategic Analytics for Lab Operations**
    - Identify high-risk vegetable-pesticide combinations
    - Forecast seasonal contamination risks
    - Optimize testing schedules and resource allocation
    """)
    
    # Load data
    df = load_data_from_path()
    
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
    
    # Ensure violations are calculated
    df = calculate_violations(df)
    
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
    
    # Map seasons
    df = map_season_names(df)
    
    # Show data preview
    with st.expander("🔍 Data Preview & Validation"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Interactive Heatmap
    st.subheader("🔥 Risk Combination Heatmap")
    
    # Group by vegetable and pesticide to see violation counts
    if 'vegetable_english' in df.columns and 'pesticide' in df.columns and 'is_violation' in df.columns:
        risk_matrix = df.groupby(['vegetable_english', 'pesticide'])['is_violation'].sum().unstack(fill_value=0)
        
        # Filter to TOP 15 vegetables and TOP 20 pesticides for readability
        top_vegs = df.groupby('vegetable_english')['is_violation'].sum().sort_values(ascending=False).head(15).index
        top_pests = df.groupby('pesticide')['is_violation'].sum().sort_values(ascending=False).head(20).index
        
        risk_matrix_filtered = risk_matrix.loc[risk_matrix.index.isin(top_vegs), risk_matrix.columns.isin(top_pests)]
        
        fig = px.imshow(risk_matrix_filtered,
                       labels=dict(x="Pesticide", y="Vegetable", color="Violations"),
                       color_continuous_scale="Reds",
                       title="Top Vegetable-Pesticide Risk Combinations")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Heatmap requires 'vegetable_english', 'pesticide', and 'is_violation' columns.")

    # Seasonal Risk Distribution
    st.subheader("📅 Seasonal Risk Distribution")
    if 'season' in df.columns and 'is_violation' in df.columns:
        seasonal_risk = df.groupby('season')['is_violation'].mean().reset_index()
        seasonal_risk.columns = ['Season', 'Violation Rate']
        
        fig = px.bar(seasonal_risk, x='Season', y='Violation Rate',
                    color='Violation Rate', color_continuous_scale='Viridis',
                    title="Average Violation Rate by Season")
        st.plotly_chart(fig, use_container_width=True)
