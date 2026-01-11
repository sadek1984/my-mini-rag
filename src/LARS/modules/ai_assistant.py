import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import json
import requests
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from openai import OpenAI
import google.generativeai as genai
import random


def generate_bar_chart(df: pd.DataFrame, query: str):
    """Generate an interactive bar chart from DataFrame"""
    try:
        import plotly.graph_objects as go
        
        # Determine x and y columns intelligently
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for visualization")
            return
        
        # Use first text column for x-axis, first numeric for y-axis
        x_col = text_cols[0] if text_cols else df.columns[0]
        y_col = numeric_cols[0]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=df[x_col],
                y=df[y_col],
                marker=dict(
                    color=df[y_col],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df[y_col],
                textposition='outside',
                texttemplate='%{text:.1f}'
            )
        ])
        
        fig.update_layout(
            title=f"Bar Chart: {y_col} by {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating bar chart: {e}")

def generate_line_chart(df: pd.DataFrame, query: str):
    """Generate an interactive line chart from DataFrame"""
    try:
        import plotly.graph_objects as go
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for visualization")
            return
        
        x_col = text_cols[0] if text_cols else df.columns[0]
        y_col = numeric_cols[0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            line=dict(color='#2e7d32', width=3),
            marker=dict(size=10, color='#4caf50'),
            name=y_col
        ))
        
        fig.update_layout(
            title=f"Line Chart: {y_col} Trend",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating line chart: {e}")

def generate_pie_chart(df: pd.DataFrame, query: str):
    """Generate an interactive pie chart from DataFrame"""
    try:
        import plotly.graph_objects as go
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols or not text_cols:
            st.warning("⚠️ Need both text and numeric columns for pie chart")
            return
        
        labels_col = text_cols[0]
        values_col = numeric_cols[0]
        
        # Limit to top 10 for readability
        df_top = df.nlargest(10, values_col) if len(df) > 10 else df
        
        fig = go.Figure(data=[go.Pie(
            labels=df_top[labels_col],
            values=df_top[values_col],
            hole=0.3,
            textinfo='label+percent',
            marker=dict(
                colors=px.colors.sequential.Viridis,
                line=dict(color='white', width=2)
            )
        )])
        
        fig.update_layout(
            title=f"Pie Chart: {values_col} Distribution",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating pie chart: {e}")

def generate_radar_chart(df: pd.DataFrame, query: str):
    """Generate an interactive radar chart from DataFrame"""
    try:
        import plotly.graph_objects as go
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols or not text_cols:
            st.warning("⚠️ Need both text and numeric columns for radar chart")
            return
        
        # Limit to top 10 for readability
        df_top = df.head(10)
        
        categories = df_top[text_cols[0]].tolist()
        
        # If we have multiple numeric columns, create multi-series radar
        if len(numeric_cols) > 1:
            fig = go.Figure()
            
            for col in numeric_cols[:3]:  # Limit to 3 series for clarity
                # Normalize values to 0-100 scale
                values = df_top[col].tolist()
                max_val = max(values) if max(values) > 0 else 1
                normalized = [(v / max_val) * 100 for v in values]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized,
                    theta=categories,
                    fill='toself',
                    name=col
                ))
        else:
            # Single series radar
            values = df_top[numeric_cols[0]].tolist()
            max_val = max(values) if max(values) > 0 else 1
            normalized = [(v / max_val) * 100 for v in values]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=normalized,
                theta=categories,
                fill='toself',
                marker=dict(color='#4caf50'),
                line=dict(color='#2e7d32', width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Radar Chart: Comparative Analysis",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Values are normalized to 0-100 scale for comparison")
        
    except Exception as e:
        st.error(f"Error generating radar chart: {e}")

def generate_heatmap(df: pd.DataFrame, query: str = ""):
    """Generate an interactive general heatmap from DataFrame"""
    
    st.subheader("📈 Data Exploration Heatmap")
    
    if query:
        try:
            df_filtered = df.query(query)
            if len(df_filtered) == 0:
                st.warning("Query returned no results. Showing full dataset.")
                df_filtered = df
        except:
            st.error("Invalid query syntax")
            df_filtered = df
    else:
        df_filtered = df
    
    # Select columns for heatmap
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.warning("No numeric columns available for heatmap")
        return
    
    # Let user choose what to visualize
    col1, col2 = st.columns(2)
    
    with col1:
        row_col = st.selectbox(
            "Select row grouping:",
            options=categorical_cols,
            index=0 if categorical_cols else None,
            help="Group data by this categorical column for rows",
            key="heatmap_row_col"
        )
    
    with col2:
        value_col = st.selectbox(
            "Select value to visualize:",
            options=numeric_cols,
            index=0,
            help="Numeric value to show in heatmap",
            key="heatmap_value_col"
        )
    
    # Aggregate data
    if row_col and value_col:
        agg_data = df_filtered.groupby(row_col)[value_col].mean().sort_values(ascending=False)
        
        # Get top 20 categories for readability
        top_categories = agg_data.head(20)
        
        # Create heatmap (single column)
        fig = go.Figure(data=go.Heatmap(
            z=[top_categories.values],
            x=[f"{idx}: {val:.1f}" for idx, val in zip(top_categories.index, top_categories.values)],
            y=["Value"],
            colorscale='Viridis',
            text=[top_categories.values],
            texttemplate='%{text:.1f}',
            textfont={"size": 12},
            colorbar=dict(title=value_col),
            hovertemplate='<b>Category:</b> %{x}<br>' +
                         '<b>Avg Value:</b> %{z:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Average {value_col} by {row_col}",
            height=400,
            xaxis_tickangle=-45,
            margin=dict(t=50, b=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("View Data Table"):
            st.dataframe(top_categories.reset_index().rename(
                columns={row_col: 'Category', value_col: f'Avg_{value_col}'}
            ))

def generate_scatter_plot(df: pd.DataFrame, query: str):
    """Generate an interactive scatter plot from DataFrame"""
    try:
        import plotly.express as px
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for scatter plot")
            return
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        # Use third numeric column for size if available
        size_col = numeric_cols[2] if len(numeric_cols) >= 3 else None
        
        # Use text column for color if available
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        color_col = text_cols[0] if text_cols else None
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            title=f"Scatter Plot: {y_col} vs {x_col}",
            height=600,
            hover_data=df.columns.tolist()
        )
        
        fig.update_traces(marker=dict(
            line=dict(width=1, color='white'),
            opacity=0.7
        ))
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating scatter plot: {e}")

def process_data_query_sql(query: str, model, include_risk: bool = False):
    """
    🆕 Process data analysis queries using SQL + DuckDB
    Uses Schema-Only Prompting for privacy protection
    Enhanced to handle Arabic queries with sample types and pesticide counts
    """
    
    with st.spinner("🔄 Generating SQL query..."):
        try:
            # Connect to DuckDB
            con = get_duckdb_connection()
            if con is None:
                st.error("❌ Database not found. Please run migrate_db.py first.")
                return
            
            # Arabic numeral conversion
            arabic_to_english = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
                                 '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
            query_normalized = query
            for ar, en in arabic_to_english.items():
                query_normalized = query_normalized.replace(ar, en)
            
            # Sample type mapping (Arabic to search patterns)
            # Comprehensive sample type mappings from database
            sample_types = {
                # Vegetables
                'خيار': 'خيار',
                'طماطم': 'طماطم',
                'باذنجان': 'باذنجان',
                'فلفل': 'فلفل',
                'كوسة': 'كوسة',
                'جزر': 'جزر',
                'خس': 'خس',
                'بقدونس': 'بقدونس',
                'سبانخ': 'سبانخ',
                'بطاطس': 'بطاطس',
                'فاصوليا': 'فاصوليا',
                'فاصووليا': 'فاصوليا',
                'بامية': 'بامية',
                'بروكلي': 'بروكلي',
                'بصل': 'بصل',
                'جرجير': 'جرجير',
                'زهرة': 'زهرة',
                'شبت': 'شبت',
                'كزبرة': 'كزبرة',
                'ملوخية': 'ملوخية',
                'نعناع': 'نعناع',
                # Fruits
                'فراولة': 'فراولة',
                'عنب': 'عنب',
                'تفاح': 'تفاح',
                'برتقال': 'برتقال',
                'رمان': 'رمان',
                'كمثرى': 'كمثرى',
                'ليمون': 'ليمون',
                'توت': 'توت',
                'تمر': 'تمر',
                # Grains
                'قمح': 'قمح',
                'رز': 'رز',
                'ذرة': 'ذرة',
                'عدس': 'عدس',
                'دقيق': 'دقيق',
                'طحين': 'طحين',
                'شوفان': 'شوفان',
                # Spices
                'كمون': 'كمون',
                'زعتر': 'زعتر',
                'بهارات': 'بهارات',
                'قرنفل': 'قرنفل',
                'هيل': 'هيل',
                'يانسون': 'يانسون',
                'شمر': 'شمر',
                'محلب': 'محلب',
                # Nuts
                'لوز': 'لوز',
                'فستق': 'فستق',
                'كاجو': 'كاجو',
                'بندق': 'بندق',
                'بيكان': 'بيكان',
                'فول سوداني': 'فول سوداني',
                'سمسم': 'سمسم',
            }
            
            # English to Arabic sample type mapping
            english_sample_types = {
                # Vegetables
                'cucumber': 'خيار',
                'tomato': 'طماطم',
                'eggplant': 'باذنجان',
                'aubergine': 'باذنجان',
                'pepper': 'فلفل',
                'zucchini': 'كوسة',
                'squash': 'كوسة',
                'carrot': 'جزر',
                'lettuce': 'خس',
                'parsley': 'بقدونس',
                'spinach': 'سبانخ',
                'potato': 'بطاطس',
                'beans': 'فاصوليا',
                'bean': 'فاصوليا',
                'okra': 'بامية',
                'broccoli': 'بروكلي',
                'onion': 'بصل',
                'arugula': 'جرجير',
                'rocket': 'جرجير',
                'cauliflower': 'زهرة',
                'dill': 'شبت',
                'coriander': 'كزبرة',
                'cilantro': 'كزبرة',
                'molokhia': 'ملوخية',
                'mint': 'نعناع',
                # Fruits
                'strawberry': 'فراولة',
                'grape': 'عنب',
                'apple': 'تفاح',
                'orange': 'برتقال',
                'pomegranate': 'رمان',
                'pear': 'كمثرى',
                'lemon': 'ليمون',
                'lime': 'ليمون',
                'berry': 'توت',
                'date': 'تمر',
                'dates': 'تمر',
                # Grains
                'wheat': 'قمح',
                'rice': 'رز',
                'corn': 'ذرة',
                'lentil': 'عدس',
                'flour': 'دقيق',
                'oat': 'شوفان',
                'oats': 'شوفان',
                # Spices
                'cumin': 'كمون',
                'thyme': 'زعتر',
                'spices': 'بهارات',
                'clove': 'قرنفل',
                'cardamom': 'هيل',
                'anise': 'يانسون',
                'fennel': 'شمر',
                # Nuts
                'almond': 'لوز',
                'pistachio': 'فستق',
                'cashew': 'كاجو',
                'hazelnut': 'بندق',
                'pecan': 'بيكان',
                'peanut': 'فول سوداني',
                'peanuts': 'فول سوداني',
                'sesame': 'سمسم',
            }
            
            # Detect sample type from query (Arabic first)
            detected_sample_type = None
            for ar_name, pattern in sample_types.items():
                if ar_name in query:
                    detected_sample_type = pattern
                    break
            
            # If not found in Arabic, check English
            if not detected_sample_type:
                query_lower_check = query.lower()
                for en_name, ar_pattern in english_sample_types.items():
                    if en_name in query_lower_check:
                        detected_sample_type = ar_pattern
                        break
            
            # Check for specific query patterns
            query_lower = query_normalized.lower()
            
            # Pattern 0: Count of samples above/below limit for a sample type
            # Matches: "what are the count of tomato samples that reading more than limits?"
            # Matches: "how many cucumber samples exceeding limits"
            # Matches: "عدد عينات الطماطم فوق الحد"
            count_phrases = ['count', 'how many', 'عدد', 'كم']
            above_limit_phrases_p0 = ['more than limit', 'exceeding', 'above limit', 'over limit', 
                                      'reading more', 'exceed', 'فوق الحد', 'تجاوز']
            below_limit_phrases_p0 = ['below limit', 'under limit', 'within limit', 'تحت الحد']
            
            is_count_query = any(phrase in query_lower for phrase in count_phrases) or any(phrase in query for phrase in ['عدد', 'كم'])
            is_above_limit_query = any(phrase in query_lower for phrase in above_limit_phrases_p0) or any(phrase in query for phrase in ['فوق الحد', 'تجاوز'])
            is_below_limit_query = any(phrase in query_lower for phrase in below_limit_phrases_p0) or 'تحت الحد' in query
            
            if is_count_query and detected_sample_type and (is_above_limit_query or is_below_limit_query):
                limit_condition = "is_above_limit = 1" if is_above_limit_query else "is_above_limit = 0"
                limit_label = "فوق الحد" if is_above_limit_query else "تحت الحد"
                
                st.info(f"🔍 Counting **{detected_sample_type}** samples **{limit_label}**...")
                
                sql_query = f"""
                SELECT 
                    '{detected_sample_type}' as نوع_العينة,
                    COUNT(DISTINCT "كود العينة") as عدد_العينات,
                    SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
                    SUM(CASE WHEN is_above_limit = 0 AND is_detected = 1 THEN 1 ELSE 0 END) as تحت_الحد
                FROM chemistry_tidy
                WHERE "اسم العينة" LIKE '%{detected_sample_type}%'
                AND is_detected = 1
                """
                
                result = con.execute(sql_query).df()
                
                if not result.empty:
                    total = result['عدد_العينات'].values[0]
                    above = result['فوق_الحد'].values[0]
                    below = result['تحت_الحد'].values[0]
                    
                    st.success(f"✅ **{detected_sample_type}**: إجمالي {total} عينة — {above} فوق الحد، {below} تحت الحد")
                    st.dataframe(result, use_container_width=True)
                else:
                    st.warning(f"⚠️ لم يتم العثور على عينات {detected_sample_type}")
                
                con.close()
                return
            
            # Pattern 1: Find samples with exactly N pesticides (may include sample type filter)
            # Matches: "عينات الخيار التي تحتوي علي عدد ٢ مبيد" or "samples with 2 pesticides"
            pesticide_count_pattern = r'(\d+)\s*(?:مبيد|pesticides?)'
            match = re.search(pesticide_count_pattern, query_normalized)
            
            if match:
                n_pesticides = int(match.group(1))
                
                # Build query with optional sample type filter
                sample_filter = ""
                if detected_sample_type:
                    sample_filter = f"AND \"اسم العينة\" LIKE '%{detected_sample_type}%'"
                    st.info(f"🔍 Looking for **{detected_sample_type}** samples with exactly {n_pesticides} pesticides...")
                else:
                    st.info(f"🔍 Looking for samples with exactly {n_pesticides} pesticides...")
                
                query_sql = f"""
                SELECT 
                    "كود العينة" as sample_code,
                    "اسم العينة" as sample_name,
                    COUNT(*) as pesticide_count
                FROM chemistry_tidy
                WHERE is_detected = 1 AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
                {sample_filter}
                GROUP BY "كود العينة", "اسم العينة"
                HAVING COUNT(*) = {n_pesticides}
                ORDER BY "كود العينة" DESC
                LIMIT 10
                """
                
                result = con.execute(query_sql).df()
                
                if not result.empty:
                    st.success(f"✅ Found {len(result)} samples with exactly {n_pesticides} pesticides:")
                    
                    # Display the samples
                    st.dataframe(result, use_container_width=True)
                    
                    # Show detailed pesticide information for these samples
                    st.subheader("🔬 Detailed Pesticide Information")
                    
                    sample_codes = result['sample_code'].tolist()
                    sample_codes_str = ','.join([str(code) for code in sample_codes])
                    detailed_query = f"""
                    SELECT 
                        "كود العينة" as sample_code,
                        "اسم العينة" as sample_name,
                        pesticide_name,
                        concentration,
                        limit_value,
                        ROUND(exceedance_ratio, 2) as ratio,
                        CASE WHEN is_compliant = 1 THEN 'مطابق' ELSE 'غير مطابق' END as status
                    FROM chemistry_tidy
                    WHERE "كود العينة" IN ({sample_codes_str})
                    AND is_detected = 1
                    ORDER BY "كود العينة", concentration DESC
                    """
                    
                    detailed_result = con.execute(detailed_query).df()
                    st.dataframe(detailed_result, use_container_width=True)
                    
                    # Store in session state
                    st.session_state.last_query_result = result
                    st.session_state.last_query = query
                    st.session_state.last_query_type = 'pesticide_count'
                    
                else:
                    filter_msg = f" for {detected_sample_type}" if detected_sample_type else ""
                    st.warning(f"⚠️ No samples found with exactly {n_pesticides} pesticides{filter_msg}")
                
                con.close()
                return
            
            # Pattern 2: Extract specific pesticide readings with sample type filter
            # Arabic pesticide name to English mapping
            arabic_pesticide_map = {
                'بابروفيزن': 'bupro',
                'بايفنثرين': 'bifenthrin',
                'كلوربيريفوس': 'chlorpyrifos',
                'إيميداكلوبريد': 'imidacloprid',
                'اميداكلوبريد': 'imidacloprid',
                'اسيتاميبريد': 'acetamiprid',
                'ازوكسيستروبين': 'azoxystrobin',
                'دلتاميثرين': 'deltamethrin',
                'سايبرمثرين': 'cypermethrin',
                'ثياميثوكسام': 'thiamethoxam',
                'بروفينوفوس': 'profenofos',
                'ميتالاكسيل': 'metalaxyl',
                'كاربندازيم': 'carbendazim',
                'ابامكتين': 'abamectin',
                'الفيبرونيل': 'fipronil',
                'فيبرونيل': 'fipronil',
            }
            
            # All pesticide names from the database (135 unique)
            pesticide_names = [
                'abamectin', 'abemictin', 'abimactin', 'aceatamiprid', 'acetamipeid',
                'acetamiprid', 'acetamprid', 'afla', 'afla B1', 'afla B2', 'afla b1',
                'afla b2', 'aflaB1', 'aflaB2', 'aflaG1', 'alfaB1', 'azolxystrobin',
                'azoxystobin', 'azoxystrobin', 'bifenazate', 'bifenthrin', 'biphenyl',
                'boscalid', 'buprofezin', 'buprofuzibn', 'buprofuzin', 'carbaryl',
                'carbendazim', 'carbofuran', 'chlorantraniliprle', 'chlorantraniliproel',
                'chlorantraniliprole', 'chlorfenapyr', 'chlorpyrifos', 'chlorpyriphos',
                'clothiandin', 'clothianidin', 'clothindin', 'cyanantraniliprole',
                'cyantraniliprole', 'cyflufenamid', 'cyflulthrin', 'cyflumetofen',
                'cyhalothrin', 'cypermethrin', 'cyprodianil', 'cyprtmethrin',
                'deltamethrin', 'deltamithrin', 'diazinon', 'dichloranilin',
                'difenoconazole', 'diflubenzoran', 'dimethomorph', 'dinotefuran',
                'diphenylamine', 'ethion', 'ethofumesite', 'etofenoprox', 'etofenprox',
                'etoxazole', 'famoxadon', 'fenazaquin', 'fenhexamid', 'fenhexamide',
                'fenpyroximate', 'fenpyroxymate', 'fipronil', 'fludioxonil', 'flumetaquin',
                'fluoptram', 'fluopyram', 'flutriafol', 'hexythiazox', 'imidacloprid',
                'imidaclopride', 'imidaclprid', 'indoxacarb', 'linuron', 'malathion',
                'mandiprobamid', 'mandipropamid', 'metalaxyl', 'metalazyl', 'methomyl',
                'methoxyfenozid', 'methoxyfenozide', 'metrafenon', 'mrthoxyfenoxide',
                'myclobutanil', 'myclobutanyl', 'ometheoate', 'penconazole', 'phoxim',
                'pipronil butoxid', 'pipronil-butoxid', 'pipronyl butoxide', 'pirimiphos',
                'pirimiphos methyl', 'pirimiphos-methyl', 'pirmiphos', 'profenfos',
                'profenofos', 'propamocarb', 'propiconazole', 'pyraclostrobin',
                'pyridaben', 'pyridabin', 'pyrimethanil', 'pyrimethanyl', 'pyriproxyfen',
                'pyrmethanil', 'spinosad A', 'spinosad D', 'spinosad d', 'spiromesfin',
                'spirotetramat', 'spirotetramate', 'sulfoxaflor', 'tebuconazole',
                'thiabendazole', 'thiamethoxam', 'thiamithoxam', 'thiaophonat-methyl',
                'thioamethoxam', 'thiopendazole', 'thiophanate', 'thiophonat methyl',
                'thiophonate', 'thiophonate methyl', 'thiophonate-methyl', 'tricyclazole',
                'trifloxistrobin', 'trifloxystobin', 'trifloxystrobin', 'emamectin', 'bupro'
            ]
            
            # Check for Arabic pesticide names first
            detected_pesticide = None
            for ar_name, en_pattern in arabic_pesticide_map.items():
                if ar_name in query:
                    detected_pesticide = en_pattern
                    st.info(f"🔍 البحث عن مبيد **{ar_name}** ({en_pattern})...")
                    break
            
            # If no Arabic, check English names
            if not detected_pesticide:
                for pesticide in pesticide_names:
                    if pesticide in query_lower:
                        detected_pesticide = pesticide
                        break
            
            # Normalize pesticide patterns to handle variants (e.g., buprofezin vs buprofuzin)
            pesticide_pattern_map = {
                'buprofezin': 'bupro',
                'buprofuzin': 'bupro', 
                'bifenazate': 'bifenazate',
                'chlorpyrifos': 'chlorpyrifos',
                'imidacloprid': 'imidacloprid',
            }
            # Use the normalized pattern if available, otherwise use the detected name
            search_pattern = pesticide_pattern_map.get(detected_pesticide, detected_pesticide)
            
            if detected_pesticide:
                # Check for ratio/multiplier filter (e.g., "more than 2 times", "أكثر من 3 أضعاف")
                ratio_filter = ""
                ratio_value = None
                
                # English patterns for ratio
                ratio_match = re.search(r'more\s+than\s+(\d+)\s*(?:times|x)', query_lower)
                if ratio_match:
                    ratio_value = float(ratio_match.group(1))
                
                # Arabic patterns for ratio (أكثر من X أضعاف)
                if not ratio_value:
                    ratio_match_ar = re.search(r'أكثر\s+من\s+(\d+)\s*(?:ضعف|أضعاف|مرات)', query)
                    if ratio_match_ar:
                        ratio_value = float(ratio_match_ar.group(1))
                
                if ratio_value:
                    ratio_filter = f"AND exceedance_ratio > {ratio_value}"
                    st.info(f"🔍 Filtering for **> {ratio_value}x** the limit...")
                
                # Check for compliance filter (non-compliant, exceeding, above limit, etc.)
                compliance_filter = ""
                above_limit_phrases = [
                    'non compliant', 'non-compliant', 'noncompliant',
                    'exceeding', 'above limit', 'over limit', 'more than limit',
                    'reading more', 'exceed', 'exceeded', 'exceeds',
                    'غير مطابق', 'فوق الحد', 'تجاوز', 'يتجاوز'
                ]
                below_limit_phrases = [
                    'compliant', 'below limit', 'under limit', 'within limit',
                    'مطابق', 'تحت الحد', 'ضمن الحد'
                ]
                
                if any(phrase in query_lower for phrase in above_limit_phrases) or any(phrase in query for phrase in ['غير مطابق', 'فوق الحد', 'تجاوز']):
                    # Filter for samples where pesticide exceeds limit
                    compliance_filter = "AND is_above_limit = 1"
                    st.info("🔍 Filtering for samples **above the limit** (فوق الحد)...")
                elif any(phrase in query_lower for phrase in below_limit_phrases) and 'non' not in query_lower:
                    compliance_filter = "AND is_above_limit = 0"
                    st.info("🔍 Filtering for samples **below the limit** (تحت الحد)...")
                
                # Build query with optional sample type filter
                sample_filter = ""
                if detected_sample_type:
                    sample_filter = f'AND "اسم العينة" LIKE \'%{detected_sample_type}%\''
                    st.info(f"🔍 Extracting **{detected_pesticide}** readings for **{detected_sample_type}**...")
                else:
                    st.info(f"🔍 Extracting **{detected_pesticide}** readings...")
                
                sql_query = f"""
                SELECT 
                    "كود العينة" as كود_العينة,
                    "اسم العينة" as اسم_العينة,
                    pesticide_name as المبيد,
                    concentration as التركيز,
                    limit_value as الحد_المسموح,
                    ROUND(exceedance_ratio, 2) as نسبة_التجاوز,
                    CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'تحت الحد' END as الحالة
                FROM chemistry_tidy
                WHERE pesticide_name ILIKE '%{search_pattern}%'
                AND is_detected = 1
                {sample_filter}
                {ratio_filter}
                {compliance_filter}
                ORDER BY exceedance_ratio DESC
                LIMIT 30
                """
                
                result = con.execute(sql_query).df()
                
                if not result.empty:
                    # Count above/below limit
                    above_limit = len(result[result['الحالة'] == 'فوق الحد'])
                    below_limit = len(result[result['الحالة'] == 'تحت الحد'])
                    
                    filter_msg = f" في {detected_sample_type}" if detected_sample_type else ""
                    st.success(f"✅ وجد {len(result)} نتيجة{filter_msg}: {above_limit} فوق الحد، {below_limit} تحت الحد")
                    st.dataframe(result, use_container_width=True)
                    st.session_state.last_query = query
                    st.session_state.last_query_type = 'pesticide_specific'
                else:
                    filter_msg = f" في عينات {detected_sample_type}" if detected_sample_type else ""
                    st.warning(f"⚠️ لم يتم العثور على مبيد {detected_pesticide}{filter_msg}")
                    
                con.close()
                return
            
            # Pattern 3: List all pesticides for a sample type (e.g., "ماهي المبيدات الموجوده في عينة الباذنجان")
            pesticide_list_patterns = ['المبيدات الموجوده', 'المبيدات في عينة', 'تكرار المبيدات', 'المبيدات المكتشفة']
            if any(p in query for p in pesticide_list_patterns) and detected_sample_type:
                st.info(f"🔍 البحث عن المبيدات في عينات **{detected_sample_type}**...")
                
                sql_query = f"""
                SELECT 
                    pesticide_name as المبيد,
                    COUNT(*) as عدد_التكرار,
                    ROUND(AVG(concentration), 3) as متوسط_التركيز,
                    ROUND(MAX(concentration), 3) as اعلى_تركيز,
                    SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as عدد_التجاوزات
                FROM chemistry_tidy
                WHERE "اسم العينة" LIKE '%{detected_sample_type}%'
                AND is_detected = 1
                GROUP BY pesticide_name
                ORDER BY عدد_التكرار DESC, عدد_التجاوزات DESC
                """
                
                result = con.execute(sql_query).df()
                
                if not result.empty:
                    total_detections = result['عدد_التكرار'].sum()
                    total_violations = result['عدد_التجاوزات'].sum()
                    unique_pesticides = len(result)
                    
                    st.success(f"✅ وجد **{unique_pesticides}** مبيد في عينات {detected_sample_type}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("عدد المبيدات المختلفة", unique_pesticides)
                    with col2:
                        st.metric("إجمالي الاكتشافات", int(total_detections))
                    with col3:
                        st.metric("إجمالي التجاوزات", int(total_violations))
                    
                    st.dataframe(result, use_container_width=True)
                    st.session_state.last_query_result = result
                else:
                    # Check if samples exist but have no detected pesticides
                    sample_count = con.execute(f"""
                        SELECT COUNT(DISTINCT "كود العينة") 
                        FROM chemistry_tidy 
                        WHERE "اسم العينة" LIKE '%{detected_sample_type}%'
                    """).fetchone()[0]
                    
                    if sample_count > 0:
                        st.info(f"ℹ️ يوجد {sample_count} عينة من {detected_sample_type}، لكن لم يتم اكتشاف أي مبيدات فيها.")
                    else:
                        st.warning(f"⚠️ لا توجد عينات من نوع {detected_sample_type}")
                
                con.close()
                return
            
            # Pattern 4: Pesticides in multiple neighborhoods (with optional "each separately")
            # Matches: "ماهي المبيدات في حي الإسكان و النهضة"
            # Matches: "ماهي المبيدات في حي الإسكان و النهضة كل علي حده"
            # Actual neighborhoods in database (with spelling variants)
            neighborhood_map = {
                # النهضة (most samples)
                'النهضة': 'النهضة', 'النهضه': 'النهضة',
                # الإسكان
                'الإسكان': 'الإسكان', 'الاسكان': 'الإسكان',
                # الأخضر
                'الأخضر': 'الأخضر', 'الاخضر': 'الأخضر',
                # الروضة
                'الروضة': 'الروضة', 'الروضه': 'الروضة',
                # النخيل
                'النخيل': 'النخيل',
                # الريان
                'الريان': 'الريان',
                # الفلاح
                'الفلاح': 'الفلاح',
                # المنتزه
                'المنتزه': 'المنتزه',
                # الجردة
                'الجردة': 'الجردة', 'الجرده': 'الجردة',
                # الأفق
                'الأفق': 'الأفق', 'الافق': 'الأفق',
                # الضاحي
                'الضاحي': 'الضاحي',
                # السالمية
                'السالمية': 'السالمية', 'السالميه': 'السالمية',
                # الفايزية
                'الفايزية': 'الفايزية', 'الفايزيه': 'الفايزية',
                # الشقة
                'الشقة': 'الشقة', 'الشقه': 'الشقة',
                # المنار
                'المنار': 'المنار',
            }
            
            if 'المبيدات' in query and 'حي' in query:
                # Extract all mentioned neighborhoods
                found_neighborhoods = []
                for ar_name, normalized in neighborhood_map.items():
                    if ar_name in query:
                        if normalized not in found_neighborhoods:
                            found_neighborhoods.append(normalized)
                
                if found_neighborhoods:
                    # Check if "each separately" is requested
                    show_separately = any(phrase in query for phrase in ['كل علي حده', 'كل على حده', 'بشكل منفصل', 'separately', 'each'])
                    
                    if show_separately:
                        # Show pesticides for EACH neighborhood separately
                        st.info(f"🔍 عرض المبيدات في كل حي على حده: {', '.join(found_neighborhoods)}")
                        
                        for neighborhood in found_neighborhoods:
                            st.subheader(f"📍 حي {neighborhood}")
                            
                            sql_query = f"""
                            SELECT 
                                pesticide_name as المبيد,
                                COUNT(*) as عدد_التكرار,
                                ROUND(AVG(concentration), 3) as متوسط_التركيز,
                                SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
                                SUM(CASE WHEN is_above_limit = 0 THEN 1 ELSE 0 END) as تحت_الحد
                            FROM chemistry_tidy
                            WHERE "الحى" LIKE '%{neighborhood}%'
                            AND is_detected = 1
                            AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
                            GROUP BY pesticide_name
                            ORDER BY عدد_التكرار DESC
                            """
                            
                            result = con.execute(sql_query).df()
                            
                            if not result.empty:
                                total = result['عدد_التكرار'].sum()
                                violations = result['فوق_الحد'].sum()
                                st.success(f"✅ حي {neighborhood}: {len(result)} مبيد، {total} اكتشاف، {violations} تجاوز")
                                st.dataframe(result, use_container_width=True)
                            else:
                                st.warning(f"⚠️ لم يتم العثور على مبيدات في حي {neighborhood}")
                            
                            st.markdown("---")
                    else:
                        # Show combined results for all neighborhoods
                        neighborhood_conditions = " OR ".join([f'"الحى" LIKE \'%{n}%\'' for n in found_neighborhoods])
                        
                        st.info(f"🔍 البحث عن المبيدات في: {', '.join(found_neighborhoods)}")
                        
                        sql_query = f"""
                        SELECT 
                            "الحى" as الحي,
                            pesticide_name as المبيد,
                            COUNT(*) as عدد_التكرار,
                            SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد
                        FROM chemistry_tidy
                        WHERE ({neighborhood_conditions})
                        AND is_detected = 1
                        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
                        GROUP BY "الحى", pesticide_name
                        ORDER BY "الحى", عدد_التكرار DESC
                        """
                        
                        result = con.execute(sql_query).df()
                        
                        if not result.empty:
                            st.success(f"✅ وجد {len(result)} سجل في {len(found_neighborhoods)} أحياء")
                            st.dataframe(result, use_container_width=True)
                            st.info("💡 أضف 'كل علي حده' للسؤال لعرض كل حي بشكل منفصل")
                        else:
                            st.warning(f"⚠️ لم يتم العثور على مبيدات في الأحياء المحددة")
                    
                    con.close()
                    return
            
            # Get schema info for prompt
            schema_info = get_sql_schema_info(con)
            
            # Build risk instruction
            risk_instruction = ""
            if include_risk:
                risk_instruction = """
Also include risk assessment in the query:
- Calculate exceedance_ratio (concentration/limit_value) 
- Add CASE WHEN for Risk_Level: 'High' (>2x limit), 'Medium' (1-2x), 'Low' (<1x)
"""
            
            # Pesticide mapping for Arabic names
            pesticide_mapping_str = """
**Arabic Pesticide Name Mapping (use English names in SQL):**
- البايفنثرن = Bifenthrin
- الكلوربيريفوس = Chlorpyrifos  
- الديمثويت = Dimethoate
- اللامبدا = Lambda-cyhalothrin
- الميثوميل = Methomyl
- الكاربندزيم = Carbendazim
- البروفيزين / بوبروفيزين = Buprofezin
- الأباميكتين = Abamectin
- الأسيتامبريد = Acetamiprid
- الإيماميكتين = Emamectin
- الأيميداكلوبريد = Imidacloprid
- الثيامثوكسام = Thiamethoxam
- الدلتامثرين = Deltamethrin
- السيبرمثرين = Cypermethrin
- الفيبرونيل = Fipronil

**Arabic Vegetable Name Mapping (use English names in SQL):**
- طماطم = Tomato
- خيار = Cucumber
- فلفل = Pepper
- باذنجان = Eggplant
- كوسة / كوسا = Squash / Zucchini
- بطاطس = Potato
- جزر = Carrot
- خس = Lettuce
- بقدونس = Parsley
- سبانخ = Spinach
- ملفوف / كرنب = Cabbage
- فراولة = Strawberry
- عنب = Grape
- تفاح = Apple
- برتقال = Orange
"""
            
            # Build SQL prompt with Schema-Only information (TIDY FORMAT)
            prompt = f"""Generate a SQL query to answer this question about the CHEMISTRY testing database.

🔒 Database Schema (Metadata Only - No Actual Data Exposed):

{schema_info}

⚠️ PRIVACY NOTE: You are receiving ONLY schema information.
DO NOT assume specific values exist. Write generic SQL based on column names only.

**TABLE NAME:** chemistry_tidy

**User Question:** {query}
{risk_instruction}

**IMPORTANT SQL RULES:**
1. Use standard SQL syntax (DuckDB compatible)
2. ALWAYS use double quotes for Arabic column names: "اسم العينة"
3. For Arabic text matching, use: WHERE "column" LIKE '%طماطم%'
4. For compliance check: use is_compliant = 1 (compliant) or is_compliant = 0 (violation)
5. Always include ORDER BY for sorted results
6. Use LIMIT 100 for large result sets
7. For Arabic output column names, use AS: SELECT COUNT(*) AS عدد_العينات
8. Filter by year using: WHERE YEAR("التاريخ") = 2024
9. **CASE-INSENSITIVE MATCHING:** Use ILIKE instead of LIKE for case-insensitive search
10. **EXCEEDANCE:** Use exceedance_ratio > 1 to find violations, > 2 for 2x limit violations
11. **ARABIC CHARACTER VARIANTS:** Arabic has interchangeable characters. For name searches, use multiple OR conditions:
    - ى and ي are often interchangeable (الرشيدى = الرشيدي)
    - ة and ه are often interchangeable (منشأة = منشاه)
    - أ/إ/آ and ا are often interchangeable
    - Example: WHERE "اسم المستلم" LIKE '%الرشيد%' (use partial match to avoid ى/ي issue)

**COLUMN REFERENCE (TIDY FORMAT - Separate columns for pesticide data!):**
- `"كود العينة"`: Sample code (unique ID)
- `"التاريخ"`: Sample date (TIMESTAMP)
- `"اسم العينة"`: Sample name in Arabic (طماطم, بقدونس, كمون, etc.)
- `"نوع العينة"`: Sample type (خضراوات, فواكهة, توابل, مكسرات, حبوب, ورقيات)
- `"تصنيف العينة"`: Sample classification (خدمية, تجارية, etc.)
- `"نوع الاختبار"`: Test type (متبقيات مبيدات, افلاتوكسين, سموم فطرية)
- `"الحى"`: Neighborhood name
- `"اسم المنشاة"`: Establishment/facility name
- `"اسم البلدية"`: Municipality name (e.g., بلدية غرب بريدة)
- `"رقم الرخصة"`: License number
- `"حالة العينة"`: Sample condition/status
- `"اسم المستلم"`: Receiver name
- `pesticide_name`: Pesticide/contaminant name (e.g., 'chlorpyrifos', 'imidacloprid', 'bifenthrin')
- `concentration`: Measured concentration value (DOUBLE - numeric!)
- `limit_value`: Maximum allowed limit (DOUBLE - numeric!)
- `exceedance_ratio`: concentration / limit_value (DOUBLE)
- `is_above_limit`: 1 = فوق الحد (concentration > limit), 0 = تحت الحد - CALCULATED FROM DATA
- `is_compliant`: 1 = مطابق (compliant), 0 = غير مطابق (non-compliant) - ORIGINAL LAB JUDGMENT
- `is_detected`: 1 = pesticide detected with value, 0 = no detection (e.g., 'NO CBD')
- `sample_result`: Original lab result text (compliant/non-compliant)

**EXAMPLE QUERIES:**

-- 1. Find all violations (above limit)
SELECT 
    "اسم العينة" AS sample,
    pesticide_name,
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) AS ratio
FROM chemistry_tidy
WHERE is_compliant = 0
ORDER BY exceedance_ratio DESC
LIMIT 20;

-- 2. Count violations by sample type
SELECT 
    "نوع العينة" AS sample_type,
    COUNT(*) AS total,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) AS violations,
    ROUND(SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS violation_rate
FROM chemistry_tidy
WHERE is_detected = 1
GROUP BY "نوع العينة"
ORDER BY violations DESC;

-- 3. Most common pesticides
SELECT 
    pesticide_name AS المبيد,
    COUNT(*) AS عدد_الكشف,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) AS عدد_المخالفات
FROM chemistry_tidy
WHERE is_detected = 1
GROUP BY pesticide_name
ORDER BY عدد_الكشف DESC
LIMIT 15;

-- 4. Find samples with specific pesticide
SELECT 
    "اسم العينة" AS sample,
    "الحى" AS neighborhood,
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) AS ratio,
    is_compliant
FROM chemistry_tidy
WHERE pesticide_name ILIKE '%chlorpyrifos%'
ORDER BY exceedance_ratio DESC;

-- 5. Samples exceeding 2x the limit
SELECT 
    "اسم العينة" AS sample,
    pesticide_name,
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) AS ratio
FROM chemistry_tidy
WHERE exceedance_ratio > 2
ORDER BY exceedance_ratio DESC
LIMIT 20;

-- 6. Average concentration by pesticide
SELECT 
    pesticide_name,
    COUNT(*) AS samples,
    ROUND(AVG(concentration), 4) AS avg_concentration,
    ROUND(AVG(limit_value), 4) AS avg_limit,
    ROUND(AVG(exceedance_ratio), 2) AS avg_ratio
FROM chemistry_tidy
WHERE is_detected = 1
GROUP BY pesticide_name
ORDER BY avg_ratio DESC
LIMIT 15;

-- 7. Violations by neighborhood
-- Neighborhood violations in descending order (Arabic query):
SELECT 
    "الحى" AS الحي,
    COUNT(DISTINCT "كود العينة") AS عدد_العينات,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) AS عدد_المخالفات,
    ROUND(SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(DISTINCT "كود العينة"), 2) AS نسبة_المخالفات
FROM chemistry_tidy
WHERE "الحى" IS NOT NULL
GROUP BY "الحى"
ORDER BY عدد_المخالفات DESC
LIMIT 15;

-- For your specific query: "اعرض الاحياء في ترتيب تنازلي من الاكثر الي الاقل مخالفه"
SELECT 
    "الحى" AS الحي,
    COUNT(*) AS عدد_المخالفات
FROM chemistry_tidy
WHERE is_compliant = 0 AND "الحى" IS NOT NULL
GROUP BY "الحى"
ORDER BY عدد_المخالفات DESC;


-- 8. Total record count
SELECT COUNT(*) AS total_records FROM chemistry_tidy;

-- 9. Samples with multiple pesticides count
SELECT 
    "اسم العينة" AS sample,
    "كود العينة" AS code,
    COUNT(*) AS pesticide_count
FROM chemistry_tidy
WHERE is_detected = 1
GROUP BY "كود العينة", "اسم العينة"
HAVING COUNT(*) > 1
ORDER BY pesticide_count DESC
LIMIT 10;

-- 10. **IMPORTANT: Show last N SAMPLES with ALL their pesticides**
-- NOTE: Each row = 1 pesticide. Use subquery to get all pesticides for last N samples
-- Example: "اعرض آخر 5 عينات مع جميع المبيدات" or "Show last 5 samples with all pesticides"
SELECT 
    "كود العينة" AS code,
    "اسم العينة" AS sample,
    "نوع العينة" AS type,
    pesticide_name,
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) AS ratio,
    is_compliant
FROM chemistry_tidy
WHERE "كود العينة" IN (
    SELECT DISTINCT "كود العينة" 
    FROM chemistry_tidy 
    ORDER BY "كود العينة" DESC 
    LIMIT 5
)
ORDER BY "كود العينة" DESC, contaminant_number;

-- 11. Get ALL pesticides for a specific sample code
-- Example: "اعرض جميع الملوثات في عينة كود 6433"
SELECT 
    "كود العينة" AS code,
    "اسم العينة" AS sample,
    contaminant_number,
    pesticide_name,
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) AS ratio,
    is_compliant
FROM chemistry_tidy
WHERE "كود العينة" = 6433
ORDER BY contaminant_number;

-- 12. **Find samples with exactly N pesticides**
-- Example: "Find samples with exactly 2 pesticides"
SELECT 
    "كود العينة" as sample_code,
    "اسم العينة" as sample_name,
    COUNT(*) as pesticide_count
FROM chemistry_tidy
WHERE is_detected = 1 AND pesticide_name != 'NO DETECTION'
GROUP BY "كود العينة", "اسم العينة"
HAVING COUNT(*) = 2
ORDER BY "كود العينة" DESC
LIMIT 3;

-- 13. **Extract chlorpyrifos readings**
SELECT 
    "كود العينة",
    "اسم العينة",
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) as ratio,
    is_compliant
FROM chemistry_tidy
WHERE pesticide_name ILIKE '%chlorpyrifos%'
ORDER BY concentration DESC;

-- 14. **Count unique samples by neighborhood**
SELECT 
    "الحى" as neighborhood,
    COUNT(DISTINCT "كود العينة") as sample_count
FROM chemistry_tidy
WHERE "الحى" IS NOT NULL
GROUP BY "الحى"
ORDER BY sample_count DESC;

-- 15. **Get all pesticides for sample code X**
SELECT 
    pesticide_name,
    concentration,
    limit_value,
    ROUND(exceedance_ratio, 2) as ratio
FROM chemistry_tidy
WHERE "كود العينة" = 6433
AND is_detected = 1
ORDER BY concentration DESC;

-- 16. **عدد العينات الفريدة لنوع معين (مثل الخيار)**
-- Example: "كم عدد عينات الخيار الفريدة؟"
SELECT COUNT(DISTINCT "كود العينة") as unique_sample_count
FROM chemistry_tidy
WHERE "اسم العينة" LIKE '%خيار%';

-- 17. **المبيدات الموجودة في عينة معينة و تكرارها**
-- Example: "ماهي المبيدات الموجودة في عينة الباذنجان وعدد تكرارها"
SELECT 
    pesticide_name as المبيد,
    COUNT(*) as عدد_التكرار,
    SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as عدد_المخالفات
FROM chemistry_tidy
WHERE "اسم العينة" LIKE '%باذنجان%'
AND is_detected = 1
AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
GROUP BY pesticide_name
ORDER BY عدد_التكرار DESC;

-- 18. **العينات فوق الحد وتحت الحد لنوع معين**
-- Example: "ما هي عينات الطماطم فوق الحد وتحت الحد المسموح"
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    pesticide_name as المبيد,
    concentration as التركيز,
    limit_value as الحد_المسموح,
    CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'تحت الحد' END as الحالة
FROM chemistry_tidy
WHERE "اسم العينة" LIKE '%طماطم%'
AND is_detected = 1
ORDER BY is_above_limit DESC, concentration DESC;

-- 19. **انواع الخضروات في حي معين**
-- Example: "ما هي انواع الخضروات في حي الإسكان وعددها"
SELECT 
    "اسم العينة" as نوع_العينة,
    COUNT(DISTINCT "كود العينة") as عدد_العينات,
    SUM(CASE WHEN is_above_limit = 1 THEN 1 ELSE 0 END) as فوق_الحد,
    SUM(CASE WHEN is_above_limit = 0 AND is_detected = 1 THEN 1 ELSE 0 END) as تحت_الحد
FROM chemistry_tidy
WHERE "الحى" LIKE '%إسكان%' OR "الحى" LIKE '%الاسكان%'
GROUP BY "اسم العينة"
ORDER BY عدد_العينات DESC;

-- 20. **عينات تحتوي على مبيد معين فوق/تحت الحد**
-- Example: "ما هي عينات الطماطم التي تحتوي علي مبيد البايفنثرن"
SELECT 
    "كود العينة" as كود_العينة,
    "اسم العينة" as اسم_العينة,
    pesticide_name as المبيد,
    concentration as التركيز,
    limit_value as الحد,
    CASE WHEN is_above_limit = 1 THEN 'فوق الحد' ELSE 'تحت الحد' END as الحالة
FROM chemistry_tidy
WHERE "اسم العينة" LIKE '%طماطم%'
AND pesticide_name ILIKE '%bifenthrin%'
ORDER BY is_compliant, concentration DESC;

Generate ONLY the SQL query wrapped in ```sql``` markers:"""

            # Generate SQL using AI model
            generated_text = ""
            
            try:
                if hasattr(model, 'chat') and hasattr(model.chat, 'completions'):
                    model_name = st.session_state.get('model_name', 'gpt-4o-mini')
                    response = model.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a SQL expert. Generate ONLY valid SQL queries. Use DuckDB syntax."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0
                    )
                    generated_text = response.choices[0].message.content
                else:
                    response = model.generate_content(prompt)
                    generated_text = response.text
            except Exception as gen_error:
                st.error(f"Generation failed: {gen_error}")
                raise gen_error

            # Extract SQL from response
            sql_match = re.search(r'```sql\n(.*?)```', generated_text, re.DOTALL | re.IGNORECASE)
            
            if sql_match:
                generated_sql = sql_match.group(1).strip()
            else:
                sql_match = re.search(r'```\n(.*?)```', generated_text, re.DOTALL)
                if sql_match:
                    generated_sql = sql_match.group(1).strip()
                else:
                    generated_sql = generated_text.strip()
            
            # Clean up SQL
            generated_sql = generated_sql.replace('```', '').strip()
            
            # Execute SQL
            try:
                result = con.execute(generated_sql).df()
            except Exception as sql_error:
                st.error(f"❌ SQL Error: {sql_error}")
                st.code(generated_sql, language='sql')
                st.info("💡 The AI-generated SQL had an error. Try rephrasing your question.")
                con.close()
                return
            
            # Store results
            st.session_state.last_query_result = result
            st.session_state.last_query = query
            st.session_state.last_generated_code = generated_sql
            st.session_state.last_query_type = 'sql'
            
            st.success("✅ Query executed successfully!")
            st.markdown("### 📊 Results:")
            
            if result is None or result.empty:
                st.warning("⚠️ No results found for this query.")
            else:
                # Show metrics for common result types
                if len(result.columns) > 0:
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    
                    # Handle new metric columns
                    if 'sample_count' in result.columns or 'pesticide_count' in result.columns:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if 'sample_count' in result.columns:
                                st.metric("📊 Total Samples", result['sample_count'].sum())
                            elif 'pesticide_count' in result.columns:
                                st.metric("🧪 Total Pesticides", result['pesticide_count'].sum())
                        
                        with col2:
                            if 'violations' in result.columns:
                                st.metric("🔴 Violations", result['violations'].sum())
                        
                        with col3:
                            if 'is_compliant' in result.columns:
                                compliant = result['is_compliant'].sum()
                                total = len(result)
                                st.metric("🟢 Compliance Rate", f"{compliant/total*100:.1f}%" if total > 0 else "0%")
                    
                    elif 'عدد_العينات' in result.columns or 'عدد العينات' in result.columns:
                        col_name = 'عدد_العينات' if 'عدد_العينات' in result.columns else 'عدد العينات'
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Total Records", result[col_name].sum() if pd.api.types.is_numeric_dtype(result[col_name]) else len(result))
                        with col2:
                            if 'فوق_الحد' in result.columns or 'فوق الحد' in result.columns:
                                above_col = 'فوق_الحد' if 'فوق_الحد' in result.columns else 'فوق الحد'
                                st.metric("🔴 Above Limit", result[above_col].sum())
                        with col3:
                            if 'تحت_الحد' in result.columns or 'تحت الحد' in result.columns:
                                below_col = 'تحت_الحد' if 'تحت_الحد' in result.columns else 'تحت الحد'
                                st.metric("🟢 Below Limit", result[below_col].sum())
                
                # Display results
                st.dataframe(result, use_container_width=True)
                
                # Download button
                csv = result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            
            # Show generated SQL
            with st.expander("🔍 View Generated SQL"):
                st.code(generated_sql, language='sql')
                st.caption("This SQL was AI-generated and executed on DuckDB")
            
            # Close connection
            con.close()
            
        except Exception as e:
            st.error(f"❌ Error in SQL analysis: {str(e)}")
            with st.expander("🔍 Debug Details"):
                st.exception(e)
                if 'generated_sql' in locals():
                    st.markdown("**Generated SQL:**")
                    st.code(generated_sql, language='sql')


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
            
            # First, detect the query type to apply the right analysis approach
            query_lower = query.lower()
            
            # Detect query type
            is_trend_query = any(kw in query_lower for kw in ['trend', 'by year', 'over time', 'yearly', 'annual', 'year over year', 'from 2', 'to 2'])
            is_pesticide_query = any(kw in query_lower for kw in ['pesticide', 'which pesticide', 'top pesticide', 'most violations', 'pesticides with'])
            is_vegetable_arabic_query = any(kw in query for kw in ['الخضروات', 'عدد', 'فوق الحد', 'تحت الحد', 'الحي'])
            is_compliance_query = any(kw in query_lower for kw in ['compliance', 'compliant', 'non-compliant', 'violation rate'])
            
            # Build context-specific instructions
            if is_trend_query:
                specific_instructions = """
**QUERY TYPE DETECTED: TREND/YEARLY ANALYSIS**
- Group data by 'year' column
- Calculate counts, rates, or averages per year
- Result should show trends: year | metric1 | metric2 | ...
- Sort by year ascending
- DO NOT use Arabic column names for this query
- Example columns: 'year', 'total_samples', 'violations', 'compliance_rate'
"""
            elif is_pesticide_query:
                specific_instructions = """
**QUERY TYPE DETECTED: PESTICIDE ANALYSIS**
- Group by 'pesticide_standardized' or similar pesticide column
- Count violations using 'is_compliant' == 0 or 'result' containing 'غير مطابق'
- Show: pesticide name | total tests | violations | violation_rate
- Sort by violations or violation_rate descending
- DO NOT use Arabic column names unless specifically asked
- Show top 10-20 results
"""
            elif is_compliance_query:
                specific_instructions = """
**QUERY TYPE DETECTED: COMPLIANCE ANALYSIS**
- Calculate compliance metrics (compliant vs non-compliant)
- Use 'is_compliant' column if available (1=compliant, 0=violation)
- Or check 'result' column for 'مطابق' (compliant) vs 'غير مطابق' (non-compliant)
- Show rates as percentages
- Group by the relevant dimension (year, vegetable, pesticide, etc.)
"""
            elif is_vegetable_arabic_query:
                specific_instructions = """
**QUERY TYPE DETECTED: VEGETABLE STATISTICS (ARABIC FORMAT)**
- ONLY use this format when user explicitly asks in Arabic or wants Arabic output
- Group by vegetable (use 'vegetable_arabic' column)
- Calculate: total samples, above limit, below limit
- Use Arabic column names: 'عدد العينات', 'فوق الحد', 'تحت الحد'
- Filter by neighborhood if mentioned in query
"""
            else:
                specific_instructions = """
**QUERY TYPE: GENERAL ANALYSIS**
- Analyze the data based on the specific question
- Use appropriate groupby, aggregation, or filtering
- Use English column names unless Arabic is specifically requested
- Return a clear, well-formatted DataFrame
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
            
            prompt = f"""Generate pandas code to answer this question about the pesticide testing dataset.

**IMPORTANT: READ THE QUESTION CAREFULLY AND ANSWER EXACTLY WHAT IS ASKED!**

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

User Question: {query}
{risk_instruction}

{specific_instructions}

**AVAILABLE VARIABLES IN SCOPE:**
- df: The main DataFrame (already loaded)
- pd: pandas module
- np: numpy module  
- query: The user's question as a string
- pesticide_mapping: Dictionary mapping Arabic pesticide names to English
  Example: pesticide_mapping['البايفنثرن'] = 'Bifenthrin'
  Use this to translate Arabic pesticide names when filtering

**CRITICAL REQUIREMENTS:**
1. ANSWER THE EXACT QUESTION - don't provide generic summaries
2. Store final result in variable 'result'
3. Execute code DIRECTLY - do NOT wrap in a function
4. Do NOT use 'return' statement
5. Handle missing values with .fillna() or dropna()
6. **STRING FILTERING - CRITICAL:** 
   - ALWAYS use `na=False` parameter with .str.contains() to handle NaN values
   - Example: df[df['vegetable_english'].str.contains('Cucumber', case=False, na=False)]
   - Never use .str.contains() without na=False parameter
7. **PESTICIDE NAME MATCHING:** 
   - If query mentions pesticide name in Arabic (e.g., 'البايفنثرن'), use pesticide_mapping
   - Example: pesticide_name = pesticide_mapping.get('البايفنثرن', 'Bifenthrin')
   - Then filter: df[df['pesticide_standardized'].str.contains(pesticide_name, case=False, na=False)]

**KEY COLUMNS TO USE:**
- 'year' - for yearly analysis
- 'is_compliant' - 1=compliant, 0=violation
- 'pesticide_standardized' - pesticide name
- 'vegetable_english' - vegetable name in English
- 'reading' - measured value
- 'limits' - maximum allowed limit

**EXAMPLE CODE PATTERNS:**

For "compliance trends by year":
```python
result = df.groupby('year').agg(
    total_samples=('is_compliant', 'count'),
    compliant=('is_compliant', 'sum'),
    violations=('is_compliant', lambda x: (x == 0).sum())
).reset_index()
result['compliance_rate'] = (result['compliant'] / result['total_samples'] * 100).round(2)
result = result.sort_values('year')
```

For "pesticides with most violations":
```python
result = df[df['is_compliant'] == 0].groupby('pesticide_standardized').size().reset_index(name='violations')
result = result.sort_values('violations', ascending=False).head(15)
```

For "total number of cucumber samples" (simple count):
```python
# Count total samples for cucumber (IMPORTANT: use na=False to handle NaN)
cucumber_samples = df[df['vegetable_english'].str.contains('Cucumber', case=False, na=False)]
total_count = len(cucumber_samples)

# Create result DataFrame
result = pd.DataFrame({{
    'الخضار': ['خيار (Cucumber)'],
    'عدد العينات الكلي': [total_count]
}})
```

For "tomato samples with Bifenthrin above/below limit" (Arabic query):
```python
# Extract pesticide name from Arabic
pesticide_name = pesticide_mapping.get('البايفنثرن', 'Bifenthrin')

# Filter for tomato and this pesticide (IMPORTANT: use na=False)
filtered = df[
    (df['vegetable_english'].str.contains('Tomato', case=False, na=False)) &
    (df['pesticide_standardized'].str.contains(pesticide_name, case=False, na=False))
].copy()

# Calculate above/below limit
above_limit = (filtered['is_compliant'] == 0).sum()
below_limit = (filtered['is_compliant'] == 1).sum()

# Create result DataFrame
result = pd.DataFrame({{
    'الخضار': ['الطماطم'],
    'المبيد': [pesticide_name],
    'عدد العينات': [len(filtered)],
    'فوق الحد': [above_limit],
    'تحت الحد': [below_limit]
}})
```

For "vegetables in neighborhood with compliance stats" (complex Arabic query):
```python
# Example: "ما هي انواع الخضروات الموجودة في حي الاسكان و ما عددها و ماهي العينات فوق الحد المسموح وتحت الحد المسموح"
# Filter by neighborhood (IMPORTANT: use na=False)
neighborhood_data = df[df['neighborhood_arabic'].str.contains('الإسكان', case=False, na=False)].copy()

# Group by vegetable and calculate compliance stats
result = neighborhood_data.groupby('vegetable_arabic').agg(
    عدد_العينات=('vegetable_arabic', 'count'),
    فوق_الحد=('is_compliant', lambda x: (x == 0).sum()),
    تحت_الحد=('is_compliant', lambda x: (x == 1).sum())
).reset_index()

# Rename columns for clarity
result.columns = ['الخضار', 'عدد العينات', 'فوق الحد', 'تحت الحد']
result = result.sort_values('عدد العينات', ascending=False)
```

For "vegetables affected by pesticide with compliance" (Arabic query):
```python
# Example: "ما هي الخضروات التي يشملها مبيد البايفنثرن و ماهي العينات تحت الحد و فوق الحد المسموح"
# Translate pesticide name
pesticide_name = pesticide_mapping.get('البايفنثرن', 'Bifenthrin')

# Filter by pesticide
pesticide_data = df[df['pesticide_standardized'].str.contains(pesticide_name, case=False, na=False)].copy()

# Group by vegetable and calculate compliance
result = pesticide_data.groupby('vegetable_arabic').agg(
    عدد_العينات=('vegetable_arabic', 'count'),
    تحت_الحد=('is_compliant', lambda x: (x == 1).sum()),
    فوق_الحد=('is_compliant', lambda x: (x == 0).sum())
).reset_index()

result.columns = ['الخضار', 'عدد العينات', 'تحت الحد', 'فوق الحد']
result = result.sort_values('عدد العينات', ascending=False)
```

Generate ONLY the Python code wrapped in ```python``` markers:"""


            # Generate response using DeepSeek (first priority) or Gemini (fallback)
            generated_text = ""
            
            try:
                # Check if it looks like an OpenAI/DeepSeek client
                if hasattr(model, 'chat') and hasattr(model.chat, 'completions'):
                    # Get model name from session state (either gpt-4o-mini or qwen2.5-coder:1.5b)
                    model_name = st.session_state.get('model_name', 'gpt-4o-mini')
                    
                    response = model.chat.completions.create(
                        model=model_name,  # Use dynamic model name
                        messages=[
                            {"role": "system", "content": "You are a Python data analysis expert. Output ONLY python code. 'df' (DataFrame) and 'query' (str) are already defined. Do not define functions."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0
                    )
                    generated_text = response.choices[0].message.content
                else:
                    # Assume Gemini GenerativeModel
                    response = model.generate_content(prompt)
                    generated_text = response.text
            except Exception as gen_error:
                # If DeepSeek model name fails, try a generic one or fallback
                st.error(f"Generation failed: {gen_error}")
                raise gen_error

            code_match = re.search(r'```python\n(.*?)```', generated_text, re.DOTALL)
            
            if code_match:
                generated_code = code_match.group(1).strip()
            else:
                code_match = re.search(r'```\n(.*?)```', generated_text, re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1).strip()
                else:
                    generated_code = generated_text.strip()
            
            # Create pesticide name mapping for Arabic queries
            pesticide_mapping = {
                'البايفنثرن': 'Bifenthrin',
                'الكلوربيريفوس': 'Chlorpyrifos',
                'الديمثويت': 'Dimethoate',
                'اللامبدا': 'Lambda-cyhalothrin',
                'الميثوميل': 'Methomyl',
                'الكاربندزيم': 'Carbendazim',
                'الإيماميكتين': 'Emamectin',
                'الأيميداكلوبريد': 'Imidacloprid',
                'الثيامثوكسام': 'Thiamethoxam',
                'البروفيزين': 'Buprofezin',
                'الأباميكتين': 'Abamectin',
                'الأسيتامبريد': 'Acetamiprid',
                'الدلتامثرين': 'Deltamethrin',
                'السيبرمثرين': 'Cypermethrin'
            }
            
            exec_globals = {
                'df': df.copy(),
                'pd': pd,
                'np': np,
                'query': query,
                'pesticide_mapping': pesticide_mapping,  # Add mapping to globals
                'result': None
            }
            
            exec(generated_code, exec_globals)
            
            result = exec_globals.get('result')
            
            # Store results in session state for persistence across button clicks
            st.session_state.last_query_result = result
            st.session_state.last_query = query
            st.session_state.last_generated_code = generated_code
            st.session_state.last_include_risk = include_risk
            
            st.success("✅ Analysis completed!")
            st.markdown("### 📊 Results:")
            
            # Always show generated code when debugging (moved up for None results)
            if result is None:
                st.warning("⚠️ The generated code did not assign a value to the 'result' variable. Check the code below:")
                st.code(generated_code, language='python')
                st.info("""
                **Tip:** The AI should assign the final output to a variable called `result`. 
                Example: `result = df.groupby(...).agg(...)`
                """)
            elif isinstance(result, pd.DataFrame):
                # Smart metrics display based on result type
                result_cols = set(result.columns)
                
                # Type 1: Arabic vegetable statistics format
                if 'vegetable_arabic' in result_cols and 'عدد العينات' in result_cols:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        try:
                            total = result['عدد العينات'].sum() if pd.api.types.is_numeric_dtype(result['عدد العينات']) else len(result)
                            st.metric("🟢 Total Samples", total)
                        except:
                            st.metric("🟢 Total Samples", len(result))
                    with col2:
                        try:
                            above = result['فوق الحد'].sum() if 'فوق الحد' in result_cols else 0
                            st.metric("🔴 Above Limit", above)
                        except:
                            st.metric("🔴 Above Limit", "N/A")
                    with col3:
                        try:
                            below = result['تحت الحد'].sum() if 'تحت الحد' in result_cols else 0
                            st.metric("🟡 Below Limit", below)
                        except:
                            st.metric("🟡 Below Limit", "N/A")
                
                # Type 2: Yearly trend data
                elif 'year' in result_cols and len(result) <= 10:
                    cols = result_cols - {'year'}
                    metric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(result[c])]
                    if metric_cols:
                        display_cols = metric_cols[:4]  # Show up to 4 metrics
                        metric_columns = st.columns(len(display_cols))
                        for i, col_name in enumerate(display_cols):
                            with metric_columns[i]:
                                try:
                                    if 'rate' in col_name.lower() or 'percent' in col_name.lower():
                                        latest = result[col_name].iloc[-1]
                                        st.metric(f"📈 Latest {col_name.replace('_', ' ').title()}", f"{latest:.1f}%")
                                    else:
                                        total = result[col_name].sum()
                                        st.metric(f"📊 Total {col_name.replace('_', ' ').title()}", f"{total:,.0f}")
                                except:
                                    pass
                
                # Type 3: Pesticide/violation analysis
                elif 'pesticide_standardized' in result_cols or 'pesticide' in str(result_cols).lower():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🧪 Pesticides Analyzed", len(result))
                    with col2:
                        viol_col = [c for c in result_cols if 'violation' in c.lower()]
                        if viol_col:
                            total_violations = result[viol_col[0]].sum()
                            st.metric("⚠️ Total Violations", f"{total_violations:,.0f}")
                
                # Type 4: Generic result - just show row count
                else:
                    st.metric("📋 Results Found", len(result))
                
                st.dataframe(result, use_container_width=True)
                
                # ADDED: Show detailed individual samples for vegetable/neighborhood queries
                if 'vegetable_arabic' in result_cols and 'عدد العينات' in result_cols:
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Individual Samples")
                    st.caption("Showing all individual test samples (not grouped)")
                    
                    # Try to re-query the original data to show individual samples
                    try:
                        # Extract neighborhood from query if present
                        neighborhood_keywords = {
                            'الاسكان': 'Al Iskan',
                            'الموطأ': 'Al Mowata', 
                            'الصفراء': 'Al Safra',
                            'الأفق': 'Al Ufuq',
                            'الريان': 'Al Rayyan',
                            'السلام': 'Al Salam'
                        }
                        
                        neighborhood_found = None
                        for arabic, english in neighborhood_keywords.items():
                            if arabic in query:
                                neighborhood_found = english
                                break
                        
                        if neighborhood_found and 'neighborhood_english' in df.columns:
                            # Filter for this specific neighborhood
                            detailed_df = df[df['neighborhood_english'] == neighborhood_found].copy()
                            
                            # Select relevant columns
                            detail_cols = []
                            if 'vegetable_arabic' in detailed_df.columns:
                                detail_cols.append('vegetable_arabic')
                            if 'pesticide_standardized' in detailed_df.columns:
                                detail_cols.append('pesticide_standardized')
                            if 'reading' in detailed_df.columns:
                                detail_cols.append('reading')
                            if 'limits' in detailed_df.columns:
                                detail_cols.append('limits')
                            if 'result' in detailed_df.columns:
                                detail_cols.append('result')
                            if 'is_compliant' in detailed_df.columns:
                                detail_cols.append('is_compliant')
                            if 'date' in detailed_df.columns:
                                detail_cols.append('date')
                            
                            if detail_cols:
                                detailed_display = detailed_df[detail_cols].copy()
                                
                                # Add above/below limit indicator
                                if 'is_above_limit' in detailed_display.columns:
                                    detailed_display['حالة العينة'] = detailed_display['is_above_limit'].map({
                                        1: '🔴 فوق الحد',
                                        0: '🟢 تحت الحد'
                                    })
                                elif 'is_compliant' in detailed_display.columns:
                                    # Fallback for old data
                                    detailed_display['حالة العينة'] = detailed_display['is_compliant'].map({
                                        0: '🔴 فوق الحد',
                                        1: '🟢 تحت الحد'
                                    })
                                
                                st.info(f"📊 Total individual samples found: **{len(detailed_display)}** samples")
                                st.dataframe(detailed_display, use_container_width=True)
                                
                                # Add download button for detailed data
                                csv = detailed_display.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label="📥 Download Detailed Data (CSV)",
                                    data=csv,
                                    file_name=f"detailed_samples_{neighborhood_found}.csv",
                                    mime="text/csv",
                                )
                    except Exception as e:
                        st.info(f"💡 Showing grouped summary. Individual samples: {st.session_state.get('last_query', 'N/A')}")
                
                # Note: Chart buttons are now handled by display_stored_results_with_charts()
                # which is called after this function to persist across button clicks
                
                st.markdown("---")
                
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
                
                # Show code in expander for successful queries
                with st.expander("👨‍💻 View Generated Code"):
                    st.code(generated_code, language='python')
                    st.caption("This code was AI-generated and executed on your dataset")
                
            elif isinstance(result, (pd.Series, list, dict)):
                st.dataframe(pd.DataFrame(result), use_container_width=True)
                with st.expander("👨‍💻 View Generated Code"):
                    st.code(generated_code, language='python')
            else:
                st.write(result)
                with st.expander("👨‍💻 View Generated Code"):
                    st.code(generated_code, language='python')
            
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


# In your display_stored_results_with_charts function:
def show_chat_page(api_client):
    """
    Fixed chat page that works with simple query system
    Just copy-paste this to replace your existing show_chat_page function
    
    Args:
        api_client: Your existing API client (or None)
    """
    
    st.header("💬 AI Assistant 🌱")
    st.caption("Powered by GPT-4o-mini - Ask questions in natural language")
    
    # ISO Document Management in Sidebar
    with st.sidebar:
        with st.expander("📄 ISO Document Upload"):
            uploaded_file = st.file_uploader(
                "Upload ISO Document",
                type=['pdf', 'docx', 'xlsx'],
                help="Limit 200MB per file"
            )
            
            doc_type = st.selectbox(
                "Document Type",
                ["Quality Manual", "Procedure", "Work Instruction", "Form"]
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
            
        # Settings
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        risk_mode = st.radio(
            "Risk Assessment Mode",
            ["🔵 Auto-detect", "✅ Always On", "❌ Always Off"]
        )
        
        # Apply risk mode to detection
        if risk_mode == "✅ Always On":
            has_risk = True  # Force risk assessment
        elif risk_mode == "❌ Always Off":
            has_risk = False  # Disable risk assessment
        else:
            has_risk = None # Auto-detect will handle this later
        
        # 🆕 Query Engine Selection (SQL vs Pandas)
        st.markdown("---")
        st.markdown("### 🗄️ Query Engine")
        
        # Check if DuckDB database exists
        # Note: ai_assistant.py is in modules/, database is in ../data/
        db_path = Path(__file__).parent.parent / 'data' / 'lars_data.duckdb'
        db_exists = db_path.exists()

        
        if db_exists:
            query_engine = st.radio(
                "Choose Query Engine",
                ["🚀 SQL (DuckDB)", "🐼 Pandas"],
                help="SQL is faster and more secure. Pandas is the legacy option."
            )
            st.session_state.query_engine = 'sql' if 'SQL' in query_engine else 'pandas'
            
            if 'SQL' in query_engine:
                st.success("⚡ SQL mode: Faster queries with DuckDB")
            else:
                st.info("🐼 Pandas mode: Legacy query engine")
        else:
            st.warning("⚠️ DuckDB database not found. Using Pandas.")
            st.caption("Run `python migrate_db.py` to enable SQL mode.")
            st.session_state.query_engine = 'pandas'
        
        # ========================================================================
        # DATE FILTER SECTION
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📅 Date Filter")
        
        # Initialize date filter state
        if 'date_filter_enabled' not in st.session_state:
            st.session_state.date_filter_enabled = False
        if 'date_filter_start' not in st.session_state:
            st.session_state.date_filter_start = pd.Timestamp('2022-01-01').date()
        if 'date_filter_end' not in st.session_state:
            st.session_state.date_filter_end = pd.Timestamp.now().date()
        
        # Enable/Disable toggle
        date_filter_enabled = st.checkbox(
            "🔘 Enable Date Filter",
            value=st.session_state.date_filter_enabled,
            help="When enabled, your queries will be filtered to the selected date range",
            key="date_filter_checkbox"
        )
        st.session_state.date_filter_enabled = date_filter_enabled
        
        if date_filter_enabled:
            from datetime import timedelta
            today = pd.Timestamp.now().date()
            
            # Quick Date Buttons
            st.markdown("**⚡ Quick Select:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📅 Today", use_container_width=True, key="qd_today"):
                    st.session_state.date_filter_start = today
                    st.session_state.date_filter_end = today
                
                if st.button("📆 This Week", use_container_width=True, key="qd_week"):
                    st.session_state.date_filter_start = today - timedelta(days=today.weekday())
                    st.session_state.date_filter_end = today
            
            with col2:
                if st.button("🗓️ This Month", use_container_width=True, key="qd_month"):
                    st.session_state.date_filter_start = today.replace(day=1)
                    st.session_state.date_filter_end = today
                
                if st.button("📊 Last 3 Months", use_container_width=True, key="qd_3months"):
                    st.session_state.date_filter_start = today - timedelta(days=90)
                    st.session_state.date_filter_end = today
            
            with col3:
                if st.button("📈 This Year", use_container_width=True, key="qd_year"):
                    st.session_state.date_filter_start = today.replace(month=1, day=1)
                    st.session_state.date_filter_end = today
                
                if st.button("📚 All Data", use_container_width=True, key="qd_all"):
                    st.session_state.date_filter_start = pd.Timestamp('2022-01-01').date()
                    st.session_state.date_filter_end = today
            
            # Show current filter with formatting
            st.markdown("---")
            start_str = st.session_state.date_filter_start.strftime("%Y-%m-%d")
            end_str = st.session_state.date_filter_end.strftime("%Y-%m-%d")
            st.success(f"📅 **{start_str}** → **{end_str}**")
            
            # Custom date input using text input (more reliable)
            with st.expander("📆 Custom Date Range"):
                col1, col2 = st.columns(2)
                with col1:
                    custom_start = st.date_input(
                        "Start Date", 
                        value=st.session_state.date_filter_start,
                        key="custom_start_date"
                    )
                with col2:
                    custom_end = st.date_input(
                        "End Date", 
                        value=st.session_state.date_filter_end,
                        key="custom_end_date"
                    )
                
                if st.button("✅ Apply Custom Range", use_container_width=True, key="apply_custom"):
                    st.session_state.date_filter_start = custom_start
                    st.session_state.date_filter_end = custom_end
                    st.success(f"✅ Applied: {custom_start} → {custom_end}")
                    st.rerun()
    
    # ========================================================================
    # STEP 1: Get DataFrame (adjust based on your app)
    # ========================================================================
    
    try:
        # Try multiple ways to get the dataframe
        if api_client and hasattr(api_client, 'get_data'):
            df = api_client.get_data()
        elif api_client and hasattr(api_client, 'df'):
            df = api_client.df
        elif 'current_data' in st.session_state:
            df = st.session_state.current_data
        elif 'df' in st.session_state:
            df = st.session_state.df
        else:
            st.warning("⚠️ No data loaded. Please load data first.")
            st.info("💡 Go to the data loading page or upload your Excel file.")
            return
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # ========================================================================
    # STEP 2: Initialize AI Model (OpenAI or Ollama)
    # ========================================================================
    
    # Add model selection toggle in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Model Selection")
        
        model_choice = st.radio(
            "Select AI Model",
            ["🟢 OpenAI (gpt-4o-mini)", "🔵 Ollama (mistral:7b-instruct)"],
            help="Choose between OpenAI API or local Ollama model"
        )
        
        if model_choice == "🟢 OpenAI (gpt-4o-mini)":
            st.info("Using OpenAI Cloud API")
        else:
            st.info("Using Local Ollama Model")
    
    # Initialize based on selection
    model = None
    model_name = None
    
    if model_choice == "🟢 OpenAI (gpt-4o-mini)":
        # OpenAI Configuration
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not set")
            st.info("""
            **How to fix:**
            1. Get your API key from OpenAI
            2. Set it in your .env.app file as `OPENAI_API_KEY`
            """)
            return
        
        # Configure OpenAI Client
        try:
            model = OpenAI(api_key=api_key)
            model_name = "gpt-4o-mini"
            st.success("✅ OpenAI model ready")
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return
    
    else:  # Ollama
        # Ollama Configuration
        try:
            from openai import OpenAI as OllamaClient
            
            # Initialize Ollama client (defaults to http://localhost:11434)
            model = OllamaClient(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Ollama doesn't need real API key
            )
            model_name = "mistral:7b-instruct"
            
            # Test if Ollama is running
            try:
                # Quick test call
                test_response = model.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.success(f"✅ Ollama model ready: {model_name}")
            except Exception as test_error:
                st.error(f"⚠️ Ollama not responding: {test_error}")
                st.info("""
                **How to fix:**
                1. Install Ollama: https://ollama.ai
                2. Run: `ollama pull mistral:7b-instruct`
                3. Ollama should auto-start, or run: `ollama serve`
                """)
                return
                
        except Exception as e:
            st.error(f"Failed to initialize Ollama client: {e}")
            st.info("Make sure Ollama is installed and running on http://localhost:11434")
            return
    
    # Store model info in session state for use in process_data_query
    st.session_state.model_name = model_name
    
    # ========================================================================
    # STEP 3: Show Instructions
    # ========================================================================
    
    st.info("""
    **🌱 How to use:**
    1. Type your question about the pesticide data
    2. Press Enter or click "Ask"
    3. View the AI-generated code and results
    
    **Example Questions:**
    - "What are the top 5 vegetables with highest violation rates?"
    - "Show me compliance trends by year"
    - "Which pesticides have the most violations?"
    - "Calculate average exceedance ratio by vegetable"
    - "Find vegetables with more than 10 violations"
    """)
    
    # ========================================================================
    # STEP 4: Get User Input (THIS IS THE KEY PART!)
    # ========================================================================
    
    # Method 1: Chat input (modern Streamlit way)
    chat_question = st.chat_input("Ask a question about the data...")
    
    # Method 2: Text area + button (alternative)
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        text_question = st.text_area(
            "Or type your question here:",
            height=100,
            placeholder="Example: What are the top 10 vegetables by violation count?"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    # Determine which input to use
    user_question = chat_question or (text_question if ask_button else None)
    
    # ========================================================================
    # STEP 5: Process Question
    # ========================================================================
    
    # ================================================================
    # HANDLE QUICK QUERY (from buttons at bottom)
    # ================================================================
    if 'quick_query' in st.session_state and st.session_state.quick_query:
        user_question = st.session_state.quick_query
        st.session_state.quick_query = None  # Clear it immediately
        # Don't rerun - just process it below
    
    if user_question:
        # DETECT QUERY TYPE
        iso_keywords = ['iso', '17025', 'standard', 'procedure', 'manual', 
                        'quality', 'calibration', 'validation', 'document', 'sop']
        risk_keywords = ['risk', 'assessment', 'health', 'safety', 'adi', 'arfd']
        
        query_lower = user_question.lower()
        is_iso = any(kw in query_lower for kw in iso_keywords)
        
        # Handle auto-detect risk
        if has_risk is None:
            has_risk = any(kw in query_lower for kw in risk_keywords)
        
        # HANDLE ISO QUERIES WITH RAG
        if is_iso:
            with st.spinner("🔍 Searching ISO documents..."):
                if hasattr(api_client, 'nlp_controller'):
                    try:
                        result = api_client.nlp_controller.query(user_question)
                        st.success("✅ Found ISO information")
                        st.markdown("### 📋 Answer:")
                        st.write(result.get('answer', 'No information found'))
                        
                        if result.get('sources'):
                            with st.expander("📚 Sources"):
                                for src in result['sources']:
                                    st.write(f"• {src}")
                    except Exception as e:
                        st.error(f"ISO search error: {e}")
                else:
                    st.info("ISO documents not configured. Falling back to general query.")
                    is_iso = False  # Fall back to Gemini
        
        # HANDLE PESTICIDE QUERIES WITH GEMINI (your existing code)
        if not is_iso:  # Only run if not an ISO query
            # Apply risk assessment settings
            risk_mode = st.session_state.get('risk_mode', '🤖 Auto-detect')
            if risk_mode == "✅ Always On":
                include_risk = True
            elif risk_mode == "❌ Always Off":
                include_risk = False
            else:  # Auto-detect
                risk_keywords = ['risk', 'safety', 'safe', 'health', 'assess', 'eat', 'consume', 'adult', 'child', 'population']
                include_risk = any(kw in user_question.lower() for kw in risk_keywords)
            
            if include_risk:
                st.success("🎯 Risk assessment enabled")
            
            # ================================================================
            # APPLY DATE FILTER IF ENABLED
            # ================================================================
            
            df_filtered = df.copy()
            
            if st.session_state.get('date_filter_enabled', False):
                start_date = st.session_state.get('date_filter_start')
                end_date = st.session_state.get('date_filter_end')
                
                if start_date and end_date and 'document_date' in df_filtered.columns:
                    # Convert to datetime for comparison
                    df_filtered['document_date'] = pd.to_datetime(df_filtered['document_date'], errors='coerce')
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include full end date
                    
                    # Apply date filter
                    df_filtered = df_filtered[
                        (df_filtered['document_date'] >= start_ts) & 
                        (df_filtered['document_date'] <= end_ts)
                    ]
                    
                    # Show filter status
                    st.info(f"📅 Date filter active: {start_date} → {end_date} ({len(df_filtered):,} records)")
                    
                    if len(df_filtered) == 0:
                        st.warning("⚠️ No data found in the selected date range. Try a different range.")
                        return
            
            # Choose query engine based on sidebar selection
            query_engine = st.session_state.get('query_engine', 'pandas')
            
            if query_engine == 'sql':
                # 🆕 Use SQL (DuckDB) - faster and more secure
                st.info("🚀 Using SQL (DuckDB) query engine")
                process_data_query_sql(user_question, model, include_risk)
            else:
                # Legacy: Use Pandas
                st.info("🐼 Using Pandas query engine")
                process_data_query(user_question, df_filtered, model, include_risk)

    # ========================================================================
    # STEP 6.5: Display stored results with chart buttons (PERSISTENT)
    # ========================================================================
    
    # Always display stored results - this persists across button clicks
    display_stored_results_with_charts()

    # ========================================================================
    # STEP 6.6: Data Entry Form (when toggled)
    # ========================================================================
    
    # Show data entry button in sidebar
    show_data_entry_sidebar()
    
    # Display data entry form if toggled
    if st.session_state.get('show_data_entry_form', False):
        show_data_entry_form()

    # ========================================================================
    # STEP 7: Show Chat History (Optional)
    # ========================================================================
    
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Recent Queries")
        
        # Show last 5 queries
        for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q{i+1}: {item['question'][:60]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                if isinstance(item['result'], pd.DataFrame):
                    st.dataframe(item['result'].head(10))
                else:
                    st.write(item['result'])
                
                show_code = st.checkbox("👨‍💻 Show generated code", value=True, key=f"show_code_hist_{i}")
                if show_code:
                    st.code(item['code'], language='python')
                        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # ========================================================================
    # STEP 8: Quick Query Buttons (Optional)
    # ========================================================================
    
    st.markdown("---")
    st.subheader("⚡ Quick Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Top Vegetables"):
            # Trigger quick query by setting it in session state
            st.session_state.quick_query = "What are the top 10 vegetables by violation count?"
            st.rerun()
    
    with col2:
        if st.button("🧪 Top Pesticides"):
            st.session_state.quick_query = "What are the top 10 pesticides by violation count?"
            st.rerun()
    
    with col3:
        if st.button("📈 Yearly Trends"):
            st.session_state.quick_query = "Show me compliance rate by year"
            st.rerun()


def format_result_as_table(df: pd.DataFrame, query_type: str = "statistics") -> pd.DataFrame:
    """Format DataFrame as a clean table for display"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Remove any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    if query_type == "vegetable_statistics":
        # Ensure we have the right columns for vegetable statistics
        expected_cols = ['vegetable_arabic', 'عدد العينات', 'فوق الحد', 'تحت الحد']
        for expected in expected_cols:
            if expected not in df.columns:
                # Try to find similar columns
                for col in df.columns:
                    if any(word in str(col) for word in ['vegetable', 'خضروات', 'نوع']):
                        df = df.rename(columns={col: 'vegetable_arabic'})
                    elif any(word in str(col) for word in ['count', 'عدد', 'عينات']):
                        df = df.rename(columns={col: 'عدد العينات'})
                    elif any(word in str(col) for word in ['above', 'فوق', 'exceed']):
                        df = df.rename(columns={col: 'فوق الحد'})
                    elif any(word in str(col) for word in ['below', 'تحت', 'compliant']):
                        df = df.rename(columns={col: 'تحت الحد'})
    
    # Format numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if 'reading' in str(col).lower() or 'limits' in str(col).lower():
            df[col] = df[col].apply(lambda x: f"{float(x):.2f}" if pd.notnull(x) else "N/A")
        elif any(word in str(col).lower() for word in ['edi', 'hq', 'ratio', 'percentage']):
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}" if pd.notnull(x) else "N/A")
    
    return df
def get_duckdb_connection():
    """Get DuckDB connection to the LARS database"""
    db_path = Path(__file__).parent.parent / 'data' / 'lars_data.duckdb'
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)

def get_duckdb_connection_write():
    """Get DuckDB connection to the LARS database with write access"""
    db_path = Path(__file__).parent.parent / 'data' / 'lars_data.duckdb'
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=False)

def transform_pesticide_columns(df):
    """
    Transform pesticide columns (ملوث 1 to ملوث 8) into separate rows
    Each row will have: sample_code, pesticide_name, concentration, limit_value
    """
    # Find pesticide columns dynamically (they contain 'ملوث' in their names)
    pesticide_columns = [col for col in df.columns if 'ملوث' in col]
    
    transformed_data = []
    
    for idx, row in df.iterrows():
        sample_code = row.get('كود العينة', None)
        if pd.isna(sample_code):
            continue
            
        for pesticide_col in pesticide_columns:
            cell_value = row[pesticide_col]
            if pd.notna(cell_value) and str(cell_value).strip() != '':
                # Split the pesticide info: "chlorpyrifos,0.7,0.01"
                parts = str(cell_value).split(',')
                
                # Extract contaminant number (e.g., "ملوث 1" -> "1")
                contaminant_match = re.search(r'ملوث\s*(\d+)', pesticide_col)
                contaminant_num = contaminant_match.group(1) if contaminant_match else '0'
                
                if len(parts) >= 3:
                    pesticide_name = parts[0].strip()
                    try:
                        concentration = float(parts[1].strip())
                        limit_value = float(parts[2].strip())
                        
                        transformed_data.append({
                            'كود العينة': sample_code,
                            'التاريخ': row.get('التاريخ', None),
                            'اسم العينة': row.get('اسم العينة', ''),
                            'نوع العينة': row.get('نوع العينة', ''),
                            'الحى': row.get('الحى', ''),
                            'اسم المنشاة': row.get('اسم المنشاة', ''),
                            'pesticide_name': pesticide_name,
                            'concentration': concentration,
                            'limit_value': limit_value,
                            'exceedance_ratio': concentration / limit_value if limit_value > 0 else 0,
                            'is_compliant': 1 if concentration <= limit_value else 0,
                            'contaminant_number': contaminant_num,
                            'is_detected': 1
                        })
                    except (ValueError, TypeError):
                        # Handle cases like "NO CBD"
                        if 'NO' in str(cell_value).upper():
                            transformed_data.append({
                                'كود العينة': sample_code,
                                'التاريخ': row.get('التاريخ', None),
                                'اسم العينة': row.get('اسم العينة', ''),
                                'نوع العينة': row.get('نوع العينة', ''),
                                'الحى': row.get('الحى', ''),
                                'اسم المنشاة': row.get('اسم المنشاة', ''),
                                'pesticide_name': 'NO DETECTION',
                                'concentration': 0,
                                'limit_value': 0,
                                'exceedance_ratio': 0,
                                'is_compliant': 1,
                                'contaminant_number': contaminant_num,
                                'is_detected': 0
                            })
                else:
                    # Handle "NO CBD" or other non-parseable values
                    if 'NO' in str(cell_value).upper():
                        transformed_data.append({
                            'كود العينة': sample_code,
                            'التاريخ': row.get('التاريخ', None),
                            'اسم العينة': row.get('اسم العينة', ''),
                            'نوع العينة': row.get('نوع العينة', ''),
                            'الحى': row.get('الحى', ''),
                            'اسم المنشاة': row.get('اسم المنشاة', ''),
                            'pesticide_name': 'NO DETECTION',
                            'concentration': 0,
                            'limit_value': 0,
                            'exceedance_ratio': 0,
                            'is_compliant': 1,
                            'contaminant_number': contaminant_num,
                            'is_detected': 0
                        })
    
    return pd.DataFrame(transformed_data)

def find_samples_with_n_pesticides(con, n_pesticides=2, limit=3):
    """
    Find samples that contain exactly N pesticides
    Returns unique sample codes
    """
    query = f"""
    SELECT 
        "كود العينة" as sample_code,
        "اسم العينة" as sample_name,
        COUNT(*) as pesticide_count
    FROM chemistry_tidy
    WHERE is_detected = 1 AND pesticide_name != 'NO DETECTION'
    GROUP BY "كود العينة", "اسم العينة"
    HAVING COUNT(*) = {n_pesticides}
    ORDER BY "كود العينة" DESC
    LIMIT {limit}
    """
    
    return con.execute(query).df()

def load_and_transform_data():
    """
    Load data and transform pesticide columns into tidy format
    This should be called when the app starts or when new data is loaded
    """
    if 'df' not in st.session_state:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed_data_with_neighborhood.xlsx')
        if os.path.exists(data_path):
            df = pd.read_excel(data_path)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Transform the data
            df_tidy = transform_pesticide_columns(df)
            st.session_state.df_tidy = df_tidy
            
            # Save to DuckDB
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'lars_data.duckdb')
            if os.path.exists(db_path):
                try:
                    con = duckdb.connect(db_path, read_only=False)
                    # Create the tidy table
                    con.execute("CREATE OR REPLACE TABLE chemistry_tidy AS SELECT * FROM df_tidy")
                    con.close()
                    st.success(f"✅ Transformed data loaded: {len(df_tidy):,} pesticide records from {len(df):,} samples")
                except Exception as e:
                    st.error(f"❌ Error saving to DuckDB: {e}")
            else:
                st.warning("⚠️ DuckDB database not found. Create it with: `python migrate_db.py`")

def get_sql_schema_info(con) -> str:
    """
    Get database schema information for Schema-Only Prompting
    Returns metadata for both original and transformed data
    Uses CHEMISTRY_TIDY table (transformed with separate columns)
    """
    schema_parts = []
    
    # Get information about chemistry_tidy table (transformed data)
    try:
        columns_df = con.execute("DESCRIBE chemistry_tidy").df()
        schema_parts.append("**Table: chemistry_tidy (Transformed - One row per pesticide)**")
        schema_parts.append("| Column | Type | Description |")
        schema_parts.append("|--------|------|-------------|")
        
        column_descriptions = {
            'كود العينة': 'Sample code (unique identifier)',
            'التاريخ': 'Sample date (TIMESTAMP)',
            'اسم العينة': 'Sample name in Arabic (e.g., طماطم, خيار, كمون)',
            'نوع العينة': 'Sample type (خضراوات, فواكهة, توابل, etc.)',
            'تصنيف العينة': 'Sample classification (خدمية, تجارية, etc.)',
            'نوع الاختبار': 'Test type (متبقيات مبيدات, افلاتوكسين, etc.)',
            'الحى': 'Neighborhood name',
            'اسم المنشاة': 'Establishment/facility name',
            'اسم البلدية': 'Municipality name (e.g., بلدية غرب بريدة)',
            'رقم الرخصة': 'License number',
            'حالة العينة': 'Sample condition/status',
            'اسم المستلم': 'Receiver name',
            'pesticide_name': 'Pesticide/contaminant name',
            'concentration': 'Measured concentration (DOUBLE)',
            'limit_value': 'Maximum allowed limit (DOUBLE)',
            'exceedance_ratio': 'concentration / limit_value (DOUBLE)',
            'is_above_limit': '1 = above limit (فوق الحد), 0 = below limit (تحت الحد) - CALCULATED',
            'is_compliant': '1 = compliant (مطابق), 0 = non-compliant (غير مطابق) - LAB JUDGMENT',
            'is_detected': '1 = pesticide detected, 0 = no detection',
            'contaminant_number': 'Which contaminant column (1-10)',
            'sample_result': 'Original lab result text'
        }
        
        for _, row in columns_df.iterrows():
            col_name = row['column_name']
            desc = column_descriptions.get(col_name, '')
            schema_parts.append(f"| {col_name} | {row['column_type']} | {desc} |")
        
        # Get row count
        count = con.execute("SELECT COUNT(*) as count FROM chemistry_tidy").fetchone()[0]
        schema_parts.append(f"\n**Total Records:** {count:,} pesticide-sample combinations")
        
        # Get unique sample count
        unique_samples = con.execute("SELECT COUNT(DISTINCT \"كود العينة\") as samples FROM chemistry_tidy").fetchone()[0]
        schema_parts.append(f"**Unique Samples:** {unique_samples:,} samples")
        
        # Get pesticide statistics
        pesticide_stats = con.execute("""
            SELECT 
                pesticide_name,
                COUNT(*) as count,
                SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as violations,
                ROUND(AVG(CASE WHEN is_detected = 1 THEN exceedance_ratio END), 2) as avg_exceedance
            FROM chemistry_tidy
            WHERE pesticide_name != 'NO DETECTION'
            GROUP BY pesticide_name
            ORDER BY count DESC
            LIMIT 10
        """).df()
        
        if not pesticide_stats.empty:
            schema_parts.append("\n**Top Pesticides Detected:**")
            for _, row in pesticide_stats.iterrows():
                schema_parts.append(f"- {row['pesticide_name']}: {row['count']} detections ({row['violations']} violations)")
        
        # Get sample types
        sample_type_stats = con.execute("""
            SELECT 
                "نوع العينة" as sample_type,
                COUNT(DISTINCT "كود العينة") as samples,
                SUM(CASE WHEN is_compliant = 0 THEN 1 ELSE 0 END) as violations
            FROM chemistry_tidy
            WHERE "نوع العينة" IS NOT NULL
            GROUP BY "نوع العينة"
            ORDER BY samples DESC
            LIMIT 5
        """).df()
        
        if not sample_type_stats.empty:
            schema_parts.append("\n**Sample Types:**")
            for _, row in sample_type_stats.iterrows():
                schema_parts.append(f"- {row['sample_type']}: {row['samples']} samples ({row['violations']} violations)")
        
        # NEW: Add information about original structure for reference
        schema_parts.append("\n---")
        schema_parts.append("**Original Data Structure (for reference):**")
        schema_parts.append("- Each sample can have 0-8 pesticides in columns ملوث 1 to ملوث 8")
        schema_parts.append("- Format in each ملوث column: 'pesticide_name,concentration,limit'")
        schema_parts.append("- Example: 'chlorpyrifos,0.7,0.01'")
        schema_parts.append("- 'NO CBD' means no detection")
        
    except Exception as e:
        schema_parts.append(f"Error reading schema: {e}")
    
    return '\n'.join(schema_parts)

def generate_map_chart(df: pd.DataFrame, query: str = ""):
    """Generate an interactive map chart for neighborhoods in القصيم, السعودية"""
    try:
        import plotly.graph_objects as go
        
        # Enhanced neighborhood coordinates with Arabic names and proper positioning
        NEIGHBORHOOD_COORDS = {
            'الإسكان': (26.3380, 43.9720),
            'الموطأ': (26.3650, 43.9850),
            'الصفراء': (26.3700, 43.9400),
            'الأفق': (26.3550, 43.9650),
            'الريان': (26.3300, 43.9600),
            'النخيل': (26.3400, 43.9900),
            'السلام': (26.3500, 43.9500),
            'الفيصلية': (26.3250, 43.9550),
            'المنار': (26.3200, 43.9450),
            'الراشد': (26.3150, 43.9550),
            'الحمر': (26.3400, 43.9350),
            'الضاحي': (26.3050, 43.9650),
            'قرطبة': (26.2950, 43.9750),
            'الوادي': (26.3550, 43.9800),
            'المروج': (26.3600, 43.9950),
            'الجردة': (26.3350, 43.9250),
            'البصر': (26.3750, 43.9850),
            'الغدير': (26.2900, 43.9500),
            'المحمدية': (26.3480, 43.9480),
            'الربوة': (26.3380, 43.9380),
            'النزهة': (26.3280, 43.9680),
            'الورود': (26.3180, 43.9480),
            'الروضة': (26.3080, 43.9380),
            'الحزم': (26.2980, 43.9580),
            'العليا': (26.3880, 43.9580),
            'واسط': (26.3350, 43.9700),
            'الملك فيصل': (26.3300, 43.9750),
            'الملك عبدالله': (26.3400, 43.9800),
            'الملك سلمان': (26.3200, 43.9750),
            'الخالدية': (26.3100, 43.9700),
            'النهضة': (26.3000, 43.9650),
            'الرمال': (26.2900, 43.9600),
            'الوريف': (26.2800, 43.9550),
            'الغربية': (26.2700, 43.9500),
            'الشمالية': (26.2600, 43.9450),
            'الجنوبية': (26.2500, 43.9400),
            'الشرقية': (26.2400, 43.9350),
            'الوسطى': (26.2300, 43.9300),
        }
        
        DEFAULT_COORDS = (26.3300, 43.9750)
        
        def extract_neighborhood(address):
            if pd.isna(address): return "غير محدد"
            address_str = str(address)
            clean_address = re.sub(r'\d+', '', address_str).strip()
            for neighborhood in NEIGHBORHOOD_COORDS.keys():
                if neighborhood in clean_address: return neighborhood
            return clean_address[:30] if len(clean_address) > 30 else clean_address
        
        neighborhood_col = None
        for col in df.columns:
            if any(word in str(col).lower() for word in ['حي', 'neighborhood', 'الحي', 'بقعة']):
                neighborhood_col = col
                break
        
        if neighborhood_col is None:
            st.error("⚠️ لم يتم العثور على بيانات الأحياء")
            return
        
        violation_col = None
        for col in df.columns:
            if any(word in str(col).lower() for word in ['violation', 'مخالفة', 'غير مطابق']):
                violation_col = col
                break
        
        if violation_col:
            df['violation_count'] = pd.to_numeric(df[violation_col], errors='coerce').fillna(0)
            agg_data = df.groupby(neighborhood_col).agg({'violation_count': ['sum', 'count']}).reset_index()
            agg_data.columns = ['neighborhood', 'violation_count', 'sample_count']
        else:
            agg_data = df.groupby(neighborhood_col).size().reset_index(name='sample_count')
            agg_data['violation_count'] = 0
        
        agg_data['violation_rate'] = agg_data.apply(lambda row: row['violation_count'] / row['sample_count'] if row['sample_count'] > 0 else 0, axis=1)
        agg_data['lat'] = agg_data['neighborhood'].apply(lambda x: NEIGHBORHOOD_COORDS.get(str(x), DEFAULT_COORDS)[0])
        agg_data['lon'] = agg_data['neighborhood'].apply(lambda x: NEIGHBORHOOD_COORDS.get(str(x), DEFAULT_COORDS)[1])
        
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=agg_data['lat'], lon=agg_data['lon'], mode='markers+text',
            text=agg_data['neighborhood'], textposition="top center",
            marker=dict(size=agg_data['violation_count'].apply(lambda x: max(20, min(60, x * 5))),
            color=agg_data['violation_rate'].apply(lambda r: 'red' if r > 0.5 else 'orange' if r > 0.3 else 'yellow' if r > 0.1 else 'green')),
            hoverinfo='text',
            hovertext=[f"<b>{r['neighborhood']}</b><br>Samples: {r['sample_count']}<br>Violations: {r['violation_count']}" for _, r in agg_data.iterrows()]
        ))
        
        fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=26.33, lon=43.97), zoom=11), height=600)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating map: {e}")

def display_stored_results_with_charts():
    if 'last_query_result' not in st.session_state or st.session_state.last_query_result is None:
        return
    
    result = st.session_state.last_query_result
    query = st.session_state.get('last_query', '')
    
    st.markdown("---")
    st.subheader("📊 نتائج آخر استعلام (Stored Results)")
    st.dataframe(result, use_container_width=True)
    
    if 'selected_chart' not in st.session_state:
        st.session_state.selected_chart = None
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("📊 أعمدة", key="btn_bar_stored"): st.session_state.selected_chart = "bar"
    with col2:
        if st.button("📈 خطي", key="btn_line_stored"): st.session_state.selected_chart = "line"
    with col3:
        if st.button("🥧 دائري", key="btn_pie_stored"): st.session_state.selected_chart = "pie"
    with col4:
        if st.button("🎯 رادار", key="btn_radar_stored"): st.session_state.selected_chart = "radar"
    with col5:
        if st.button("💠 مبعثر", key="btn_scatter_stored"): st.session_state.selected_chart = "scatter"
    with col6:
        if st.button("🗺️ خريطة", key="btn_map_stored"): st.session_state.selected_chart = "map"
    
    if st.session_state.selected_chart:
        chart_func = {
            "bar": generate_bar_chart, "line": generate_line_chart, 
            "pie": generate_pie_chart, "radar": generate_radar_chart,
            "scatter": generate_scatter_plot, "map": generate_map_chart
        }.get(st.session_state.selected_chart)
        
        if chart_func:
            chart_func(result, query)
        
        if st.button("🗑️ مسح الرسم البياني"):
            st.session_state.selected_chart = None
            st.rerun()

def show_data_entry_sidebar():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ➕ Data Entry")
        if st.button("📝 Add New Sample", use_container_width=True):
            st.session_state.show_data_entry_form = not st.session_state.get('show_data_entry_form', False)
            st.rerun()

def show_data_entry_form():
    if not st.session_state.get('show_data_entry_form', False):
        return
    
    st.markdown("---")
    st.subheader("📝 Add New Sample Data")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No dataset loaded for dropdowns.")
        return
        
    df = st.session_state.df
    vegetables = sorted(df['vegetable_arabic'].dropna().unique().tolist()) if 'vegetable_arabic' in df.columns else []
    pesticides = sorted(df['pesticide'].dropna().unique().tolist()) if 'pesticide' in df.columns else []
    
    with st.form("data_entry_form"):
        col1, col2 = st.columns(2)
        with col1:
            veg = st.selectbox("نوع الخضروات", ["-- Select --"] + vegetables)
            pest = st.selectbox("المبيد الحشري", ["-- Select --"] + pesticides)
        with col2:
            reading = st.number_input("القراءة", min_value=0.0, format="%.4f")
            limits = st.number_input("الحد الأقصى", min_value=0.0, format="%.4f")
        
        submitted = st.form_submit_button("➕ Add Sample")
        if submitted:
            st.success("✅ Sample added (Demo implementation - please run migrate_db.py for persistent storage)")

def upload_iso_document_ui(api_client, uploaded_file, doc_type: str, project_id: str):
    with st.spinner(f"📤 Uploading {uploaded_file.name}..."):
        try:
            url = f"{api_client.base_url}/api/v1/iso/upload/{project_id}"
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(url, files=files, data={'document_type': doc_type}, timeout=120)
            response.raise_for_status()
            st.success(f"✅ Successfully uploaded and indexed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")


