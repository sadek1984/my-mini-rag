"""
Risk Windows Module for LARS Application

Provides specialized UI windows for Health Risk and Quality Index display.
Triggered by keyword detection in user queries.
Features sample type detection, parameter inputs, and PRIMo 4 population classes.

Author: LARS Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import re

# Import risk assessment functions
try:
    from modules.risk_assessment_service import (
        calculate_lars_metrics,
        detect_query_type,
        get_saudi_ir,
        SAUDI_IR_DATA,
        SAUDI_COMMODITY_MAPPING,
        POPULATION_CLASSES
    )
    RISK_SERVICE_AVAILABLE = True
except ImportError as e:
    RISK_SERVICE_AVAILABLE = False
    print(f"⚠️ Risk service not available: {e}")


# ============================================================================
# SAMPLE TYPE DETECTION
# ============================================================================

# Arabic sample type mapping
SAMPLE_TYPE_MAPPING = {
    'طماطم': ('Tomato', 'طماطم'),
    'خيار': ('Cucumber', 'خيار'),
    'فلفل': ('Pepper', 'فلفل'),
    'باذنجان': ('Eggplant', 'باذنجان'),
    'كوسة': ('Zucchini', 'كوسة'),
    'فاصوليا': ('Beans', 'فاصوليا'),
    'بامية': ('Okra', 'بامية'),
    'خس': ('Lettuce', 'خس'),
    'بقدونس': ('Parsley', 'بقدونس'),
    'كزبرة': ('Coriander', 'كزبرة'),
    'نعناع': ('Mint', 'نعناع'),
    'جرجير': ('Leafy Greens', 'جرجير'),
    'سبانخ': ('Leafy Greens', 'سبانخ'),
    'ملفوف': ('Cabbage', 'ملفوف'),
    'جزر': ('Root Vegetables', 'جزر'),
    'بطاطس': ('Root Vegetables', 'بطاطس'),
    # English mappings
    'tomato': ('Tomato', 'طماطم'),
    'cucumber': ('Cucumber', 'خيار'),
    'pepper': ('Pepper', 'فلفل'),
    'eggplant': ('Eggplant', 'باذنجان'),
    'zucchini': ('Zucchini', 'كوسة'),
    'beans': ('Beans', 'فاصوليا'),
    'okra': ('Okra', 'بامية'),
    'lettuce': ('Lettuce', 'خس'),
    'parsley': ('Parsley', 'بقدونس'),
}


def detect_sample_type(query: str) -> Optional[tuple]:
    """
    Detect sample type from query.
    
    Returns:
        Tuple of (commodity_english, arabic_pattern) or None
    """
    query_lower = query.lower()
    
    for keyword, (commodity, pattern) in SAMPLE_TYPE_MAPPING.items():
        if keyword in query or keyword in query_lower:
            return (commodity, pattern)
    
    return None


# ============================================================================
# RISK INPUT FORM
# ============================================================================

def get_residue_data_for_sample(sample_type_arabic: str) -> List[Dict[str, Any]]:
    """
    Query database for residue data filtered by sample type.
    
    Parameters:
        sample_type_arabic: Arabic name of sample type for filtering
    
    Returns:
        List of dicts with name, concentration, mrl
    """
    try:
        import duckdb
        from pathlib import Path
        
        # Find database path
        possible_paths = [
            Path(__file__).parent.parent / 'data' / 'lars_data.duckdb',
            Path(__file__).parent.parent / 'lars_data.duckdb',
            Path('/app/src/LARS/lars_data.duckdb'),
        ]
        
        db_path = None
        for p in possible_paths:
            if p.exists():
                db_path = p
                break
        
        if not db_path:
            return []
        
        con = duckdb.connect(str(db_path), read_only=True)
        
        # Build query with sample filter
        sample_filter = f"AND \"اسم العينة\" LIKE '%{sample_type_arabic}%'" if sample_type_arabic else ""
        
        query = f"""
        SELECT 
            pesticide_name as name,
            concentration,
            limit_value as mrl
        FROM chemistry_tidy 
        WHERE is_detected = 1 
        AND concentration > 0
        AND pesticide_name NOT IN ('NO DETECTION', 'NO DATA')
        {sample_filter}
        LIMIT 200
        """
        
        result_df = con.execute(query).df()
        con.close()
        
        return result_df.to_dict('records')
    except Exception as e:
        print(f"Error querying database: {e}")
        return []


# Reverse mapping: English commodity to Arabic sample name
COMMODITY_TO_ARABIC = {
    'Tomato': 'طماطم',
    'Cucumber': 'خيار',
    'Pepper': 'فلفل',
    'Eggplant': 'باذنجان',
    'Zucchini': 'كوسة',
    'Beans': 'فاصوليا',
    'Okra': 'بامية',
    'Lettuce': 'خس',
    'Parsley': 'بقدونس',
    'Coriander': 'كزبرة',
    'Mint': 'نعناع',
    'Leafy Greens': 'ورقيات',
    'Cabbage': 'ملفوف',
    'Root Vegetables': 'جزر',
    'Total Vegetables': '',  # No filter for total
    'Total Fruits': '',
}


def show_risk_input_form(
    query: str,
    detected_sample: Optional[tuple] = None
) -> Dict[str, Any]:
    """
    Show input form for risk assessment parameters.
    
    Returns:
        Dictionary with selected parameters
    """
    st.markdown("### ⚙️ Risk Assessment Parameters / معلمات تقييم المخاطر")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Population class dropdown with all PRIMo 4 options
        pop_options = {}
        for key, pc in POPULATION_CLASSES.items():
            pop_options[key] = f"{pc.name_en} ({pc.name_ar}) - {pc.default_weight_kg} kg"
        
        selected_pop = st.selectbox(
            "👥 Population Class / الفئة السكانية",
            options=list(pop_options.keys()),
            format_func=lambda x: pop_options[x],
            index=list(pop_options.keys()).index('saudi_adults') if 'saudi_adults' in pop_options else 4,
            help="Select population for body weight. Saudi adults=53kg, Saudi children=16kg",
            key="risk_population_selector"
        )
        
        body_weight = POPULATION_CLASSES[selected_pop].default_weight_kg
    
    with col2:
        # Commodity selector with detected default
        commodity_options = [k for k in SAUDI_IR_DATA.keys() if k != "Default"]
        
        default_commodity = "Total Vegetables"
        if detected_sample:
            if detected_sample[0] in commodity_options:
                default_commodity = detected_sample[0]
        
        # Use session state to remember selection
        default_idx = commodity_options.index(default_commodity) if default_commodity in commodity_options else 0
        
        selected_commodity = st.selectbox(
            "🥬 Commodity / المنتج",
            options=commodity_options,
            index=default_idx,
            help="⚠️ Changing this will re-query database for that sample type!",
            key="risk_commodity_selector"
        )
        
        ir = get_saudi_ir(selected_commodity)
        
        # Check if commodity changed from original query
        original_commodity = st.session_state.get('risk_original_commodity', default_commodity)
        if selected_commodity != original_commodity:
            # Commodity changed - need to re-query database
            arabic_sample = COMMODITY_TO_ARABIC.get(selected_commodity, '')
            new_data = get_residue_data_for_sample(arabic_sample)
            if new_data:
                st.session_state['risk_residue_data'] = new_data
                st.session_state['risk_original_commodity'] = selected_commodity
                st.rerun()  # Rerun to display updated data
    
    with col3:
        # Custom ingestion rate override
        custom_ir = st.number_input(
            "📊 Ingestion Rate (g/day)",
            min_value=1.0,
            max_value=500.0,
            value=ir * 1000,  # Convert to g/day for display
            step=5.0,
            help="Daily consumption rate. Default from Saudi IR data.",
            key="risk_ir_input"
        )
        ir = custom_ir / 1000  # Convert back to kg/day
    
    # Show selected parameters
    st.info(f"""
    **Selected Parameters:**
    - 👥 Population: {POPULATION_CLASSES[selected_pop].name_en} ({POPULATION_CLASSES[selected_pop].name_ar})
    - ⚖️ Body Weight: {body_weight} kg
    - 🥬 Commodity: {selected_commodity}
    - 📊 Ingestion Rate: {ir*1000:.1f} g/day ({ir:.4f} kg/day)
    """)
    
    return {
        'population': selected_pop,
        'body_weight': body_weight,
        'commodity': selected_commodity,
        'ingestion_rate': ir,
        'detected_sample': detected_sample
    }


# ============================================================================
# HEALTH RISK WINDOW
# ============================================================================

def show_health_risk_window(
    residue_data: List[Dict[str, Any]],
    commodity: str = "Total Vegetables",
    target: str = "adult",
    query: str = ""
):
    """
    Display Health Risk assessment window with HIc and HQc calculations.
    Uses session state to persist data across filter changes.
    """
    # Use session state data if passed data is empty
    if not residue_data:
        residue_data = st.session_state.get('risk_residue_data', [])
        query = st.session_state.get('risk_query', query)
        commodity = st.session_state.get('risk_commodity', commodity)
    
    st.markdown("---")
    
    # Header with close button
    col_title, col_close = st.columns([6, 1])
    with col_title:
        st.subheader("🏥 Health Risk Index (HRI) Assessment / مؤشر الخطر الصحي")
    with col_close:
        if st.button("❌ Close", key="close_risk_window"):
            st.session_state['risk_show_window'] = False
            st.session_state.pop('risk_residue_data', None)
            st.rerun()
    
    # Detect sample type from query
    detected_sample = detect_sample_type(query) if query else None
    
    if detected_sample:
        st.success(f"🔍 Detected sample type: **{detected_sample[0]}** ({detected_sample[1]})")
    
    # Show parameter input form
    params = show_risk_input_form(query, detected_sample)
    
    # Use the residue data
    filtered_data = residue_data
    
    st.markdown("---")
    
    # Calculate LARS metrics with selected parameters
    if filtered_data:
        target_type = "adult" if params['body_weight'] >= 50 else "child"
        analysis = calculate_lars_metrics(
            filtered_data, 
            params['commodity'], 
            target_type
        )
        
        # Override with actual body weight if different
        # Recalculate with custom IR if needed
        
        # Display summary metrics
        st.markdown("### 📊 Health Risk Summary / ملخص الخطر الصحي")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hi_total = analysis["hi_total"]
            st.metric(
                "Total HIc / مؤشر الخطر",
                f"{hi_total:.4f}",
                delta="⚠️ Exceeds 1" if hi_total >= 1 else "✅ Safe",
                delta_color="inverse" if hi_total >= 1 else "normal"
            )
        
        with col2:
            if analysis["risk_driver"]:
                driver_name = analysis["risk_driver"].split(" (")[0]
                st.metric(
                    "🎯 Risk Driver",
                    driver_name[:12] + "..." if len(driver_name) > 12 else driver_name,
                    "Highest HQc"
                )
            else:
                st.metric("Risk Driver", "N/A")
        
        with col3:
            st.metric(
                "Pesticides Analyzed",
                len(analysis["individual_results"])
            )
        
        with col4:
            # Show current commodity, not original detection
            sample_info = params['commodity'] if params.get('commodity') else (detected_sample[0] if detected_sample else "All")
            st.metric(
                "Sample Type",
                sample_info
            )
        
        # Risk interpretation banner
        if hi_total >= 1:
            st.error(f"⚠️ **مؤشر الخطر غير مقبول** - HIc = {hi_total:.4f} (≥1)")
        elif hi_total >= 0.5:
            st.warning(f"⚡ **خطر مرتفع** - HIc = {hi_total:.4f} (≥0.5)")
        else:
            st.success(f"✅ **الخطر مقبول** - HIc = {hi_total:.4f} (<0.5)")
        
        # Chemical group statistics
        st.markdown("### 🧪 Risk by Chemical Group / الخطر حسب المجموعة الكيميائية")
        
        if analysis["stats"]:
            stats_data = []
            for group, data in analysis["stats"].items():
                group_hqc = sum(
                    r["hqc"] for r in analysis["individual_results"] 
                    if r["chemical_group"] == group
                )
                stats_data.append({
                    "Group / المجموعة": group,
                    "Count": data["count"],
                    "Min": f"{data['min']:.4f}",
                    "Max": f"{data['max']:.4f}",
                    "Mean": f"{data['mean']:.4f}",
                    "Median": f"{data['median']:.4f}",
                    "HQc Sum": f"{group_hqc:.4f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Bar chart
            chart_data = pd.DataFrame([
                {"Group": group, "HQc": sum(r["hqc"] for r in analysis["individual_results"] if r["chemical_group"] == group)}
                for group in analysis["stats"].keys()
            ])
            chart_data = chart_data.sort_values("HQc", ascending=False)
            
            fig = px.bar(
                chart_data,
                x="Group",
                y="HQc",
                color="HQc",
                color_continuous_scale="Reds",
                title="Hazard Quotient by Chemical Group"
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual pesticide table
        with st.expander("📋 Individual Pesticide Risk Data / بيانات المبيدات الفردية"):
            individual_df = pd.DataFrame(analysis["individual_results"])
            if not individual_df.empty:
                individual_df = individual_df.sort_values("hqc", ascending=False)
                display_cols = ["pesticide", "chemical_group", "concentration", "adi", "edi", "hqc", "hqc_percent"]
                individual_df = individual_df[[c for c in display_cols if c in individual_df.columns]]
                individual_df.columns = ["Pesticide", "Group", "Conc", "ADI", "EDI", "HQc", "HQc %"]
                st.dataframe(individual_df, use_container_width=True)
    else:
        st.warning("No residue data available for analysis. / لا توجد بيانات متبقيات للتحليل")


# ============================================================================
# QUALITY INDEX WINDOW
# ============================================================================

def show_quality_index_window(
    residue_data: List[Dict[str, Any]],
    commodity: str = "Total Vegetables",
    query: str = ""
):
    """
    Display Quality Index (IqR) assessment window.
    Uses session state to persist data across filter changes.
    """
    # Use session state data if passed data is empty
    if not residue_data:
        residue_data = st.session_state.get('iqr_residue_data', [])
        query = st.session_state.get('iqr_query', query)
        commodity = st.session_state.get('iqr_commodity', commodity)
    
    st.markdown("---")
    
    # Header with close button
    col_title, col_close = st.columns([6, 1])
    with col_title:
        st.subheader("📐 Quality Index (IqR) Assessment / مؤشر الجودة")
    with col_close:
        if st.button("❌ Close", key="close_iqr_window"):
            st.session_state['iqr_show_window'] = False
            st.session_state.pop('iqr_residue_data', None)
            st.rerun()
    
    st.markdown("""
    **Index of Quality for Residues (IqR)** = Concentration / MRL  
    مؤشر جودة المتبقيات = التركيز / الحد الأقصى المسموح
    """)
    
    # Detect sample type from query
    detected_sample = detect_sample_type(query) if query else None
    
    if detected_sample:
        st.success(f"🔍 Detected sample type: **{detected_sample[0]}** ({detected_sample[1]})")
    
    # Show parameter input form (simplified for IqR - no body weight needed)
    col1, col2 = st.columns(2)
    
    with col1:
        commodity_options = [k for k in SAUDI_IR_DATA.keys() if k != "Default"]
        default_commodity = detected_sample[0] if detected_sample and detected_sample[0] in commodity_options else "Total Vegetables"
        
        selected_commodity = st.selectbox(
            "🥬 Commodity / المنتج",
            options=commodity_options,
            index=commodity_options.index(default_commodity) if default_commodity in commodity_options else 0,
            key="iqr_commodity_select"
        )
    
    with col2:
        sample_info = detected_sample[0] if detected_sample else "All Samples"
        st.info(f"🔍 Sample Filter: **{sample_info}**")
    
    st.markdown("---")
    
    # Calculate metrics
    if residue_data:
        analysis = calculate_lars_metrics(residue_data, selected_commodity, "adult")
        
        # Summary metrics
        st.markdown("### 📊 Quality Summary / ملخص الجودة")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            iqr_total = analysis["iqr_total"]
            st.metric(
                "Total IqR",
                f"{iqr_total:.4f}",
                delta="⚠️ >1" if iqr_total >= 1 else "✅ OK",
                delta_color="inverse" if iqr_total >= 1 else "normal"
            )
        
        with col2:
            if analysis["iqr_driver"]:
                driver_name = analysis["iqr_driver"].split(" (")[0]
                st.metric(
                    "🎯 Quality Driver",
                    driver_name[:12] + "..." if len(driver_name) > 12 else driver_name,
                    "Highest IqR"
                )
            else:
                st.metric("Quality Driver", "N/A")
        
        with col3:
            st.metric(
                "Pesticides",
                len(analysis["individual_results"])
            )
        
        # Quality interpretation
        if iqr_total >= 1:
            st.error(f"⚠️ **مشكلة جودة** - Total IqR = {iqr_total:.4f} (≥1)")
        else:
            st.success(f"✅ **الجودة مقبولة** - Total IqR = {iqr_total:.4f} (<1)")
        
        # Chemical group statistics table
        st.markdown("### 🧪 Statistics by Chemical Group / إحصائيات حسب المجموعة")
        
        if analysis["stats"]:
            stats_data = []
            for group, data in analysis["stats"].items():
                group_iqr = sum(
                    r["iqr"] for r in analysis["individual_results"] 
                    if r["chemical_group"] == group
                )
                stats_data.append({
                    "Group": group,
                    "Count": data["count"],
                    "Min": f"{data['min']:.4f}",
                    "Max": f"{data['max']:.4f}",
                    "Mean": f"{data['mean']:.4f}",
                    "Median": f"{data['median']:.4f}",
                    "IqR Sum": f"{group_iqr:.4f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Pie chart
            chart_data = pd.DataFrame([
                {"Group": group, "IqR": sum(r["iqr"] for r in analysis["individual_results"] if r["chemical_group"] == group)}
                for group in analysis["stats"].keys()
            ])
            
            fig = px.pie(
                chart_data,
                names="Group",
                values="IqR",
                title="IqR Distribution by Chemical Group",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual pesticide IqR table
        with st.expander("📋 Individual IqR Data / بيانات مؤشر الجودة الفردية"):
            individual_df = pd.DataFrame(analysis["individual_results"])
            if not individual_df.empty:
                individual_df = individual_df.sort_values("iqr", ascending=False)
                display_cols = ["pesticide", "chemical_group", "concentration", "mrl", "iqr", "iqr_percent"]
                individual_df = individual_df[[c for c in display_cols if c in individual_df.columns]]
                individual_df.columns = ["Pesticide", "Group", "Conc", "MRL", "IqR", "IqR %"]
                st.dataframe(individual_df, use_container_width=True)
    else:
        st.warning("No residue data available for analysis. / لا توجد بيانات متبقيات للتحليل")


# ============================================================================
# MAIN TRIGGER FUNCTION
# ============================================================================

def process_triggered_query(
    query: str,
    residue_data: List[Dict[str, Any]],
    commodity: str = "Total Vegetables"
) -> bool:
    """
    Check if query triggers a specialized window and display it.
    Passes query for sample type detection.
    """
    trigger = detect_query_type(query)
    
    if trigger == "[TRIGGER_UI: HEALTH_RISK_WINDOW]":
        show_health_risk_window(residue_data, commodity, query=query)
        return True
    elif trigger == "[TRIGGER_UI: QUALITY_INDEX_WINDOW]":
        show_quality_index_window(residue_data, commodity, query=query)
        return True
    
    return False
