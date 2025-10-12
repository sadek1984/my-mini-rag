import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import pymannkendall as mk

# Load your data
df = pd.read_excel('/app/data/processed_data_output.xlsx')
df['document_date'] = pd.to_datetime(df['document_date'])

# ====================
# Mann-Kendall Trend Test (Time Series)
# ====================
def mann_kendall_trend(df, pesticide_group=None):
    """Detect trends in readings over time"""

    # Filter data by group if provided
    if pesticide_group:
        data = df[df['pesticide_group'] == pesticide_group].copy()
    else:
        data = df.copy()

    # Aggregate by month
    monthly = data.groupby(data['document_date'].dt.to_period('M'))['reading'].mean()

    # Run Mann-Kendall test only if there’s enough data
    if len(monthly) > 1:
        result = mk.original_test(monthly.values)
    else:
        result = None

    # Prepare trend data safely
    if result:
        trend_data = {
            'trend': result.trend,
            'p_value': result.p,
            'tau': result.Tau,
            'significant': result.h is True
        }
    else:
        trend_data = {
            'trend': 'insufficient data',
            'p_value': None,
            'tau': None,
            'significant': False
        }

    return trend_data

# Test for pyrethroids
pyrethroid_trend = mann_kendall_trend(df, pesticide_group='pyrethroid')
print(f"Pyrethroid Trend: {pyrethroid_trend}")

# ====================
# ANOVA - Seasonal Comparison
# ====================
def seasonal_anova(df, pesticide_group=None):
    """Compare readings across seasons"""
    
    if pesticide_group:
        data = df[df['pesticide_group'] == pesticide_group].copy()
    else:
        data = df.copy()
    
    # Group by season
    seasons = [data[data['season'] == s]['reading'].dropna() for s in data['season'].unique()]
    
    # Run ANOVA
    f_stat, p_value = f_oneway(*seasons)
    
    # Get means per season
    season_means = data.groupby('season')['reading'].mean().to_dict()
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'season_means': season_means
    }

# Test seasonal differences
seasonal_results = seasonal_anova(df)
print(f"\nSeasonal ANOVA: {seasonal_results}")

# ====================
# ANOVA - Vegetable Category Comparison
# ====================
def vegetable_category_anova(df):
    """Compare readings across vegetable categories"""
    
    categories = [df[df['vegetable_category'] == cat]['reading'].dropna() 
                  for cat in df['vegetable_category'].unique()]
    
    f_stat, p_value = f_oneway(*categories)
    
    category_means = df.groupby('vegetable_category')['reading'].mean().to_dict()
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.01,
        'category_means': category_means
    }

veg_results = vegetable_category_anova(df)
print(f"\nVegetable Category ANOVA: {veg_results}")

# ====================
# Generate Insights for App
# ====================

def generate_statistical_insights(df):
    """Generate statistical insights for pesticide readings"""

    insights = []

    # Test all pesticide groups for trends
    for group in df['pesticide_group'].unique():
        trend = mann_kendall_trend(df, pesticide_group=group)

        if trend and trend.get('significant'):
            direction = "increasing" if trend['trend'] == 'increasing' else "decreasing"
            insights.append({
                'type': 'trend',
                'pesticide_group': group,
                'message': f"⚠️ Significant {direction} trend detected in {group} readings (p={trend['p_value']:.3f})",
                'severity': 'high' if direction == 'increasing' else 'medium'
            })
        else:
            # Optional: record that data was insufficient
            insights.append({
                'type': 'trend',
                'pesticide_group': group,
                'message': f"ℹ️ No significant trend detected for {group} (insufficient or stable data)",
                'severity': 'low'
            })

    # Seasonal analysis
    seasonal = seasonal_anova(df)
    if seasonal and seasonal.get('significant'):
        highest_season = max(seasonal['season_means'], key=seasonal['season_means'].get)
        insights.append({
            'type': 'seasonal',
            'message': f"📊 Significant seasonal variation detected (p={seasonal['p_value']:.3f}). Highest readings in season {highest_season}",
            'severity': 'medium',
            'season_data': seasonal['season_means']
        })

    # Vegetable category analysis
    veg = vegetable_category_anova(df)
    if veg and veg.get('significant'):
        highest_cat = max(veg['category_means'], key=veg['category_means'].get)
        insights.append({
            'type': 'category',
            'message': f"🥬 Significant differences across vegetable categories (p={veg['p_value']:.3f}). Highest in {highest_cat}",
            'severity': 'medium',
            'category_data': veg['category_means']
        })

    return insights


# === Example execution ===
insights = generate_statistical_insights(df)
for insight in insights:
    print(f"\n{insight['message']}")