"""
Streamlit + Plotly Dashboard for Energy Theft Detection (With Child-Friendly Explanations and Suspicious Points Highlighted)
File: app/streamlit_app.py

- Shows top suspicious houses by risk_score
- Interactive threshold sliders for normal / suspicious / highly suspicious
- Per-house hourly consumption line chart with suspicious points highlighted
- Summary histogram of risk scores
- Zone scatter plot for risk visualization
- Simple, clear explanations for each section
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
features = pd.read_csv('output/features.csv')
predictions = pd.read_csv('output/predictions.csv')
usage = pd.read_csv('output/synthetic_usage.csv')

# Merge predictions with metadata
features_pred = pd.merge(features, predictions, on='house_id')

st.set_page_config(page_title="Energy Theft Dashboard", layout="wide")
st.title("🔌 Energy Theft Detection Dashboard")

st.markdown("This dashboard helps identify houses that may be stealing electricity. The AI model gives each house a risk score (higher score = more suspicious).")

# Threshold sliders
st.sidebar.header("Risk Thresholds")
st.sidebar.markdown("Use these sliders to decide which houses are Normal, Suspicious, or Highly Suspicious based on risk score.")
low = st.sidebar.slider("Low → Normal", float(features_pred['risk_score'].min()), float(features_pred['risk_score'].max()), float(features_pred['risk_score'].quantile(0.33)))
med = st.sidebar.slider("Medium → Suspicious", low, float(features_pred['risk_score'].max()), float(features_pred['risk_score'].quantile(0.66)))

# Categorize houses
def categorize(score):
    if score <= low:
        return 'Normal'
    elif score <= med:
        return 'Suspicious'
    else:
        return 'Highly Suspicious'

features_pred['category'] = features_pred['risk_score'].apply(categorize)

# Top suspicious houses
st.header("Top Suspicious Houses")
st.markdown("Bar chart showing the 10 houses with the highest risk scores. Taller bars = houses more likely to be stealing electricity.")
top_susp = features_pred.sort_values('risk_score', ascending=False).head(10)
fig_bar = px.bar(top_susp, x='house_id', y='risk_score', color='risk_score', color_continuous_scale='Reds', title='Top Suspicious Houses')
st.plotly_chart(fig_bar, use_container_width=True)

# Risk score distribution
st.header("Risk Score Distribution")
st.markdown("Histogram showing how risk scores are spread across all houses. Each color shows the category: Normal, Suspicious, Highly Suspicious.")
fig_hist = px.histogram(features_pred, x='risk_score', nbins=30, title='Risk Score Distribution', color='category')
st.plotly_chart(fig_hist, use_container_width=True)

# Zone scatter plot
st.header("House Risk by Zone")
st.markdown("Scatter plot showing houses by their location (zone_x, zone_y). Each dot = a house. Color and size show how suspicious the house is. Bigger/redder dots = more likely to be stealing.")
features_pred['risk_score_size'] = features_pred['risk_score'] - features_pred['risk_score'].min() + 0.1
fig_map = px.scatter(features_pred, x='zone_x', y='zone_y', color='risk_score', size='risk_score_size', color_continuous_scale='Reds', title='House Risk by Zone')
st.plotly_chart(fig_map, use_container_width=True)

# Select house for hourly consumption
st.header("House Hourly Consumption")
st.markdown("Select a house to see its electricity usage over time. Sudden drops or unusual patterns may indicate electricity theft. Suspicious points are highlighted in red.")
house_options = top_susp['house_id'].tolist()
selected_house = st.selectbox("Select House ID", house_options)

house_usage = usage[usage['house_id']==selected_house].copy()
# Detect suspicious points: large drops between consecutive hours
house_usage['prev'] = house_usage['consumption_kwh'].shift(1)
house_usage['drop'] = house_usage['prev'] - house_usage['consumption_kwh']
threshold_drop = house_usage['drop'].mean() + 2*house_usage['drop'].std()  # significant drop
house_usage['suspicious'] = house_usage['drop'] > threshold_drop

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=house_usage['timestamp'], y=house_usage['consumption_kwh'], mode='lines+markers', name='Consumption'))
fig_line.add_trace(go.Scatter(x=house_usage.loc[house_usage['suspicious'],'timestamp'],
                              y=house_usage.loc[house_usage['suspicious'],'consumption_kwh'],
                              mode='markers', name='Suspicious Drop', marker=dict(color='red', size=10)))
fig_line.update_layout(title=f'Hourly Consumption for House {selected_house} (Red points are suspicious drops)', xaxis_title='Time', yaxis_title='kWh')
st.plotly_chart(fig_line, use_container_width=True)