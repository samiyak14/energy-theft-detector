"""
Interactive Streamlit + Plotly Energy Theft Dashboard
- Generate realistic synthetic data or upload CSVs
- Choose special occasion for realistic patterns
- Download generated CSVs
- Choose algorithm: Isolation Forest or K-Means (with cluster selection)
- Top suspicious houses, risk distribution, zone heatmap
- Hourly consumption with suspicious points highlighted
- Child-friendly explanations, hover info, and "Read more..." expanders
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np

# --------------------------
# Synthetic Data Generator
# --------------------------
def generate_synthetic_data(num_houses=50, weeks=1, theft_prob=0.05, occasion="None"):
    hours = 24*7*weeks
    records = []
    for house_id in range(1, num_houses+1):
        base = np.random.uniform(0.5, 3.0)
        daily_pattern = np.sin(np.linspace(0, 2*np.pi, 24)) + 1
        noise = np.random.normal(0, 0.2, hours)
        usage = []
        for h in range(hours):
            hour_of_day = h % 24

            # Occasion factor
            if occasion == "Diwali":
                factor = 1 + 0.5*np.exp(-((hour_of_day-20)/2)**2)  # peak around 8-10 PM
            elif occasion == "Summer":
                factor = 1 + 0.3*np.exp(-((hour_of_day-15)/3)**2)  # peak afternoon
            elif occasion == "Winter":
                factor = 1 + 0.3*np.exp(-((hour_of_day-22)/2)**2)  # peak night
            else:
                factor = 1

            val = base * daily_pattern[hour_of_day] * factor + noise[h]

            # Theft drops
            if np.random.rand() < theft_prob:
                val *= np.random.uniform(0.1, 0.5)
            val = max(val, 0)
            usage.append(val)

        for h, val in enumerate(usage):
            records.append({"house_id": house_id, "timestamp": h, "consumption_kwh": val})

    usage_df = pd.DataFrame(records)
    features_df = usage_df.groupby("house_id")["consumption_kwh"].agg(
        mean_consumption="mean",
        std_consumption="std",
        min_consumption="min",
        max_consumption="max",
        pct_zero_hours=lambda x: (x==0).mean(),
        max_drop=lambda x: np.max(np.diff(x)),
        avg_hourly_change=lambda x: np.mean(np.abs(np.diff(x))),
        consumption_range=lambda x: np.max(x)-np.min(x),
        std_ratio=lambda x: np.std(x)/np.mean(x)
    ).reset_index()
    features_df["zone_x"] = np.random.randint(0, 5, size=len(features_df))
    features_df["zone_y"] = np.random.randint(0, 5, size=len(features_df))
    return usage_df, features_df

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="Energy Theft Dashboard", layout="wide")
st.title("🔌 Energy Theft Detection Dashboard")
st.markdown("This dashboard shows which houses might be stealing electricity. Higher risk score = more suspicious.")

# --------------------------
# Sidebar: Data input
# --------------------------
st.sidebar.header("📂 Data Input")
data_option = st.sidebar.radio("Choose data source:", ["Upload CSVs", "Generate Synthetic Data"])

if data_option == "Upload CSVs":
    features_file = st.sidebar.file_uploader("Upload features CSV", type=["csv"])
    usage_file = st.sidebar.file_uploader("Upload usage CSV", type=["csv"])
    if features_file and usage_file:
        features = pd.read_csv(features_file)
        usage = pd.read_csv(usage_file)
    else:
        st.warning("Please upload both features and usage CSVs.")
        st.stop()
else:
    st.sidebar.subheader("Synthetic Data Options")
    num_houses = st.sidebar.slider("Number of houses", 10, 200, 50)
    weeks = st.sidebar.slider("Number of weeks", 1, 4, 1)
    theft_prob = st.sidebar.slider("Theft probability per hour", 0.01, 0.2, 0.05)
    occasion = st.sidebar.selectbox("Special occasion", ["None", "Diwali", "Summer", "Winter"])
    
    if st.sidebar.button("Generate Synthetic Data"):
        usage, features = generate_synthetic_data(num_houses=num_houses, weeks=weeks, theft_prob=theft_prob, occasion=occasion)
        st.session_state['usage'] = usage
        st.session_state['features'] = features
        st.success(f"Synthetic data generated for {num_houses} houses ({weeks} week(s), occasion={occasion})!")

# --------------------------
# Download CSV buttons
# --------------------------
if 'features' in st.session_state and 'usage' in st.session_state:
    st.download_button("Download features.csv", st.session_state['features'].to_csv(index=False), file_name="features.csv")
    st.download_button("Download usage.csv", st.session_state['usage'].to_csv(index=False), file_name="usage.csv")

# Use uploaded or generated data
if data_option == "Generate Synthetic Data" and 'features' in st.session_state:
    features = st.session_state['features']
    usage = st.session_state['usage']

# --------------------------
# Algorithm selection
# --------------------------
st.sidebar.header("⚙️ Model Settings")
algorithm = st.sidebar.selectbox("Choose Algorithm", ["Isolation Forest", "K-Means Clustering"])
if algorithm == "K-Means Clustering":
    n_clusters = st.sidebar.slider("Number of clusters for K-Means", 2, 5, 2)

# --------------------------
# Feature selection & scaling
# --------------------------
if 'features' in locals():
    features_to_use = ['mean_consumption', 'std_consumption', 'max_drop',
                       'avg_hourly_change', 'consumption_range', 'std_ratio']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features[features_to_use])
else:
    st.warning("No features available. Please generate or upload data first.")
    st.stop()

# --------------------------
# Model computation
# --------------------------
if algorithm == "Isolation Forest":
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features_scaled)
    features['risk_score'] = -model.decision_function(features_scaled)
else:  # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(features_scaled - centers[clusters], axis=1)
    features['risk_score'] = distances

features_pred = features.copy()

# --------------------------
# Threshold sliders
# --------------------------
st.sidebar.header("📊 Risk Thresholds")
low = st.sidebar.slider("Low → Normal", float(features_pred['risk_score'].min()),
                        float(features_pred['risk_score'].max()),
                        float(features_pred['risk_score'].quantile(0.33)))
med = st.sidebar.slider("Medium → Suspicious", low,
                        float(features_pred['risk_score'].max()),
                        float(features_pred['risk_score'].quantile(0.66)))

def categorize(score):
    if score <= low:
        return 'Normal'
    elif score <= med:
        return 'Suspicious'
    else:
        return 'Highly Suspicious'

features_pred['category'] = features_pred['risk_score'].apply(categorize)

# --------------------------
# Top suspicious houses
# --------------------------
st.header("🏠 Top Suspicious Houses")
st.markdown("Shows the 10 houses most likely stealing electricity. Taller bars = higher risk.")
top_susp = features_pred.sort_values('risk_score', ascending=False).head(10)
fig_bar = px.bar(top_susp, x='house_id', y='risk_score', color='risk_score', color_continuous_scale='Reds',
                 hover_data={'house_id': True, 'risk_score': ':.2f'},
                 title="Top 10 Suspicious Houses")
st.plotly_chart(fig_bar, use_container_width=True)
with st.expander("Read more about this chart"):
    st.markdown("""
    **Type of chart:** Bar chart  
    **What it shows:** Each bar represents a house. The height corresponds to its risk score (higher = more suspicious).  
    **How to interpret:** Taller bars indicate houses that are more likely to be stealing electricity. Hover over a bar to see exact score.  
    **Meaningful observations:** Houses that stand out significantly from the rest are high-priority for investigation.
    """)

# --------------------------
# Risk score distribution
# --------------------------
st.header("📉 Risk Score Distribution")
st.markdown("Shows how all houses score. Colors tell if a house is Normal, Suspicious, or Highly Suspicious.")
fig_hist = px.histogram(features_pred, x='risk_score', nbins=30, color='category',
                        hover_data={'risk_score': ':.2f'}, title="Risk Score Distribution")
st.plotly_chart(fig_hist, use_container_width=True)
with st.expander("Read more about this chart"):
    st.markdown("""
    **Type of chart:** Histogram  
    **What it shows:** How risk scores are distributed among all houses.  
    **How to interpret:** Helps see if most houses are normal or if many are suspicious.  
    **Meaningful observations:** Peaks in the red area indicate many highly suspicious houses.
    """)

# --------------------------
# Zone heatmap
# --------------------------
st.header("🗺️ Zone-Level Risk Heatmap")
st.markdown("Shows average risk per zone. Darker/redder zones have more suspicious houses.")
zone_summary = features_pred.groupby(['zone_x','zone_y'])['risk_score'].mean().reset_index()
fig_zone = px.density_heatmap(zone_summary, x='zone_x', y='zone_y', z='risk_score',
                              color_continuous_scale='Reds',
                              hover_data={'zone_x': True, 'zone_y': True, 'risk_score': ':.2f'},
                              title="Average Risk per Zone")
st.plotly_chart(fig_zone, use_container_width=True)
with st.expander("Read more about this chart"):
    st.markdown("""
    **Type of chart:** Heatmap  
    **What it shows:** Each cell represents a zone; color intensity = average risk.  
    **How to interpret:** Darker/redder zones have higher-risk houses.  
    **Meaningful observations:** Identify regions with multiple suspicious houses for targeted inspections.
    """)

# --------------------------
# Hourly consumption
# --------------------------
st.header("⏰ House Hourly Consumption")
st.markdown("Select a house to see its hourly electricity usage. Red points are sudden drops, possibly theft. Hover to see kWh.")
house_options = top_susp['house_id'].tolist()
selected_house = st.selectbox("Select House ID", house_options)

house_usage = usage[usage['house_id']==selected_house].copy()
house_usage['prev'] = house_usage['consumption_kwh'].shift(1)
house_usage['drop'] = house_usage['prev'] - house_usage['consumption_kwh']
threshold_drop = house_usage['drop'].mean() + 2*house_usage['drop'].std()
house_usage['suspicious'] = house_usage['drop'] > threshold_drop

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=house_usage['timestamp'], y=house_usage['consumption_kwh'],
                              mode='lines+markers', name='Consumption',
                              hovertemplate='Hour: %{x}<br>Consumption: %{y:.2f} kWh'))
fig_line.add_trace(go.Scatter(x=house_usage.loc[house_usage['suspicious'],'timestamp'],
                              y=house_usage.loc[house_usage['suspicious'],'consumption_kwh'],
                              mode='markers', name='Suspicious Drop',
                              marker=dict(color='red', size=10),
                              hovertemplate='Suspicious Drop<br>Hour: %{x}<br>Consumption: %{y:.2f} kWh'))
fig_line.update_layout(title=f'Hourly Consumption for House {selected_house} (Red = Suspicious Drop)',
                       xaxis_title='Hour', yaxis_title='kWh')
st.plotly_chart(fig_line, use_container_width=True)
with st.expander("Read more about this chart"):
    st.markdown("""
    **Type of chart:** Line chart with markers  
    **What it shows:** Each point = electricity consumed in an hour. Red points = unusually large drops.  
    **How to interpret:** Sudden drops in consumption may indicate electricity theft.  
    **Meaningful observations:** Investigate hours with red points to understand possible tampering or anomalies.
    """)
