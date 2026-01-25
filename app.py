import streamlit as st
import hopsworks
import joblib
import pandas as pd
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide"  # Use the full screen width
)

# Custom CSS to hide default menu and style the metrics
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stAlert {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. CONNECT TO BACKEND (CACHED)
# ------------------------------------------------------------------
@st.cache_resource
def load_resources():
    load_dotenv()
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    
    # Get Model
    mr = project.get_model_registry()
    model = mr.get_model("aqi_hourly_predictor", version=1)
    model_dir = model.download()
    loaded_model = joblib.load(model_dir + "/best_aqi_model.pkl")
    
    # Get Data
    fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
    df = fg.select_all().read()
    df = df.sort_values(by="timestamp")
    
    return loaded_model, df

# Load resources with a spinner
with st.spinner("📡 Connecting to Hopsworks AI Cloud..."):
    model, df = load_resources()

# ------------------------------------------------------------------
# 3. DATA PROCESSING
# ------------------------------------------------------------------
# Get the most recent data point for prediction
latest_data = df.tail(1)
features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity']

# Predict (Handles both 3-hour and 72-hour models automatically)
try:
    preds = model.predict(latest_data[features])[0]
except:
    preds = model.predict(latest_data[features]) # Handle different array shapes

num_preds = len(preds)

# Create Forecast DataFrame
future_hours = range(1, num_preds + 1)
forecast_df = pd.DataFrame({
    "Hours Ahead": future_hours,
    "Predicted AQI": preds
})

# Get History (Last 3 days / 72 hours)
history_df = df.tail(72).copy()
history_df['Hours_Relative'] = range(-len(history_df), 0)

# ------------------------------------------------------------------
# 4. DASHBOARD UI
# ------------------------------------------------------------------

# HEADER
st.title("🌫️ Karachi Air Quality Index Forecast")
st.markdown(f"**Status:** Real-time AI prediction based on data from **{latest_data['city'].values[0]}**.")

# KEY METRICS ROW
col1, col2, col3, col4 = st.columns(4)

# Calculate Stats
current_aqi = int(latest_data['aqi'].values[0])
max_pred = int(max(preds))
avg_next_24 = int(sum(preds[:24]) / 24) if num_preds >= 24 else int(sum(preds) / num_preds)

with col1:
    st.metric(label="📍 Current AQI", value=current_aqi, delta="Verified Now")
with col2:
    st.metric(label="🔮 Next 24h Average", value=avg_next_24, delta=avg_next_24-current_aqi, delta_color="inverse")
with col3:
    st.metric(label="⚠️ Peak Predicted", value=max_pred, delta="Highest Forecast", delta_color="off")
with col4:
    st.metric(label="🕒 Forecast Horizon", value=f"{num_preds} Hours", delta="Live Model")

st.divider()

# INTERACTIVE GRAPH (PLOTLY)
st.subheader("📈 3-Day Trends (History vs. AI Forecast)")

fig = go.Figure()

# 1. Plot History (Blue Area)
fig.add_trace(go.Scatter(
    x=history_df['Hours_Relative'], 
    y=history_df['aqi'],
    mode='lines',
    name='Past 72h (Actual)',
    line=dict(color='#00B4D8', width=3),
    fill='tozeroy', # Fills area under line
    fillcolor='rgba(0, 180, 216, 0.1)'
))

# 2. Plot Forecast (Red Dotted Line)
fig.add_trace(go.Scatter(
    x=forecast_df['Hours Ahead'], 
    y=forecast_df['Predicted AQI'],
    mode='lines+markers',
    name='Next 72h (AI Forecast)',
    line=dict(color='#FF4B4B', width=3, dash='dot'),
    marker=dict(size=6)
))

# 3. Add "Current Time" Line
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white", annotation_text="NOW")

# 4. Make it pretty
fig.update_layout(
    xaxis_title="Time (Hours from Now)",
    yaxis_title="AQI Level",
    hovermode="x unified",
    height=500,
    showlegend=True,
    template="plotly_dark", # Fits your dark theme requirement
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# RAW DATA SECTION
with st.expander("📂 View Raw Sensor Data & Predictions"):
    tab1, tab2 = st.tabs(["🔮 Future Forecast", "📜 Past History"])
    
    with tab1:
        st.dataframe(forecast_df, use_container_width=True)
    
    with tab2:
        st.dataframe(history_df[['timestamp', 'aqi', 'pm25', 'temp', 'humidity']].sort_values(by='timestamp', ascending=False), use_container_width=True)

# ------------------------------------------------------------------
# 5. EXPLAINABILITY (Why is AQI High?) - CORRECTED SECTION
# ------------------------------------------------------------------
st.divider()
st.subheader("🤖 Model Explanations (Feature Importance)")

try:
    # 1. Get the specific model for the first hour (T+1)
    if hasattr(model, 'estimators_'):
        inner_model = model.estimators_[0]
        
        # 2. Check: Is it a Tree (Random Forest) or Line (Linear Regression)?
        if hasattr(inner_model, 'feature_importances_'):
            # It's a Tree Model
            scores = inner_model.feature_importances_
        elif hasattr(inner_model, 'coef_'):
            # It's Linear Regression (Use absolute value of coefficients)
            scores = [abs(x) for x in inner_model.coef_]
        else:
            scores = [0] * 5

        feature_names = ['aqi', 'pm25', 'pm10', 'temp', 'humidity']

        # 3. Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': scores
        }).sort_values(by='Importance', ascending=True)

        # 4. Plot Bar Chart
        fig_imp = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color='#00CC96')
        ))

        fig_imp.update_layout(
            title="Which weather factors influence the AI most?",
            xaxis_title="Impact Score (Magnitude)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark"
        )

        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("This chart explains which inputs have the biggest impact on the prediction.")
    else:
        st.info("This model type does not support simple feature importance.")

except Exception as e:
    st.warning(f"Could not load explanations: {e}")