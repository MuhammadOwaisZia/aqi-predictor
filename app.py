import streamlit as st
import hopsworks
import joblib
import pandas as pd
import plotly.graph_objects as go
import os
import shutil
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="wide")

# CSS: Cleaner Look & Visible Menu
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; overflow: visible !important; }
    [data-testid="column"] { min-width: 150px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. FAST LOADING SYSTEM (DISK CACHE)
# ------------------------------------------------------------------
MODEL_FILE = "best_aqi_model.pkl"

def get_hopsworks_project():
    load_dotenv()
    return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))

def download_model_from_cloud(project):
    """Force downloads the latest model from Hopsworks"""
    with st.spinner("☁️ Downloading latest AI Brain... (This takes time)"):
        mr = project.get_model_registry()
        models = mr.get_models("aqi_hourly_predictor")
        
        # Get the absolute latest version
        best_model = sorted(models, key=lambda x: x.version)[-1]
        
        # Download to a temporary folder
        temp_dir = best_model.download()
        
        # Move it to our main folder for next time
        source_path = os.path.join(temp_dir, MODEL_FILE)
        if os.path.exists(source_path):
            shutil.copy(source_path, MODEL_FILE)
            
        return joblib.load(MODEL_FILE), best_model.version

@st.cache_resource(show_spinner=False)
def load_resources():
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    
    # --- 1. MODEL LOADING STRATEGY ---
    model = None
    version_info = "Local Cache"
    
    # Strategy A: Load from Local Disk (FAST)
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            print("✅ Loaded Model from Disk (Fast Mode)")
        except:
            pass # If corrupted, fall back to download

    # Strategy B: Download from Cloud (SLOW but necessary once)
    if model is None:
        model, version_info = download_model_from_cloud(project)
    
    # --- 2. DATA LOADING ---
    with st.spinner("📊 Fetching History Data..."):
        fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
        
        # ✅ CRITICAL FIX: Force HTTP mode to bypass Streamlit Firewall
        df = fg.select_all().read(read_options={"use_hive": True})
        
        df = df.sort_values(by="timestamp")

    return model, df, version_info

# Load Resources
model, df, version = load_resources()

# ------------------------------------------------------------------
# 3. LIVE DATA BRIDGE (The Fix for "87 vs 156")
# ------------------------------------------------------------------
def get_live_satellite_data():
    """Bypasses database lag and fetches Real-Time AQI directly"""
    try:
        # Fetch Real-Time Data for Karachi
        aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&current=us_aqi,pm10,pm2_5&timezone=Asia%2FKarachi"
        weather_url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&current=temperature_2m,relativehumidity_2m&timezone=Asia%2FKarachi"
        
        aqi_resp = requests.get(aqi_url).json()['current']
        weather_resp = requests.get(weather_url).json()['current']
        
        # Create a single-row DataFrame with the LIVE numbers
        return pd.DataFrame({
            'aqi': [aqi_resp['us_aqi']],
            'pm25': [aqi_resp['pm2_5']],
            'pm10': [aqi_resp['pm10']],
            'temp': [weather_resp['temperature_2m']],
            'humidity': [weather_resp['relativehumidity_2m']],
            'hour': [pd.Timestamp.now().hour]
        })
    except Exception:
        return None # If API fails, we fall back to database

# Try to get LIVE data
live_data = get_live_satellite_data()

if live_data is not None:
    latest_data = live_data
    source_msg = "✅ Using Live Satellite Data"
else:
    # Fallback to Database if internet fails
    latest_data = df.tail(1).copy()
    latest_data['hour'] = pd.to_datetime(latest_data['timestamp']).dt.hour
    source_msg = "⚠️ Using Database (Might be delayed)"

# ------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ------------------------------------------------------------------
all_features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity', 'hour']

# Auto-Fix Features
try:
    if hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
    elif hasattr(model, "estimators_"): 
        required = list(model.estimators_[0].feature_names_in_)
    else: required = all_features
except: required = all_features

final_features = [f for f in all_features if f in required]

# Predict
try: preds = model.predict(latest_data[final_features])[0]
except: preds = model.predict(latest_data[final_features])

num_preds = len(preds)
future_hours = range(1, num_preds + 1)
forecast_df = pd.DataFrame({"Hours Ahead": future_hours, "Predicted AQI": preds})
history_df = df.tail(72).copy()
history_df['Hours_Relative'] = range(-len(history_df), 0)

# ------------------------------------------------------------------
# 5. DASHBOARD UI
# ------------------------------------------------------------------
st.title("🌫️ Karachi AQI Forecast")
st.caption(f"Status: Live | Source: {source_msg}")

current_aqi = int(latest_data['aqi'].values[0])
max_pred = int(max(preds))
avg_next_24 = int(sum(preds[:24]) / 24)

# 🚨 HAZARDOUS ALERTS (Visual Banners) 
if current_aqi >= 300:
    st.error("🚨 HAZARDOUS AIR QUALITY WARNING: Avoid all outdoor exertion.")
elif current_aqi >= 200:
    st.error("⚠️ VERY UNHEALTHY: Active children and adults should avoid outdoor exertion.")
elif current_aqi >= 150:
    st.warning("😷 UNHEALTHY: Sensitive groups should wear masks.")

# --- Helper to get AQI Label & Color ---
def get_aqi_label(aqi_value):
    if aqi_value <= 50: return "Good 🌱", "normal"
    elif aqi_value <= 100: return "Moderate 😐", "off" 
    elif aqi_value <= 150: return "Unhealthy (Sens.) 😷", "inverse"
    elif aqi_value <= 200: return "Unhealthy 🔴", "inverse"
    elif aqi_value <= 300: return "Very Unhealthy 🟣", "inverse"
    else: return "Hazardous ☠️", "inverse"

col1, col2, col3, col4 = st.columns(4)

# Get dynamic labels
curr_label, curr_color = get_aqi_label(current_aqi)
peak_label, peak_color = get_aqi_label(max_pred)

with col1:
    st.metric("📍 Current AQI", current_aqi, delta=curr_label, delta_color=curr_color)
with col2:
    st.metric("🔮 Next 24h Avg", avg_next_24, delta=f"{avg_next_24-current_aqi} vs Now", delta_color="inverse")
with col3:
    st.metric("⚠️ Peak Predicted", max_pred, delta=peak_label, delta_color=peak_color)
with col4:
    st.metric("🕒 Horizon", f"{num_preds} Hours")

st.divider()

# ✅ HEADING
st.subheader("📈 Next 3 Days Forecast")

# ✅ ENHANCED GRAPH (Clean & Professional)
fig = go.Figure()

# 1. Past Data (Blue Area)
fig.add_trace(go.Scatter(
    x=history_df['Hours_Relative'], 
    y=history_df['aqi'],
    mode='lines',
    name='History (Past 72h)',
    line=dict(color='#00B4D8', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 180, 216, 0.1)',
    hovertemplate='<b>Past</b><br>Hour: %{x}<br>AQI: %{y}<extra></extra>' 
))

# 2. Future Forecast (Red Dotted)
fig.add_trace(go.Scatter(
    x=forecast_df['Hours Ahead'], 
    y=forecast_df['Predicted AQI'],
    mode='lines+markers',
    name='Forecast (Next 72h)',
    line=dict(color='#FF4B4B', width=3, dash='dot'),
    marker=dict(size=5, color='#FF4B4B'),
    hovertemplate='<b>Future</b><br>Hour: +%{x}<br>AQI: %{y}<extra></extra>' 
))

# 3. "NOW" Line
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white", annotation_text="NOW")

# 4. Professional Layout
fig.update_layout(
    xaxis_title="Timeline (Hours)",
    yaxis_title="AQI Level",
    hovermode="x unified", 
    height=450,
    showlegend=True,
    # ✅ Legend moved to the RIGHT side
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0)"), 
    template="plotly_dark",
    margin=dict(l=0,r=0,t=30,b=0)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📂 Raw Data"):
    st.dataframe(forecast_df, use_container_width=True)

st.divider()
st.subheader("🤖 Feature Importance")
if hasattr(model, 'estimators_'):
    try:
        inner = model.estimators_[0]
        scores = inner.feature_importances_ if hasattr(inner, 'feature_importances_') else [abs(x) for x in inner.coef_]
        
        # Ensure lengths match before plotting
        if len(scores) == len(final_features):
            imp_df = pd.DataFrame({'Feature': final_features, 'Importance': scores}).sort_values(by='Importance', ascending=True)
            fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h', marker=dict(color='#00CC96')))
            fig_imp.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), template="plotly_dark")
            st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.write("Feature importance not available for this model type.")