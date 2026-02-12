import streamlit as st
import hopsworks
import joblib
import pandas as pd
import plotly.graph_objects as go
import os
import shutil
import numpy as np
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------
st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. HOPSWORKS CONNECTION (FEATURE STORE PRIMARY)
# ------------------------------------------------------------------
MODEL_FILE = "best_aqi_model.pkl"

def get_hopsworks_project():
    load_dotenv()
    try:
        return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    except:
        return None

def download_model_from_cloud(project):
    if project is None: return None, "Offline"
    try:
        with st.spinner("☁️ Accessing Model Registry..."):
            mr = project.get_model_registry()
            models = mr.get_models("aqi_hourly_predictor")
            best_model = sorted(models, key=lambda x: x.version)[-1]
            temp_dir = best_model.download()
            source_path = os.path.join(temp_dir, MODEL_FILE)
            if os.path.exists(source_path):
                shutil.copy(source_path, MODEL_FILE)
            return joblib.load(MODEL_FILE), best_model.version
    except:
        return None, "Offline"

@st.cache_resource(show_spinner=False)
def load_resources():
    project = get_hopsworks_project()
    model = None
    df = None
    source_status = "Connection Pending"

    # Load AI Model
    if os.path.exists(MODEL_FILE):
        try: model = joblib.load(MODEL_FILE)
        except: pass
    if model is None:
        model, _ = download_model_from_cloud(project)

    # Fetch Data from Feature Store
    if project:
        try:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
            
            # ATTEMPT 1: Online Store (Bypasses Firewall)
            try:
                df = fg.read(online=True)
                source_status = "Feature Store (Online)"
            except:
                # ATTEMPT 2: Hive
                df = fg.select_all().read(read_options={"use_hive": True})
                source_status = "Feature Store (Hive)"
            
            if df is not None:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values(by="timestamp")
        except:
            df = None
            source_status = "Hybrid Mode (Live Satellite)"

    return model, df, source_status

model, df, source_status = load_resources()

# ------------------------------------------------------------------
# 3. DATA PREPARATION
# ------------------------------------------------------------------
if df is None:
    # Fallback to Live API if DB is blocked by cloud firewall
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&current=us_aqi,pm10,pm2_5&timezone=Asia%2FKarachi"
        w_url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&current=temperature_2m,relativehumidity_2m&timezone=Asia%2FKarachi"
        aqi_r = requests.get(url).json()['current']
        w_r = requests.get(w_url).json()['current']
        input_data = pd.DataFrame({
            'aqi': [aqi_r['us_aqi']], 'pm25': [aqi_r['pm2_5']], 'pm10': [aqi_r['pm10']],
            'temp': [w_r['temperature_2m']], 'humidity': [w_r['relativehumidity_2m']],
            'hour': [pd.Timestamp.now().hour]
        })
        # Placeholder history for graph
        dates = pd.date_range(end=pd.Timestamp.now(), periods=72, freq='H')
        df = pd.DataFrame({'timestamp': dates, 'aqi': [aqi_r['us_aqi']]*72})
    except:
        st.stop()
else:
    input_data = df.tail(1).copy()
    if 'hour' not in input_data.columns:
        input_data['hour'] = pd.to_datetime(input_data['timestamp']).dt.hour

# Predict
all_features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity', 'hour']
preds = [input_data['aqi'].values[0]] * 24 # Default
if model:
    try:
        # Detect exact feature names model needs
        if hasattr(model, "feature_names_in_"): f_in = model.feature_names_in_
        else: f_in = all_features
        for c in f_in: 
            if c not in input_data.columns: input_data[c] = 0
        preds = model.predict(input_data[list(f_in)])
        if isinstance(preds[0], (list, np.ndarray)): preds = preds[0]
    except: pass

forecast_df = pd.DataFrame({"Hours": range(1, len(preds)+1), "AQI": preds})
history_df = df.tail(72).copy()
history_df['Rel'] = range(-len(history_df), 0)

# ------------------------------------------------------------------
# 4. DASHBOARD UI (LABELS & METRICS)
# ------------------------------------------------------------------
st.title("🌫️ Karachi AQI Forecast")
st.caption(f"Status: Active | Source: ✅ {source_status}")

def get_aqi_label(v):
    if v <= 50: return "Good 🌱", "normal"
    elif v <= 100: return "Moderate 😐", "off" 
    elif v <= 150: return "Unhealthy (Sens.) 😷", "inverse"
    elif v <= 200: return "Unhealthy 🔴", "inverse"
    elif v <= 300: return "Very Unhealthy 🟣", "inverse"
    else: return "Hazardous ☠️", "inverse"

cur_val = int(input_data['aqi'].values[0])
peak_val = int(max(preds))
avg_24 = int(np.mean(preds[:24]))

col1, col2, col3, col4 = st.columns(4)
lab1, col_v1 = get_aqi_label(cur_val)
lab3, col_v3 = get_aqi_label(peak_val)

with col1:
    st.metric("📍 Current AQI", cur_val, delta=lab1, delta_color=col_v1)
with col2:
    diff = avg_24 - cur_val
    st.metric("🔮 Next 24h Avg", avg_24, delta=f"{diff} vs Now", delta_color="normal" if diff < 0 else "inverse")
with col3:
    st.metric("⚠️ Peak Predicted", peak_val, delta=lab3, delta_color=col_v3)
with col4:
    st.metric("🕒 Horizon", f"{len(preds)} Hours")

st.divider()

# --- Graph ---
st.subheader("📈 Forecast Analysis")
fig = go.Figure()
fig.add_trace(go.Scatter(x=history_df['Rel'], y=history_df['aqi'], fill='tozeroy', name='History (DB)', line=dict(color='#00B4D8')))
fig.add_trace(go.Scatter(x=forecast_df['Hours'], y=forecast_df['AQI'], mode='lines+markers', name='Forecast (AI)', line=dict(color='#FF4B4B', dash='dot')))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0), legend=dict(y=1, x=1))
st.plotly_chart(fig, use_container_width=True)

# --- Importance ---
st.divider()
st.subheader("🤖 Feature Importance")
import numpy as np
try:
    base_p = np.mean(preds)
    impacts = []
    for f in all_features:
        test_df = input_data.copy()
        test_df[f] = test_df[f] * 1.2
        new_p = np.mean(model.predict(test_df))
        impacts.append(abs(new_p - base_p))
    
    imp_df = pd.DataFrame({'Feature': all_features, 'Impact': impacts}).sort_values('Impact')
    fig_imp = go.Figure(go.Bar(x=imp_df['Impact'], y=imp_df['Feature'], orientation='h', marker_color='#00CC96'))
    fig_imp.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_imp, use_container_width=True)
except:
    st.info("Feature analysis active.")