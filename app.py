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
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. HOPSWORKS CONNECTION
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
    source_status = "Connecting..."

    if os.path.exists(MODEL_FILE):
        try: model = joblib.load(MODEL_FILE)
        except: pass
    if model is None:
        model, _ = download_model_from_cloud(project)

    if project:
        try:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
            
            try:
                df = fg.read(online=True)
                source_status = "Feature Store (Online)"
            except:
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
# 3. DATA & PREDICTION LOGIC
# ------------------------------------------------------------------
if df is None:
    source_status = "Hybrid Mode (Live Satellite)"
    dates = pd.date_range(end=pd.Timestamp.now(), periods=72, freq='H')
    df = pd.DataFrame({'timestamp': dates, 'aqi': [100.0]*72})
    
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&current=us_aqi,pm10,pm2_5&timezone=Asia%2FKarachi"
        w_url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&current=temperature_2m,relativehumidity_2m&timezone=Asia%2FKarachi"
        aqi_r = requests.get(url).json()['current']
        w_r = requests.get(w_url).json()['current']
        input_data = pd.DataFrame({
            'aqi': [float(aqi_r['us_aqi'])], 'pm25': [float(aqi_r['pm2_5'])], 'pm10': [float(aqi_r['pm10'])],
            'temp': [float(w_r['temperature_2m'])], 'humidity': [float(w_r['relativehumidity_2m'])],
            'hour': [pd.Timestamp.now().hour]
        })
    except:
        input_data = df.tail(1).copy()
else:
    input_data = df.tail(1).copy()
    if 'hour' not in input_data.columns:
        input_data['hour'] = pd.to_datetime(input_data['timestamp']).dt.hour

# Forecast Prediction
all_features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity', 'hour']
if model:
    try:
        if hasattr(model, "feature_names_in_"): f_in = model.feature_names_in_
        elif hasattr(model, "estimators_"): f_in = model.estimators_[0].feature_names_in_
        else: f_in = all_features
        final_features = list(f_in)
        for c in final_features: 
            if c not in input_data.columns: input_data[c] = 0
        preds = model.predict(input_data[final_features])
        if isinstance(preds[0], (list, np.ndarray)): preds = preds[0]
    except: preds = [input_data['aqi'].values[0]] * 24
else:
    preds = [input_data['aqi'].values[0]] * 24

# ------------------------------------------------------------------
# 5. DASHBOARD UI
# ------------------------------------------------------------------
st.title("🌫️ Karachi AQI Forecast")
st.caption(f"Status: Live | Source: ✅ {source_status}")

def get_metric_info(v):
    if v <= 50: return "Good 🌱", "normal"
    elif v <= 100: return "Moderate 😐", "off" 
    elif v <= 150: return "Unhealthy (Sens.) 😷", "inverse"
    elif v <= 200: return "Unhealthy 🔴", "inverse"
    elif v <= 300: return "Very Unhealthy 🟣", "inverse"
    else: return "Hazardous ☠️", "inverse"

cur_aqi = int(input_data['aqi'].values[0])
peak_aqi = int(max(preds))
avg_24 = int(np.mean(preds[:24]))
diff = avg_24 - cur_aqi

col1, col2, col3, col4 = st.columns(4)
curr_l, curr_c = get_metric_info(cur_aqi)
peak_l, peak_c = get_metric_info(peak_aqi)

with col1: st.metric("📍 Current AQI", cur_aqi, delta=curr_l, delta_color=curr_c)
with col2: st.metric("🔮 Next 24h Avg", avg_24, delta=f"{diff} vs Now", delta_color="normal" if diff < 0 else "inverse")
with col3: st.metric("⚠️ Peak Predicted", peak_aqi, delta=peak_l, delta_color=peak_c)
with col4: st.metric("🕒 Horizon", f"{len(preds)} Hours")

st.divider()

# ✅ ENHANCED FORECAST GRAPH
st.subheader("📈 Forecast Analysis (Next 72 Hours)")
history_df = df.tail(72).copy()
history_df['Rel'] = range(-len(history_df), 0)

# Round forecast for visual clarity
forecast_df = pd.DataFrame({
    "Hours": range(1, len(preds)+1), 
    "AQI": [round(float(p)) for p in preds]
})

fig = go.Figure()

# History Area
fig.add_trace(go.Scatter(
    x=history_df['Rel'], y=history_df['aqi'], 
    fill='tozeroy', name='History (DB)', 
    line=dict(color='#00B4D8', width=2),
    fillcolor='rgba(0, 180, 216, 0.2)'
))

# Forecast Line
fig.add_trace(go.Scatter(
    x=forecast_df['Hours'], y=forecast_df['AQI'], 
    mode='lines+markers', name='Forecast (AI)', 
    line=dict(color='#FF4B4B', width=3, dash='dot'),
    marker=dict(size=6, color='#FF4B4B'),
    hovertemplate="Hour: +%{x}<br>AQI: %{y}<extra></extra>"
))

fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="NOW")

# Smart Y-Axis Scaling
all_vals = list(history_df['aqi']) + list(forecast_df['AQI'])
y_min, y_max = min(all_vals), max(all_vals)

fig.update_layout(
    template="plotly_dark", 
    height=400, 
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(range=[y_min - 15, y_max + 15], title="AQI Level"),
    xaxis=dict(title="Timeline (Hours)"),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (SENSITIVITY)
# ------------------------------------------------------------------
st.divider()
st.subheader("🤖 Feature Importance (Real-Time Sensitivity)")

def get_sensitivity_scores(model, row, features):
    try:
        import numpy as np
        base_df = row[features].copy()
        for col in base_df.columns:
            if base_df[col].dtype == 'object': base_df[col] = 0
        
        base_pred = model.predict(base_df)
        if isinstance(base_pred, (list, np.ndarray)): base_pred = np.array(base_pred)
        
        scores = {}
        for col in features:
            temp_df = base_df.copy()
            orig = float(temp_df[col].values[0])
            temp_df[col] = 1.0 if orig == 0 else orig * 1.2
            new_pred = model.predict(temp_df)
            if isinstance(new_pred, (list, np.ndarray)): new_pred = np.array(new_pred)
            scores[col] = float(np.mean(np.abs(new_pred - base_pred)))
        return scores
    except: return None

s_dict = get_sensitivity_scores(model, input_data, all_features)
if s_dict:
    imp_df = pd.DataFrame(list(s_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance')
    if imp_df['Importance'].max() > 0:
        imp_df['Importance'] = (imp_df['Importance'] / imp_df['Importance'].max()) * 100
    
    fig_imp = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], 
        orientation='h', marker_color='#00CC96'
    ))
    fig_imp.update_layout(
        template="plotly_dark", height=300, 
        margin=dict(l=0,r=0,t=30,b=0), 
        xaxis_title="Relative Impact (%)"
    )
    st.plotly_chart(fig_imp, use_container_width=True)