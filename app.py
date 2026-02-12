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
            
            # ✅ Try Online Store first (Allowed by Firewall)
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
    df = pd.DataFrame({'timestamp': dates, 'aqi': [100]*72})
    
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
    except:
        input_data = df.tail(1)
else:
    input_data = df.tail(1).copy()
    if 'hour' not in input_data.columns:
        input_data['hour'] = pd.to_datetime(input_data['timestamp']).dt.hour

# Forecast Prediction
all_features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity', 'hour']
final_features = all_features
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
# 5. DASHBOARD UI - ENHANCED FORECAST GRAPH
# ------------------------------------------------------------------
st.subheader("📈 Forecast Analysis (Next 72 Hours)")

# Prepare data for plotting
history_df = df.tail(72).copy()
history_df['Rel'] = range(-len(history_df), 0)
forecast_df = pd.DataFrame({"Hours": range(1, len(preds)+1), "AQI": preds})

fig = go.Figure()

# 1. Past Data (Blue Area)
fig.add_trace(go.Scatter(
    x=history_df['Rel'], 
    y=history_df['aqi'], 
    fill='tozeroy', 
    name='History (DB)', 
    line=dict(color='#00B4D8', width=3),
    fillcolor='rgba(0, 180, 216, 0.2)' # Subtle transparency
))

# 2. Future Forecast (Red Dotted Line)
fig.add_trace(go.Scatter(
    x=forecast_df['Hours'], 
    y=forecast_df['AQI'], 
    mode='lines+markers', 
    name='Forecast (AI)', 
    line=dict(color='#FF4B4B', width=4, dash='dot'),
    marker=dict(size=6, color='#FF4B4B')
))

# 3. Vertical "NOW" Line
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white", annotation_text=" NOW", annotation_position="top right")

# 4. Layout Adjustments
fig.update_layout(
    template="plotly_dark", 
    height=450, 
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis=dict(
        title="Timeline (Hours)",
        gridcolor='rgba(255, 255, 255, 0.1)',
        zeroline=False
    ),
    yaxis=dict(
        title="AQI Level",
        gridcolor='rgba(255, 255, 255, 0.1)',
        # Auto-scale Y axis to fit the data
        range=[min(min(history_df['aqi']), min(preds)) - 10, max(max(history_df['aqi']), max(preds)) + 20]
    ),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
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
        base_df = row[features].copy()
        for col in base_df.columns:
            if base_df[col].dtype == 'object': base_df[col] = 0
        base_pred = model.predict(base_df)
        if isinstance(base_pred, list): base_pred = np.array(base_pred)
        
        scores = {}
        for col in features:
            temp_df = base_df.copy()
            orig = temp_df[col].values[0]
            temp_df[col] = 1.0 if orig == 0 else orig * 1.2
            new_pred = model.predict(temp_df)
            if isinstance(new_pred, list): new_pred = np.array(new_pred)
            scores[col] = np.mean(np.abs(new_pred - base_pred))
        return scores
    except: return None

s_dict = get_sensitivity_scores(model, input_data, final_features)
if s_dict:
    imp_df = pd.DataFrame(list(s_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance')
    if imp_df['Importance'].max() > 0:
        imp_df['Importance'] = (imp_df['Importance'] / imp_df['Importance'].max()) * 100
    fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h', marker_color='#00CC96'))
    fig_imp.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0), xaxis_title="Relative Impact (%)")
    st.plotly_chart(fig_imp, use_container_width=True)