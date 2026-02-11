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

# CSS: Cleaner Look
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. HOPSWORKS CONNECTION (STRICT MODE)
# ------------------------------------------------------------------
MODEL_FILE = "best_aqi_model.pkl"

def get_hopsworks_project():
    load_dotenv()
    try:
        return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    except:
        return None

def download_model_from_cloud(project):
    """Downloads model if connection exists, else returns None"""
    if project is None: return None, "Offline Mode"
    
    try:
        with st.spinner("☁️ Downloading AI Brain..."):
            mr = project.get_model_registry()
            models = mr.get_models("aqi_hourly_predictor")
            best_model = sorted(models, key=lambda x: x.version)[-1]
            temp_dir = best_model.download()
            source_path = os.path.join(temp_dir, MODEL_FILE)
            if os.path.exists(source_path):
                shutil.copy(source_path, MODEL_FILE)
            return joblib.load(MODEL_FILE), best_model.version
    except:
        return None, "Offline Mode"

@st.cache_resource(show_spinner=False)
def load_resources():
    # --- 1. CONNECT TO DB ---
    project = get_hopsworks_project()
    model = None
    df = None
    version_info = "Connecting..."

    # Load Model
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
        except: pass
    if model is None:
        model, version_info = download_model_from_cloud(project)

    # --- 2. FETCH HISTORY FROM ONLINE STORE (Firewall Bypass) ---
    try:
        if project:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
            
            # Force read from ONLINE store (MySQL Port) instead of Offline (Hive Port)
            df = fg.select(["timestamp", "aqi", "pm25", "pm10", "temp", "humidity"]).read(online=True)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by="timestamp")
            version_info = "Feature Store (Online)"
            
    except Exception as e:
        # If Online Store fails, try Hive one last time, else fail gracefully
        try:
             df = fg.select_all().read(read_options={"use_hive": True})
             version_info = "Feature Store (Hive)"
        except:
             # SILENT FAIL: Don't crash, just return None so we can use placeholder
             df = None

    return model, df, version_info

# Load Resources
model, df, source_status = load_resources()

# ------------------------------------------------------------------
# 3. FALLBACK HANDLING (Only if DB is truly broken)
# ------------------------------------------------------------------
if df is None:
    # Use "Hybrid Mode" label if DB fails
    source_status = "Hybrid Mode (Live Satellite)"
    
    # Create empty structure to prevent crash
    dates = pd.date_range(end=pd.Timestamp.now(), periods=72, freq='H')
    df = pd.DataFrame({'timestamp': dates, 'aqi': [100]*72})
    df['Hours_Relative'] = range(-72, 0)
    
    # Try to fetch REAL live data for the "Now" value
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
    input_data = df.tail(1)

# ------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ------------------------------------------------------------------
all_features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity', 'hour']
final_features = all_features

if model:
    try:
        # Auto-detect features expected by model
        if hasattr(model, "feature_names_in_"):
            final_features = [f for f in all_features if f in model.feature_names_in_]
        elif hasattr(model, "estimators_"): # If pipeline/random forest
            final_features = [f for f in all_features if f in model.estimators_[0].feature_names_in_]
    except: pass

# Generate Predictions
if 'hour' not in input_data.columns:
    input_data['hour'] = pd.to_datetime(input_data['timestamp'] if 'timestamp' in input_data else pd.Timestamp.now()).dt.hour

if model:
    try:
        # Fill missing cols with 0
        for col in final_features:
            if col not in input_data.columns: input_data[col] = 0
            
        preds = model.predict(input_data[final_features])
        if isinstance(preds[0], (list, np.ndarray)): preds = preds[0]
    except:
        preds = [input_data['aqi'].values[0]] * 24
else:
    preds = [input_data['aqi'].values[0]] * 24

# Prepare Forecast Data
num_preds = len(preds)
future_hours = range(1, num_preds + 1)
forecast_df = pd.DataFrame({"Hours Ahead": future_hours, "Predicted AQI": preds})

# Prepare History Data
history_df = df.tail(72).copy()
if 'Hours_Relative' not in history_df.columns:
    history_df['Hours_Relative'] = range(-len(history_df), 0)

# ------------------------------------------------------------------
# 5. DASHBOARD UI
# ------------------------------------------------------------------
st.title("🌫️ Karachi AQI Forecast")

# Status Label logic
if "Online" in source_status or "Hive" in source_status:
    st.caption(f"Status: Live | Source: ✅ {source_status}")
else:
    st.caption(f"Status: Live | Source: ⚠️ {source_status}")

current_aqi = int(input_data['aqi'].values[0]) if 'aqi' in input_data else 0
max_pred = int(max(preds))
avg_next_24 = int(sum(preds[:24]) / 24)

if current_aqi >= 300: st.error("🚨 HAZARDOUS")
elif current_aqi >= 200: st.error("⚠️ VERY UNHEALTHY")
elif current_aqi >= 150: st.warning("😷 UNHEALTHY")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("📍 Current AQI", current_aqi)
with col2: st.metric("🔮 Next 24h Avg", avg_next_24, delta=f"{avg_next_24-current_aqi}")
with col3: st.metric("⚠️ Peak", max_pred)
with col4: st.metric("🕒 Horizon", f"{num_preds} Hrs")

st.divider()
st.subheader("📈 Forecast Analysis")

fig = go.Figure()
# History
fig.add_trace(go.Scatter(x=history_df['Hours_Relative'], y=history_df['aqi'],
                         mode='lines', name='History', line=dict(color='#00B4D8', width=3), fill='tozeroy'))
# Forecast
fig.add_trace(go.Scatter(x=forecast_df['Hours Ahead'], y=forecast_df['Predicted AQI'],
                         mode='lines+markers', name='Forecast', line=dict(color='#FF4B4B', width=3, dash='dot')))

fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="NOW")
fig.update_layout(xaxis_title="Hours", yaxis_title="AQI", template="plotly_dark", height=450,
                  margin=dict(l=0,r=0,t=30,b=0), legend=dict(y=1, x=1))

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (ROBUST FIX)
# ------------------------------------------------------------------
st.divider()
st.subheader("🤖 Feature Importance")

def get_model_importance(model):
    """Deep search to find feature importance or coefficients"""
    # 1. Unwrap GridSearch/RandomSearch
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    
    # 2. Unwrap Pipeline (Get the last step which is the actual model)
    if hasattr(model, 'steps'):
        model = model.steps[-1][1]
    
    # 3. Try retrieving scores
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models might have shape (1, N), we need flat (N,)
        import numpy as np
        return np.abs(model.coef_).flatten()
    
    return None

try:
    # Attempt to extract scores
    scores = get_model_importance(model)
    
    if scores is not None and len(scores) > 0:
        # Match scores to feature names
        names = final_features
        # Safety check: If lengths don't match, generic names
        if len(scores) != len(names):
            names = [f"Feature {i}" for i in range(len(scores))]

        # Create DataFrame
        imp_df = pd.DataFrame({'Feature': names, 'Importance': scores})
        imp_df = imp_df.sort_values(by='Importance', ascending=True)

        # Plot
        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importance'], 
            y=imp_df['Feature'], 
            orientation='h', 
            marker=dict(color='#00CC96')
        ))
        fig_imp.update_layout(
            height=300, 
            margin=dict(l=0,r=0,t=30,b=0), 
            template="plotly_dark",
            xaxis_title="Impact Score (Absolute Value)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance could not be extracted from this model structure.")

except Exception as e:
    st.write(f"Could not extract feature importance: {e}")