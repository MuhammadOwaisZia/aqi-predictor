import hopsworks
import pandas as pd
import requests
import os
import time
import numpy as np
from dotenv import load_dotenv

# 1. Connect to Hopsworks
load_dotenv()
try:
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit()

# 2. Fetch Data
print("🌍 Fetching latest data from Satellite...")
try:
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=us_aqi,pm10,pm2_5&timezone=Asia%2FKarachi&past_days=1&forecast_days=3"
    weather_url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,relativehumidity_2m&timezone=Asia%2FKarachi&past_days=1&forecast_days=3"

    aqi_data = requests.get(aqi_url).json()
    weather_data = requests.get(weather_url).json()
except Exception as e:
    print(f"❌ Failed to fetch weather data: {e}")
    exit()

# 3. Create DataFrames
df_aqi = pd.DataFrame({
    'timestamp': aqi_data['hourly']['time'],
    'aqi': aqi_data['hourly']['us_aqi'],
    'pm25': aqi_data['hourly']['pm2_5'],
    'pm10': aqi_data['hourly']['pm10']
})

df_weather = pd.DataFrame({
    'timestamp': weather_data['hourly']['time'],
    'temp': weather_data['hourly']['temperature_2m'],
    'humidity': weather_data['hourly']['relativehumidity_2m']
})

# 4. Merge & Fix Types (Strict Schema)
df = pd.merge(df_aqi, df_weather, on='timestamp')
df['city'] = 'Karachi'

# Convert to correct types for Hopsworks
df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6 
df['aqi'] = df['aqi'].fillna(0).astype('int64')
df['humidity'] = df['humidity'].fillna(0).astype('int64')
df['pm25'] = df['pm25'].astype(float)
df['pm10'] = df['pm10'].astype(float)
df['temp'] = df['temp'].astype(float)

# 5. Upload with AUTO-RETRY Logic
MAX_RETRIES = 3
print(f"🚀 Preparing to upload {len(df)} rows...")

for attempt in range(MAX_RETRIES):
    try:
        print(f"🔄 Attempt {attempt + 1}/{MAX_RETRIES}...")
        
        # We assume the previous partial upload might have worked, 
        # so we force it through.
        fg.insert(df, write_options={"wait_for_job": False})
        
        print("✅ SUCCESS! Data sent to Hopsworks.")
        break # Exit loop if successful
        
    except Exception as e:
        print(f"⚠️ Connection Error on Attempt {attempt + 1}: {e}")
        if attempt < MAX_RETRIES - 1:
            print("⏳ Waiting 10 seconds before retrying...")
            time.sleep(10)
        else:
            print("❌ All attempts failed. Please check your internet connection.")