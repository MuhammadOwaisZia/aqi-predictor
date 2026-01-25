import pandas as pd
import hopsworks
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. Setup
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def backfill_hourly_data():
    print("⏳ Fetching 4 months of HOURLY data from Open-Meteo...")
    
    # Karachi Coordinates
    lat, lon = 24.8607, 67.0011
    
    # DATES: Stop 5 days ago to avoid "Archive Delay" errors
    end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    
    print(f"   📅 Requesting period: {start_date} to {end_date}")

    # 1. Get Air Quality Data (from Air Quality API)
    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=pm10,pm2_5,us_aqi&timezone=auto"
    aqi_response = requests.get(aqi_url).json()
    
    if 'hourly' not in aqi_response:
        print("❌ Error fetching AQI:", aqi_response)
        return pd.DataFrame()

    # 2. Get Weather Data (from Archive API)
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m&timezone=auto"
    weather_response = requests.get(weather_url).json()
    
    if 'hourly' not in weather_response:
        print("❌ Error fetching Weather:", weather_response)
        return pd.DataFrame()

    # 3. Create DataFrame (Safely)
    hourly_aqi = aqi_response['hourly']
    hourly_weather = weather_response['hourly']
    
    min_len = min(len(hourly_aqi['time']), len(hourly_weather['time']))
    
    df = pd.DataFrame({
        "timestamp": hourly_aqi['time'][:min_len],
        "pm25": hourly_aqi['pm2_5'][:min_len],
        "pm10": hourly_aqi['pm10'][:min_len],
        "aqi": hourly_aqi['us_aqi'][:min_len],
        "temp": hourly_weather['temperature_2m'][:min_len],
        "humidity": hourly_weather['relative_humidity_2m'][:min_len],
        "city": "Karachi"
    })

    # Cleanup: Convert time strings to Millisecond Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
    
    # Drop rows with missing values
    df = df.dropna()
    
    print(f"✅ Downloaded {len(df)} hourly rows.")
    return df

def upload_to_hopsworks(df):
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    # Create Feature Group VERSION 1
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features_hourly",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly Air Quality Data for Karachi"
    )
    
    print("📤 Uploading data (Async Mode)...")
    
    # --- THE FIX IS HERE ---
    # write_options={"wait_for_job": False} tells Python NOT to wait for the server
    aqi_fg.insert(df, write_options={"wait_for_job": False})
    
    print("✅ Success! Data upload started in background.")

if __name__ == "__main__":
    df = backfill_hourly_data()
    if not df.empty:
        upload_to_hopsworks(df)