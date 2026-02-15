import os
import hopsworks
import pandas as pd
import requests
import time
from datetime import datetime

# --- 1. ROBUST RETRY FUNCTION ---
def fetch_with_retry(url, retries=3, delay=5):
    """Tries to fetch data 3 times before failing."""
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network hiccup (Attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
    raise ConnectionError(f"❌ Failed to fetch data from {url}")

# --- 2. MAIN EXECUTION ---
def run():
    # A. Fetch Data
    print("⏳ Fetching live weather data...")
    weather_data = fetch_with_retry("https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&current=temperature_2m,relativehumidity_2m&timezone=Asia%2FKarachi")
    
    print("⏳ Fetching live AQI data...")
    aqi_data = fetch_with_retry("https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&current=us_aqi,pm10,pm2_5&timezone=Asia%2FKarachi")
    
    # B. Parse Data
    timestamp = datetime.now()
    data = {
        'timestamp': [int(timestamp.timestamp() * 1000)], # Hopsworks expects millis
        'aqi': [aqi_data['current']['us_aqi']],
        'pm25': [aqi_data['current']['pm2_5']],
        'pm10': [aqi_data['current']['pm10']],
        'temp': [weather_data['current']['temperature_2m']],
        'humidity': [weather_data['current']['relativehumidity_2m']],
        'hour': [timestamp.hour]
    }
    df = pd.DataFrame(data)
    print(f"✅ Data processed: AQI={data['aqi'][0]}")

    # C. Save to Hopsworks
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Get the Feature Group (Ensure version matches your backfill)
    fg = fs.get_feature_group(name="aqi_features_hourly", version=2)
    
    print("💾 Inserting data...")
    fg.insert(df)
    print("🎉 Success! Live data uploaded.")

if __name__ == "__main__":
    run()