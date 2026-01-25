import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def connect_to_feature_store():
    # This logs you into the cloud
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    return project.get_feature_store()

# Test the connection
if __name__ == "__main__":
    fs = connect_to_feature_store()
    print(f"✅ Connected to: {fs.name}")

load_dotenv()
API_TOKEN = os.getenv("AQI_TOKEN")

def fetch_aqi_data(city="Karachi"):
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    response = requests.get(url)
    res = response.json()
    
    if res['status'] == 'ok':
        data = res['data']
        # We are creating a 'Feature Dictionary' 
        features = {
            "city": city,
            "aqi": data['aqi'],
            "pm25": data['iaqi'].get('pm25', {}).get('v', 0),
            "pm10": data['iaqi'].get('pm10', {}).get('v', 0),
            "temp": data['iaqi'].get('t', {}).get('v', 0),
            "humidity": data['iaqi'].get('h', {}).get('v', 0),
            "wind": data['iaqi'].get('w', {}).get('v', 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return features
    return None

if __name__ == "__main__":
    data = fetch_aqi_data()
    if data:
        # Convert to a DataFrame (Standard for ML data) 
        df = pd.DataFrame([data])
        print("--- New Data Point Collected ---")
        print(df)
        
        # Save to a local CSV file for now (This is our local backup)
        df.to_csv("data_pipeline/local_data_log.csv", mode='a', header=not os.path.exists("data_pipeline/local_data_log.csv"), index=False)
        print("\n✅ Data saved to data_pipeline/local_data_log.csv")

def push_to_hopsworks(df):
    fs = connect_to_feature_store()
    
    # Create or get the 'table' in the cloud
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city"], # Unique identifier
        event_time="timestamp", # When this happened
        description="Air Quality features for Karachi"
    )
    
    # Upload the data
    aqi_fg.insert(df)
    print("🚀 Data successfully pushed to Hopsworks!")