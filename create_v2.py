import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv

# Load your API Key from .env
load_dotenv()

# 1. Connect to Hopsworks
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

print("📦 Fetching data from Version 1...")
old_fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
df = old_fg.read()

# 2. Create Version 2 with Online Store ENABLED
print("🚀 Creating Version 2 with Online Store active...")
new_fg = fs.get_or_create_feature_group(
    name="aqi_features_hourly",
    version=2, 
    primary_key=['timestamp'], 
    event_time='timestamp',
    online_enabled=True # <--- This fixes the error you saw on the website
)

# 3. Insert the data
new_fg.insert(df)

print("✅ DONE! Version 2 is now live and online-ready.")