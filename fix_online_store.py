import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# 1. Get the data from Version 1
print("Reading data from Version 1...")
old_fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
df = old_fg.read()

# 2. Create Version 2 with Online Enabled
print("Creating Version 2 with Online Store ENABLED...")
new_fg = fs.get_or_create_feature_group(
    name="aqi_features_hourly",
    version=2, # <--- Incremented version
    primary_key=['timestamp'], 
    event_time='timestamp',
    online_enabled=True # <--- The magic switch
)

# 3. Upload the data
new_fg.insert(df)
print("✅ Success! Version 2 is created and Online.")