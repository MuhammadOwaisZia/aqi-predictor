import hopsworks
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Connect to Cloud
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# 2. Download the Trained Model
print("🧠 Downloading model from Model Registry...")
mr = project.get_model_registry()
model_meta = mr.get_model("aqi_predictor_3days", version=1)
model_dir = model_meta.download()
model = joblib.load(model_dir + "/aqi_model_3days.pkl")

# 3. Get the VERY LATEST Data point from Feature Store
print("📊 Fetching latest weather data...")
aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
# We read everything, sort by time, and take the LAST row (most recent)
df = aqi_fg.select_all().read()
df = df.sort_values(by="timestamp")
latest_data = df.tail(1)

print(f"📅 Based on data from: {pd.to_datetime(latest_data['timestamp'].values[0], unit='ms')}")

# 4. Make a Prediction
# We need to pass the same features we trained on
features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity']
X_new = latest_data[features]

preds = model.predict(X_new)

# 5. Show the Future
print("\n🔮 AI PREDICTION FOR KARACHI:")
print(f"   Day 1: {round(preds[0][0])} AQI")
print(f"   Day 2: {round(preds[0][1])} AQI")
print(f"   Day 3: {round(preds[0][2])} AQI")