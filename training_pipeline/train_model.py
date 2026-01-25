import hopsworks
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
from dotenv import load_dotenv

# 1. Connect
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# 2. Get the Data (History)
try:
    aqi_fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
    df = aqi_fg.select_all().read()
except:
    print("⚠️ Data not found. Run backfill first!")
    exit()

df = df.sort_values(by="timestamp")

# ---------------------------------------------------------
# 🎯 THE CRITICAL CHANGE FOR YOUR GOAL
# We create 72 separate targets (Hour 1 to Hour 72)
# ---------------------------------------------------------
print("⏳ Creating targets for next 72 Hours (3 Days)...")
targets = []
for i in range(1, 73):  # Loop from 1 to 72
    col_name = f'target_h{i}'
    # Shift data 'i' steps back so the model learns to look 'i' hours ahead
    df[col_name] = df['aqi'].shift(-i) 
    targets.append(col_name)

df = df.dropna()
print(f"✅ Created {len(targets)} future targets for the model to learn.")

# 3. Define Input Features (What we know NOW)
features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity']

X = df[features]       # Input: Current conditions
y = df[targets]        # Output: Next 72 hours of AQI

# 4. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("🥊 Training 3-Day Forecaster (Gradient Boosting)...")
# We use MultiOutputRegressor so it can output 72 numbers at once
model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100))
model.fit(X_train, y_train)

# 5. Evaluate & Save
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"🏆 Average Error across 3 Days: {round(mae, 2)}")

joblib.dump(model, "best_aqi_model.pkl")

# Register
mr = project.get_model_registry()
hopsworks_model = mr.python.create_model(
    name="aqi_hourly_predictor",
    metrics={"mae": mae},
    description="Predicts next 72 hours (3 Days) based on hourly data"
)
hopsworks_model.save("best_aqi_model.pkl")

print("✅ 72-Hour Model Successfully Saved!")