import hopsworks
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from dotenv import load_dotenv

# 1. Connect
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# 2. Get the Data (History)
try:
    print("📊 Fetching data from Feature Store...")
    aqi_fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
    df = aqi_fg.select_all().read()
except:
    print("⚠️ Data not found. Run backfill first!")
    exit()

# Sort by time to ensure order
df = df.sort_values(by="timestamp")

# ---------------------------------------------------------
# 🛠️ FEATURE ENGINEERING (Fixing the "Blindness")
# We explicitly tell the model what time it is.
# ---------------------------------------------------------
# Convert timestamp to datetime objects just in case
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Extract the Hour (0-23) so the model learns daily cycles (Rush Hour vs Night)
df['hour'] = df['timestamp'].dt.hour 

# ---------------------------------------------------------
# 🎯 TARGET CREATION (72 Hours Ahead)
# ---------------------------------------------------------
print("⏳ Creating targets for next 72 Hours (3 Days)...")
targets = []
for i in range(1, 73):  # Loop from 1 to 72
    col_name = f'target_h{i}'
    # Shift data 'i' steps back so the model learns to look 'i' hours ahead
    df[col_name] = df['aqi'].shift(-i) 
    targets.append(col_name)

df = df.dropna()
print(f"✅ Training Data Ready: {len(df)} rows. Model will learn 72-step patterns.")

# 3. Define Input Features
# Added 'hour' to this list so the model sees time-of-day
features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity', 'hour']

X = df[features]       # Input: Current conditions + Time
y = df[targets]        # Output: Next 72 hours of AQI

# 4. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("🥊 Training Stronger 3-Day Forecaster (Gradient Boosting)...")

# 🚀 UPGRADE: Using a stronger configuration to catch spikes (156 AQI)
# n_estimators=200: Tries 200 times to correct its errors
# max_depth=5: Looks deeper into complex relationships (Temperature vs Traffic)
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)

model = MultiOutputRegressor(gb_model)
model.fit(X_train, y_train)

# 5. Evaluate & Save
print("🔍 Evaluating Model Performance...")
preds = model.predict(X_test)

# Calculate Metrics
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"🏆 Average Error (MAE): {round(mae, 2)}")
print(f"📊 Model Accuracy (R² Score): {round(r2, 4)}")

joblib.dump(model, "best_aqi_model.pkl")

# Register Model
mr = project.get_model_registry()
hopsworks_model = mr.python.create_model(
    name="aqi_hourly_predictor",
    metrics={"mae": mae, "r2_score": r2},
    description="Gradient Boosting 72-Hour Forecaster (Includes Hour Feature)"
)
hopsworks_model.save("best_aqi_model.pkl")

print("✅ Stronger Model Saved & Registered!")