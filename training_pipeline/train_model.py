import hopsworks
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
from dotenv import load_dotenv

# 1. Connect to Hopsworks
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# 2. Retrieve NEW Hourly Data
print("📊 Fetching Hourly Data from Feature Store...")
try:
    aqi_fg = fs.get_feature_group(name="aqi_features_hourly", version=1)
    df = aqi_fg.select_all().read()
except:
    print("⚠️ Could not find 'aqi_features_hourly'. Did you run the backfill?")
    exit()

# 3. Prepare Data
df = df.sort_values(by="timestamp")

# Create Targets: Predict Next 3 HOURS
# (We shift -1, -2, -3 to predict h+1, h+2, h+3)
df['target_h1'] = df['aqi'].shift(-1)
df['target_h2'] = df['aqi'].shift(-2)
df['target_h3'] = df['aqi'].shift(-3)
df = df.dropna()

print(f"✅ Training Data Created: {len(df)} rows")

features = ['aqi', 'pm25', 'pm10', 'temp', 'humidity']
targets = ['target_h1', 'target_h2', 'target_h3']

X = df[features]
y = df[targets]

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Define the 3 Models to Fight
models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
    "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100)
}

best_mae = float("inf")
best_model = None
best_name = ""

print(f"\n🥊 Starting Model Battle...")

for name, model in models.items():
    # Wrap in MultiOutput for 3-step prediction
    wrapped_model = MultiOutputRegressor(model)
    wrapped_model.fit(X_train, y_train)
    
    preds = wrapped_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"   👉 {name}: MAE = {round(mae, 2)}")
    
    if mae < best_mae:
        best_mae = mae
        best_model = wrapped_model
        best_name = name

print(f"\n🏆 WINNER: {best_name} (Error: {round(best_mae, 2)})")

# 5. Save the Winner
joblib.dump(best_model, "best_aqi_model.pkl")

# Register Model
print("💾 Saving best model to Registry...")
mr = project.get_model_registry()
hopsworks_model = mr.python.create_model(
    name="aqi_hourly_predictor",
    metrics={"mae": best_mae},
    description=f"Best model ({best_name}) for 3-Hour Prediction"
)
hopsworks_model.save("best_aqi_model.pkl")

print("✅ Model Training Complete!")