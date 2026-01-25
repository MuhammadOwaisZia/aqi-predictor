import hopsworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# 1. Connect to Hopsworks
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# 2. Retrieve the Feature Group
aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

# 3. Get the Data
query = aqi_fg.select_all()
df = query.read()

# 4. Sort and Format
df = df.sort_values(by="timestamp")
df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

print(f"✅ Data Retrieved. Rows: {len(df)}")

# 5. Plot the Daily Trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="aqi", marker="o", color="orange", label="Daily AQI")
plt.title(f"AQI History (4 Months) - {df['city'].iloc[0]}")
plt.xlabel("Date")
plt.ylabel("AQI Value")
plt.grid(True)
plt.xticks(rotation=45)

# Save the image
plt.tight_layout()
plt.savefig("aqi_trend.png")
print("📊 Graph saved as 'aqi_trend.png'. Check your folder!")