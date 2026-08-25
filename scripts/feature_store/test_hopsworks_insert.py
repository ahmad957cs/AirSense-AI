import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("HOPSWORKS INSERT TEST")
print("=" * 60)

# Connect
project = hopsworks.login(
    host=os.getenv("HOPSWORKS_HOST"),
    project=os.getenv("HOPSWORKS_PROJECT"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    engine="python",
    cert_folder=os.getenv("HOPSWORKS_CERT_FOLDER"),
)

print("Connected:", project.name)

fs = project.get_feature_store()

# Get existing v3
fg = fs.get_feature_group(
    name="pm25_hourly_features",
    version=3,
)

print("Feature Group:", fg.name)
print("Version:", fg.version)

# Tiny test dataset using the actual schema
df = pd.read_csv("data/processed/pm25_features.csv")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

test_df = df.head(2).copy()

print("\nTest rows:")
print(test_df[["timestamp", "pm25"]])

print("\nAttempting insert...")

fg.insert(test_df)

print("\n" + "=" * 60)
print("TEST INSERT SUCCESSFUL")
print("=" * 60)