from __future__ import annotations

import os

import hopsworks
from dotenv import load_dotenv


load_dotenv()

project = hopsworks.login(
    host=os.environ["HOPSWORKS_HOST"],
    project=os.environ["HOPSWORKS_PROJECT"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    engine="python",
)

print("Connected project:", project.name)

fs = project.get_feature_store()

feature_view = fs.get_feature_view(
    name="pm25_forecasting_view",
    version=1,
)

print("Feature View:", feature_view.name)
print("Version:", feature_view.version)

print("Recreating training dataset version 1...")

job = feature_view.recreate_training_dataset(
    training_dataset_version=1
)
print("Training dataset recreation job completed successfully.")