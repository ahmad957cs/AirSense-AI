import os
from pathlib import Path

import pandas as pd
import hopsworks
from dotenv import load_dotenv


# ============================================================
# AirSense-AI
# Training Dataset Creation
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

FEATURE_VIEW_NAME = "pm25_forecasting_view"
FEATURE_VIEW_VERSION = 1

TRAINING_DATASET_VERSION = 1

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_X_PATH = OUTPUT_DIR / "training_features.csv"
OUTPUT_Y_PATH = OUTPUT_DIR / "training_target.csv"


# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------

load_dotenv(ENV_PATH)

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_CERT_FOLDER = os.getenv("HOPSWORKS_CERT_FOLDER")


if not HOPSWORKS_HOST:
    raise ValueError("HOPSWORKS_HOST not found in .env")

if not HOPSWORKS_PROJECT:
    raise ValueError("HOPSWORKS_PROJECT not found in .env")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("AIRSENSE-AI — TRAINING DATASET CREATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Connect to Hopsworks
    # --------------------------------------------------------

    print("\n[1/6] Connecting to Hopsworks...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        engine="python",
        cert_folder=HOPSWORKS_CERT_FOLDER,
    )

    print(f"Connected to project: {project.name}")

    fs = project.get_feature_store()

    print("Feature Store connected.")

    # --------------------------------------------------------
    # 2. Retrieve Feature View
    # --------------------------------------------------------

    print("\n[2/6] Retrieving Feature View...")

    feature_view = fs.get_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
    )

    if feature_view is None:
        raise RuntimeError(
            f"Feature View '{FEATURE_VIEW_NAME}' "
            f"version {FEATURE_VIEW_VERSION} was not found."
        )

    print(f"Feature View: {feature_view.name}")
    print(f"Version: {feature_view.version}")

    # --------------------------------------------------------
    # 3. Create / retrieve training dataset
    # --------------------------------------------------------

    print("\n[3/6] Getting training data...")

    X, y = feature_view.get_training_data(
        training_dataset_version=TRAINING_DATASET_VERSION
    )

    print(f"Raw feature rows: {len(X)}")
    print(f"Raw feature columns: {len(X.columns)}")
    print(f"Target rows: {len(y)}")

    # --------------------------------------------------------
    # 4. Ensure timestamp is available
    # --------------------------------------------------------

    print("\n[4/6] Preparing training data...")

    if "timestamp" in X.columns:
        X["timestamp"] = pd.to_datetime(
            X["timestamp"],
            utc=True
        )

    if isinstance(y, pd.DataFrame) and "timestamp" in y.columns:
        y["timestamp"] = pd.to_datetime(
            y["timestamp"],
            utc=True
        )

    # --------------------------------------------------------
    # 5. Remove incomplete rows
    # --------------------------------------------------------

    print("\n[5/6] Removing incomplete historical rows...")

    print("\nMissing values before cleaning:")

    missing_x = X.isna().sum()
    missing_x = missing_x[missing_x > 0]

    if len(missing_x) > 0:
        print(missing_x)
    else:
        print("No missing values in features.")

    if isinstance(y, pd.DataFrame):
        missing_y = y.isna().sum()
    else:
        missing_y = pd.Series(
            {
                y.name if y.name else "target": y.isna().sum()
            }
        )

    print("\nTarget missing values:")
    print(missing_y)

    # Combine X and y so the same rows are removed from both.
    combined = pd.concat(
        [
            X.reset_index(drop=True),
            y.reset_index(drop=True),
        ],
        axis=1,
    )

    rows_before = len(combined)

    combined = combined.dropna()

    rows_after = len(combined)

    rows_removed = rows_before - rows_after

    print("\nRows before cleaning:", rows_before)
    print("Rows after cleaning:", rows_after)
    print("Rows removed:", rows_removed)

    if rows_after == 0:
        raise RuntimeError(
            "No complete rows remain after removing missing values."
        )

    # --------------------------------------------------------
    # Separate features and target again
    # --------------------------------------------------------

    target_name = y.name

    if target_name is None:
        target_name = "pm25"

    if target_name not in combined.columns:
        raise RuntimeError(
            f"Target column '{target_name}' was not found "
            "after combining the datasets."
        )

    y_clean = combined[target_name].copy()

    X_clean = combined.drop(
        columns=[target_name]
    ).copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if "timestamp" in X_clean.columns:

        X_clean = X_clean.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        y_clean = y_clean.loc[
            X_clean.index
        ].reset_index(drop=True)

    else:
        print(
            "\nWARNING: timestamp is not present in X. "
            "Chronological sorting cannot be performed."
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\nFinal validation:")

    print("X shape:", X_clean.shape)
    print("y shape:", y_clean.shape)

    print(
        "Remaining feature NaNs:",
        X_clean.isna().sum().sum()
    )

    print(
        "Remaining target NaNs:",
        y_clean.isna().sum()
    )

    if "timestamp" in X_clean.columns:

        print(
            "First timestamp:",
            X_clean["timestamp"].min()
        )

        print(
            "Last timestamp:",
            X_clean["timestamp"].max()
        )

    if X_clean.isna().sum().sum() != 0:
        raise RuntimeError(
            "Feature dataset still contains NaN values."
        )

    if y_clean.isna().sum() != 0:
        raise RuntimeError(
            "Target dataset still contains NaN values."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[6/6] Saving training dataset...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_clean.to_csv(
        OUTPUT_X_PATH,
        index=False
    )

    y_clean.to_csv(
        OUTPUT_Y_PATH,
        index=False,
        header=True
    )

    print("\n" + "=" * 60)
    print("TRAINING DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Features: {OUTPUT_X_PATH}")
    print(f"Target:   {OUTPUT_Y_PATH}")

    print(f"\nTraining rows: {len(X_clean)}")
    print(f"Feature columns: {len(X_clean.columns)}")

    print("\nFirst 5 feature rows:")
    print(X_clean.head())

    print("\nFirst 5 target values:")
    print(y_clean.head())


if __name__ == "__main__":
    main()