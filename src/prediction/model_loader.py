from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PREPROCESSING_DIR = (
    ARTIFACTS_DIR / "preprocessing"
)

MODEL_SELECTION_PATH = (
    ARTIFACTS_DIR
    / "model_selection"
    / "final_model_selection.json"
)


FEATURE_COLUMNS_PATH = (
    PREPROCESSING_DIR
    / "feature_columns.json"
)

FEATURE_SCALER_PATH = (
    PREPROCESSING_DIR
    / "feature_scaler.pkl"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR
    / "target_scaler.pkl"
)

TABULAR_IMPUTER_PATH = (
    PREPROCESSING_DIR
    / "tabular_imputer.joblib"
)


RF_MODEL_PATH = (
    MODELS_DIR
    / "random_forest"
    / "random_forest_72h.joblib"
)

LSTM_MODEL_PATH = (
    MODELS_DIR
    / "lstm"
    / "airsense_lstm_72h_best.keras"
)


class ModelBundle:
    """
    Loads all artifacts required by the production
    forecasting engine.
    """

    def __init__(self) -> None:

        self.rf_model: Any = None
        self.lstm_model: Any = None

        self.feature_scaler: Any = None
        self.target_scaler: Any = None
        self.tabular_imputer: Any = None

        self.feature_columns: list[str] = []
        self.selection_map: dict[str, str] = {}

    def load(self) -> None:

        self._check_required_files()

        print("Loading Random Forest...")

        self.rf_model = joblib.load(
            RF_MODEL_PATH
        )

        print("Loading LSTM...")

        self.lstm_model = (
            tf.keras.models.load_model(
                LSTM_MODEL_PATH
            )
        )

        print("Loading preprocessing artifacts...")

        with FEATURE_SCALER_PATH.open(
            "rb"
        ) as f:
            self.feature_scaler = (
                pickle.load(f)
            )

        with TARGET_SCALER_PATH.open(
            "rb"
        ) as f:
            self.target_scaler = (
                pickle.load(f)
            )

        self.tabular_imputer = (
            joblib.load(
                TABULAR_IMPUTER_PATH
            )
        )

        with FEATURE_COLUMNS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            raw_feature_columns = json.load(f)

        # --------------------------------------------------------
        # Normalize feature_columns.json
        #
        # The saved artifact may be:
        #   1. a plain list
        #   2. a dict containing feature_columns/features/columns
        #   3. a dict mapping indices -> feature names
        # --------------------------------------------------------
        if isinstance(raw_feature_columns, list):
            self.feature_columns = raw_feature_columns
        elif isinstance(raw_feature_columns, dict):
            # Common structured format
            for key in (
                "feature_columns",
                "features",
                "columns",
            ):
                if (
                    key in raw_feature_columns
                    and isinstance(
                        raw_feature_columns[key],
                        list,
                    )
                ):
                    self.feature_columns = (
                        raw_feature_columns[key]
                    )
                    break
            else:
                # Handle mappings such as:
                # {"0": "coverage_percent", ...}
                try:
                    ordered_items = sorted(
                        raw_feature_columns.items(),
                        key=lambda item: int(item[0]),
                    )

                    values = [
                        value
                        for _, value in ordered_items
                    ]
                    if all(
                        isinstance(value, str)
                        for value in values
                    ):
                        self.feature_columns = values
                    else:
                        raise ValueError
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        "Unsupported feature_columns.json format. "
                        "Expected a list, a dictionary containing "
                        "'feature_columns'/'features'/'columns', "
                        "or an index-to-feature-name mapping."
                    ) from exc
        else:
            raise TypeError(
                "feature_columns.json must contain "
                "a list or dictionary."
            )

        # --------------------------------------------------------
        # Final validation
        # --------------------------------------------------------
        if not isinstance(
            self.feature_columns,
            list,
        ):
            raise TypeError(
                "feature_columns must be a list."
            )
        self.feature_columns = [
            str(column)
            for column in self.feature_columns
        ]
        print(
            "Feature columns loaded:",
            len(self.feature_columns),
        )
        if len(self.feature_columns) != 43:
            raise ValueError(
                f"Expected 43 model features, "
                f"but found {len(self.feature_columns)}."
            )

        with MODEL_SELECTION_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.selection_map = json.load(f)

        print("Models and preprocessing loaded.")

    def _check_required_files(self) -> None:

        required = [
            RF_MODEL_PATH,
            LSTM_MODEL_PATH,
            FEATURE_COLUMNS_PATH,
            FEATURE_SCALER_PATH,
            TARGET_SCALER_PATH,
            TABULAR_IMPUTER_PATH,
            MODEL_SELECTION_PATH,
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:

            raise FileNotFoundError(
                "Required production artifacts are missing:\n"
                + "\n".join(missing)
            )