from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.aqi.aqi_calculator import (
    pm25_to_aqi,
)

from src.prediction.hopsworks_features import (
    get_latest_feature_window,
)
from src.prediction.model_loader import (
    ModelBundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_WINDOW = 72
FORECAST_HORIZON = 72


class ForecastEngine:

    def __init__(self) -> None:

        self.bundle = ModelBundle()
        self.bundle.load()

    # ========================================================
    # Load latest history
    # ========================================================

    def load_history(self) -> pd.DataFrame:
        """
        Load the latest forecasting features from Hopsworks.

        This is the production inference path. Local CSV files
        are not used for production feature retrieval.
        """
        df = get_latest_feature_window(
            hours=INPUT_WINDOW
        )

        required = [
            "timestamp",
            "pm25",
        ]

        required.extend(
            self.bundle.feature_columns
        )

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Required columns missing from Hopsworks Feature View:\n"
                + "\n".join(missing)
            )

        return (
            df.sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ========================================================
    # Validate latest 72 hours
    # ========================================================

    def _get_latest_window(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        window = df.tail(
            INPUT_WINDOW
        ).copy()

        if len(window) != INPUT_WINDOW:

            raise ValueError(
                "Not enough historical rows "
                "for a 72-hour input window."
            )

        timestamps = (
            pd.to_datetime(
                window["timestamp"]
            )
        )

        differences = (
            timestamps.diff()
            .dropna()
        )

        if not (
            differences
            == pd.Timedelta(hours=1)
        ).all():

            raise ValueError(
                "Latest 72-hour window is not "
                "hourly continuous."
            )

        return window

    # ========================================================
    # Prepare feature matrix
    # ========================================================

    def _prepare_features(
        self,
        window: pd.DataFrame,
    ) -> np.ndarray:

        X = window[
            self.bundle.feature_columns
        ].copy()

        # The same tabular imputer generated during
        # model preparation is used as a safety layer
        # for current inference data.

        X_imputed = (
            self.bundle.tabular_imputer
            .transform(X)
        )

        return np.asarray(
            X_imputed,
            dtype=np.float32,
        )

    # ========================================================
    # Random Forest prediction
    # ========================================================

    def _predict_rf(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        prediction = (
            self.bundle.rf_model
            .predict(
                X[-1:].reshape(
                    1,
                    -1,
                )
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        ).reshape(-1)

        if len(prediction) < FORECAST_HORIZON:

            raise ValueError(
                "Random Forest did not return "
                "72 predictions."
            )

        return prediction[
            :FORECAST_HORIZON
        ]

    # ========================================================
    # LSTM prediction
    # ========================================================

    def _predict_lstm(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        X_scaled = (
            self.bundle.feature_scaler
            .transform(
                pd.DataFrame(
                    X,
                    columns=self.bundle.feature_columns
                )
            )
        )

        X_sequence = (
            X_scaled
            .reshape(
                1,
                INPUT_WINDOW,
                len(
                    self.bundle.feature_columns
                ),
            )
        )

        prediction_scaled = (
            self.bundle.lstm_model.predict(
                X_sequence,
                verbose=0,
            )
        )

        prediction_scaled = (
            np.asarray(
                prediction_scaled
            )
            .reshape(1, -1)
        )

        prediction = (
            self.bundle.target_scaler
            .inverse_transform(
                prediction_scaled
            )
            .reshape(-1)
        )

        if len(prediction) < FORECAST_HORIZON:

            raise ValueError(
                "LSTM did not return "
                "72 predictions."
            )

        return prediction[
            :FORECAST_HORIZON
        ]

    # ========================================================
    # Create 24-hour forecast AQI
    # ========================================================

    def _add_aqi(
        self,
        result: pd.DataFrame,
        history_pm25: pd.Series,
    ) -> pd.DataFrame:

        actual_history = (
            history_pm25
            .astype(float)
            .tolist()
        )

        predicted_values = (
            result["pm25"]
            .astype(float)
            .tolist()
        )

        combined = (
            actual_history
            + predicted_values
        )

        aqi_values = []
        categories = []
        rolling_pm25 = []

        history_length = len(
            actual_history
        )

        for i in range(
            len(predicted_values)
        ):

            current_index = (
                history_length + i
            )

            start = max(
                0,
                current_index - 23,
            )

            window = combined[
                start : current_index + 1
            ]

            mean_24h = (
                float(
                    np.mean(window)
                )
            )

            aqi, category = (
                pm25_to_aqi(
                    mean_24h
                )
            )

            rolling_pm25.append(
                mean_24h
            )

            aqi_values.append(
                aqi
            )

            categories.append(
                category
            )

        result[
            "pm25_24h_average"
        ] = rolling_pm25

        result["aqi"] = aqi_values

        result["aqi_category"] = categories

        return result

    # ========================================================
    # Main prediction
    # ========================================================

    def forecast(self) -> pd.DataFrame:

        df = self.load_history()

        window = (
            self._get_latest_window(
                df
            )
        )

        X = self._prepare_features(
            window
        )

        rf_prediction = (
            self._predict_rf(X)
        )

        lstm_prediction = (
            self._predict_lstm(X)
        )

        current_time = pd.to_datetime(
            window["timestamp"].iloc[-1]
        )

        future_timestamps = pd.date_range(
            start=(
                current_time
                + pd.Timedelta(hours=1)
            ),
            periods=FORECAST_HORIZON,
            freq="h",
        )

        rows = []

        for i in range(
            FORECAST_HORIZON
        ):

            hour = i + 1

            model_key = (
                f"hour_{hour:02d}"
            )

            selected_model = (
                self.bundle.selection_map
                .get(model_key)
            )

            if selected_model == "Random Forest":

                prediction = (
                    rf_prediction[i]
                )

                model_used = (
                    "Random Forest"
                )

            elif selected_model == "LSTM":

                prediction = (
                    lstm_prediction[i]
                )

                model_used = "LSTM"

            else:

                raise ValueError(
                    f"No model selection for "
                    f"{model_key}"
                )

            rows.append(
                {
                    "timestamp":
                        future_timestamps[i],
                    "hour_ahead":
                        hour,
                    "day":
                        (
                            "Day 1"
                            if hour <= 24
                            else
                            "Day 2"
                            if hour <= 48
                            else
                            "Day 3"
                        ),
                    "model_used":
                        model_used,
                    "pm25":
                        max(
                            0.0,
                            float(
                                prediction
                            ),
                        ),
                }
            )

        result = pd.DataFrame(
            rows
        )

        result = self._add_aqi(
            result,
            df["pm25"].tail(23),
        )

        return result