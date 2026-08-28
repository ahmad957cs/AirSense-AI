from __future__ import annotations

from typing import Any


# ============================================================
# AirSense-AI — AQI Health Alerts
# ============================================================


def get_health_alert(aqi: float) -> dict[str, Any]:
    """
    Generate a health-oriented alert from an AQI value.

    Uses the same AQI categories already defined by
    src.aqi.aqi_calculator.classify_aqi().
    """

    if aqi <= 50:
        return {
            "level": "normal",
            "category": "Good",
            "title": "Air quality is good",
            "message": (
                "Air quality is considered satisfactory. "
                "No special precautions are needed."
            ),
            "action": (
                "Normal outdoor activities are appropriate."
            ),
        }

    if aqi <= 100:
        return {
            "level": "advisory",
            "category": "Moderate",
            "title": "Air quality is acceptable",
            "message": (
                "Air quality is generally acceptable, "
                "although unusually sensitive people "
                "may experience mild effects."
            ),
            "action": (
                "Sensitive individuals may consider "
                "reducing prolonged or heavy outdoor exertion."
            ),
        }

    if aqi <= 150:
        return {
            "level": "warning",
            "category": "Unhealthy for Sensitive Groups",
            "title": "Sensitive groups should take care",
            "message": (
                "Members of sensitive groups may experience "
                "health effects."
            ),
            "action": (
                "Sensitive individuals should reduce "
                "prolonged or heavy outdoor exertion."
            ),
        }

    if aqi <= 200:
        return {
            "level": "warning",
            "category": "Unhealthy",
            "title": "Unhealthy air quality",
            "message": (
                "Everyone may begin to experience health effects, "
                "with sensitive groups potentially experiencing "
                "more serious effects."
            ),
            "action": (
                "Consider reducing prolonged or heavy outdoor "
                "exertion and limit exposure where possible."
            ),
        }

    if aqi <= 300:
        return {
            "level": "critical",
            "category": "Very Unhealthy",
            "title": "Very unhealthy air quality",
            "message": (
                "Health alert: the risk of health effects "
                "is increased for everyone."
            ),
            "action": (
                "Avoid prolonged or heavy outdoor activity "
                "and reduce exposure."
            ),
        }

    return {
        "level": "hazardous",
        "category": "Hazardous",
        "title": "Hazardous air quality",
        "message": (
            "Health emergency conditions may exist. "
            "Everyone is more likely to be affected."
        ),
        "action": (
            "Avoid outdoor exposure and follow local "
            "health guidance."
        ),
    }


def get_forecast_alert(
    forecast_df,
) -> dict[str, Any]:
    """
    Evaluate the highest predicted AQI in a forecast.

    Returns the alert corresponding to the peak predicted AQI.
    """

    if forecast_df.empty:
        raise ValueError(
            "Forecast dataframe is empty."
        )

    if "aqi" not in forecast_df.columns:
        raise ValueError(
            "Forecast dataframe must contain an 'aqi' column."
        )

    peak_index = (
        forecast_df["aqi"]
        .astype(float)
        .idxmax()
    )

    peak_aqi = float(
        forecast_df.loc[peak_index, "aqi"]
    )

    alert = get_health_alert(
        peak_aqi
    )

    alert["peak_aqi"] = int(
        round(peak_aqi)
    )

    if "timestamp" in forecast_df.columns:
        alert["peak_timestamp"] = str(
            forecast_df.loc[
                peak_index,
                "timestamp",
            ]
        )

    return alert