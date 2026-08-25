from pathlib import Path

from src.prediction.forecast_engine import (
    ForecastEngine,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 70)
print(
    "AIRSENSE-AI — 72-HOUR PRODUCTION FORECAST TEST"
)
print("=" * 70)


engine = ForecastEngine()

forecast = engine.forecast()


print("\nForecast generated successfully.")

print("\nShape:")
print(forecast.shape)

print("\nModel selection:")
print(
    forecast[
        "model_used"
    ].value_counts()
)

print("\nFirst 10 hours:")
print(
    forecast.head(10).to_string(
        index=False
    )
)

print("\nLast 10 hours:")
print(
    forecast.tail(10).to_string(
        index=False
    )
)

print("\nAQI categories:")
print(
    forecast[
        "aqi_category"
    ].value_counts()
)

print("\nFull forecast:")
print(
    forecast.to_string(
        index=False
    )
)


output_path = (
    OUTPUT_DIR
    / "latest_72h_forecast.csv"
)

forecast.to_csv(
    output_path,
    index=False,
)

print("\nSaved:")
print(output_path)

print("\n" + "=" * 70)
print("✅ FORECAST ENGINE TEST COMPLETE")
print("=" * 70)