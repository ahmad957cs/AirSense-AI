from pathlib import Path

import pandas as pd


# ============================================================
# AirSense-AI — Final Model Selection Report
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "model_selection"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {
    "Random Forest": (
        METRICS_DIR / "random_forest_validation_metrics.csv"
    ),
    "XGBoost": (
        METRICS_DIR / "xgboost_validation_metrics.csv"
    ),
    "LSTM": (
        METRICS_DIR / "lstm_validation_metrics.csv"
    ),
}


def load_metrics(model_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} validation metrics not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "hour_ahead",
        "day",
        "MAE",
        "RMSE",
        "R2",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} metrics missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["model"] = model_name

    return df


def main() -> None:

    print("=" * 60)
    print("AIRSENSE-AI — FINAL MODEL SELECTION")
    print("=" * 60)

    frames = []

    for model_name, path in FILES.items():

        try:
            frame = load_metrics(
                model_name,
                path,
            )

            print(
                f"{model_name}: "
                f"{len(frame)} hourly metric rows loaded."
            )

            frames.append(frame)

        except FileNotFoundError as exc:
            print(f"\nWARNING: {exc}")

    if len(frames) < 2:
        raise RuntimeError(
            "At least two model validation metric files "
            "are required for model comparison."
        )

    all_metrics = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Overall validation leaderboard
    # --------------------------------------------------------

    overall = (
        all_metrics
        .groupby("model")[
            ["MAE", "RMSE", "R2"]
        ]
        .mean()
        .sort_values("RMSE")
    )

    print("\n" + "=" * 60)
    print("OVERALL VALIDATION LEADERBOARD")
    print("=" * 60)

    print(
        overall.to_string()
    )

    # --------------------------------------------------------
    # Day-level leaderboard
    # --------------------------------------------------------

    daily = (
        all_metrics
        .groupby(
            ["model", "day"]
        )[
            ["MAE", "RMSE", "R2"]
        ]
        .mean()
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("DAY-LEVEL VALIDATION")
    print("=" * 60)

    print(
        daily.to_string(index=False)
    )

    # --------------------------------------------------------
    # Best model for every hour
    # --------------------------------------------------------

    winners = []

    for hour in sorted(
        all_metrics["hour_ahead"].unique()
    ):

        hour_metrics = all_metrics[
            all_metrics["hour_ahead"] == hour
        ].copy()

        if hour_metrics.empty:
            continue

        best = hour_metrics.loc[
            hour_metrics["RMSE"].idxmin()
        ]

        winners.append(
            {
                "hour_ahead": int(hour),
                "day": best["day"],
                "selected_model": best["model"],
                "validation_RMSE": float(best["RMSE"]),
                "validation_MAE": float(best["MAE"]),
                "validation_R2": float(best["R2"]),
            }
        )

    selection_map = pd.DataFrame(winners)

    print("\n" + "=" * 60)
    print("MODEL SELECTION BY FORECAST HOUR")
    print("=" * 60)

    print(
        selection_map.to_string(index=False)
    )

    # --------------------------------------------------------
    # Summary by selected model
    # --------------------------------------------------------

    selected_counts = (
        selection_map["selected_model"]
        .value_counts()
    )

    print("\n" + "=" * 60)
    print("SELECTION COUNTS")
    print("=" * 60)

    print(
        selected_counts.to_string()
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    all_metrics.to_csv(
        OUTPUT_DIR / "all_validation_metrics.csv",
        index=False,
    )

    overall.to_csv(
        OUTPUT_DIR / "overall_validation_leaderboard.csv"
    )

    daily.to_csv(
        OUTPUT_DIR / "daily_validation_summary.csv",
        index=False,
    )

    selection_map.to_csv(
        OUTPUT_DIR / "hourly_model_selection.csv",
        index=False,
    )

    # --------------------------------------------------------
    # JSON model-selection map
    # --------------------------------------------------------

    model_map = {
        f"hour_{int(row.hour_ahead):02d}": row.selected_model
        for row in selection_map.itertuples()
    }

    import json

    with (
        OUTPUT_DIR / "model_selection.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            model_map,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("MODEL SELECTION ARTIFACTS CREATED")
    print("=" * 60)

    print(
        OUTPUT_DIR / "all_validation_metrics.csv"
    )

    print(
        OUTPUT_DIR / "overall_validation_leaderboard.csv"
    )

    print(
        OUTPUT_DIR / "daily_validation_summary.csv"
    )

    print(
        OUTPUT_DIR / "hourly_model_selection.csv"
    )

    print(
        OUTPUT_DIR / "model_selection.json"
    )


if __name__ == "__main__":
    main()