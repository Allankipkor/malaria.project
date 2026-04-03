"""
Automatic prediction script for the malaria outbreak model.

This script:
- Loads the trained sklearn Pipeline bundle saved by training (`notebook.py` / `notebook.ipynb`)
  as `outputs/malaria_outbreak_random_forest.joblib`.
- Automatically reads NEW input data from a CSV file.
- Aligns the input DataFrame columns with `feature_names` from training.
- Calls `pipeline.predict()` directly (no manual scaler usage).
- Prints predicted outbreak risk (0 = no outbreak, 1 = outbreak) for each row.

You can point `NEW_DATA_CSV` to:
- a new malaria cases CSV with the same feature columns as training, or
- a CSV of processed climate + malaria features.

Beginner-friendly usage:
    1. Train once: `python notebook.py` (from this folder).
    2. Either create `new_malaria_features.csv` with the same columns as training, or run
       `python auto_predict.py` with no file — it will use a short demo sample from
       `malaria_cases.csv` (+ climate merge if available).
    3. Or pass an explicit path: `python auto_predict.py path/to/your.csv`
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import joblib
import pandas as pd


BUNDLE_PATH = os.path.join("outputs", "malaria_outbreak_random_forest.joblib")
MALARIA_CSV = "malaria_cases.csv"
CLIMATE_MERGE_CSV = os.path.join("outputs", "climate_avgua_by_year_m49.csv")
# Default filename if you prepare your own rows for prediction:
NEW_DATA_CSV = "new_malaria_features.csv"
DEMO_TAIL_ROWS = 5


def load_bundle(path: str) -> Dict[str, Any]:
    """Load the saved bundle: {'pipeline': ..., 'feature_names': [...]}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'.\n"
            "Train the model first by running notebook.py from this folder."
        )
    bundle = joblib.load(path)
    if "pipeline" not in bundle or "feature_names" not in bundle:
        raise ValueError(
            "Loaded bundle does not contain the expected keys "
            "('pipeline', 'feature_names')."
        )
    return bundle


def load_new_data(csv_path: str, feature_names: List[str]) -> pd.DataFrame:
    """
    Load new data for prediction and align it with the training feature names.

    - Reads the CSV into a DataFrame.
    - Ensures all required feature columns are present.
    - If there are extra columns, they are ignored.
    - Feature order is exactly `feature_names`, which avoids sklearn warnings.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"New data CSV not found at '{csv_path}'. "
            f"Create it with columns: {feature_names}"
        )

    df_new = pd.read_csv(csv_path)
    print(f"[P] Loaded new data from '{csv_path}': {df_new.shape[0]} rows, {df_new.shape[1]} columns.")

    missing = [col for col in feature_names if col not in df_new.columns]
    if missing:
        raise ValueError(
            "The following required feature columns are missing in the new data CSV: "
            f"{missing}\nExpected columns (from training): {feature_names}"
        )

    # Keep only the columns used during training and in the same order
    df_new = df_new[feature_names]
    print("[P] New data columns aligned to training feature_names.\n")
    return df_new


def demo_feature_frame(feature_names: List[str], n: int = DEMO_TAIL_ROWS) -> pd.DataFrame:
    """
    Build a sample feature matrix like training: malaria_cases + optional climate merge,
    then take the last ``n`` rows (so you can run auto_predict without new_malaria_features.csv).
    """
    if not os.path.isfile(MALARIA_CSV):
        raise FileNotFoundError(
            f"Demo mode needs '{MALARIA_CSV}' in the working directory "
            f"(current: {os.getcwd()})."
        )
    df = pd.read_csv(MALARIA_CSV)
    if os.path.isfile(CLIMATE_MERGE_CSV):
        cl = pd.read_csv(CLIMATE_MERGE_CSV)
        df = df.merge(
            cl,
            left_on=["DIM_TIME", "DIM_GEO_CODE_M49"],
            right_on=["year", "m49"],
            how="left",
        )
        df = df.drop(columns=["year", "m49"], errors="ignore")
    df = df.tail(n)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            "Demo frame is missing columns expected by the trained model: "
            f"{missing}\n"
            f"Run extract_climate_features.py so '{CLIMATE_MERGE_CSV}' exists, "
            "or pass a full CSV via: python auto_predict.py your_file.csv"
        )
    return df[feature_names]


def run_prediction() -> None:
    """
    AUTOMATIC PREDICTION PIPELINE:
    - Load trained Pipeline bundle.
    - Load and align new input data (or use a built-in demo sample).
    - Call `pipeline.predict()` on the new data.
    - Print predicted outbreak risk for each row.
    """
    parser = argparse.ArgumentParser(
        description="Predict malaria outbreak risk using the saved Random Forest pipeline."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help=f"CSV with the same feature columns as training (optional; default tries {NEW_DATA_CSV})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=f"Always use the last {DEMO_TAIL_ROWS} rows of {MALARIA_CSV} (+ climate merge).",
    )
    args = parser.parse_args()

    bundle = load_bundle(BUNDLE_PATH)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    print("[P] Loaded trained Pipeline bundle from:", BUNDLE_PATH)
    print("[P] Feature names used during training:")
    print(feature_names, "\n")

    if args.demo:
        print(
            f"[P] --demo: predicting on last {DEMO_TAIL_ROWS} rows of "
            f"{MALARIA_CSV} (with climate merge if available).\n"
        )
        df_new = demo_feature_frame(feature_names)
    elif args.input_csv is not None:
        df_new = load_new_data(args.input_csv, feature_names)
    elif os.path.isfile(NEW_DATA_CSV):
        df_new = load_new_data(NEW_DATA_CSV, feature_names)
    else:
        print(
            f"[P] No '{NEW_DATA_CSV}' and no input path given - "
            f"using demo: last {DEMO_TAIL_ROWS} rows of {MALARIA_CSV}.\n"
        )
        df_new = demo_feature_frame(feature_names)

    # IMPORTANT: We do NOT call any scaler/encoder manually here.
    # All preprocessing is encapsulated in the Pipeline, so we only call:
    predictions = pipeline.predict(df_new)

    # Print predictions row by row
    print("[P] Predicted outbreak risk for each new record (0 = no outbreak, 1 = outbreak):")
    for idx, pred in enumerate(predictions):
        print(f"    Row {idx}: Predicted outbreak risk: {int(pred)}")


if __name__ == "__main__":
    run_prediction()

