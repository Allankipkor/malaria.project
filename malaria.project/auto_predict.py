"""
Automatic prediction script for the malaria outbreak model.

This script:
- Loads the trained sklearn Pipeline bundle saved by `malaria_outbreak_prediction.py`
  from `outputs/malaria_outbreak_random_forest.joblib`.
- Automatically reads NEW input data from a CSV file.
- Aligns the input DataFrame columns with `feature_names` from training.
- Calls `pipeline.predict()` directly (no manual scaler usage).
- Prints predicted outbreak risk (0 = no outbreak, 1 = outbreak) for each row.

You can point `NEW_DATA_CSV` to:
- a new malaria cases CSV with the same feature columns as training, or
- a CSV of processed climate + malaria features.

Beginner-friendly usage:
    1. Run the training script once:
           python malaria_outbreak_prediction.py
    2. Prepare a new CSV with the same feature columns used in training.
    3. Update `NEW_DATA_CSV` below if needed.
    4. Run:
           python auto_predict.py
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import pandas as pd
import joblib


BUNDLE_PATH = os.path.join("outputs", "malaria_outbreak_random_forest.joblib")
# Change this to the CSV that contains new samples for prediction:
NEW_DATA_CSV = "new_malaria_features.csv"


def load_bundle(path: str) -> Dict[str, Any]:
    """Load the saved bundle: {'pipeline': ..., 'feature_names': [...]}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'.\n"
            f"Train the model first by running malaria_outbreak_prediction.py."
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


def run_prediction() -> None:
    """
    AUTOMATIC PREDICTION PIPELINE:
    - Load trained Pipeline bundle.
    - Load and align new input data.
    - Call `pipeline.predict()` on the new data.
    - Print predicted outbreak risk for each row.
    """
    bundle = load_bundle(BUNDLE_PATH)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    print("[P] Loaded trained Pipeline bundle from:", BUNDLE_PATH)
    print("[P] Feature names used during training:")
    print(feature_names, "\n")

    df_new = load_new_data(NEW_DATA_CSV, feature_names)

    # IMPORTANT: We do NOT call any scaler/encoder manually here.
    # All preprocessing is encapsulated in the Pipeline, so we only call:
    predictions = pipeline.predict(df_new)

    # Print predictions row by row
    print("[P] Predicted outbreak risk for each new record (0 = no outbreak, 1 = outbreak):")
    for idx, pred in enumerate(predictions):
        print(f"    Row {idx}: Predicted outbreak risk: {int(pred)}")


if __name__ == "__main__":
    run_prediction()

