"""
Simple prediction script for the trained Random Forest malaria outbreak model.

How it works:
1. Loads `outputs/malaria_outbreak_random_forest.joblib` (pipeline + feature_names from training).
2. Builds one input row using the same columns as training: last row of `malaria_cases.csv`,
   optionally merged with `outputs/climate_avgua_by_year_m49.csv` if that file exists
   (same logic as the notebook / auto_predict.py).
3. Reorders columns to match `feature_names` and calls `pipeline.predict()`.
4. Prints the predicted class: 0 = no outbreak, 1 = outbreak.

Run from anywhere; paths are resolved relative to this script's folder.

If you see no "Predicted outbreak class" line, the script usually exited earlier with an error
(e.g. missing bundle, wrong working directory was a problem before we fixed paths, or column mismatch).
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

MALARIA_CSV = "malaria_cases.csv"
CLIMATE_MERGE_CSV = os.path.join("outputs", "climate_avgua_by_year_m49.csv")


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    root = _project_root()
    bundle_path = os.path.join(root, "outputs", "malaria_outbreak_random_forest.joblib")

    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(
            f"Model bundle not found:\n  {bundle_path}\n"
            "Train the model first (e.g. run notebook.py or the training notebook), "
            "then run this script again."
        )

    model_bundle = joblib.load(bundle_path)
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]

    print("Model bundle loaded from:", bundle_path)
    print("Features expected by the model:", feature_names)

    malaria_path = os.path.join(root, MALARIA_CSV)
    if not os.path.isfile(malaria_path):
        raise FileNotFoundError(
            f"Need '{MALARIA_CSV}' next to this script to build a demo row:\n  {malaria_path}"
        )

    df = pd.read_csv(malaria_path)
    climate_path = os.path.join(root, CLIMATE_MERGE_CSV)
    if os.path.isfile(climate_path):
        cl = pd.read_csv(climate_path)
        df = df.merge(
            cl,
            left_on=["DIM_TIME", "DIM_GEO_CODE_M49"],
            right_on=["year", "m49"],
            how="left",
        )
        df = df.drop(columns=["year", "m49"], errors="ignore")

    new_data = df.tail(1)
    missing = [c for c in feature_names if c not in new_data.columns]
    if missing:
        raise ValueError(
            "Demo row is missing columns the trained model expects:\n"
            f"  {missing}\n"
            f"Ensure climate merge output exists ({CLIMATE_MERGE_CSV}) if training used it, "
            "or retrain after aligning data with notebook."
        )

    new_data = new_data[feature_names]

    prediction = pipeline.predict(new_data)[0]
    print("Predicted outbreak class (0 = no outbreak, 1 = outbreak):", int(prediction))


if __name__ == "__main__":
    main()
