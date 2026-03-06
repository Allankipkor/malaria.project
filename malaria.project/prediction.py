"""
Simple prediction script for the trained Random Forest malaria outbreak model.

This script:
- Loads the saved model bundle from `outputs/malaria_outbreak_random_forest.joblib`
- Builds a single input sample as a pandas DataFrame
- Applies the same scaling as used during training
- Runs `model.predict` and prints the predicted outbreak class (0 = no outbreak, 1 = outbreak)

NOTE:
- The feature names and value ranges must match the features used during training.
- After you train the model with `malaria_outbreak_prediction.py`, the bundle will
  contain: `model`, `scaler`, and `feature_names`.
"""

import joblib
import pandas as pd


def main() -> None:
    # Load the trained Random Forest bundle (model + scaler + feature names)
    bundle_path = "outputs/malaria_outbreak_random_forest.joblib"
    model_bundle = joblib.load(bundle_path)

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    print("Model bundle loaded from:", bundle_path)
    print("Features expected by the model:", feature_names)

    sample = {
    "temperature_c": 27.5,
    "rainfall_mm": 120.0,
    "humidity_pct": 78.0,
    "population_density": 350.0,
    "region": 1,
    "altitude_m": 450.0,
    "cases_last_month": 22
 # if encoded as integer during training
    }

    if not sample:
        raise ValueError(
            "The 'sample' dictionary is empty.\n"
            "Edit prediction.py and fill 'sample' with keys matching 'feature_names' "
            "and appropriate numeric values."
        )

    # Create DataFrame and ensure columns are in the same order as during training
    new_data = pd.DataFrame([sample])
    # Reorder / subset columns to match training feature order
    new_data = new_data[feature_names]

    # Apply the same scaling and remove feature names to avoid sklearn warning
    new_data_scaled = scaler.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_scaled)[0]

    print("Predicted outbreak class (0 = no outbreak, 1 = outbreak):", prediction)


if __name__ == "__main__":
    main()