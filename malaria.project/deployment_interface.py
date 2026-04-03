"""
Streamlit deployment interface for the trained malaria outbreak model.

What it does:
1) Loads `outputs/malaria_outbreak_random_forest.joblib` (pipeline + feature_names)
2) Lets you upload a CSV with the SAME raw feature columns as training
3) Aligns columns to `feature_names` and runs `pipeline.predict()`
4) Displays predicted class (0 = no outbreak, 1 = outbreak)

Run:
  streamlit run deployment_interface.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import pandas as pd
import streamlit as st


def _root_dir() -> str:
    # Resolve paths relative to this file, so it works regardless of where you run from.
    return os.path.dirname(os.path.abspath(__file__))


def load_bundle(bundle_path: str) -> Dict[str, Any]:
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(
            f"Model bundle not found at: {bundle_path}\n"
            "Train the model first, then re-run this UI."
        )
    bundle = joblib.load(bundle_path)
    if "pipeline" not in bundle or "feature_names" not in bundle:
        raise ValueError("Bundle must contain keys: 'pipeline' and 'feature_names'.")
    return bundle


def align_to_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns for prediction:\n"
            f"{missing}\n\n"
            "Use a CSV that contains all training feature columns, or export from your preprocessing."
        )
    return df[feature_names]


@st.cache_resource(show_spinner=False)
def get_bundle_cached() -> Dict[str, Any]:
    bundle_path = os.path.join(_root_dir(), "outputs", "malaria_outbreak_random_forest.joblib")
    return load_bundle(bundle_path)


def demo_dataframe(feature_names: List[str], tail_rows: int = 5) -> pd.DataFrame:
    """
    Build a demo row set using the same approach as auto_predict.py:
    - Read `malaria_cases.csv`
    - If climate merge file exists, merge it on (DIM_TIME, DIM_GEO_CODE_M49) ~ (year, m49)
    - Take the last `tail_rows`
    - Align columns to `feature_names`
    """
    root = _root_dir()
    malaria_csv = os.path.join(root, "malaria_cases.csv")
    climate_merge_csv = os.path.join(root, "outputs", "climate_avgua_by_year_m49.csv")

    if not os.path.isfile(malaria_csv):
        raise FileNotFoundError(f"Demo needs '{malaria_csv}'.")

    df = pd.read_csv(malaria_csv)
    if os.path.isfile(climate_merge_csv):
        cl = pd.read_csv(climate_merge_csv)
        df = df.merge(
            cl,
            left_on=["DIM_TIME", "DIM_GEO_CODE_M49"],
            right_on=["year", "m49"],
            how="left",
        )
        df = df.drop(columns=["year", "m49"], errors="ignore")

    df = df.tail(tail_rows)
    return align_to_features(df, feature_names)


def main() -> None:
    st.set_page_config(page_title="Malaria Outbreak Predictor", layout="wide")
    st.title("Malaria Outbreak Predictor (Random Forest)")
    st.write("This app uses your saved scikit-learn Pipeline to produce predictions (0/1).")

    bundle = get_bundle_cached()
    pipeline = bundle["pipeline"]
    feature_names: List[str] = bundle["feature_names"]

    with st.expander("Model expects these feature columns"):
        st.write(feature_names)
        st.caption(f"Total features: {len(feature_names)}")

    st.subheader("Input data")
    uploaded = st.file_uploader("Upload a CSV with the SAME feature columns used in training", type=["csv"])

    colA, colB = st.columns(2)
    with colB:
        use_demo = st.button("Run demo using last 5 rows of malaria_cases.csv")

    if uploaded is None and not use_demo:
        st.info("Upload a CSV or click 'Run demo' to see predictions.")
        return

    try:
        if use_demo:
            df_new = demo_dataframe(feature_names, tail_rows=5)
            st.success(f"Demo loaded: {df_new.shape[0]} rows")
        else:
            df_new = pd.read_csv(uploaded)
            st.write(f"Uploaded CSV shape: {df_new.shape}")
            df_new = align_to_features(df_new, feature_names)
            st.success(f"Columns aligned for prediction: {df_new.shape[0]} rows")

        # Run predictions
        preds = pipeline.predict(df_new)

        # Try to show probabilities if supported by the trained pipeline
        prob_col = None
        proba_vals = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(df_new)
                # For binary classification, scikit-learn orders columns by classes_.
                # Usually class 1 probability is proba[:, 1], but we try to be robust.
                if proba.shape[1] >= 2:
                    proba_vals = proba[:, 1]
                    prob_col = "p(outbreak=1)"
            except Exception:
                proba_vals = None

        out = df_new.copy()
        out["prediction_outbreak"] = preds.astype(int)
        if prob_col is not None and proba_vals is not None:
            out[prob_col] = proba_vals

        st.subheader("Predictions")
        st.write(out.head(25))

        # Simple summary counts
        counts = out["prediction_outbreak"].value_counts().sort_index()
        st.subheader("Prediction counts")
        st.write(counts)

        # Show legend
        st.caption("Interpretation: 0 = no outbreak, 1 = outbreak.")
    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()

