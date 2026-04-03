# ============================
# Imports & Setup
# ============================
# This file teaches the computer how to predict malaria outbreaks using past data.
# Every section is explained for someone with no coding experience.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import joblib

try:
    from IPython.display import display
except ImportError:
    display = print  # running as `python notebook.py` outside Jupyter

# Display settings for readability
pd.set_option("display.max_columns", 50)
sns.set(style="whitegrid")

# Paths
DATA_PATH = "malaria_cases.csv"  # <-- CSV dataset path
CLIMATE_FEATURES_PATH = os.path.join(
    "outputs", "climate_avgua_by_year_m49.csv"
)  # from extract_climate_features.py
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "malaria_outbreak_random_forest.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL = "outbreak"
THRESHOLD_RATE = 50  # threshold for automatic outbreak creation (RATE_PER_1000_NL > 50)

"""
## Load Dataset

In this section we:

- Load the malaria dataset from CSV using pandas.
- Inspect the first few rows.
- Confirm that the required incidence column (`RATE_PER_1000_NL`) is present for automatic target creation.
"""
# ============================
# Load Dataset
# ============================

# Load CSV into DataFrame
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'. "
        "Please ensure the CSV file is placed in the working directory."
    )

df = pd.read_csv(DATA_PATH)

if os.path.isfile(CLIMATE_FEATURES_PATH):
    climate_df = pd.read_csv(CLIMATE_FEATURES_PATH)
    df = df.merge(
        climate_df,
        left_on=["DIM_TIME", "DIM_GEO_CODE_M49"],
        right_on=["year", "m49"],
        how="left",
    )
    df = df.drop(columns=["year", "m49"])
    print(
        f"Merged climate features from '{CLIMATE_FEATURES_PATH}' "
        f"(rows still {len(df)})."
    )
else:
    print(
        f"Note: '{CLIMATE_FEATURES_PATH}' not found. "
        "Run `python extract_climate_features.py` to add ERA5 temperature features."
    )

print(f"Loaded dataset from '{DATA_PATH}' with shape: {df.shape}")
print("\nFirst 5 rows:")
display(df.head())

# Check for required column used to create 'outbreak'
if "RATE_PER_1000_NL" not in df.columns:
    raise ValueError(
        "Column 'RATE_PER_1000_NL' not found in dataset. "
        "This column is required to automatically create the 'outbreak' target."
    )

"""
## Automatic Target Creation

We automatically create a **binary target column** called `outbreak`:

LaTeX:

outbreak =
    1 if RATE_PER_1000_NL > 50
    0 otherwise

This converts the continuous malaria incidence into a binary outbreak indicator.
"""
# ============================
# Automatic Target Creation
# ============================

# Create binary target column based on incidence threshold
df[TARGET_COL] = (df["RATE_PER_1000_NL"] > THRESHOLD_RATE).astype(int)

print("Created binary target column 'outbreak' using rule: RATE_PER_1000_NL > 50")

print("\nTarget value counts:")
print(df[TARGET_COL].value_counts())
print("\nTarget proportions:")
print(df[TARGET_COL].value_counts(normalize=True))

"""
## Exploratory Data Analysis (EDA)

In this section we:

- Inspect the overall structure of the dataset (`info`, `describe`).
- Examine the distribution of the target variable `outbreak`.
- Produce a simple correlation heatmap for numeric features.

This helps us understand the data, potential class imbalance, and relationships between variables.
"""
# ============================
# Exploratory Data Analysis (EDA)
# ============================

print("DataFrame info:")
print(df.info())

print("\nNumeric summary statistics:")
display(df.describe(include=[np.number]))

print("\nNon-numeric columns:")
display(df.select_dtypes(exclude=[np.number]).columns.tolist())

print(f"\nTarget '{TARGET_COL}' distribution:")
print(df[TARGET_COL].value_counts())
print("\nTarget proportions:")
print(df[TARGET_COL].value_counts(normalize=True))

# Correlation matrix for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    plt.show()

"""
## Feature Engineering & Preprocessing

We now prepare the data for modeling.

Steps:

1. **Split features and target**:
   - `X` = all columns except `outbreak`
   - `y` = `outbreak`

2. **Automatically identify numeric and categorical features**:
   - Numeric: `df.select_dtypes(include=[np.number])`
   - Categorical: all other columns

3. **Build a `ColumnTransformer` for preprocessing**:
   - Numeric pipeline:
     - `SimpleImputer(strategy="median")`
     - `StandardScaler()`
   - Categorical pipeline:
     - `SimpleImputer(strategy="most_frequent")`
     - `OneHotEncoder(handle_unknown="ignore")`

4. Combine preprocessing and model in a single `Pipeline` to avoid manual scaling
   and feature-name mismatches.
"""
# ============================
# Feature Engineering & Preprocessing
# ============================

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# Preprocessing for numeric data
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ColumnTransformer to apply the appropriate transformations to each column type
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

print("\nPreprocessing pipeline (ColumnTransformer) defined.")

# ============================
# Model Training (RandomForestClassifier)
# ============================

from sklearn.utils import class_weight

# Train/test split (stratify by target to handle imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test size : {X_test.shape[0]} rows")

# Compute class weights to help with class imbalance (optional but often useful)
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {cls: w for cls, w in zip(classes, cw)}
print("\nClass weights used for RandomForest:")
print(class_weights)

# Define the RandomForestClassifier
rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight=class_weights,
    n_jobs=-1,
)

# Build the full Pipeline: preprocessing + model
rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf_clf),
    ]
)

# Fit the pipeline on the training data
print("\nFitting RandomForest Pipeline...")
rf_pipeline.fit(X_train, y_train)
print("Model training complete.")

"""
## Model Evaluation

We evaluate the trained RandomForest model using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion matrix** (visualized with seaborn)

We use `rf_pipeline.predict()` directly, which ensures that all preprocessing
is applied consistently to the test data.
"""
# ============================
# Model Evaluation
# ============================

# Predictions on the test set using the trained Pipeline
y_pred = rf_pipeline.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("Evaluation Metrics on Test Set:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("RandomForest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

"""
## Save Trained Model

We save the **entire Pipeline** (preprocessing + RandomForest model) using `joblib` to:

`outputs/malaria_outbreak_random_forest.joblib`

This allows us to perform **automatic prediction** later without
rebuilding or re-fitting the model, and **without manual scaling**.
"""
# ============================
# Save Trained Model
# ============================

bundle = {
    "pipeline": rf_pipeline,
    "feature_names": X.columns.tolist(),  # original feature names before preprocessing
}

joblib.dump(bundle, MODEL_PATH)
print(f"Saved trained Pipeline bundle to: {MODEL_PATH}")

"""
## Automatic Prediction Example

In this final section, we demonstrate **automatic prediction**:

1. Load the saved Pipeline bundle from disk.
2. Select an example sample from the original dataset (excluding `outbreak`).
3. Run `pipeline.predict()` directly on a pandas DataFrame.
4. Display the predicted outbreak risk for the example.

This simulates how you can deploy the trained model for new, unseen data.
"""
# ============================
# Automatic Prediction Example
# ============================

# Load the saved pipeline bundle
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Saved model bundle not found at '{MODEL_PATH}'. "
        "Run the training cells above first."
    )

loaded_bundle = joblib.load(MODEL_PATH)
loaded_pipeline = loaded_bundle["pipeline"]
loaded_feature_names = loaded_bundle["feature_names"]

print("Loaded Pipeline bundle from disk.")
print("Feature names stored in bundle:")
print(loaded_feature_names)

# Create an example sample automatically from the dataset
# Here we simply take the first row of X (features only).
example_sample = X.iloc[[0]].copy()  # keep as DataFrame

print("\nExample sample (first row of feature set):")
display(example_sample)

# IMPORTANT:
# We do NOT call scaler/encoder manually.
# All preprocessing is handled inside the Pipeline.
example_prediction = loaded_pipeline.predict(example_sample)[0]

print("\nPredicted outbreak risk for example sample (0 = no outbreak, 1 = outbreak):")
print(int(example_prediction))

"""
---

### Summary

In this notebook, we:

- Loaded a malaria-related CSV dataset and automatically created a binary `outbreak` target.
- Performed basic exploratory data analysis.
- Built a **single scikit-learn Pipeline** that encapsulates:
  - Numeric and categorical preprocessing via `ColumnTransformer`.
  - A `RandomForestClassifier` with class balancing.
- Evaluated the model using standard classification metrics and a confusion matrix.
- Saved the trained Pipeline to disk for reuse.
- Demonstrated **automatic prediction** using `pipeline.predict()` directly on a pandas DataFrame.

This structure can be integrated into an undergraduate final year project as both
a methodological and practical implementation component of a machine learning-based
malaria outbreak prediction model.
"""
