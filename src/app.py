"""
app.py
======
Phase 4: functional Streamlit application.

The app loads the artifacts trained in the earlier phases (scaler + final model)
and lets a user enter a new wine's properties, runs the model, and shows the
predicted wine profile clearly. Run it with:

    streamlit run src/app.py
"""

import json
import sys
import os
from pathlib import Path

# Ensure src/ is on the path regardless of the working directory or Streamlit reruns.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from preprocessing import FEATURE_COLUMNS

# Paths to the artifacts produced during training (absolute, relative to repo root).
_ROOT = Path(__file__).resolve().parent.parent
SCALER_PATH = str(_ROOT / "models" / "scaler.pkl")
# Final model chosen in Phase 3: Logistic Regression had the best F1-macro (0.993)
# and was also the fastest/smallest. Swap to another .pkl to use a different model.
MODEL_PATH = str(_ROOT / "models" / "logreg_model.pkl")
# Maps the model's numeric classes (0..k-1) back to readable profile names.
LABEL_MAP_PATH = str(_ROOT / "models" / "label_mapping.json")

# UI ranges per feature: (min, max, default). Taken from the real training data
# so the user can't enter values the model never saw.
FEATURE_RANGES = {
    "fixed acidity":        (4.0, 16.0, 7.0),
    "volatile acidity":     (0.1, 1.6, 0.5),
    "citric acid":          (0.0, 1.0, 0.3),
    "residual sugar":       (0.5, 22.0, 2.5),
    "chlorides":            (0.01, 0.6, 0.05),
    "free sulfur dioxide":  (1.0, 100.0, 30.0),
    "total sulfur dioxide": (6.0, 290.0, 115.0),
    "density":              (0.987, 1.039, 0.995),
    "pH":                   (2.7, 4.0, 3.2),
    "sulphates":            (0.2, 2.0, 0.5),
    "alcohol":              (8.0, 15.0, 10.0),
}


@st.cache_resource
def load_artifacts():
    """Load the scaler, model and label map once, caching them across reruns.

    `st.cache_resource` keeps the objects in memory so the (slow) disk load only
    happens on the first interaction, not on every slider move.
    """
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        # JSON keys are strings; convert them back to int to match model classes.
        label_map = {int(k): v for k, v in json.load(f).items()}
    return scaler, model, label_map


def collect_user_input() -> pd.DataFrame:
    """Render the input widgets and return the wine as a one-row DataFrame.

    Building a DataFrame with the exact FEATURE_COLUMNS order guarantees the
    features reach the scaler/model in the same order they were trained on.
    """
    wine_type = st.selectbox("Wine type", ["red", "white"])

    values = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(FEATURE_COLUMNS):
        lo, hi, default = FEATURE_RANGES[feature]
        target_col = col1 if i % 2 == 0 else col2
        values[feature] = target_col.slider(feature, lo, hi, default)

    # Encode wine_type the same way Phase 1 did (red=0, white=1).
    values["wine_type"] = 0 if wine_type == "red" else 1

    # Column order must match training: the 11 features, then wine_type.
    ordered = FEATURE_COLUMNS + ["wine_type"]
    return pd.DataFrame([values])[ordered]


def main():
    st.set_page_config(page_title="Wine Profile Classifier", page_icon="🍷")
    st.title("🍷 Wine Profile Classifier")
    st.write("Enter a wine's physicochemical properties to predict its profile.")

    # Fail clearly if the model hasn't been trained yet.
    if not Path(MODEL_PATH).exists() or not Path(SCALER_PATH).exists():
        st.error("Trained artifacts not found. Run the training notebooks first "
                 "to generate models/scaler.pkl and models/rf_model.pkl.")
        return

    scaler, model, label_map = load_artifacts()
    user_df = collect_user_input()

    if st.button("Classify"):
        # Scale ONLY the 11 numeric features, exactly as in training.
        X = user_df.copy()
        X[FEATURE_COLUMNS] = scaler.transform(X[FEATURE_COLUMNS])

        # Pass a plain array (no column names) to match how the model was trained.
        X_values = X.values

        # The model returns a numeric class; translate it to the readable name.
        prediction = model.predict(X_values)[0]
        profile = label_map.get(int(prediction), str(prediction))
        st.success(f"Predicted profile: **{profile}**")

        # Show the probability for each profile, if the model supports it.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_values)[0]
            readable = [label_map.get(int(c), str(c)) for c in model.classes_]
            proba_df = pd.DataFrame(
                {"probability": proba}, index=readable
            ).sort_values("probability", ascending=False)
            st.bar_chart(proba_df)


if __name__ == "__main__":
    main()
