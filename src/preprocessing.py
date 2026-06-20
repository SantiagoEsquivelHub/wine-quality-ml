"""
preprocessing.py
================
Phase 1: load, clean and prepare the Wine Quality dataset.

This module groups ALL data transformations into reusable functions, so that both
the notebooks and the app produce EXACTLY the same preprocessing. That consistency
is critical: if the app scales the data differently from how the model was trained,
the predictions will be wrong.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Global seed: makes random results (splits, models) reproducible.
# The rubric/checklist requires random_state=42 across the whole codebase.
RANDOM_STATE = 42

# The 11 physicochemical features. We name them explicitly so we don't depend on
# the column order of the CSV and so we can validate they are all present.
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_and_combine(red_path: str, white_path: str) -> pd.DataFrame:
    """Load both CSV files (red and white) and merge them into a single DataFrame.

    Critical detail: the original UCI CSV files use ';' as the separator, not ','.
    If `sep=';'` is forgotten, pandas loads everything into a single column.

    A `wine_type` column is added BEFORE merging so we don't lose track of which
    file each row came from.
    """
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    red["wine_type"] = "red"
    white["wine_type"] = "white"

    df = pd.concat([red, white], ignore_index=True)
    return df


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Basic cleaning: duplicates and missing-value validation.

    The Wine Quality dataset officially has no missing values, but it DOES have
    duplicate rows (wines with identical properties). We drop them by default so
    clustering and models don't give extra weight to those repeated rows.
    """
    df = df.copy()

    # Missing-value check (should be 0, but we must verify, not assume).
    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        # Simple median imputation (robust to outliers) if anything shows up.
        for col in FEATURE_COLUMNS:
            df[col] = df[col].fillna(df[col].median())

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def winsorize_outliers(df: pd.DataFrame, columns: list[str] | None = None,
                       percentile: float = 0.99) -> pd.DataFrame:
    """Handle outliers by clipping (winsorizing) instead of removing them.

    Rather than deleting extreme rows (which shrinks the sample), we cap values
    above the 99th percentile down to that percentile. This keeps every sample but
    smooths absurdly high values (typical in 'residual sugar' and
    'total sulfur dioxide' for white wines).
    """
    df = df.copy()
    if columns is None:
        columns = ["residual sugar", "total sulfur dioxide"]

    for col in columns:
        upper_bound = df[col].quantile(percentile)
        lower_bound = df[col].quantile(1 - percentile)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def encode_wine_type(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `wine_type` (red/white) to numeric 0/1.

    Since there are only 2 categories, a binary mapping is enough; One-Hot encoding
    is unnecessary (it would create two redundant columns for a binary variable).
    """
    df = df.copy()
    df["wine_type"] = df["wine_type"].map({"red": 0, "white": 1}).astype(int)
    return df


def scale(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame] | None = None,
          scaler_path: str = "models/scaler.pkl"):
    """Standardize the 11 features (mean 0, std 1) with StandardScaler.

    GOLDEN RULE against data leakage: the scaler is fitted (.fit) ONLY on the
    training data. If we fitted it on the full dataset, we would be leaking
    information from the test set into training.

    The fitted scaler is saved to disk (joblib) so the app applies exactly the
    same transformation to the user's new data.
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    train_scaled[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])

    # Save the scaler so it can be reused in the app.
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    results = [train_scaled]
    if other_dfs:
        for d in other_dfs:
            d_scaled = d.copy()
            # NOTE: only .transform here (NO .fit), using the train mean/std.
            d_scaled[FEATURE_COLUMNS] = scaler.transform(d[FEATURE_COLUMNS])
            results.append(d_scaled)

    return results[0] if len(results) == 1 else tuple(results)


def split_data(df: pd.DataFrame, target_column: str,
               test_size: float = 0.15, val_size: float = 0.15):
    """Split into train / validation / test in a stratified way.

    Stratifying means the proportion of each class is preserved across the three
    splits. This is vital with imbalanced classes: otherwise the test set could end
    up with no examples of a minority class.
    """
    # First cut: separate the test set.
    temp_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RANDOM_STATE,
        stratify=df[target_column],
    )
    # Second cut: from the remainder, separate the validation set.
    # relative_val rescales the size so val ends up being val_size of the original.
    relative_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        temp_df, test_size=relative_val, random_state=RANDOM_STATE,
        stratify=temp_df[target_column],
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example of the full Phase 1 pipeline (paths relative to the repo root).
    df = load_and_combine("data/raw/winequality-red.csv",
                          "data/raw/winequality-white.csv")
    print(f"Rows after combining: {len(df)}")

    df = clean(df)
    print(f"Rows after dropping duplicates: {len(df)}")

    df = winsorize_outliers(df)
    df = encode_wine_type(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/wine_combined.csv", index=False)
    print("Clean combined dataset saved to data/processed/wine_combined.csv")
