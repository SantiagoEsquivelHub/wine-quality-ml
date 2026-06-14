"""
evaluation.py
=============
Phase 3 (second half): evaluate and compare the trained models.

The rubric weights "evaluation and comparison of results" heavily, so this module
computes the same metrics for every model and builds a single comparison table.
Because the neural network outputs probabilities and the sklearn models output
class labels, we normalize how we get predictions before scoring.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def predict_labels(model, X, is_keras: bool = False) -> np.ndarray:
    """Return predicted class labels, regardless of model type.

    sklearn models already return labels via .predict(). A Keras classifier returns
    a probability matrix (one column per class), so we take argmax to get the most
    likely class for each row.
    """
    if is_keras:
        proba = model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)
    return model.predict(X)


def predict_proba(model, X, is_keras: bool = False) -> np.ndarray:
    """Return predicted class probabilities (needed for ROC-AUC).

    sklearn exposes .predict_proba(); Keras already outputs probabilities.
    """
    if is_keras:
        return model.predict(X, verbose=0)
    return model.predict_proba(X)


def evaluate_model(model, X_test, y_test, is_keras: bool = False,
                   model_name: str = "model") -> dict:
    """Compute all metrics for one model on the test set.

    We use average='macro' for precision/recall/F1: it averages the score of each
    class with equal weight, so a tiny minority class counts as much as a big one.
    That's the honest way to score imbalanced data (plain accuracy can look great
    while ignoring rare classes).
    """
    y_pred = predict_labels(model, X_test, is_keras)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    # ROC-AUC needs probabilities; one-vs-rest handles the multiclass case.
    try:
        y_proba = predict_proba(model, X_test, is_keras)
        metrics["roc_auc_ovr"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        # Some edge cases (e.g. a class missing in y_test) make AUC undefined.
        metrics["roc_auc_ovr"] = float("nan")

    return metrics


def measure_inference_time(model, X, is_keras: bool = False) -> float:
    """Measure how long the model takes to predict the whole set (seconds).

    Inference time matters for the final-model choice: a slightly better model that
    is far slower may be worse for a real-time app.
    """
    start = time.perf_counter()
    predict_labels(model, X, is_keras)
    return time.perf_counter() - start


def get_confusion_matrix(model, X_test, y_test, is_keras: bool = False) -> np.ndarray:
    """Confusion matrix: rows = true class, columns = predicted class.

    The diagonal holds correct predictions; off-diagonal cells show which classes
    get confused with which. Great for the report as a heatmap.
    """
    y_pred = predict_labels(model, X_test, is_keras)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(model, X_test, y_test,
                              is_keras: bool = False) -> str:
    """Per-class precision/recall/F1 as a readable text table."""
    y_pred = predict_labels(model, X_test, is_keras)
    return classification_report(y_test, y_pred, zero_division=0)


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Stack the per-model metric dicts into one comparison DataFrame.

    `results` is a list of the dicts returned by evaluate_model (optionally with a
    'train_time' / 'inference_time' key added). Sorted by F1-macro so the best
    model floats to the top — this is the table that goes in the report.
    """
    df = pd.DataFrame(results)
    if "f1_macro" in df.columns:
        df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return df
