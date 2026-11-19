"""Validation metrics for the corrected 501-image split.

The sample standard deviation uses ddof=1. These helpers do not read a
historical result directory or an external photograph set.
"""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.corrected_baselines.protocol import (
    CLASS_NAMES,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    REPORT_NOTE,
    TRAINING_SEEDS,
)


def sample_mean_std(values):
    if len(values) != len(TRAINING_SEEDS):
        raise ValueError(f"Expected {len(TRAINING_SEEDS)} values, received {len(values)}")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def format_mean_std(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def validation_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != EXPECTED_VAL_N:
        raise ValueError(f"Validation predictions must cover {EXPECTED_VAL_N} images")
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        if int(support[index]) != EXPECTED_VAL_COUNTS[name]:
            raise ValueError(
                f"{name} support is {int(support[index])}, "
                f"expected {EXPECTED_VAL_COUNTS[name]}"
            )
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "metric_set": "VALIDATION",
        "independent_test_set": False,
        "note": REPORT_NOTE,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "n_validation": int(len(y_true)),
    }
