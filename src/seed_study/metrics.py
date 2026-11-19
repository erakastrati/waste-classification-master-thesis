"""Validation metrics and the three-seed aggregate.

Accuracy and macro F1 are computed from the same predictions. The standard
deviation is the sample standard deviation (ddof=1). Historical runs are not
an input to this module.
"""

import csv
import json
import math
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.seed_study.protocol import (
    CLASS_NAMES,
    EXPERIMENTS,
    TRAINING_SEEDS,
    VALIDATION_METRIC_NOTE,
    run_directory,
)


def sample_mean_std(values):
    if len(values) != len(TRAINING_SEEDS):
        raise ValueError(
            f"Expected {len(TRAINING_SEEDS)} values, received {len(values)}"
        )
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def first_strict_maximum(phase_histories):
    """First epoch that strictly improves val_accuracy, across phases.

    This is the ModelCheckpoint(mode='max') rule: numpy.greater, and the
    checkpoint object is not reset between phases.
    """
    best_value = float("-inf")
    best = None
    for phase_number, history in enumerate(phase_histories, start=1):
        for epoch, value in enumerate(history["val_accuracy"], start=1):
            value = float(value)
            if value > best_value:
                best_value = value
                best = {
                    "phase": phase_number,
                    "epoch": epoch,
                    "val_accuracy": value,
                }
    if best is None:
        raise ValueError("No validation accuracy was recorded")
    return best


def validation_metrics(y_true, y_pred):
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
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
        "note": VALIDATION_METRIC_NOTE,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "n_validation": int(len(y_true)),
    }


def report_text(y_true, y_pred, heading_lines):
    table = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    )
    return "\n".join(heading_lines) + "\n\n" + table


def save_confusion_matrix(matrix, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("VALIDATION confusion matrix")
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_confusion_csv(matrix, path):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row])


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_run_metrics(experiment, training_seed):
    path = run_directory(experiment, training_seed) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation metrics: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("metric_set") != "VALIDATION":
        raise RuntimeError(f"{path} is not labeled as VALIDATION metrics")
    if payload.get("independent_test_set") is not False:
        raise RuntimeError(f"{path} is not marked as a non-independent validation score")
    return payload


def aggregate_results():
    """Aggregate the nine controlled runs. Refuses to read historical results."""
    summary = {
        "metric_set": "VALIDATION",
        "standard_deviation": "sample, ddof=1",
        "seeds": list(TRAINING_SEEDS),
        "historical_runs_included": False,
        "note": VALIDATION_METRIC_NOTE,
        "experiments": {},
    }
    for experiment, spec in EXPERIMENTS.items():
        accuracy_values = []
        f1_values = []
        for seed in TRAINING_SEEDS:
            metrics = load_run_metrics(experiment, seed)
            accuracy_values.append(float(metrics["accuracy"]))
            f1_values.append(float(metrics["macro_f1"]))
        accuracy_mean, accuracy_std = sample_mean_std(accuracy_values)
        f1_mean, f1_std = sample_mean_std(f1_values)
        block = {
            "title": spec["title"],
            "seeds": list(TRAINING_SEEDS),
            "accuracy_values": accuracy_values,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "accuracy_mean_pm_std": f"{accuracy_mean:.4f} ± {accuracy_std:.4f}",
            "macro_f1_values": f1_values,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "macro_f1_mean_pm_std": f"{f1_mean:.4f} ± {f1_std:.4f}",
        }
        summary["experiments"][experiment] = block
        dump_json(run_directory(experiment, TRAINING_SEEDS[0]).parent / "aggregate.json", block)
    from src.seed_study.protocol import SEED_STUDY_ROOT

    dump_json(SEED_STUDY_ROOT / "summary.json", summary)
    return summary
