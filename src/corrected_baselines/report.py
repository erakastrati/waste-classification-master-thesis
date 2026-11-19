"""Four-experiment validation report for the corrected TrashNet split.

The report reads only experiments/EXP-001 through EXP-004. It refuses the
historical seed study, the 78-image set, and the 600-image set. It does not
write a summary until all 12 validation files exist.
"""

import json
from pathlib import Path

from src.corrected_baselines.metrics import format_mean_std, sample_mean_std
from src.corrected_baselines.protocol import (
    ALL_EXPERIMENTS,
    CLASS_NAMES,
    COMPARISON_DIR,
    EVALUATED_FILE,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    FORBIDDEN_MARKERS,
    REPORT_NOTE,
    SELECTION_RULE,
    TRAINING_SEEDS,
    assert_safe_output,
    run_directory,
)


def assert_validation_record(record):
    if record.get("comparison") != "corrected_trashnet_seed42_dedup":
        raise RuntimeError("Record is not part of the corrected comparison")
    if record.get("metric_set") != "VALIDATION":
        raise RuntimeError("Record is not labeled VALIDATION")
    if record.get("independent_test_set") is not False:
        raise RuntimeError("Record is not marked as a non-independent validation score")
    if record.get("n_validation") != EXPECTED_VAL_N:
        raise RuntimeError(f"Record does not cover the {EXPECTED_VAL_N}-image validation split")
    if record.get("held_out_600_used") is not False:
        raise RuntimeError("Record used the held-out 600-image set")
    if record.get("historical_78_used") is not False:
        raise RuntimeError("Record used the historical 78-image set")
    if record.get("historical_seed_study_included") is not False:
        raise RuntimeError("Record includes the historical seed study")
    blob = json.dumps(record)
    for marker in FORBIDDEN_MARKERS:
        if marker in blob:
            raise RuntimeError(f"Record references excluded data: {marker}")
    per_class = record.get("per_class")
    if set(per_class or {}) != set(CLASS_NAMES):
        raise RuntimeError("Record is missing per-class metrics")
    for name in CLASS_NAMES:
        if int(per_class[name]["support"]) != EXPECTED_VAL_COUNTS[name]:
            raise RuntimeError(f"{name} support is not the corrected validation count")
    return record


def aggregate(records_by_experiment):
    if tuple(records_by_experiment) != ALL_EXPERIMENTS and set(records_by_experiment) != set(ALL_EXPERIMENTS):
        raise RuntimeError(f"The comparison requires {ALL_EXPERIMENTS}")
    summary = {
        "comparison": "corrected_trashnet_seed42_dedup",
        "metric_set": "VALIDATION",
        "independent_test_set": False,
        "standard_deviation": "sample, ddof=1",
        "seeds": list(TRAINING_SEEDS),
        "n_validation": EXPECTED_VAL_N,
        "historical_seed_study_included": False,
        "historical_78_used": False,
        "held_out_600_used": False,
        "note": REPORT_NOTE,
        "experiments": {},
    }
    for experiment in ALL_EXPERIMENTS:
        records = [assert_validation_record(record) for record in records_by_experiment[experiment]]
        if [record["training_seed"] for record in records] != list(TRAINING_SEEDS):
            raise RuntimeError(f"{experiment} is not in seed order {TRAINING_SEEDS}")
        accuracy = [float(record["accuracy"]) for record in records]
        macro_f1 = [float(record["macro_f1"]) for record in records]
        accuracy_mean, accuracy_std = sample_mean_std(accuracy)
        f1_mean, f1_std = sample_mean_std(macro_f1)
        per_class = {}
        for name in CLASS_NAMES:
            block = {}
            for metric in ("precision", "recall", "f1"):
                values = [float(record["per_class"][name][metric]) for record in records]
                mean, std = sample_mean_std(values)
                block[metric] = {
                    "values": values,
                    "mean": mean,
                    "std": std,
                    "mean_pm_std": format_mean_std(mean, std),
                }
            block["support"] = EXPECTED_VAL_COUNTS[name]
            per_class[name] = block
        summary["experiments"][experiment] = {
            "selection_rule": SELECTION_RULE[experiment],
            "evaluated_file": EVALUATED_FILE[experiment],
            "accuracy_values": accuracy,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "accuracy_mean_pm_std": format_mean_std(accuracy_mean, accuracy_std),
            "macro_f1_values": macro_f1,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "macro_f1_mean_pm_std": format_mean_std(f1_mean, f1_std),
            "per_class": per_class,
        }
    blob = json.dumps(summary)
    for marker in FORBIDDEN_MARKERS:
        if marker in blob:
            raise RuntimeError(f"Summary references excluded data: {marker}")
    return summary


def load_completed_runs():
    loaded = {}
    for experiment in ALL_EXPERIMENTS:
        rows = []
        for seed in TRAINING_SEEDS:
            path = run_directory(experiment, seed) / "evaluation.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Corrected validation file is missing: {path}. "
                    "The 600-image set is not a substitute."
                )
            rows.append(assert_validation_record(json.loads(path.read_text())))
        loaded[experiment] = rows
    return loaded


def write_summary():
    summary = aggregate(load_completed_runs())
    destination = assert_safe_output(COMPARISON_DIR / "summary.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    return destination


def score_selected_checkpoint(experiment, training_seed):
    """Score one selected checkpoint on the 501-image validation manifest.

    This is the post-training path for per-class metrics. It is not called
    by preflight, and it has no input other than the corrected validation list.
    """
    if experiment not in ALL_EXPERIMENTS or training_seed not in TRAINING_SEEDS:
        raise ValueError("Unknown corrected run")
    checkpoint = run_directory(experiment, training_seed) / EVALUATED_FILE[experiment]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint does not exist yet: {checkpoint}. "
            "Refusing to score any other dataset."
        )
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.models import load_model

    from src.corrected_baselines.data import PREPROCESS, load_batch, load_split, steps_per_epoch, validation_dataset
    from src.corrected_baselines.evidence import build_evaluation, write_predictions

    _train_paths, _train_labels, val_paths, val_labels = load_split()
    preprocess = PREPROCESS.get(experiment, efficientnet_preprocess)
    model = load_model(checkpoint)
    probabilities = []
    for start in range(0, len(val_paths), 32):
        images, _labels = load_batch(
            val_paths[start:start + 32],
            val_labels[start:start + 32],
            preprocess,
        )
        probabilities.append(model(images, training=False).numpy())
    import numpy as np

    probabilities = np.concatenate(probabilities, axis=0)
    val_loss, _accuracy = model.evaluate(
        validation_dataset(val_paths, val_labels, preprocess),
        steps=steps_per_epoch(len(val_paths)),
        verbose=0,
    )
    metrics = build_evaluation(
        experiment, training_seed, val_labels, probabilities, val_loss, {"file": checkpoint.name}
    )
    destination = assert_safe_output(
        COMPARISON_DIR / "per_class" / f"{experiment}_seed_{training_seed}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(destination.with_name(f"{experiment}_seed_{training_seed}.csv"), val_paths, val_labels, probabilities)
    destination.write_text(json.dumps(assert_validation_record(metrics), indent=2) + "\n")
    return destination


def package_text():
    root = Path(__file__).resolve().parent
    return "\n".join(path.read_text() for path in sorted(root.glob("*.py")))


if __name__ == "__main__":
    raise SystemExit(
        "The comparison report is not written until all 12 corrected runs exist. "
        "This module does not score the 600-image set."
    )
