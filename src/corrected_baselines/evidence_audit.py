"""Static and synthetic checks for the corrected-run evidence files.

Does not call fit, does not create experiment run directories, and does not
read the historical 78-image set or the 600-image set.
"""

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np

from src.corrected_baselines.evidence import (
    PREDICTION_COLUMNS,
    HistoryStore,
    build_evaluation,
    capture_warnings,
    configuration,
    dataset_provenance,
    environment_record,
    manifest_paths_and_labels,
    mcnemar_exact_p,
    paired_contingency,
    read_predictions,
    write_predictions,
    write_status,
)
from src.corrected_baselines.metrics import sample_mean_std
from src.corrected_baselines.protocol import (
    ALL_EXPERIMENTS,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_VAL_COUNTS,
    PROJECT_ROOT,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)
from src.corrected_baselines.report import assert_validation_record
from src.corrected_baselines.train import trainer_uses_shared_exp002_callbacks

CHECKS = []


def check(name, ok, detail):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return bool(ok)


def _one_hot(labels):
    probabilities = np.zeros((len(labels), 6), dtype=np.float64)
    probabilities[np.arange(len(labels)), labels] = 1.0
    return probabilities


def _source(relative):
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def main():
    ok = True
    train_paths, _train_labels = manifest_paths_and_labels(TRAIN_MANIFEST)
    val_paths, val_labels = manifest_paths_and_labels(VAL_MANIFEST)
    provenance = dataset_provenance(train_paths, val_paths, 42)
    ok &= check(
        "provenance hashes and class counts",
        provenance["train_txt_sha256"]
        and provenance["val_txt_sha256"]
        and provenance["train_class_counts"] == EXPECTED_TRAIN_COUNTS
        and provenance["val_class_counts"] == EXPECTED_VAL_COUNTS
        and provenance["training_seed_redraws_split"] is False
        and provenance["n_train"] == 2016
        and provenance["n_validation"] == 501,
        provenance["train_manifest"],
    )
    configs = {experiment: configuration(experiment, 123) for experiment in ALL_EXPERIMENTS}
    config_ok = all(
        configs[name]["class_weight"] is None
        and configs[name]["loss"] == "sparse_categorical_crossentropy"
        and configs[name]["batch_size"] == 32
        and "preprocessing" in configs[name]
        and "batch_normalization" in configs[name]
        and "early_stopping" in configs[name]
        and configs[name]["training_seed"] == 123
        for name in ALL_EXPERIMENTS
    )
    config_ok = config_ok and configs["EXP-001"]["augmentation"] is None
    config_ok = config_ok and configs["EXP-001"]["phases"][0]["max_epochs"] == 10
    config_ok = config_ok and configs["EXP-004"]["augmentation"]["RandomContrast"]["factor"] == [0.0, 0.2]
    config_ok = config_ok and configs["EXP-002"]["checkpoint"]["shared_across_phases"] is True
    config_ok = config_ok and configs["EXP-003"]["checkpoint"]["tie"] == "equal val_accuracy keeps phase 1"
    ok &= check("configuration records the approved rules", config_ok, "four experiments, class_weight null")

    environment = environment_record()
    ok &= check(
        "environment records software versions",
        all(environment.get(key) for key in ("python_version", "tensorflow", "keras", "numpy"))
        and environment["op_determinism_enabled"] is False,
        f"python {environment['python_version']} tf {environment['tensorflow']} keras {environment['keras']}",
    )

    probabilities = _one_hot(val_labels)
    evaluation = build_evaluation(
        "EXP-003", 42, val_labels, probabilities, 0.25, {"phase": 1, "epoch_in_phase": 4}
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prediction_path = root / "val_predictions.csv"
        predicted = write_predictions(prediction_path, val_paths, val_labels, probabilities)
        rows = read_predictions(prediction_path)
        other = root / "other.csv"
        flipped = probabilities.copy()
        flipped[0] = np.roll(flipped[0], 1)
        write_predictions(other, val_paths, val_labels, flipped)
        other_rows = read_predictions(other)
        contingency = paired_contingency(rows, other_rows)
        store = HistoryStore(root / "history.json", "EXP-002", 42)
        store.start_phase(1)
        store.record_epoch(1, 1, 1, {
            "loss": 1.0, "accuracy": 0.2, "val_loss": 0.9, "val_accuracy": 0.3,
        })
        store.start_phase(2)
        try:
            raise RuntimeError("stopped during phase 2")
        except RuntimeError as exc:
            write_status(root / "status.json", "failed", exception=exc)
        history = json.loads((root / "history.json").read_text())
        status = json.loads((root / "status.json").read_text())
        warning_path = root / "warnings.txt"
        with capture_warnings(warning_path):
            warnings.warn("synthetic evidence warning", UserWarning)
        warning_text = warning_path.read_text()
        leftover = list(root.glob("*.tmp"))

    ok &= check(
        "predictions follow val.txt and keep six probabilities",
        list(predicted) == val_labels
        and [row["path"] for row in rows] == val_paths
        and tuple(rows[0].keys()) == PREDICTION_COLUMNS
        and len(PREDICTION_COLUMNS) == 12,
        f"{len(rows)} rows",
    )
    ok &= check(
        "paired predictions support McNemar",
        contingency["n"] == 501
        and contingency["left_only_correct"] == 1
        and contingency["right_only_correct"] == 0
        and contingency["mcnemar_exact_p"] == 1.0
        and mcnemar_exact_p(0, 6) == 0.03125,
        f"discordant 1/0 p={contingency['mcnemar_exact_p']}",
    )
    ok &= check(
        "evaluation schema matches the four-experiment report",
        assert_validation_record(evaluation)["accuracy"] == 1.0
        and evaluation["macro_f1"] == 1.0
        and evaluation["val_accuracy"] == 1.0
        and set(evaluation["per_class"]) == set(EXPECTED_VAL_COUNTS)
        and evaluation["confusion_matrix"][0][0] == EXPECTED_VAL_COUNTS["cardboard"],
        "accuracy 1.0000 on a synthetic perfect score",
    )
    mean, std = sample_mean_std([0.5, 0.6, 0.7])
    ok &= check(
        "three-seed summary uses the sample standard deviation",
        abs(mean - 0.6) < 1e-12 and abs(std - 0.1) < 1e-12,
        f"{mean:.4f} ± {std:.4f}",
    )
    ok &= check(
        "a phase failure keeps completed epochs and the exception",
        history["epochs_completed"] == 1
        and history["phase1"]["val_accuracy"] == [0.3]
        and history["phase2"] == {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        and status["state"] == "failed"
        and "stopped during phase 2" in status["exception"]
        and "RuntimeError" in status["traceback"]
        and "synthetic evidence warning" in warning_text
        and leftover == [],
        status["exception"],
    )

    baseline = _source("src/corrected_baselines/train.py")
    efficient = _source("src/retraining/train.py")
    directory_call = "image_dataset_from_directory" + "("
    split_argument = "validation_split" + "="
    held_out = "data/" + "real_test_dataset"
    old_photos = "results/" + "real_world_test"
    ok &= check(
        "trainers flush history and write the standard artifacts",
        "HistoryStore" in baseline
        and "HistoryStore" in efficient
        and "val_predictions.csv" in baseline
        and "val_predictions.csv" in efficient
        and "write_run_header" in baseline
        and "write_run_header" in efficient
        and "capture_warnings" in baseline
        and 'write_status(run_dir / "status.json", "failed"' in baseline
        and 'write_status(run_dir / "status.json", "failed"' in efficient
        and "on_record=store.adopt" in efficient
        and trainer_uses_shared_exp002_callbacks(),
        "both trainers",
    )
    ok &= check(
        "fit arguments and checkpoint rules are unchanged",
        "class_weight=" not in baseline
        and "class_weight=" not in efficient
        and "epochs=PHASE1_EPOCHS" in baseline
        and "epochs=PHASE2_EPOCHS" in baseline
        and "epochs=PHASE1_EPOCHS" in efficient
        and "epochs=PHASE2_EPOCHS" in efficient
        and 'tie_rule": "equal val_accuracy keeps phase 1"' in efficient
        and "make_callbacks(run_dir, 1)" in efficient
        and "make_callbacks(run_dir, 2)" in efficient
        and directory_call not in baseline
        and directory_call not in efficient
        and split_argument not in baseline
        and split_argument not in efficient
        and held_out not in baseline
        and held_out not in efficient
        and old_photos not in baseline
        and old_photos not in efficient
        and "run(" not in baseline.split("if __name__")[-1]
        and "run(" not in efficient.split("if __name__")[-1],
        "no class weights and no held-out paths",
    )
    created = list((PROJECT_ROOT / "experiments").glob("*")) if (PROJECT_ROOT / "experiments").exists() else []
    ok &= check("no experiment directory was created", created == [], str(created))

    report = {
        "passed": ok and all(item["passed"] for item in CHECKS),
        "checks": CHECKS,
        "training_started": False,
    }
    destination = PROJECT_ROOT / "results/retraining_2026/audit/evidence_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", destination.relative_to(PROJECT_ROOT))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
