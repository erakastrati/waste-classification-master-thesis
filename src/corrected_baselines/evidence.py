"""Run records for the corrected EXP-001 through EXP-004 comparison.

Writing these files does not change a fit call, a seed, or a checkpoint rule.
This module does not read the historical 78-image set or the 600-image set.
"""

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.corrected_baselines.metrics import validation_metrics
from src.corrected_baselines.protocol import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    EVALUATED_FILE,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_TRAIN_N,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    IMAGE_SIZE,
    PHASE1_EPOCHS,
    PHASE1_LEARNING_RATE,
    PHASE2_EPOCHS,
    PHASE2_LEARNING_RATE,
    PROJECT_ROOT,
    REPORT_NOTE,
    SELECTION_RULE,
    TRAIN_MANIFEST,
    TRAIN_TXT_SHA256,
    VAL_MANIFEST,
    VAL_TXT_SHA256,
)

COMPARISON_NAME = "corrected_trashnet_seed42_dedup"
PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
PREDICTION_COLUMNS = (
    "index",
    "path",
    "true_index",
    "predicted_index",
    "true_class",
    "predicted_class",
    *PROBABILITY_COLUMNS,
)
HISTORY_KEYS = ("loss", "accuracy", "val_loss", "val_accuracy")
ADAM_DEFAULTS = {
    "name": "Adam",
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
    "amsgrad": False,
}


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_counts(paths):
    return dict(Counter(path.split("/")[2] for path in paths))


def dataset_provenance(train_paths, val_paths, training_seed):
    """Hash the approved manifests and record the membership this run used."""
    train_hash = sha256_file(TRAIN_MANIFEST)
    val_hash = sha256_file(VAL_MANIFEST)
    if train_hash != TRAIN_TXT_SHA256 or val_hash != VAL_TXT_SHA256:
        raise RuntimeError("Manifest hash does not match the approved deduplicated split")
    train_counts = _class_counts(train_paths)
    val_counts = _class_counts(val_paths)
    if len(train_paths) != EXPECTED_TRAIN_N or train_counts != EXPECTED_TRAIN_COUNTS:
        raise RuntimeError("Training membership does not match the approved split")
    if len(val_paths) != EXPECTED_VAL_N or val_counts != EXPECTED_VAL_COUNTS:
        raise RuntimeError("Validation membership does not match the approved split")
    return {
        "train_manifest": TRAIN_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "val_manifest": VAL_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "train_txt_sha256": train_hash,
        "val_txt_sha256": val_hash,
        "manifest_seed": 42,
        "training_seed": int(training_seed),
        "training_seed_redraws_split": False,
        "n_train": len(train_paths),
        "n_validation": len(val_paths),
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "validation_order": "val.txt order",
    }


def environment_record():
    import keras
    import tensorflow as tf

    devices = [
        {"type": device.device_type, "name": device.name}
        for device in tf.config.list_physical_devices()
    ]
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "devices": devices,
        "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
        "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        "op_determinism_enabled": False,
        "tf_data_pipeline_deterministic": True,
        "bit_for_bit_guarantee": False,
    }


def configuration(experiment, training_seed):
    """Static training record. It does not alter the constants it describes."""
    if experiment not in SELECTION_RULE:
        raise ValueError(f"Unknown experiment {experiment}")
    shared = {
        "comparison": COMPARISON_NAME,
        "experiment": experiment,
        "training_seed": int(training_seed),
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "loss": "sparse_categorical_crossentropy",
        "label_smoothing": None,
        "metrics": ["accuracy"],
        "class_weight": None,
        "training_shuffle": "full index list shuffled by random.Random(training_seed) every epoch",
        "validation_order": "sorted val.txt",
        "selection_rule": SELECTION_RULE[experiment],
        "evaluated_file": EVALUATED_FILE[experiment],
        "metric_set": "VALIDATION",
    }
    if experiment == "EXP-001":
        shared.update({
            "architecture": (
                "Sequential CNN, input 224x224x3, Conv2D 32/64/128 with 3x3 relu "
                "and MaxPool 2x2, GlobalAveragePooling2D, Dense 128 relu, "
                "Dropout 0.3, Dense 6 softmax"
            ),
            "initialization": "keras.utils.set_random_seed(training_seed) before build",
            "preprocessing": "pixel values divided by 255",
            "augmentation": None,
            "optimizer": {**ADAM_DEFAULTS, "construction": "optimizer='adam'", "learning_rate": PHASE1_LEARNING_RATE},
            "phases": [{"phase": 1, "max_epochs": PHASE1_EPOCHS, "learning_rate": PHASE1_LEARNING_RATE}],
            "batch_normalization": "The CNN has no batch-norm layers.",
            "early_stopping": None,
            "checkpoint": None,
        })
    elif experiment == "EXP-002":
        shared.update({
            "architecture": (
                "MobileNetV2 include_top=False, ImageNet weights, input 224x224x3; "
                "GlobalAveragePooling2D, Dense 128 relu, Dropout 0.3, Dense 6 softmax"
            ),
            "initialization": "keras.utils.set_random_seed(training_seed) before build; base weights are ImageNet",
            "preprocessing": "MobileNetV2 preprocess_input, TensorFlow mode: x/127.5 - 1",
            "augmentation": None,
            "optimizer": ADAM_DEFAULTS,
            "phases": [
                {
                    "phase": 1,
                    "max_epochs": PHASE1_EPOCHS,
                    "learning_rate": PHASE1_LEARNING_RATE,
                    "base_trainable": False,
                },
                {
                    "phase": 2,
                    "max_epochs": PHASE2_EPOCHS,
                    "learning_rate": PHASE2_LEARNING_RATE,
                    "unfreeze_last_n": 30,
                    "batch_norm_in_unfrozen_tail": "left trainable",
                },
            ],
            "batch_normalization": (
                "Phase 1 freezes the whole base. Phase 2 unfreezes the last 30 "
                "layers and leaves batch-norm layers inside that tail trainable. "
                "The original graph calls the base with training=False; fit "
                "supplies its own training flag, so those layers can update."
            ),
            "early_stopping": {
                "monitor": "val_accuracy",
                "patience": 5,
                "restore_best_weights": True,
                "shared_across_phases": True,
            },
            "checkpoint": {
                "file": "best.keras",
                "monitor": "val_accuracy",
                "mode": "max",
                "save_best_only": True,
                "shared_across_phases": True,
                "tie": "an equal val_accuracy does not replace the file",
            },
        })
    else:
        shared.update({
            "architecture": (
                "EfficientNetB0 include_top=False, ImageNet weights, input 224x224x3; "
                "GlobalAveragePooling2D, Dense 128 relu, Dropout 0.3, Dense 6 softmax"
            ),
            "initialization": "keras.utils.set_random_seed(training_seed) before build; base weights are ImageNet",
            "preprocessing": (
                "decode, bilinear resize to 224, keras EfficientNet preprocess_input "
                "(no-op). The model stem rescales by 1/255 and applies ImageNet normalization."
            ),
            "augmentation": None if experiment == "EXP-003" else {
                "RandomFlip": "horizontal",
                "RandomRotation": {"factor": 0.15, "range_radians": [-0.15, 0.15]},
                "RandomZoom": {"height_factor": 0.15, "width_factor": None},
                "RandomContrast": {"factor": [0.0, 0.2], "effect": "0% to +20%, not plus or minus 20%"},
                "seed": "each augmentation layer receives training_seed",
                "active": "training only",
            },
            "optimizer": ADAM_DEFAULTS,
            "phases": [
                {
                    "phase": 1,
                    "max_epochs": PHASE1_EPOCHS,
                    "learning_rate": PHASE1_LEARNING_RATE,
                    "base_trainable": False,
                },
                {
                    "phase": 2,
                    "max_epochs": PHASE2_EPOCHS,
                    "learning_rate": PHASE2_LEARNING_RATE,
                    "unfreeze_last_n": 30,
                    "batch_norm": "every base batch-norm layer set trainable false before compile",
                },
            ],
            "batch_normalization": (
                "Phase 1 freezes the whole base. Phase 2 unfreezes the last 30 "
                "layers, then sets every base batch-norm layer trainable false "
                "before the phase-2 compile. The base is called with no training "
                "keyword."
            ),
            "early_stopping": {
                "monitor": "val_accuracy",
                "patience": 5,
                "restore_best_weights": True,
                "new_object_each_phase": True,
            },
            "checkpoint": {
                "files": ["phase1_best.keras", "phase2_best.keras", "published.keras"],
                "monitor": "val_accuracy",
                "mode": "max",
                "save_best_only": True,
                "separate_phase_objects": True,
                "tie": "equal val_accuracy keeps phase 1",
            },
        })
    return shared


def git_provenance():
    """Read the repository revision when a training run starts.

    The commit is not stored in this source file. A clean worktree matches
    the committed revision; any tracked change means it does not.
    """
    def _git(args):
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    try:
        sha = _git(["git", "rev-parse", "HEAD"]).strip()
        status = _git(["git", "status", "--porcelain", "--untracked-files=all"])
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        return {
            "commit": None,
            "dirty": None,
            "matches_committed_revision": False,
            "error": str(detail).strip() or exc.__class__.__name__,
        }
    full_sha = len(sha) == 40 and all(character in "0123456789abcdef" for character in sha)
    if not full_sha:
        return {
            "commit": sha or None,
            "dirty": None,
            "matches_committed_revision": False,
            "error": "git rev-parse HEAD did not return a full SHA",
        }
    dirty = bool(status.strip())
    return {
        "commit": sha,
        "dirty": dirty,
        "matches_committed_revision": not dirty,
        "status": status,
    }


def write_run_header(run_dir, experiment, training_seed, train_paths, val_paths):
    run_dir = Path(run_dir)
    config = configuration(experiment, training_seed)
    config["provenance"] = dataset_provenance(train_paths, val_paths, training_seed)
    config["git"] = git_provenance()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_status(run_dir / "status.json", "running")
    return config


def write_status(path, state, exception=None):
    payload = {"state": state}
    if exception is not None:
        payload["exception_type"] = type(exception).__name__
        payload["exception"] = f"{type(exception).__name__}: {exception}"
        payload["traceback"] = traceback.format_exc()
    write_json(path, payload)


@contextmanager
def capture_warnings(path):
    path = Path(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield caught
        finally:
            lines = [
                f"{item.category.__name__}: {item.filename}:{item.lineno}: {item.message}"
                for item in caught
            ]
            path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


class HistoryStore:
    """Flush every completed epoch. A later exception keeps the epochs already written."""

    def __init__(self, path, experiment, training_seed):
        self.path = Path(path)
        self.experiment = experiment
        self.training_seed = int(training_seed)
        self.records = []
        self.phase_lists = {}
        self.write()

    def start_phase(self, phase):
        self.phase_lists.setdefault(phase, {key: [] for key in HISTORY_KEYS})
        self.write()

    def adopt(self, row):
        required = ("phase", "epoch_in_phase", "global_epoch", *HISTORY_KEYS)
        missing = [key for key in required if key not in row]
        if missing:
            raise KeyError(f"Epoch row is missing {missing}")
        stored = {
            "phase": int(row["phase"]),
            "epoch_in_phase": int(row["epoch_in_phase"]),
            "global_epoch": int(row["global_epoch"]),
            **{key: float(row[key]) for key in HISTORY_KEYS},
        }
        self.records.append(stored)
        bucket = self.phase_lists.setdefault(stored["phase"], {key: [] for key in HISTORY_KEYS})
        for key in HISTORY_KEYS:
            bucket[key].append(stored[key])
        self.write()
        return stored

    def record_epoch(self, phase, epoch_in_phase, global_epoch, logs):
        return self.adopt({
            "phase": phase,
            "epoch_in_phase": epoch_in_phase,
            "global_epoch": global_epoch,
            "loss": logs["loss"],
            "accuracy": logs["accuracy"],
            "val_loss": logs["val_loss"],
            "val_accuracy": logs["val_accuracy"],
        })

    def write(self):
        payload = {
            "experiment": self.experiment,
            "training_seed": self.training_seed,
            "metric_set": "VALIDATION",
            "epochs_completed": len(self.records),
            "epochs": self.records,
        }
        for phase, key in ((1, "phase1"), (2, "phase2")):
            if phase in self.phase_lists:
                payload[key] = self.phase_lists[phase]
        write_json(self.path, payload)


def history_callback(store, phase, global_offset):
    """Keras callback that flushes one epoch. It does not select a checkpoint."""
    import keras

    class _Flush(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            store.record_epoch(
                phase,
                int(epoch) + 1,
                global_offset + int(epoch) + 1,
                logs or {},
            )

    return _Flush()


def write_predictions(path, image_paths, y_true, probabilities):
    """Write validation predictions in the given path order, which must be val.txt."""
    image_paths = list(image_paths)
    y_true = [int(label) for label in y_true]
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(image_paths), len(CLASS_NAMES)):
        raise ValueError(
            f"Probabilities have shape {probabilities.shape}, "
            f"expected {(len(image_paths), len(CLASS_NAMES))}"
        )
    if len(y_true) != len(image_paths):
        raise ValueError("Label count does not match the validation paths")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each prediction row must be a six-class probability distribution")
    predicted = np.argmax(probabilities, axis=1)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for index, (image_path, truth, guess, probs) in enumerate(
            zip(image_paths, y_true, predicted, probabilities)
        ):
            writer.writerow([
                index,
                image_path,
                truth,
                int(guess),
                CLASS_NAMES[truth],
                CLASS_NAMES[int(guess)],
                *[f"{float(value):.8f}" for value in probs],
            ])
    return predicted


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError(f"Prediction columns are {reader.fieldnames}")
        rows = list(reader)
    for index, row in enumerate(rows):
        if int(row["index"]) != index:
            raise ValueError("Prediction index left val.txt order")
        if row["true_class"] != CLASS_NAMES[int(row["true_index"])]:
            raise ValueError(f"True class does not match its index at row {index}")
        if row["predicted_class"] != CLASS_NAMES[int(row["predicted_index"])]:
            raise ValueError(f"Predicted class does not match its index at row {index}")
    return rows


def paired_contingency(left_rows, right_rows):
    """Pair two prediction tables on path order for McNemar's test.

    left_only_correct and right_only_correct are the discordant cells.
    """
    if [row["path"] for row in left_rows] != [row["path"] for row in right_rows]:
        raise RuntimeError("Prediction tables are not paired on the same images in val.txt order")
    if [row["true_index"] for row in left_rows] != [row["true_index"] for row in right_rows]:
        raise RuntimeError("Paired predictions do not share true labels")
    both_correct = both_wrong = left_only = right_only = 0
    for left, right in zip(left_rows, right_rows):
        left_correct = left["true_index"] == left["predicted_index"]
        right_correct = right["true_index"] == right["predicted_index"]
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            left_only += 1
        elif right_correct:
            right_only += 1
        else:
            both_wrong += 1
    return {
        "n": len(left_rows),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": mcnemar_exact_p(left_only, right_only),
        "test": "two-sided exact McNemar on discordant validation pairs",
        "metric_set": "VALIDATION",
        "independent_test_set": False,
    }


def mcnemar_exact_p(left_only_correct, right_only_correct):
    """Two-sided exact test. Discordant counts are Binomial(n, 0.5) under the null."""
    left_only_correct = int(left_only_correct)
    right_only_correct = int(right_only_correct)
    if left_only_correct < 0 or right_only_correct < 0:
        raise ValueError("Discordant counts cannot be negative")
    discordant = left_only_correct + right_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(left_only_correct, right_only_correct)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def build_evaluation(experiment, training_seed, y_true, probabilities, val_loss, selection):
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = np.argmax(probabilities, axis=1)
    metrics = validation_metrics(y_true, predicted)
    metrics.update({
        "comparison": COMPARISON_NAME,
        "experiment": experiment,
        "training_seed": int(training_seed),
        "val_loss": float(val_loss),
        "val_accuracy": metrics["accuracy"],
        "held_out_600_used": False,
        "historical_78_used": False,
        "historical_seed_study_included": False,
        "evaluated_file": EVALUATED_FILE[experiment],
        "selection_rule": SELECTION_RULE[experiment],
        "selection": selection,
        "prediction_file": "val_predictions.csv",
        "prediction_order": "val.txt",
        "class_weight": None,
        "note": REPORT_NOTE,
    })
    return metrics


def manifest_paths_and_labels(manifest_path):
    paths = []
    labels = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        relative = line.strip()
        if not relative:
            continue
        class_name = relative.split("/")[2]
        if class_name not in CLASS_TO_INDEX:
            raise ValueError(f"Path is outside the six-class table: {relative}")
        paths.append(relative)
        labels.append(CLASS_TO_INDEX[class_name])
    return paths, labels
