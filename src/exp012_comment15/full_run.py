"""One approved EXP-012 configuration per invocation.

`--dry-run` checks the six configurations, paths, split, and model settings
without creating directories or calling fit(). Training starts only when one
configuration id is passed explicitly.
"""

import argparse
import logging
import platform
import resource
import sys
import time
import traceback
import warnings

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.exp012_comment15.data import make_dataset
from src.exp012_comment15.safety import (
    CHECKPOINT_KWARGS,
    EARLY_STOPPING_KWARGS,
    EXPECTED_HEAD_TRAINABLE_PARAMS,
    FULL_EPOCHS,
    FULL_RUNS,
    SELECTION_RULE,
    SHUFFLE_BUFFER_SIZE,
    STUDY_BATCH_SIZES,
    TRAINING_SEED,
    assert_within_full_runs,
    create_full_run_directory,
    first_strict_maximum,
    full_run_directory,
    full_runs_root,
    static_safety_report,
    termination_reason,
    write_json,
)
from src.models.efficientnet_model import build_efficientnet_model
from src.seed_study.environment import environment_record
from src.seed_study.protocol import EXPECTED_TRAIN_N, EXPECTED_VAL_N
from src.seed_study.split import (
    class_counts,
    load_frozen_split,
    verify_files_exist,
    verify_membership,
)

LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]


def parameter_count(weights):
    count = 0
    for weight in weights:
        size = 1
        for dim in weight.shape:
            size *= int(dim)
        count += size
    return count


def assert_frozen_feature_extractor(model, base_model):
    if base_model.trainable or any(layer.trainable for layer in base_model.layers):
        raise RuntimeError("Backbone is trainable. Refusing to fine-tune.")
    if parameter_count(base_model.trainable_weights) != 0:
        raise RuntimeError("Backbone has trainable weights. Refusing to fine-tune.")
    head_params = parameter_count(model.trainable_weights)
    if head_params != EXPECTED_HEAD_TRAINABLE_PARAMS:
        raise RuntimeError(
            f"Trainable head has {head_params} parameters, "
            f"expected {EXPECTED_HEAD_TRAINABLE_PARAMS}"
        )
    input_shape = tuple(model.inputs[0].shape)
    if input_shape != (None, 224, 224, 3):
        raise RuntimeError(f"Unexpected input shape: {input_shape}")
    layer_types = {type(layer).__name__ for layer in model.layers}
    if any(name.startswith("Random") for name in layer_types):
        raise RuntimeError(f"Augmentation layers are present: {sorted(layer_types)}")
    return head_params


def peak_memory():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        peak_bytes = int(raw)
        unit = "bytes"
    else:
        peak_bytes = int(raw) * 1024
        unit = "kilobytes_converted_to_bytes"
    return {
        "ru_maxrss_raw": int(raw),
        "raw_unit": unit,
        "peak_bytes": peak_bytes,
        "peak_mebibytes": round(peak_bytes / (1024 * 1024), 1),
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "includes_interpreter_and_tensorflow": True,
    }


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_seconds = []
        self._started = None
        self._train_started = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self._started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._started)

    def on_train_end(self, logs=None):
        self.total_seconds = time.perf_counter() - self._train_started


class WarningLog(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


def plain_float(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)


def count_correct(model, dataset):
    correct = 0
    seen = 0
    for images, labels in dataset:
        probabilities = model(images, training=False)
        predicted = tf.argmax(probabilities, axis=1, output_type=tf.int32)
        label_values = tf.cast(labels, tf.int32)
        correct += int(tf.reduce_sum(tf.cast(predicted == label_values, tf.int32)))
        seen += int(label_values.shape[0])
    return correct, seen


def load_verified_split():
    train_paths, val_paths = load_frozen_split()
    verify_membership(train_paths, val_paths)
    verify_files_exist(train_paths, val_paths)
    overlap = len(set(train_paths) & set(val_paths))
    if overlap != 0 or len(train_paths) != EXPECTED_TRAIN_N or len(val_paths) != EXPECTED_VAL_N:
        raise RuntimeError(
            f"Split check failed: train={len(train_paths)} "
            f"val={len(val_paths)} overlap={overlap}"
        )
    return train_paths, val_paths, overlap


def run_config(config_id):
    if FULL_EPOCHS != 15:
        raise RuntimeError("Refusing to train because the epoch budget is not 15")
    config = next(item for item in FULL_RUNS if item["id"] == config_id)
    if config["batch_size"] not in STUDY_BATCH_SIZES:
        raise RuntimeError(f"Refusing batch size {config['batch_size']}")

    started = time.perf_counter()
    train_paths, val_paths, overlap = load_verified_split()
    tf.keras.utils.set_random_seed(TRAINING_SEED)
    build_started = time.perf_counter()
    model, base_model = build_efficientnet_model()
    head_params = assert_frozen_feature_extractor(model, base_model)
    build_seconds = time.perf_counter() - build_started

    train_ds = make_dataset(
        train_paths,
        training=True,
        batch_size=config["batch_size"],
        training_seed=TRAINING_SEED,
    )
    val_ds = make_dataset(
        val_paths,
        training=False,
        batch_size=config["batch_size"],
        training_seed=None,
    )

    run_dir = create_full_run_directory(config["id"])
    checkpoint_path = assert_within_full_runs(run_dir / "best.keras")
    try:
        write_json(
            run_dir / "config.json",
            {
                "study": "exp012_comment15_full",
                "configuration_id": config["id"],
                "architecture": "EfficientNetB0",
                "weights": "imagenet",
                "backbone_frozen": True,
                "fine_tuning": False,
                "augmentation": False,
                "preprocessing": "tensorflow.keras.applications.efficientnet.preprocess_input",
                "input_shape": [224, 224, 3],
                "trainable_head_params": head_params,
                "optimizer": "Adam",
                "learning_rate": config["learning_rate"],
                "loss": LOSS,
                "metrics": METRICS,
                "batch_size": config["batch_size"],
                "epochs": FULL_EPOCHS,
                "training_seed": TRAINING_SEED,
                "training_seed_reset_before_model_build": True,
                "classification_head": "freshly initialized after the training seed reset",
                "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
                "validation_shuffle": False,
                "drop_remainder": False,
                "early_stopping": EARLY_STOPPING_KWARGS,
                "checkpoint": CHECKPOINT_KWARGS,
                "selection_rule": SELECTION_RULE,
                "n_train": len(train_paths),
                "n_validation": len(val_paths),
                "train_validation_overlap": overlap,
                "split_manifest": "results/seed_study/manifest",
                "output_directory": str(run_dir),
                "environment": environment_record(),
            },
        )

        warning_log = WarningLog()
        warning_log.setFormatter(logging.Formatter("%(name)s: %(levelname)s: %(message)s"))
        tf_logger = logging.getLogger("tensorflow")
        tf_logger.addHandler(warning_log)
        timer = EpochTimer()
        early_stop = EarlyStopping(**EARLY_STOPPING_KWARGS)
        checkpoint = ModelCheckpoint(filepath=str(checkpoint_path), **CHECKPOINT_KWARGS)
        model.compile(
            optimizer=Adam(learning_rate=config["learning_rate"]),
            loss=LOSS,
            metrics=METRICS,
        )
        caught_warnings = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=FULL_EPOCHS,
                    callbacks=[timer, checkpoint, early_stop],
                )
        finally:
            tf_logger.removeHandler(warning_log)

        history_map = history.history
        required = ("accuracy", "loss", "val_accuracy", "val_loss")
        missing = [key for key in required if key not in history_map]
        if missing or len(timer.epoch_seconds) != len(history_map["loss"]):
            raise RuntimeError(f"Incomplete epoch history. Missing={missing}")
        val_accuracy = [plain_float(value) for value in history_map["val_accuracy"]]
        best_epoch, history_best = first_strict_maximum(val_accuracy)
        completed_epochs = len(val_accuracy)
        reason = termination_reason(
            completed_epochs,
            int(early_stop.stopped_epoch),
            FULL_EPOCHS,
        )
        early_best_epoch = int(early_stop.best_epoch) + 1
        checkpoint_best = plain_float(checkpoint.best)
        count_started = time.perf_counter()
        loaded = tf.keras.models.load_model(checkpoint_path)
        n_correct, seen = count_correct(loaded, val_ds)
        count_seconds = time.perf_counter() - count_started
        reconstructed = n_correct / seen
        selection_agrees = (
            seen == len(val_paths)
            and early_best_epoch == best_epoch
            and abs(checkpoint_best - history_best) <= 1e-6
            and abs(reconstructed - history_best) <= 1e-5
        )
        epochs = []
        for index, duration in enumerate(timer.epoch_seconds):
            epochs.append(
                {
                    "epoch": index + 1,
                    "accuracy": plain_float(history_map["accuracy"][index]),
                    "loss": plain_float(history_map["loss"][index]),
                    "val_accuracy": val_accuracy[index],
                    "val_loss": plain_float(history_map["val_loss"][index]),
                    "duration_seconds": round(duration, 3),
                }
            )
        payload = {
            "configuration_id": config["id"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "training_seed": TRAINING_SEED,
            "n_train": len(train_paths),
            "n_validation": len(val_paths),
            "epochs": epochs,
            "fit_seconds": round(timer.total_seconds, 3),
            "model_build_seconds": round(build_seconds, 3),
            "validation_count_seconds": round(count_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "best_epoch": best_epoch,
            "best_val_accuracy": history_best,
            "completed_epochs": completed_epochs,
            "early_stopping": EARLY_STOPPING_KWARGS,
            "checkpoint": CHECKPOINT_KWARGS,
            "selection_rule": SELECTION_RULE,
            "termination_reason": reason,
            "early_stopping_stopped_epoch_attribute": int(early_stop.stopped_epoch),
            "early_stopping_best_epoch": early_best_epoch,
            "checkpoint_best_val_accuracy": checkpoint_best,
            "validation_correct": n_correct,
            "validation_total": seen,
            "validation_accuracy_from_checkpoint": reconstructed,
            "selection_agrees": selection_agrees,
            "peak_memory": peak_memory(),
            "python_warnings": [
                f"{item.category.__name__}: {item.message}" for item in caught_warnings
            ],
            "tensorflow_warning_logs": list(warning_log.records),
        }
        write_json(run_dir / "history.json", payload)
        if not selection_agrees:
            raise RuntimeError(
                "Checkpoint selection and reported validation accuracy disagree: "
                f"history_epoch={best_epoch}, early_stopping_epoch={early_best_epoch}, "
                f"history={history_best}, checkpoint={checkpoint_best}, "
                f"recount={n_correct}/{seen}"
            )
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Checkpoint file is missing: {checkpoint_path}")
        write_json(
            run_dir / "COMPLETE.json",
            {
                "status": "complete",
                "configuration_id": config["id"],
                "completed_epochs": completed_epochs,
                "best_epoch": best_epoch,
                "termination_reason": reason,
                "validation_correct": n_correct,
                "validation_total": seen,
                "fit_seconds": payload["fit_seconds"],
                "total_seconds": payload["total_seconds"],
            },
        )
    except Exception:
        write_json(
            run_dir / "error.json",
            {
                "configuration_id": config["id"],
                "traceback": traceback.format_exc(),
                "peak_memory": peak_memory(),
            },
        )
        raise
    print(
        f"FULL_RUN_COMPLETE config={config['id']} "
        f"best_epoch={best_epoch} correct={n_correct}/{seen} "
        f"reason={reason} fit_seconds={payload['fit_seconds']}"
    )
    return payload


def expected_batches(count, batch_size):
    return (count + batch_size - 1) // batch_size


def dry_run():
    errors = static_safety_report()
    for message in errors:
        print(f"FAIL static: {message}")
    if errors:
        return 1
    print("PASS static safety checks")

    train_paths, val_paths, overlap = load_verified_split()
    print(
        "PASS split "
        f"n_train={len(train_paths)} n_val={len(val_paths)} overlap={overlap} "
        f"train_counts={class_counts(train_paths)} val_counts={class_counts(val_paths)}"
    )

    if full_runs_root().exists():
        print(f"FAIL full-run root already exists: {full_runs_root()}")
        return 1
    print(f"PASS full-run root absent: {full_runs_root()}")
    for config in FULL_RUNS:
        path = full_run_directory(config["id"])
        assert_within_full_runs(path)
        if path.exists():
            print(f"FAIL full-run directory already exists: {path}")
            return 1
        print(
            "PASS config "
            f"{config['id']} lr={config['learning_rate']} "
            f"batch={config['batch_size']} epochs={FULL_EPOCHS} "
            f"seed={TRAINING_SEED} absent={path}"
        )

    tie_epoch, tie_value = first_strict_maximum([0.5, 0.8, 0.8, 0.7])
    if tie_epoch != 2 or tie_value != 0.8:
        raise RuntimeError("Tie handling did not keep the earliest maximum")
    if termination_reason(15, 0, 15) != "epoch_budget":
        raise RuntimeError("Completed budget was not recognized")
    if termination_reason(6, 5, 15) != "early_stopping":
        raise RuntimeError("Early stopping was not recognized")
    try:
        termination_reason(15, 14, 15)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Ambiguous termination was accepted")
    print("PASS selection rule keeps the earliest strict maximum")
    print("PASS termination rule distinguishes epoch_budget from early_stopping")

    for batch_size in STUDY_BATCH_SIZES:
        train_ds = make_dataset(
            train_paths,
            training=True,
            batch_size=batch_size,
            training_seed=TRAINING_SEED,
        )
        val_ds = make_dataset(
            val_paths,
            training=False,
            batch_size=batch_size,
            training_seed=None,
        )
        train_batches = int(train_ds.cardinality())
        val_batches = int(val_ds.cardinality())
        expected_train = expected_batches(len(train_paths), batch_size)
        expected_val = expected_batches(len(val_paths), batch_size)
        if train_batches != expected_train or val_batches != expected_val:
            raise RuntimeError(
                f"Unexpected cardinality for batch {batch_size}: "
                f"train={train_batches}/{expected_train} val={val_batches}/{expected_val}"
            )
        print(
            f"PASS dataset batch={batch_size} "
            f"train_batches={train_batches} val_batches={val_batches} "
            f"shuffle_buffer={SHUFFLE_BUFFER_SIZE}"
        )

    tf.keras.utils.set_random_seed(TRAINING_SEED)
    model, base_model = build_efficientnet_model()
    head_params = assert_frozen_feature_extractor(model, base_model)
    for config in FULL_RUNS:
        learning_rate = plain_float(Adam(learning_rate=config["learning_rate"]).learning_rate)
        requested = float(tf.cast(config["learning_rate"], tf.float32))
        if learning_rate != requested:
            raise RuntimeError(f"Adam learning rate mismatch for {config['id']}")
    print(
        "PASS model "
        f"head_params={head_params} input={tuple(model.inputs[0].shape)} "
        f"frozen=True augmentation=False loss={LOSS} metrics={METRICS}"
    )
    if full_runs_root().exists():
        raise RuntimeError("Dry-run created the full-run directory")
    print("PASS dry-run; full training was not started")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Validate or run one EXP-012 configuration. No configuration trains by default."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", choices=[item["id"] for item in FULL_RUNS])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.dry_run:
        return dry_run()
    if args.config is None:
        print("Refusing to start training. Pass --dry-run, or one --config after approval.")
        return 2
    run_config(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
