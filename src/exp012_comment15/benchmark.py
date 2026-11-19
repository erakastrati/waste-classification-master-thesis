"""Two-epoch CPU benchmark for batch sizes 16 and 64.

This entry point cannot launch the six-configuration study. The epoch budget
is fixed at 2 and the only accepted batch sizes are 16 and 64.
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
    ALLOWED_BATCH_SIZES,
    BENCHMARK_EPOCHS,
    EXPECTED_HEAD_TRAINABLE_PARAMS,
    LEARNING_RATE,
    SHUFFLE_BUFFER_SIZE,
    TRAINING_SEED,
    assert_within_benchmark,
    create_run_directory,
    write_json,
)
from src.models.efficientnet_model import build_efficientnet_model
from src.seed_study.environment import environment_record
from src.seed_study.protocol import CHECKPOINT_KWARGS, EARLY_STOPPING_KWARGS
from src.seed_study.split import load_frozen_split, verify_files_exist, verify_membership


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
    system = platform.system()
    if system == "Darwin":
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


def first_best_epoch(val_accuracy):
    best_value = max(val_accuracy)
    return val_accuracy.index(best_value) + 1


def run_benchmark(batch_size):
    if batch_size not in ALLOWED_BATCH_SIZES:
        raise RuntimeError(f"Refusing batch size {batch_size}")
    if BENCHMARK_EPOCHS != 2:
        raise RuntimeError("Refusing to run because the benchmark budget is not 2 epochs")

    train_paths, val_paths = load_frozen_split()
    verify_membership(train_paths, val_paths)
    verify_files_exist(train_paths, val_paths)
    overlap = len(set(train_paths) & set(val_paths))
    if overlap != 0 or len(train_paths) != 2022 or len(val_paths) != 505:
        raise RuntimeError(
            f"Split check failed: train={len(train_paths)} val={len(val_paths)} overlap={overlap}"
        )

    tf.keras.utils.set_random_seed(TRAINING_SEED)
    build_started = time.perf_counter()
    model, base_model = build_efficientnet_model()
    head_params = assert_frozen_feature_extractor(model, base_model)
    build_seconds = time.perf_counter() - build_started

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

    run_dir = create_run_directory(batch_size)
    checkpoint_path = assert_within_benchmark(run_dir / "best.keras")
    write_json(
        run_dir / "config.json",
        {
            "study": "exp012_comment15_benchmark",
            "full_six_configuration_study": False,
            "architecture": "EfficientNetB0",
            "weights": "imagenet",
            "backbone_frozen": True,
            "fine_tuning": False,
            "augmentation": False,
            "preprocessing": "tensorflow.keras.applications.efficientnet.preprocess_input",
            "input_shape": [224, 224, 3],
            "trainable_head_params": head_params,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
            "batch_size": batch_size,
            "epochs": BENCHMARK_EPOCHS,
            "training_seed": TRAINING_SEED,
            "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
            "validation_shuffle": False,
            "drop_remainder": False,
            "early_stopping": EARLY_STOPPING_KWARGS,
            "checkpoint": CHECKPOINT_KWARGS,
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
    callbacks = [
        timer,
        ModelCheckpoint(filepath=str(checkpoint_path), **CHECKPOINT_KWARGS),
        early_stop,
    ]
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    caught_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=BENCHMARK_EPOCHS,
                callbacks=callbacks,
            )
    except Exception:
        write_json(
            run_dir / "error.json",
            {
                "batch_size": batch_size,
                "traceback": traceback.format_exc(),
                "peak_memory": peak_memory(),
            },
        )
        raise
    finally:
        tf_logger.removeHandler(warning_log)

    history_map = history.history
    required = ("accuracy", "loss", "val_accuracy", "val_loss")
    missing = [key for key in required if key not in history_map]
    if missing or len(timer.epoch_seconds) != len(history_map["loss"]):
        raise RuntimeError(f"Incomplete epoch history. Missing={missing}")
    epochs = []
    for index, duration in enumerate(timer.epoch_seconds):
        epochs.append(
            {
                "epoch": index + 1,
                "accuracy": float(history_map["accuracy"][index]),
                "loss": float(history_map["loss"][index]),
                "val_accuracy": float(history_map["val_accuracy"][index]),
                "val_loss": float(history_map["val_loss"][index]),
                "duration_seconds": round(duration, 3),
            }
        )
    payload = {
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "training_seed": TRAINING_SEED,
        "n_train": len(train_paths),
        "n_validation": len(val_paths),
        "epochs": epochs,
        "fit_seconds": round(timer.total_seconds, 3),
        "model_build_seconds": round(build_seconds, 3),
        "best_epoch": first_best_epoch(history_map["val_accuracy"]),
        "history_length": len(epochs),
        "stopped_before_budget": len(epochs) < BENCHMARK_EPOCHS,
        "early_stopping_stopped_epoch_attribute": int(early_stop.stopped_epoch),
        "peak_memory": peak_memory(),
        "python_warnings": [
            f"{item.category.__name__}: {item.message}" for item in caught_warnings
        ],
        "tensorflow_warning_logs": list(warning_log.records),
    }
    write_json(run_dir / "history.json", payload)
    write_json(
        run_dir / "COMPLETE.json",
        {
            "status": "complete",
            "batch_size": batch_size,
            "epochs_completed": len(epochs),
            "fit_seconds": payload["fit_seconds"],
        },
    )
    print(
        f"BENCHMARK_COMPLETE batch={batch_size} "
        f"fit_seconds={payload['fit_seconds']} "
        f"epoch_seconds={[row['duration_seconds'] for row in epochs]}"
    )
    return payload


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run one 2-epoch EXP-012 CPU benchmark. Does not start the six-run study."
    )
    parser.add_argument("--batch-size", required=True, type=int, choices=ALLOWED_BATCH_SIZES)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_benchmark(args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
