"""Lightweight checks for the seed study. This module does not train."""

import inspect
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.seed_study.environment import environment_record
from src.seed_study.metrics import dump_json, first_strict_maximum, sample_mean_std, validation_metrics
from src.seed_study.protocol import (
    CLASS_NAMES,
    EXPERIMENTS,
    PROJECT_ROOT,
    SEED_STUDY_ROOT,
    SOURCE_SPLIT_JSONS,
    TRAINING_SEEDS,
    runs,
    run_directory,
)
from src.seed_study.split import (
    freeze_manifest,
    load_frozen_split,
    verify_files_exist,
    verify_membership,
)


CHECKS = []


def check(name):
    def wrap(function):
        CHECKS.append((name, function))
        return function
    return wrap


def snapshot_tree(root, skip=None):
    records = {}
    root = Path(root)
    if not root.exists():
        return records
    skip = None if skip is None else Path(skip).resolve()
    for directory, dirnames, filenames in os.walk(root):
        current = Path(directory).resolve()
        if skip is not None and (current == skip or skip in current.parents):
            dirnames.clear()
            continue
        dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            path = Path(directory) / name
            stat = path.stat()
            records[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return records


def protected_snapshot():
    records = {}
    records.update(snapshot_tree(PROJECT_ROOT / "results", skip=SEED_STUDY_ROOT))
    records.update(snapshot_tree(PROJECT_ROOT / "data" / "trashNet-dataset"))
    records.update(snapshot_tree(PROJECT_ROOT / "docs"))
    records.update(snapshot_tree(PROJECT_ROOT / "src" / "training"))
    records.update(snapshot_tree(PROJECT_ROOT / "src" / "models"))
    records.update(snapshot_tree(PROJECT_ROOT / "src" / "evaluation"))
    return records


@check("frozen membership counts, classes, and empty overlap")
def check_membership():
    train_paths, val_paths = load_frozen_split()
    verify_membership(train_paths, val_paths)
    verify_files_exist(train_paths, val_paths)
    return {"n_train": len(train_paths), "n_val": len(val_paths)}


@check("manifest matches both previously verified split files, in order")
def check_source_order():
    train_paths, val_paths = load_frozen_split()
    for source in SOURCE_SPLIT_JSONS:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["train_shuffled"] != train_paths or payload["val_shuffled"] != val_paths:
            raise RuntimeError(f"Manifest differs from {source}")
    return {"sources": [str(path.relative_to(PROJECT_ROOT)) for path in SOURCE_SPLIT_JSONS]}


@check("live Keras seed=42 split matches the frozen lists and class order")
def check_keras_reproduction():
    previous = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        from tensorflow.keras.preprocessing import image_dataset_from_directory

        keyword_args = dict(
            validation_split=0.2,
            seed=42,
            image_size=(224, 224),
            batch_size=32,
        )
        train_ds = image_dataset_from_directory(
            "data/trashNet-dataset",
            subset="training",
            **keyword_args,
        )
        val_ds = image_dataset_from_directory(
            "data/trashNet-dataset",
            subset="validation",
            **keyword_args,
        )
    finally:
        os.chdir(previous)
    train_paths, val_paths = load_frozen_split()
    if list(train_ds.file_paths) != train_paths or list(val_ds.file_paths) != val_paths:
        raise RuntimeError("Frozen lists are not the current seed=42 Keras split")
    if list(train_ds.class_names) != list(CLASS_NAMES) or list(val_ds.class_names) != list(CLASS_NAMES):
        raise RuntimeError(
            f"Class order is {train_ds.class_names}, expected {list(CLASS_NAMES)}"
        )
    overlap = set(train_ds.file_paths) & set(val_ds.file_paths)
    if overlap:
        raise RuntimeError(f"Reproduced split overlaps in {len(overlap)} paths")
    return {"class_names": list(CLASS_NAMES)}


@check("training seed does not change split membership")
def check_seed_does_not_split():
    import keras

    signature = inspect.signature(load_frozen_split)
    if len(signature.parameters) != 0:
        raise RuntimeError("load_frozen_split must not accept a seed")
    before = load_frozen_split()
    for seed in TRAINING_SEEDS:
        keras.utils.set_random_seed(seed)
        if load_frozen_split() != before:
            raise RuntimeError(f"set_random_seed({seed}) changed the frozen split")
    source = inspect.getsource(keras.utils.set_random_seed)
    for required in ("random.seed", "np.random.seed", "tf.random.set_seed"):
        if required not in source:
            raise RuntimeError(f"keras.utils.set_random_seed does not set {required}")
    return {"seeds_checked": list(TRAINING_SEEDS)}


@check("training shuffle changes order only; validation order stays fixed")
def check_dataset_order():
    from src.seed_study.datasets import path_label_dataset

    train_paths, val_paths = load_frozen_split()

    def collect(dataset):
        paths = []
        labels = []
        for path, label in dataset:
            paths.append(path.numpy().decode("utf-8"))
            labels.append(int(label.numpy()))
        return paths, labels

    val_dataset = path_label_dataset(val_paths, training=False, training_seed=None)
    val_first, val_labels = collect(val_dataset)
    val_second, _ = collect(val_dataset)
    from src.seed_study.split import resolved_path

    expected_val = [str(resolved_path(path)) for path in val_paths]
    expected_labels = [CLASS_NAMES.index(path.split("/")[-2]) for path in val_paths]
    if val_first != expected_val or val_second != expected_val or val_labels != expected_labels:
        raise RuntimeError("Validation dataset order or labels differ from the manifest")
    try:
        path_label_dataset(val_paths, training=False, training_seed=123)
    except ValueError:
        pass
    else:
        raise RuntimeError("Validation dataset accepted a training seed")

    orders = {}
    label_orders = {}
    for seed in TRAINING_SEEDS:
        dataset = path_label_dataset(train_paths, training=True, training_seed=seed)
        first_paths, first_labels = collect(dataset)
        second_paths, second_labels = collect(dataset)
        expected_train = [str(resolved_path(path)) for path in train_paths]
        if set(first_paths) != set(expected_train) or set(second_paths) != set(expected_train):
            raise RuntimeError(f"Training seed {seed} changed train membership")
        if sorted(first_labels) != sorted(CLASS_NAMES.index(path.split("/")[-2]) for path in train_paths):
            raise RuntimeError(f"Training seed {seed} changed train labels")
        if first_paths == second_paths:
            raise RuntimeError(f"Training seed {seed} did not reshuffle the next epoch")
        if first_labels != [CLASS_NAMES.index(Path(path).parent.name) for path in first_paths]:
            raise RuntimeError(f"Training seed {seed} detached a label from its path")
        if second_labels != [CLASS_NAMES.index(Path(path).parent.name) for path in second_paths]:
            raise RuntimeError(f"Training seed {seed} detached a label on the second epoch")
        orders[seed] = first_paths
        label_orders[seed] = first_labels
    if orders[42] == orders[123] or orders[42] == orders[456] or orders[123] == orders[456]:
        raise RuntimeError("Different training seeds produced the same training order")
    return {
        "validation_order_fixed": True,
        "train_orders_differ": True,
        "label_checks": {str(seed): len(labels) for seed, labels in label_orders.items()},
    }


@check("all nine runs share one manifest and write outside historical results")
def check_output_paths():
    directories = []
    for experiment, seed in runs():
        path = run_directory(experiment, seed)
        if path.exists():
            raise RuntimeError(f"Run directory already exists and would be protected: {path}")
        directories.append(str(path))
    if len(set(directories)) != 9:
        raise RuntimeError("The nine run directories are not unique")
    for experiment in EXPERIMENTS:
        for seed in TRAINING_SEEDS:
            other_seed = 123 if seed == 42 else 42
            if run_directory(experiment, seed) == run_directory(experiment, other_seed):
                raise RuntimeError("Two seeds resolve to the same directory")
    from src.seed_study.protocol import assert_safe_output

    for relative in (
        "results/cnn_baseline/cnn_baseline.keras",
        "results/mobilenet/mobilenet_best.keras",
        "results/efficientnet/efficientnet_best.keras",
    ):
        try:
            assert_safe_output(PROJECT_ROOT / relative)
        except RuntimeError:
            continue
        raise RuntimeError(f"Protected historical path was accepted as an output: {relative}")
    return {"run_directories": directories}


@check("overwrite guard refuses a non-empty run directory")
def check_overwrite_guard():
    from src.seed_study.train import require_fresh

    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "seed_42"
        target.mkdir()
        (target / "metrics.json").write_text("{}", encoding="utf-8")
        try:
            require_fresh(target)
        except SystemExit as error:
            if "Refusing to overwrite" not in str(error):
                raise
        else:
            raise RuntimeError("require_fresh allowed an existing directory")
    return {"guard": "directory existence"}


@check("EXP-002 and EXP-003 callbacks match the original checkpoint rule")
def check_callbacks():
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    from src.seed_study.callbacks import build_callbacks
    from src.seed_study.train import train_one

    checkpoint_path = SEED_STUDY_ROOT / "_callback_probe" / "best.keras"
    callbacks = build_callbacks(checkpoint_path)
    checkpoint, early_stop = callbacks
    if not isinstance(checkpoint, ModelCheckpoint) or not isinstance(early_stop, EarlyStopping):
        raise RuntimeError("Callback types do not match the original scripts")
    if checkpoint.monitor != "val_accuracy" or checkpoint.save_best_only is not True:
        raise RuntimeError("ModelCheckpoint settings differ from the original scripts")
    if checkpoint.monitor_op is not np.greater:
        raise RuntimeError("ModelCheckpoint is not using a strict maximum")
    if early_stop.monitor != "val_accuracy" or early_stop.patience != 5:
        raise RuntimeError("EarlyStopping settings differ from the original scripts")
    if early_stop.restore_best_weights is not True or early_stop.mode != "auto":
        raise RuntimeError("EarlyStopping mode or restore behavior differs")
    source = inspect.getsource(train_one)
    if source.count("callbacks=callbacks") != 2:
        raise RuntimeError("Phase 1 and Phase 2 do not share the callbacks variable")
    if "load_model(evaluated_path)" not in source:
        raise RuntimeError("Transfer experiments do not load the checkpoint file")
    if "optimizer=\"adam\"" not in source:
        raise RuntimeError("EXP-001 does not compile with optimizer='adam'")
    if "EarlyStopping" in source or "ModelCheckpoint" in source:
        raise RuntimeError("train_one should use the shared callback factory, not new callback classes")
    selection = first_strict_maximum(
        [
            {"val_accuracy": [0.50, 0.80, 0.80]},
            {"val_accuracy": [0.70, 0.90]},
        ]
    )
    if selection != {"phase": 2, "epoch": 2, "val_accuracy": 0.90}:
        raise RuntimeError(f"Checkpoint selection reconstruction is wrong: {selection}")
    tied = first_strict_maximum([{"val_accuracy": [0.20, 0.80, 0.80]}])
    if tied["epoch"] != 2:
        raise RuntimeError("An equal val_accuracy replaced the earlier checkpoint")
    for name, spec in EXPERIMENTS.items():
        if name == "exp001" and (spec["callbacks"] or spec["selection"] != "epoch_10"):
            raise RuntimeError("EXP-001 protocol enables callback selection")
        if name != "exp001" and spec["selection"] != "best_checkpoint":
            raise RuntimeError(f"{name} does not evaluate the best checkpoint")
        if name != "exp001" and spec["unfreeze_last_n"] != 30:
            raise RuntimeError(f"{name} does not unfreeze the last 30 layers")
    return {"checkpoint_rule": "strict maximum across both phases"}


@check("original training scripts still contain the protocol this study copies")
def check_original_scripts():
    root = PROJECT_ROOT / "src" / "training"
    cnn = (root / "train_cnn.py").read_text(encoding="utf-8")
    mobilenet = (root / "train_mobilenet.py").read_text(encoding="utf-8")
    efficientnet = (root / "train_efficientnet.py").read_text(encoding="utf-8")
    if "EarlyStopping" in cnn or "ModelCheckpoint" in cnn:
        raise RuntimeError("EXP-001 script now contains callbacks; the seed study must be revisited")
    for required in ("optimizer=\"adam\"", "epochs=10", "validation_split=0.2", "seed=42", "batch_size=32"):
        if required not in cnn:
            raise RuntimeError(f"train_cnn.py no longer contains {required}")
    for script in (mobilenet, efficientnet):
        for required in (
            "learning_rate=1e-3",
            "learning_rate=1e-5",
            "epochs=10",
            "epochs=15",
            "patience=5",
            "monitor=\"val_accuracy\"",
            "save_best_only=True",
            "mode=\"max\"",
            "restore_best_weights=True",
            "layers[:-30]",
            "load_model(",
            "callbacks=[checkpoint, early_stop]",
        ):
            if script.count(required) < 1:
                raise RuntimeError(f"A transfer script no longer contains {required}")
        if script.count("callbacks=[checkpoint, early_stop]") != 2:
            raise RuntimeError("A transfer script no longer reuses both callbacks in both phases")
    study_root = PROJECT_ROOT / "src" / "seed_study"
    for name in ("train.py", "datasets.py", "split.py"):
        text = (study_root / name).read_text(encoding="utf-8")
        if "image_dataset_from_directory(" in text:
            raise RuntimeError(f"{name} rebuilds the split with image_dataset_from_directory")
    return {"original_scripts": "unchanged protocol strings"}


@check("metric definitions use validation labels and sample standard deviation")
def check_metrics():
    mean, std = sample_mean_std([1.0, 2.0, 3.0])
    if mean != 2.0 or abs(std - 1.0) > 1e-12:
        raise RuntimeError(f"sample standard deviation mismatch: {mean} ± {std}")
    constant_mean, constant_std = sample_mean_std([0.5, 0.5, 0.5])
    if constant_mean != 0.5 or constant_std != 0.0:
        raise RuntimeError("sample standard deviation of a constant is not zero")
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    perfect = validation_metrics(y_true, y_pred)
    if perfect["metric_set"] != "VALIDATION" or perfect["independent_test_set"] is not False:
        raise RuntimeError("Metrics are not labeled as validation scores")
    if perfect["accuracy"] != 1.0 or perfect["macro_f1"] != 1.0:
        raise RuntimeError("Perfect predictions did not score 1")
    if perfect["n_validation"] != 6:
        raise RuntimeError("Validation count was not taken from the prediction vector")
    return {"accuracy_std_formula": "ddof=1"}


@check("seed-study modules do not reference historical output filenames for writing")
def check_study_sources_do_not_target_history():
    text = (PROJECT_ROOT / "src" / "seed_study" / "train.py").read_text(encoding="utf-8")
    for forbidden in (
        "results/cnn_baseline",
        "results/mobilenet/mobilenet",
        "results/efficientnet/efficientnet",
    ):
        if forbidden in text:
            raise RuntimeError(f"train.py contains historical output path {forbidden}")
    return {"train_outputs": "run_directory only"}


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.chdir(PROJECT_ROOT)
    before = protected_snapshot()
    freeze_manifest()
    failures = []
    results = {}
    for name, function in CHECKS:
        try:
            results[name] = function()
            print(f"PASS  {name}")
        except Exception as error:
            failures.append({"check": name, "error": str(error)})
            print(f"FAIL  {name}: {error}")
    after = protected_snapshot()
    if before != after:
        changed = sorted(set(before) ^ set(after) | {key for key in before if key in after and before[key] != after[key]})
        failures.append({"check": "historical artifacts unchanged", "error": changed[:20]})
        print(f"FAIL  historical artifacts changed ({len(changed)} paths)")
    else:
        print("PASS  historical artifacts unchanged")
        results["historical artifacts unchanged"] = {"files": len(before)}

    environment = environment_record()
    report = {
        "passed": not failures,
        "failures": failures,
        "checks": results,
        "environment": environment,
        "training_not_started": True,
    }
    SEED_STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    dump_json(SEED_STUDY_ROOT / "preflight_report.json", report)
    dump_json(SEED_STUDY_ROOT / "environment.json", environment)
    print(f"TensorFlow {environment['tensorflow']} | Keras {environment['keras']} | Python {environment['python_version']}")
    print("Devices: " + ", ".join(device["name"] for device in environment["devices"]))
    print("Operation determinism: off. Seeds do not guarantee bit-for-bit reproducibility.")
    if failures:
        print(f"Preflight failed ({len(failures)} checks).")
        raise SystemExit(1)
    print("Preflight passed. No model was trained.")


if __name__ == "__main__":
    main()
