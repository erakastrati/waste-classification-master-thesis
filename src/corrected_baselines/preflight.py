"""Pre-training checks for corrected EXP-001 and EXP-002.

Loads the approved manifest and runs forward passes. Does not call fit and
does not score the 600-image set.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout

from src.corrected_baselines.data import (
    cnn_preprocess,
    decode_resized,
    epoch_orders,
    load_batch,
    load_split,
    mobilenet_preprocess,
    steps_per_epoch,
    training_dataset,
)
from src.corrected_baselines.metrics import sample_mean_std, validation_metrics
from src.corrected_baselines.models import (
    build_exp001,
    build_exp002,
    compile_exp001,
    compile_exp002_phase1,
    compile_exp002_phase2,
    exp002_callback_list,
)
from src.corrected_baselines.protocol import (
    ALL_EXPERIMENTS,
    BATCH_SIZE,
    CLASS_NAMES,
    COMPARISON_DIR,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_TRAIN_N,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    MANIFEST_DIR,
    P038_PATHS,
    PHASE1_EPOCHS,
    PHASE1_LEARNING_RATE,
    PHASE2_EPOCHS,
    PHASE2_LEARNING_RATE,
    PROJECT_ROOT,
    TRAIN_TXT_SHA256,
    TRAINING_SEEDS,
    UNFREEZE_LAST_N,
    VAL_TXT_SHA256,
    run_directory,
)
from src.corrected_baselines.report import (
    aggregate,
    assert_validation_record,
    load_completed_runs,
    package_text,
    score_selected_checkpoint,
    write_summary,
)
from src.corrected_baselines.train import run, trainer_uses_shared_exp002_callbacks
from src.retraining.data import class_counts

CHECKS = []


def check(name, ok, detail):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return bool(ok)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def learning_rate(model):
    value = model.optimizer.learning_rate
    return float(value.numpy() if hasattr(value, "numpy") else value)


def learning_rate_is(model, expected):
    return abs(learning_rate(model) - expected) < 1e-6


def optimizer_defaults_ok(model):
    optimizer = model.optimizer
    beta_1 = float(optimizer.beta_1.numpy() if hasattr(optimizer.beta_1, "numpy") else optimizer.beta_1)
    beta_2 = float(optimizer.beta_2.numpy() if hasattr(optimizer.beta_2, "numpy") else optimizer.beta_2)
    epsilon = float(optimizer.epsilon)
    return (
        abs(beta_1 - 0.9) < 1e-6
        and abs(beta_2 - 0.999) < 1e-6
        and abs(epsilon - 1e-7) < 1e-12
        and bool(optimizer.amsgrad) is False
    )


def head_kernel(model):
    for layer in model.layers:
        if isinstance(layer, Dense) and layer.units == 128:
            return layer.get_weights()[0]
    raise RuntimeError("128-unit head was not found")


def first_conv_kernel(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) and layer.get_weights():
            return layer.get_weights()[0]
    raise RuntimeError("Conv2D kernel was not found")


def layer_call_kwargs(layer):
    node = layer._inbound_nodes[0]
    arguments = getattr(node, "arguments", None)
    if arguments is not None:
        return dict(arguments.kwargs)
    return dict(getattr(node, "call_kwargs", {}))


def snapshot(paths):
    recorded = {}
    for path in paths:
        recorded[path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)
    return recorded


def protected_paths():
    root = PROJECT_ROOT
    paths = [
        root / "src/training/train_cnn.py",
        root / "src/training/train_mobilenet.py",
        root / "src/training/train_efficientnet.py",
        root / "src/training/train_efficientnet_augmented.py",
        root / "src/training/train_taco_finetune.py",
        root / "src/models/cnn_model.py",
        root / "src/models/mobilenet_model.py",
        root / "src/models/efficientnet_model.py",
        root / "results/cnn_baseline/cnn_baseline.keras",
        root / "results/mobilenet/mobilenet.keras",
        root / "results/mobilenet/mobilenet_best.keras",
        root / "results/efficientnet/efficientnet.keras",
        root / "results/efficientnet/efficientnet_best.keras",
        root / "results/efficientnet_augmented/efficientnet_augmented.keras",
        root / "results/efficientnet_augmented/efficientnet_augmented_best.keras",
        root / "results/seed_study/summary.json",
        root / "results/retraining_2026/manifests/trashnet_seed42_dedup/train.txt",
        root / "results/retraining_2026/manifests/trashnet_seed42_dedup/val.txt",
    ]
    paths.extend(sorted((root / "src/retraining").glob("*.py")))
    return paths


def directory_listing(relative):
    path = PROJECT_ROOT / relative
    if not path.exists():
        return []
    return sorted(item.relative_to(PROJECT_ROOT).as_posix() for item in path.rglob("*") if item.is_file())


def fake_record(experiment, seed, accuracy):
    return {
        "comparison": "corrected_trashnet_seed42_dedup",
        "metric_set": "VALIDATION",
        "independent_test_set": False,
        "n_validation": EXPECTED_VAL_N,
        "held_out_600_used": False,
        "historical_78_used": False,
        "historical_seed_study_included": False,
        "experiment": experiment,
        "training_seed": seed,
        "accuracy": accuracy,
        "macro_f1": accuracy,
        "per_class": {
            name: {
                "precision": accuracy,
                "recall": accuracy,
                "f1": accuracy,
                "support": support,
            }
            for name, support in EXPECTED_VAL_COUNTS.items()
        },
    }


def main():
    before = snapshot(protected_paths())
    held_out_before = directory_listing("data/real_test_dataset")
    historical_78_before = directory_listing("results/real_world_test")
    train_paths, train_labels, val_paths, val_labels = load_split()
    ok = True

    ok &= check(
        "manifest hashes",
        sha256_file(MANIFEST_DIR / "train.txt") == TRAIN_TXT_SHA256
        and sha256_file(MANIFEST_DIR / "val.txt") == VAL_TXT_SHA256,
        "trashnet_seed42_dedup train.txt and val.txt",
    )
    ok &= check("train count", len(train_paths) == EXPECTED_TRAIN_N, str(len(train_paths)))
    ok &= check("validation count", len(val_paths) == EXPECTED_VAL_N, str(len(val_paths)))
    ok &= check(
        "class counts",
        class_counts(train_paths) == EXPECTED_TRAIN_COUNTS
        and class_counts(val_paths) == EXPECTED_VAL_COUNTS,
        f"train {class_counts(train_paths)} val {class_counts(val_paths)}",
    )
    manifest_train = [
        line.strip() for line in (MANIFEST_DIR / "train.txt").read_text().splitlines() if line.strip()
    ]
    manifest_val = [
        line.strip() for line in (MANIFEST_DIR / "val.txt").read_text().splitlines() if line.strip()
    ]
    ok &= check(
        "reader matches the manifest files",
        train_paths == manifest_train and val_paths == manifest_val,
        "same path order as train.txt and val.txt",
    )
    ok &= check(
        "labels follow the six-class folders",
        all(CLASS_NAMES[label] == path.split("/")[2] for path, label in zip(train_paths, train_labels))
        and all(CLASS_NAMES[label] == path.split("/")[2] for path, label in zip(val_paths, val_labels)),
        "cardboard 0 through trash 5",
    )
    ok &= check("path overlap", sorted(set(train_paths) & set(val_paths)) == [], "0")

    with (MANIFEST_DIR / "exclusions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    excluded = [row["path"] for row in rows]
    retained = [row["retained_path"] for row in rows]
    inventory = set(train_paths) | set(val_paths)
    ok &= check(
        "excluded files absent and retained files present",
        len(rows) == 10
        and set(excluded).isdisjoint(inventory)
        and set(retained) <= inventory,
        f"{len(rows)} exclusions",
    )
    ok &= check(
        "P038 stays in training",
        all(path in train_paths for path in P038_PATHS)
        and all(path not in val_paths for path in P038_PATHS),
        "metal149.jpg and metal9.jpg",
    )

    hashes = {}
    for relative in train_paths + val_paths:
        file_hash = sha256_file(PROJECT_ROOT / relative)
        hashes.setdefault(file_hash, []).append(relative)
    cross = []
    train_set = set(train_paths)
    val_set = set(val_paths)
    for members in hashes.values():
        if any(path in train_set for path in members) and any(path in val_set for path in members):
            cross.append(members)
    repeated = [members for members in hashes.values() if len(members) > 1]
    ok &= check("SHA-256 cross-split overlap", cross == [], str(cross))
    ok &= check("each file hash once", repeated == [], str(repeated))

    raw = decode_resized(str(PROJECT_ROOT / train_paths[0]))
    cnn_image = cnn_preprocess(raw).numpy()
    mobile_image = mobilenet_preprocess(raw).numpy()
    raw_image = raw.numpy()
    ok &= check(
        "EXP-001 preprocessing is divide by 255",
        cnn_image.shape == (224, 224, 3)
        and np.allclose(cnn_image, raw_image / 255.0)
        and float(cnn_image.min()) >= 0
        and float(cnn_image.max()) <= 1,
        f"min {float(cnn_image.min()):.4f} max {float(cnn_image.max()):.4f}",
    )
    ok &= check(
        "EXP-002 preprocessing is MobileNet tf mode",
        mobile_image.shape == (224, 224, 3)
        and np.allclose(mobile_image, raw_image / 127.5 - 1.0)
        and float(mobile_image.min()) >= -1
        and float(mobile_image.max()) <= 1
        and not np.allclose(mobile_image, cnn_image),
        f"min {float(mobile_image.min()):.4f} max {float(mobile_image.max()):.4f}",
    )
    cnn_batch, cnn_labels = load_batch(train_paths[:BATCH_SIZE], train_labels[:BATCH_SIZE], cnn_preprocess)
    mobile_batch, mobile_labels = load_batch(val_paths[:BATCH_SIZE], val_labels[:BATCH_SIZE], mobilenet_preprocess)
    partial, partial_labels = load_batch(val_paths[-21:], val_labels[-21:], cnn_preprocess)
    ok &= check(
        "batch shapes",
        cnn_batch.shape == (32, 224, 224, 3)
        and mobile_batch.shape == (32, 224, 224, 3)
        and partial.shape == (21, 224, 224, 3)
        and cnn_batch.dtype == np.float32
        and list(cnn_labels) == train_labels[:32]
        and list(mobile_labels) == val_labels[:32]
        and list(partial_labels) == val_labels[-21:],
        f"train {cnn_batch.shape} partial {partial.shape}",
    )
    ok &= check(
        "steps per epoch",
        steps_per_epoch(EXPECTED_TRAIN_N) == 63 and steps_per_epoch(EXPECTED_VAL_N) == 16,
        "63 training, 16 validation",
    )
    orders = epoch_orders(len(train_paths), 42)
    first = next(orders)
    second = next(orders)
    repeated_first = next(epoch_orders(len(train_paths), 42))
    other_seed = next(epoch_orders(len(train_paths), 123))
    dataset_batch = next(iter(training_dataset(
        train_paths, train_labels, 42, cnn_preprocess
    ).take(1)))
    ok &= check(
        "training shuffle is a full seeded permutation",
        sorted(first) == list(range(len(train_paths)))
        and first == repeated_first
        and first != second
        and first != other_seed
        and list(dataset_batch[1].numpy()) == [train_labels[index] for index in first[:32]],
        f"first index {first[0]}",
    )

    perfect = validation_metrics(val_labels, val_labels)
    ok &= check(
        "per-class support matches the validation manifest",
        perfect["n_validation"] == 501
        and perfect["accuracy"] == 1
        and {name: item["support"] for name, item in perfect["per_class"].items()} == EXPECTED_VAL_COUNTS
        and perfect["metric_set"] == "VALIDATION"
        and perfect["independent_test_set"] is False,
        "supports " + ", ".join(f"{name} {EXPECTED_VAL_COUNTS[name]}" for name in CLASS_NAMES),
    )

    planned = []
    for experiment in ("EXP-001", "EXP-002"):
        for seed in TRAINING_SEEDS:
            directory = run_directory(experiment, seed)
            planned.append(directory.relative_to(PROJECT_ROOT).as_posix() + "/")
    ok &= check(
        "six baseline run slots are empty",
        len(set(planned)) == 6 and not any((PROJECT_ROOT / path).exists() for path in planned),
        ", ".join(planned),
    )
    ok &= check(
        "EXP-003 and EXP-004 slots were not created",
        not any(run_directory(experiment, seed).exists() for experiment in ("EXP-003", "EXP-004") for seed in TRAINING_SEEDS),
        "experiments/EXP-003 and experiments/EXP-004",
    )

    cnn = build_exp001(42)
    compile_exp001(cnn)
    cnn_classes = [layer.__class__.__name__ for layer in cnn.layers]
    cnn_filters = [layer.filters for layer in cnn.layers if isinstance(layer, Conv2D)]
    dropout = [layer.rate for layer in cnn.layers if isinstance(layer, Dropout)]
    probabilities = cnn(cnn_batch, training=False).numpy()
    probabilities_again = cnn(cnn_batch, training=False).numpy()
    ok &= check(
        "EXP-001 architecture and 10-epoch Adam schedule",
        cnn_classes == [
            "Conv2D", "MaxPooling2D", "Conv2D", "MaxPooling2D",
            "Conv2D", "MaxPooling2D", "GlobalAveragePooling2D", "Dense", "Dropout", "Dense",
        ]
        and tuple(cnn.input_shape) == (None, 224, 224, 3)
        and cnn_filters == [32, 64, 128]
        and dropout == [0.3]
        and not any(isinstance(layer, BatchNormalization) for layer in cnn.layers)
        and learning_rate_is(cnn, PHASE1_LEARNING_RATE)
        and optimizer_defaults_ok(cnn)
        and PHASE1_EPOCHS == 10,
        f"lr {learning_rate(cnn)}",
    )
    ok &= check(
        "EXP-001 inference forward pass",
        probabilities.shape == (32, 6)
        and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        and np.allclose(probabilities, probabilities_again),
        f"shape {probabilities.shape}",
    )
    cnn_kernel_42 = head_kernel(cnn).copy()
    conv_kernel_42 = first_conv_kernel(cnn).copy()

    mobile, base = build_exp002(42)
    compile_exp002_phase1(mobile)
    base_layer = mobile.get_layer(base.name)
    phase1_bn = [layer.name for layer in base.layers if isinstance(layer, BatchNormalization) and layer.trainable]
    random_layers = [layer.name for layer in mobile.layers if layer.__class__.__name__.startswith("Random")]
    mobile_probabilities = mobile(mobile_batch, training=False).numpy()
    ok &= check(
        "EXP-002 phase 1 freezes the ImageNet base",
        base.trainable is False
        and phase1_bn == []
        and random_layers == []
        and learning_rate_is(mobile, PHASE1_LEARNING_RATE)
        and optimizer_defaults_ok(mobile)
        and head_kernel(mobile).shape[1] == 128
        and mobile_probabilities.shape == (32, 6)
        and np.allclose(mobile_probabilities.sum(axis=1), 1, atol=1e-5),
        f"base layers {len(base.layers)} lr {learning_rate(mobile)}",
    )
    ok &= check(
        "EXP-002 base call keeps the original training=False argument",
        layer_call_kwargs(base_layer).get("training") is False,
        str(layer_call_kwargs(base_layer)),
    )
    imagenet_kernel = first_conv_kernel(base).copy()
    mobile_head_42 = head_kernel(mobile).copy()
    compile_exp002_phase2(mobile, base)
    tail = base.layers[-UNFREEZE_LAST_N:]
    prefix = base.layers[:-UNFREEZE_LAST_N]
    tail_bn = [layer for layer in tail if isinstance(layer, BatchNormalization)]
    ok &= check(
        "EXP-002 phase 2 unfreezes the last 30 layers and their batch norm",
        all(not layer.trainable for layer in prefix)
        and any(layer.trainable for layer in tail)
        and len(tail_bn) > 0
        and all(layer.trainable for layer in tail_bn)
        and learning_rate_is(mobile, PHASE2_LEARNING_RATE)
        and PHASE2_EPOCHS == 15,
        "trainable tail BN: " + ", ".join(layer.name for layer in tail_bn),
    )
    callbacks = exp002_callback_list(run_directory("EXP-002", 42) / "best.keras")
    checkpoint, early = callbacks
    ok &= check(
        "EXP-002 uses one shared checkpoint across both phases",
        trainer_uses_shared_exp002_callbacks()
        and checkpoint.monitor == "val_accuracy"
        and checkpoint.save_best_only
        and bool(checkpoint.monitor_op(np.float64(0.5), np.float64(0.5))) is False
        and bool(checkpoint.monitor_op(np.float64(0.6), np.float64(0.5)))
        and early.patience == 5
        and early.restore_best_weights
        and early.monitor == "val_accuracy"
        and Path(checkpoint.filepath).name == "best.keras"
        and not run_directory("EXP-002", 42).exists(),
        checkpoint.filepath,
    )

    cnn_123 = build_exp001(123)
    mobile_123, base_123 = build_exp002(123)
    cnn_456 = build_exp001(456)
    mobile_456, base_456 = build_exp002(456)
    ok &= check(
        "seeds 42, 123 and 456 change EXP-001 initialization",
        not np.allclose(cnn_kernel_42, head_kernel(cnn_123))
        and not np.allclose(cnn_kernel_42, head_kernel(cnn_456))
        and not np.allclose(conv_kernel_42, first_conv_kernel(cnn_123)),
        f"head {cnn_kernel_42.shape}",
    )
    ok &= check(
        "seeds change the EXP-002 head and not the ImageNet weights",
        not np.allclose(mobile_head_42, head_kernel(mobile_123))
        and not np.allclose(mobile_head_42, head_kernel(mobile_456))
        and np.allclose(imagenet_kernel, first_conv_kernel(base_123))
        and np.allclose(imagenet_kernel, first_conv_kernel(base_456)),
        f"head {mobile_head_42.shape}",
    )
    del cnn, cnn_123, cnn_456, mobile, mobile_123, mobile_456, base, base_123, base_456

    mean, std = sample_mean_std([0.5, 0.6, 0.7])
    records = {
        experiment: [fake_record(experiment, seed, accuracy) for seed, accuracy in zip(TRAINING_SEEDS, (0.5, 0.6, 0.7))]
        for experiment in ALL_EXPERIMENTS
    }
    summary = aggregate(records)
    exp001_summary = summary["experiments"]["EXP-001"]
    ok &= check(
        "reporting protocol uses the sample standard deviation",
        abs(mean - 0.6) < 1e-12
        and abs(std - 0.1) < 1e-12
        and exp001_summary["accuracy_mean_pm_std"] == "0.6000 ± 0.1000"
        and set(exp001_summary["per_class"]) == set(CLASS_NAMES)
        and summary["historical_seed_study_included"] is False
        and summary["historical_78_used"] is False
        and summary["held_out_600_used"] is False
        and summary["metric_set"] == "VALIDATION",
        exp001_summary["accuracy_mean_pm_std"],
    )
    rejected = []
    invalid_records = [
        fake_record("EXP-001", 42, 0.5),
        fake_record("EXP-001", 42, 0.5),
        fake_record("EXP-001", 42, 0.5),
        fake_record("EXP-001", 42, 0.5),
    ]
    invalid_records[0]["n_validation"] = 78
    invalid_records[1]["n_validation"] = 600
    invalid_records[2]["historical_seed_study_included"] = True
    invalid_records[3]["manifest"] = "data/real_test_dataset"
    for record in invalid_records:
        try:
            assert_validation_record(record)
        except RuntimeError:
            rejected.append(True)
        else:
            rejected.append(False)
    ok &= check(
        "reporter rejects 78-image, 600-image and historical inputs",
        rejected == [True, True, True, True],
        str(rejected),
    )
    missing_summary = False
    try:
        write_summary()
    except FileNotFoundError:
        missing_summary = True
    missing_score = False
    try:
        score_selected_checkpoint("EXP-003", 42)
    except FileNotFoundError:
        missing_score = True
    missing_runs = False
    try:
        load_completed_runs()
    except FileNotFoundError:
        missing_runs = True
    ok &= check(
        "report is not written before the 12 runs exist",
        missing_summary and missing_score and missing_runs and not COMPARISON_DIR.exists(),
        "summary, checkpoint scoring and run loading all refused",
    )

    training_refused = False
    try:
        run("EXP-001", 42)
    except RuntimeError:
        training_refused = True
    train_source = (PROJECT_ROOT / "src/corrected_baselines/train.py").read_text()
    model_source = (PROJECT_ROOT / "src/corrected_baselines/models.py").read_text()
    historical_cnn = (PROJECT_ROOT / "src/training/train_cnn.py").read_text()
    historical_mobile = (PROJECT_ROOT / "src/training/train_mobilenet.py").read_text()
    package = "\n".join(
        path.read_text()
        for path in (PROJECT_ROOT / "src/corrected_baselines").glob("*.py")
        if path.name != "preflight.py"
    )
    ok &= check(
        "full training entry point refuses to start",
        training_refused
        and "run(" not in train_source.split("if __name__")[-1]
        and not run_directory("EXP-001", 42).exists(),
        "approve_training remains false",
    )
    ok &= check(
        "baseline package does not redraw a split",
        "image_dataset_from_directory(" not in package and "validation_split=" not in package,
        "no Keras directory split in src/corrected_baselines",
    )
    ok &= check(
        "EXP-002 recipe does not import the EfficientNet batch-norm freeze",
        "BatchNormalization" not in model_source
        and "compile_phase2" not in package
        and "src.retraining.train" not in package
        and "src.retraining.model" not in package,
        "phase 2 only sets layer.trainable on the prefix",
    )
    ok &= check(
        "historical EXP-001 and EXP-002 scripts are still the original recipes",
        "callbacks" not in historical_cnn
        and "epochs=10" in historical_cnn
        and historical_cnn.count("ModelCheckpoint") == 0
        and "base_model.trainable = True" in historical_mobile
        and "base_model.layers[:-30]" in historical_mobile
        and "BatchNormalization" not in historical_mobile,
        "train_cnn.py and train_mobilenet.py",
    )

    created = [
        path for path in (
            [run_directory(experiment, seed) for experiment in ALL_EXPERIMENTS for seed in TRAINING_SEEDS]
            + [COMPARISON_DIR]
        )
        if path.exists()
    ]
    ok &= check("no run or comparison directory was created", created == [], str(created))
    ok &= check(
        "held-out and historical photograph sets were not touched",
        directory_listing("data/real_test_dataset") == held_out_before
        and directory_listing("results/real_world_test") == historical_78_before,
        f"600-set files {len(held_out_before)}; 78-set files {len(historical_78_before)}",
    )
    after = snapshot(protected_paths())
    changed = [path for path, digest in before.items() if after[path] != digest]
    ok &= check("historical and EXP-003 pipeline files unchanged", changed == [], str(changed))

    report = {
        "passed": ok and all(item["passed"] for item in CHECKS),
        "checks": CHECKS,
        "train_class_counts": class_counts(train_paths),
        "val_class_counts": class_counts(val_paths),
        "planned_baseline_runs": planned,
        "training_started": False,
    }
    destination = PROJECT_ROOT / "results/retraining_2026/audit/EXP-001_EXP-002_preflight.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", destination.relative_to(PROJECT_ROOT))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
