"""Pre-training checks for the corrected EXP-003 and EXP-004 pipeline.

Loads data and runs one forward pass. Does not call fit.
"""

import hashlib
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

from src.retraining.data import (
    assert_expected_counts,
    class_counts,
    load_batch,
    load_image,
    load_split,
)
from src.retraining.model import build_model, compile_phase1, compile_phase2, make_callbacks
from src.retraining.protocol import (
    BATCH_SIZE,
    CLASS_NAMES,
    EXPECTED_TRAIN_N,
    EXPECTED_VAL_N,
    EXPERIMENTS,
    MANIFEST_DIR,
    P038_PATHS,
    PHASE1_LEARNING_RATE,
    PHASE2_LEARNING_RATE,
    PROJECT_ROOT,
    PROTECTED_PREFIXES,
    TRAINING_SEEDS,
    UNFREEZE_LAST_N,
    run_directory,
)

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


def head_kernel(model):
    for layer in model.layers:
        if layer.__class__.__name__ == "Dense" and layer.units == 128:
            return layer.get_weights()[0]
    raise RuntimeError("128-unit head layer was not found")


def layer_call_kwargs(layer):
    node = layer._inbound_nodes[0]
    arguments = getattr(node, "arguments", None)
    if arguments is not None:
        return dict(arguments.kwargs)
    return dict(getattr(node, "call_kwargs", {}))


def snapshot_protected():
    paths = [
        PROJECT_ROOT / "src/training/train_efficientnet.py",
        PROJECT_ROOT / "src/training/train_efficientnet_augmented.py",
        PROJECT_ROOT / "src/training/train_taco_finetune.py",
        PROJECT_ROOT / "src/models/efficientnet_model.py",
        PROJECT_ROOT / "results/efficientnet/efficientnet.keras",
        PROJECT_ROOT / "results/efficientnet/efficientnet_best.keras",
        PROJECT_ROOT / "results/efficientnet_augmented/efficientnet_augmented.keras",
        PROJECT_ROOT / "results/efficientnet_augmented/efficientnet_augmented_best.keras",
        PROJECT_ROOT / "results/retraining_2026/manifests/trashnet_seed42_dedup/train.txt",
        PROJECT_ROOT / "results/retraining_2026/manifests/trashnet_seed42_dedup/val.txt",
    ]
    return {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths}


def inspect_phase(base, phase):
    batch_norm = [layer for layer in base.layers if isinstance(layer, BatchNormalization)]
    tail = base.layers[-UNFREEZE_LAST_N:]
    prefix = base.layers[:-UNFREEZE_LAST_N]
    return {
        "phase": phase,
        "batch_norm_count": len(batch_norm),
        "batch_norm_trainable": [layer.name for layer in batch_norm if layer.trainable],
        "prefix_trainable": [layer.name for layer in prefix if layer.trainable],
        "tail_frozen_non_bn": [
            layer.name for layer in tail
            if not layer.trainable and not isinstance(layer, BatchNormalization)
        ],
        "tail_trainable_bn": [
            layer.name for layer in tail
            if layer.trainable and isinstance(layer, BatchNormalization)
        ],
        "normalization_trainable": [
            layer.name for layer in base.layers
            if layer.__class__.__name__ == "Normalization" and layer.trainable
        ],
    }


def main():
    before = snapshot_protected()
    train_paths, train_labels, val_paths, val_labels = load_split()
    assert_expected_counts(train_paths, val_paths)
    ok = True
    ok &= check("train count", len(train_paths) == EXPECTED_TRAIN_N, str(len(train_paths)))
    ok &= check("val count", len(val_paths) == EXPECTED_VAL_N, str(len(val_paths)))
    ok &= check(
        "class counts",
        class_counts(train_paths) | {"val_" + key: value for key, value in class_counts(val_paths).items()}
        == {**class_counts(train_paths), **{"val_" + key: value for key, value in class_counts(val_paths).items()}},
        f"train {class_counts(train_paths)} val {class_counts(val_paths)}",
    )
    ok &= check(
        "label range",
        set(train_labels) | set(val_labels) == set(range(6))
        and all(CLASS_NAMES[label] == path.split("/")[2] for path, label in zip(train_paths, train_labels))
        and all(CLASS_NAMES[label] == path.split("/")[2] for path, label in zip(val_paths, val_labels)),
        "indices 0-5 match parent folders",
    )
    overlap = sorted(set(train_paths) & set(val_paths))
    ok &= check("path overlap", overlap == [], str(len(overlap)))
    exclusions = (MANIFEST_DIR / "exclusions.csv").read_text().splitlines()[1:]
    excluded = [line.split(",")[0] for line in exclusions if line.strip()]
    present = sorted(set(excluded) & (set(train_paths) | set(val_paths)))
    ok &= check("excluded files absent", present == [], str(present))
    p038_train = [path for path in P038_PATHS if path in train_paths]
    p038_val = [path for path in P038_PATHS if path in val_paths]
    ok &= check("P038 stays in training", p038_train == list(P038_PATHS) and p038_val == [], str(p038_val))

    hashes = {}
    for relative in train_paths + val_paths:
        file_hash = sha256_file(PROJECT_ROOT / relative)
        hashes.setdefault(file_hash, []).append(relative)
    shared = []
    train_set = set(train_paths)
    val_set = set(val_paths)
    for members in hashes.values():
        if any(path in train_set for path in members) and any(path in val_set for path in members):
            shared.append(members)
    duplicated_inside = [members for members in hashes.values() if len(members) > 1]
    ok &= check("SHA-256 overlap", shared == [], str(shared))
    ok &= check("no repeated file hash", duplicated_inside == [], str(duplicated_inside))

    same_lists = []
    for experiment in EXPERIMENTS:
        paths, labels, val_p, val_l = load_split()
        same_lists.append((experiment, paths == train_paths, val_p == val_paths, labels == train_labels, val_l == val_labels))
    ok &= check(
        "same split for both experiments",
        all(item[1] and item[2] and item[3] and item[4] for item in same_lists),
        "both experiments read trashnet_seed42_dedup",
    )

    train_batch_x, train_batch_y = load_batch(train_paths[:BATCH_SIZE], train_labels[:BATCH_SIZE])
    val_batch_x, val_batch_y = load_batch(val_paths[:BATCH_SIZE], val_labels[:BATCH_SIZE])
    partial_x, partial_y = load_batch(val_paths[-21:], val_labels[-21:])
    ok &= check(
        "batch shapes",
        train_batch_x.shape == (32, 224, 224, 3)
        and val_batch_x.shape == (32, 224, 224, 3)
        and partial_x.shape == (21, 224, 224, 3)
        and train_batch_x.dtype == np.float32
        and train_batch_y.dtype == np.int32,
        f"train {train_batch_x.shape} {train_batch_x.dtype} val {val_batch_x.shape} partial {partial_x.shape}",
    )
    ok &= check(
        "pixels stay in 0-255",
        float(train_batch_x.min()) >= 0
        and float(train_batch_x.max()) <= 255
        and float(train_batch_x.max()) > 1
        and float(val_batch_x.max()) > 1,
        f"train min {float(train_batch_x.min()):.3f} max {float(train_batch_x.max()):.3f}",
    )
    per_class = {}
    for name in CLASS_NAMES:
        relative = next(path for path in train_paths if path.split("/")[2] == name)
        image = load_image(str(PROJECT_ROOT / relative))
        per_class[name] = list(image.shape) + [image.dtype.name]
    ok &= check(
        "one image per class",
        all(shape[:3] == [224, 224, 3] and shape[3] == "float32" for shape in per_class.values()),
        str(per_class),
    )

    planned = []
    for experiment in EXPERIMENTS:
        for seed in TRAINING_SEEDS:
            directory = run_directory(experiment, seed)
            relative = directory.relative_to(PROJECT_ROOT).as_posix() + "/"
            planned.append(relative)
    slots_ok = (
        len(set(planned)) == 6
        and not any((PROJECT_ROOT / path).exists() for path in planned)
        and not any(path.startswith(prefix) for path in planned for prefix in PROTECTED_PREFIXES)
    )
    ok &= check("six run slots", slots_ok, ", ".join(planned))

    for experiment in EXPERIMENTS:
        for phase in (1, 2):
            directory = run_directory(experiment, 42)
            checkpoint, early = make_callbacks(directory, phase)
            other, _early2 = make_callbacks(directory, 2 if phase == 1 else 1)
            tie_keeps_current = bool(checkpoint.monitor_op(np.float64(0.5), np.float64(0.5))) is False
            improves = bool(checkpoint.monitor_op(np.float64(0.6), np.float64(0.5)))
            ok &= check(
                f"{experiment} phase {phase} checkpoint",
                checkpoint.monitor == "val_accuracy"
                and checkpoint.save_best_only
                and checkpoint is not other
                and improves
                and tie_keeps_current
                and early.patience == 5
                and early.restore_best_weights
                and f"phase{phase}_best.keras" in checkpoint.filepath,
                checkpoint.filepath,
            )

    exp003, base003 = build_model("EXP-003", 42)
    compile_phase1(exp003, base003)
    aug_names = [layer.name for layer in exp003.layers if layer.__class__.__name__.startswith("Random")]
    base_kwargs = layer_call_kwargs(exp003.get_layer("efficientnetb0"))
    ok &= check("EXP-003 has no augmentation", aug_names == [], str(aug_names))
    ok &= check(
        "base call has no training keyword",
        "training" not in base_kwargs,
        str(base_kwargs),
    )
    phase1 = inspect_phase(base003, 1)
    ok &= check(
        "EXP-003 phase 1 freezes the base",
        phase1["batch_norm_trainable"] == []
        and phase1["prefix_trainable"] == []
        and phase1["batch_norm_count"] > 0
        and learning_rate_is(exp003, PHASE1_LEARNING_RATE),
        f"bn {phase1['batch_norm_count']} lr {learning_rate(exp003)}",
    )
    inference = exp003(train_batch_x, training=False).numpy()
    inference_again = exp003(train_batch_x, training=False).numpy()
    ok &= check(
        "EXP-003 inference forward pass",
        inference.shape == (32, 6)
        and np.allclose(inference.sum(axis=1), 1, atol=1e-5)
        and np.allclose(inference, inference_again),
        f"shape {inference.shape}",
    )
    kernel_42 = head_kernel(exp003).copy()
    compile_phase2(exp003, base003)
    phase2 = inspect_phase(base003, 2)
    ok &= check(
        "EXP-003 phase 2 freezes every batch norm",
        phase2["batch_norm_trainable"] == []
        and phase2["tail_trainable_bn"] == []
        and phase2["prefix_trainable"] == []
        and phase2["normalization_trainable"] == []
        and learning_rate_is(exp003, PHASE2_LEARNING_RATE),
        f"trainable bn {phase2['batch_norm_trainable']} prefix {phase2['prefix_trainable'][:5]}",
    )
    del exp003, base003

    exp004, base004 = build_model("EXP-004", 42)
    compile_phase1(exp004, base004)
    contrast = exp004.get_layer("random_contrast")
    rotation = exp004.get_layer("random_rotation")
    zoom = exp004.get_layer("random_zoom")
    flip = exp004.get_layer("random_flip")
    probe = tf.keras.Model(exp004.input, contrast.output)
    augmented = probe(train_batch_x, training=True).numpy()
    identity = probe(train_batch_x, training=False).numpy()
    ok &= check(
        "EXP-004 augmentation settings",
        flip.mode == "horizontal"
        and tuple(rotation.factor) == (-0.15, 0.15)
        and zoom.height_factor == 0.15
        and zoom.width_factor is None
        and tuple(contrast.factor) == (0.0, 0.2)
        and flip.seed == 42
        and rotation.seed == 42
        and zoom.seed == 42
        and contrast.seed == 42,
        f"contrast {tuple(contrast.factor)} rotation {tuple(rotation.factor)}",
    )
    ok &= check(
        "EXP-004 augmentation is training-only",
        np.allclose(identity, train_batch_x)
        and not np.allclose(augmented, train_batch_x),
        f"inference max abs diff {float(np.max(np.abs(identity - train_batch_x))):.6f}",
    )
    compile_phase2(exp004, base004)
    phase2_aug = inspect_phase(base004, 2)
    ok &= check(
        "EXP-004 phase 2 uses the same batch-norm freeze",
        phase2_aug["batch_norm_trainable"] == [] and phase2_aug["prefix_trainable"] == [],
        f"bn trainable {phase2_aug['batch_norm_trainable']}",
    )
    del exp004, base004

    exp003_b, _base_b = build_model("EXP-003", 123)
    kernel_123 = head_kernel(exp003_b)
    ok &= check(
        "seeds 42 and 123 change initialization",
        kernel_42.shape == kernel_123.shape and not np.allclose(kernel_42, kernel_123),
        f"kernel {kernel_42.shape}",
    )
    del exp003_b

    source = (PROJECT_ROOT / "src/training/train_taco_finetune.py").read_text()
    ok &= check(
        "EXP-005 still loads the historical checkpoint",
        'BASE_MODEL_PATH   = "results/efficientnet_augmented/efficientnet_augmented.keras"' in source,
        "train_taco_finetune.py line unchanged",
    )
    package = "\n".join(
        path.read_text()
        for path in (PROJECT_ROOT / "src/retraining").glob("*.py")
        if path.name != "preflight.py"
    )
    ok &= check(
        "pipeline does not call validation_split",
        "image_dataset_from_directory(" not in package and "validation_split=" not in package,
        "src/retraining has neither call",
    )
    after = snapshot_protected()
    changed = [path for path, digest in before.items() if after[path] != digest]
    ok &= check("historical files unchanged", changed == [], str(changed))

    report = {
        "passed": ok and all(item["passed"] for item in CHECKS),
        "checks": CHECKS,
        "train_class_counts": class_counts(train_paths),
        "val_class_counts": class_counts(val_paths),
        "planned_runs": planned,
    }
    out = PROJECT_ROOT / "results/retraining_2026/audit/pretraining_validation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out.relative_to(PROJECT_ROOT))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
