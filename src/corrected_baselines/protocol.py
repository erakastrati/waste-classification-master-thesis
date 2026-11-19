"""Shared corrected-split comparison for EXP-001 through EXP-004.

EXP-001 and EXP-002 keep their original models and schedules. The manifest,
the three training seeds, and the validation report are shared with the
approved EXP-003 and EXP-004 runs. Nothing here reads the historical seed
study, the 78-image set, or the 600-image set.
"""

from pathlib import Path

from src.retraining.protocol import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_TRAIN_N,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    IMAGE_SIZE,
    MANIFEST_DIR,
    P038_PATHS,
    PROJECT_ROOT,
    TRAIN_MANIFEST,
    TRAINING_SEEDS,
    VAL_MANIFEST,
)

LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]
PHASE1_EPOCHS = 10
PHASE1_LEARNING_RATE = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LEARNING_RATE = 1e-5
UNFREEZE_LAST_N = 30
EARLY_STOPPING_PATIENCE = 5

TRAIN_TXT_SHA256 = "bc768a67584e2be3b758a3390de3d90ffb3c4bb90b27e158e422dcf1d468e549"
VAL_TXT_SHA256 = "b4b344db8ac6e2cc99199ac12754b8fe951ff8c2cf0d36e14e88bb567a022cb5"

COMPARISON_DIR = PROJECT_ROOT / "results/retraining_2026/comparison"
REPORT_NOTE = (
    "VALIDATION accuracy on the corrected 501-image TrashNet split. "
    "This split also selects the EXP-002, EXP-003, and EXP-004 checkpoints, "
    "and it is the only score available for the EXP-001 weights after epoch 10. "
    "It is not independent test accuracy. The historical 78-image set is excluded. "
    "The 600-image real-world set stays held out and is not scored here."
)

# Outputs this package may create. Historical trees are refused.
BASELINE_EXPERIMENTS = ("EXP-001", "EXP-002")
ALL_EXPERIMENTS = ("EXP-001", "EXP-002", "EXP-003", "EXP-004")

EVALUATED_FILE = {
    "EXP-001": "model.keras",
    "EXP-002": "best.keras",
    "EXP-003": "published.keras",
    "EXP-004": "published.keras",
}

SELECTION_RULE = {
    "EXP-001": "weights after epoch 10; no ModelCheckpoint and no EarlyStopping",
    "EXP-002": (
        "one ModelCheckpoint across both phases, monitor=val_accuracy, mode=max; "
        "an equal val_accuracy does not replace the file; evaluation loads best.keras"
    ),
    "EXP-003": "separate phase checkpoints; higher val_accuracy is published; ties keep phase 1",
    "EXP-004": "separate phase checkpoints; higher val_accuracy is published; ties keep phase 1",
}

FORBIDDEN_MARKERS = (
    "data/real_test_dataset",
    "results/real_world_test",
    "results/final_held_out_evaluation",
    "results/v2_table19",
    "results/seed_study",
    "results/cnn_baseline",
    "results/mobilenet/",
    "results/efficientnet/",
    "results/efficientnet_augmented",
    "results/efficientnet_taco",
)

PROTECTED_PREFIXES = (
    "results/cnn_baseline/",
    "results/mobilenet/",
    "results/seed_study/",
    "results/efficientnet/",
    "results/efficientnet_augmented/",
    "results/efficientnet_taco/",
    "results/real_world_test/",
    "results/final_held_out_evaluation/",
    "results/v2_table19/",
    "results/retraining_2026/manifests/",
    "data/",
    "src/retraining/",
    "src/training/",
    "src/models/",
    "src/seed_study/",
)

CORRECTIONS = (
    "Membership is trashnet_seed42_dedup, 2016 training and 501 validation images. "
    "The original Keras split of 2022 and 505 is not used.",
    "keras.utils.set_random_seed(training_seed) runs before the model is built. "
    "The original scripts did not seed Python, NumPy, or TensorFlow.",
    "Each training epoch shuffles the full index list with random.Random(training_seed). "
    "The original loader used a shuffle buffer of 256.",
    "Validation stays in sorted manifest order. The original loader shuffled validation as well.",
    "tf.data.Options.deterministic is True for this input pipeline. "
    "TensorFlow operation determinism stays off, so GPU bit-identity is not promised.",
    "EXP-002 still uses one ModelCheckpoint for both phases and does not freeze "
    "BatchNormalization inside the unfrozen last 30 layers.",
    "EXP-001 still uses optimizer='adam', 10 epochs, and the weights from the final epoch.",
)


def run_directory(experiment, training_seed):
    if experiment not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment}")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"Training seed must be one of {TRAINING_SEEDS}")
    return PROJECT_ROOT / "experiments" / experiment / "runs" / f"seed_{training_seed}"


def assert_safe_output(path):
    resolved = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    relative = resolved.relative_to(root).as_posix()
    allowed = (
        relative.startswith("experiments/EXP-001/")
        or relative.startswith("experiments/EXP-002/")
        or relative.startswith("results/retraining_2026/comparison/")
        or relative.startswith("results/retraining_2026/audit/")
    )
    if not allowed:
        raise RuntimeError(f"Refusing to write outside the corrected baseline outputs: {relative}")
    for prefix in PROTECTED_PREFIXES:
        if relative.startswith(prefix) or prefix.startswith(relative):
            raise RuntimeError(f"Refusing to write into protected path {relative}")
    return resolved
