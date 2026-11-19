"""Constants for the approved corrected EXP-003 and EXP-004 protocol."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "results/retraining_2026/manifests/trashnet_seed42_dedup"
TRAIN_MANIFEST = MANIFEST_DIR / "train.txt"
VAL_MANIFEST = MANIFEST_DIR / "val.txt"

CLASS_NAMES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAINING_SEEDS = (42, 123, 456)
EXPERIMENTS = ("EXP-003", "EXP-004")

PHASE1_EPOCHS = 10
PHASE1_LEARNING_RATE = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LEARNING_RATE = 1e-5
UNFREEZE_LAST_N = 30
EARLY_STOPPING_PATIENCE = 5

ADAM_KWARGS = {
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
    "amsgrad": False,
}

EXPECTED_TRAIN_COUNTS = {
    "cardboard": 323,
    "glass": 400,
    "metal": 327,
    "paper": 471,
    "plastic": 385,
    "trash": 110,
}
EXPECTED_VAL_COUNTS = {
    "cardboard": 80,
    "glass": 100,
    "metal": 81,
    "paper": 117,
    "plastic": 96,
    "trash": 27,
}
EXPECTED_TRAIN_N = 2016
EXPECTED_VAL_N = 501

P038_PATHS = (
    "data/trashNet-dataset/metal/metal149.jpg",
    "data/trashNet-dataset/metal/metal9.jpg",
)

LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]

PROTECTED_PREFIXES = (
    "results/efficientnet/",
    "results/efficientnet_augmented/",
    "results/efficientnet_taco/",
    "data/trashNet-dataset/",
)


def run_directory(experiment, training_seed):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment}")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"Training seed must be one of {TRAINING_SEEDS}")
    return PROJECT_ROOT / "experiments" / experiment / "runs" / f"seed_{training_seed}"
