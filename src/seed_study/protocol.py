"""Fixed protocol for the controlled seed study.

Constants here are copied from the original training scripts. The training
seed is not a split seed.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_STUDY_ROOT = PROJECT_ROOT / "results" / "seed_study"
MANIFEST_DIR = SEED_STUDY_ROOT / "manifest"

# Previously verified Keras seed=42 lists. Both copies were identical.
SOURCE_SPLIT_JSONS = (
    PROJECT_ROOT / "results" / "duplicate_check" / "cache" / "trashnet_split.json",
    PROJECT_ROOT / "results" / "final_dataset_audit" / "cache" / "trashnet_split.json",
)

TRAINING_SEEDS = (42, 123, 456)
SPLIT_SEED = 42

CLASS_NAMES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8  # Keras image_dataset_from_directory default
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]
UNFREEZE_LAST_N = 30

EXPECTED_TRAIN_COUNTS = {
    "cardboard": 334,
    "glass": 399,
    "metal": 322,
    "paper": 471,
    "plastic": 393,
    "trash": 103,
}
EXPECTED_VAL_COUNTS = {
    "cardboard": 69,
    "glass": 102,
    "metal": 88,
    "paper": 123,
    "plastic": 89,
    "trash": 34,
}
EXPECTED_TRAIN_N = 2022
EXPECTED_VAL_N = 505

# ModelCheckpoint kwargs from train_mobilenet.py and train_efficientnet.py.
# mode="max" uses numpy.greater, so an equal val_accuracy does not replace
# the saved checkpoint.
CHECKPOINT_KWARGS = {
    "monitor": "val_accuracy",
    "save_best_only": True,
    "mode": "max",
    "verbose": 1,
}

# EarlyStopping kwargs. mode is left at its default "auto", as in the
# original scripts. on_train_begin resets EarlyStopping between phases.
# ModelCheckpoint.best is not reset, so one checkpoint object spans both phases.
EARLY_STOPPING_KWARGS = {
    "monitor": "val_accuracy",
    "patience": 5,
    "restore_best_weights": True,
    "verbose": 1,
}

EXPERIMENTS = {
    "exp001": {
        "directory": "exp001_cnn",
        "title": "EXP-001 CNN Baseline",
        "builder": "src.models.cnn_model.build_cnn_model",
        "preprocess": "divide_by_255",
        "selection": "epoch_10",
        "phase1_epochs": 10,
        "phase1_optimizer": "adam",
        "phase1_learning_rate": 0.001,
        "phase2_epochs": None,
        "callbacks": False,
    },
    "exp002": {
        "directory": "exp002_mobilenet",
        "title": "EXP-002 MobileNetV2",
        "builder": "src.models.mobilenet_model.build_mobilenet_model",
        "preprocess": "mobilenet_v2",
        "selection": "best_checkpoint",
        "phase1_epochs": 10,
        "phase1_optimizer": "Adam",
        "phase1_learning_rate": 1e-3,
        "phase2_epochs": 15,
        "phase2_optimizer": "Adam",
        "phase2_learning_rate": 1e-5,
        "unfreeze_last_n": UNFREEZE_LAST_N,
        "callbacks": True,
    },
    "exp003": {
        "directory": "exp003_efficientnet",
        "title": "EXP-003 EfficientNetB0",
        "builder": "src.models.efficientnet_model.build_efficientnet_model",
        "preprocess": "efficientnet_passthrough",
        "selection": "best_checkpoint",
        "phase1_epochs": 10,
        "phase1_optimizer": "Adam",
        "phase1_learning_rate": 1e-3,
        "phase2_epochs": 15,
        "phase2_optimizer": "Adam",
        "phase2_learning_rate": 1e-5,
        "unfreeze_last_n": UNFREEZE_LAST_N,
        "callbacks": True,
    },
}

PROTECTED_RELATIVE_DIRS = (
    "results/cnn_baseline",
    "results/mobilenet",
    "results/efficientnet",
    "data",
    "docs",
)

VALIDATION_METRIC_NOTE = (
    "VALIDATION metrics on the frozen 505-image TrashNet validation split. "
    "This is not an independent TrashNet test set. EXP-002 and EXP-003 select "
    "the checkpoint with this same validation split. Historical unseeded runs "
    "are excluded from the mean and standard deviation."
)

METHODOLOGICAL_NOTES = (
    "The Train/Validation membership is the original Keras shuffle with seed 42. "
    "Training seeds 42, 123, and 456 are applied only after that membership is loaded.",
    "Validation images stay in frozen manifest order. The original scripts also "
    "buffer-shuffled validation because each image_dataset_from_directory call "
    "defaulted to shuffle=True. Accuracy and macro F1 are computed from the full "
    "prediction vector.",
    "The training shuffle buffer (size 256, reshuffled every epoch) uses the "
    "training seed. The original scripts always used seed 42 for that buffer.",
    "keras.utils.set_random_seed(training_seed) runs before the model is built. "
    "The original scripts did not seed Python, NumPy, or TensorFlow. The "
    "training-seed 42 run is a new controlled run, not the historical checkpoint.",
    "tf.data.Options.deterministic is True for the input pipeline. TensorFlow "
    "operation determinism is off. A seed does not guarantee bit-for-bit "
    "reproducibility across hardware, TensorFlow builds, or non-deterministic ops.",
    "Architectures, preprocessing, image size, batch size, optimizers, learning "
    "rates, epoch limits, dropout, the 30-layer unfreeze, and the original "
    "callback rules are unchanged. No augmentation is added.",
    "EXP-001 trains for 10 epochs and evaluates those weights. It has no "
    "EarlyStopping and no ModelCheckpoint.",
    "EXP-002 and EXP-003 pass the same ModelCheckpoint and EarlyStopping objects "
    "to both phases. Reported metrics are computed from the loaded best checkpoint, "
    "not from the weights left in memory after Phase 2.",
)


def runs():
    return [(name, seed) for name in EXPERIMENTS for seed in TRAINING_SEEDS]


def run_directory(experiment, training_seed):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if isinstance(training_seed, bool) or training_seed not in TRAINING_SEEDS:
        raise ValueError(
            f"Training seed must be one of {TRAINING_SEEDS}, received {training_seed}"
        )
    path = (
        SEED_STUDY_ROOT / EXPERIMENTS[experiment]["directory"] / f"seed_{training_seed}"
    ).resolve()
    assert_safe_output(path)
    return path


def assert_safe_output(path):
    resolved = Path(path).resolve()
    seed_root = SEED_STUDY_ROOT.resolve()
    if resolved != seed_root and seed_root not in resolved.parents:
        raise RuntimeError(f"Output is outside results/seed_study: {resolved}")
    for relative in PROTECTED_RELATIVE_DIRS:
        protected = (PROJECT_ROOT / relative).resolve()
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            raise RuntimeError(f"Output collides with protected path {protected}: {resolved}")
    return resolved
