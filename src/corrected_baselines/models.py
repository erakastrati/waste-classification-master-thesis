"""Original EXP-001 and EXP-002 builders, with a training seed applied first."""

import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.models.cnn_model import build_cnn_model
from src.models.mobilenet_model import build_mobilenet_model
from src.corrected_baselines.protocol import (
    EARLY_STOPPING_PATIENCE,
    LOSS,
    METRICS,
    PHASE1_LEARNING_RATE,
    PHASE2_LEARNING_RATE,
    UNFREEZE_LAST_N,
    assert_safe_output,
)


def build_exp001(training_seed):
    keras.utils.set_random_seed(training_seed)
    return build_cnn_model()


def build_exp002(training_seed):
    keras.utils.set_random_seed(training_seed)
    return build_mobilenet_model()


def compile_exp001(model):
    model.compile(optimizer="adam", loss=LOSS, metrics=METRICS)


def compile_exp002_phase1(model):
    model.compile(
        optimizer=Adam(learning_rate=PHASE1_LEARNING_RATE),
        loss=LOSS,
        metrics=METRICS,
    )


def compile_exp002_phase2(model, base):
    """Original phase-2 rule: unfreeze the last 30 layers, including their BN."""
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LEARNING_RATE),
        loss=LOSS,
        metrics=METRICS,
    )


def exp002_callback_list(checkpoint_path):
    """One checkpoint object and one early-stopping object for both phases.

    EarlyStopping resets at the start of each fit. ModelCheckpoint does not,
    and mode='max' uses a strict improvement, so a tie keeps the earlier file.
    """
    path = assert_safe_output(checkpoint_path)
    if path.name != "best.keras":
        raise ValueError(f"EXP-002 checkpoint must be best.keras, received {path.name}")
    checkpoint = ModelCheckpoint(
        filepath=str(path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    return [checkpoint, early_stopping]
