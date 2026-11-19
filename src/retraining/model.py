"""Shared EfficientNetB0 builder for corrected EXP-003 and EXP-004."""

import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from tensorflow.keras.optimizers import Adam

from src.retraining.protocol import (
    ADAM_KWARGS,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE,
    PHASE1_LEARNING_RATE,
    PHASE2_LEARNING_RATE,
    UNFREEZE_LAST_N,
)


def build_model(experiment, training_seed):
    """Build a new ImageNet EfficientNetB0. Call set_random_seed before the head exists."""
    if experiment not in ("EXP-003", "EXP-004"):
        raise ValueError(experiment)
    keras.utils.set_random_seed(training_seed)
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = inputs
    if experiment == "EXP-004":
        x = RandomFlip("horizontal", seed=training_seed)(x)
        x = RandomRotation(0.15, seed=training_seed)(x)
        x = RandomZoom(0.15, seed=training_seed)(x)
        x = RandomContrast(0.2, seed=training_seed)(x)
    x = base(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(6, activation="softmax")(x)
    model = Model(inputs, outputs, name=experiment.lower().replace("-", "_"))
    return model, base


def _adam(learning_rate):
    return Adam(learning_rate=learning_rate, **ADAM_KWARGS)


def compile_phase1(model, base):
    base.trainable = False
    model.compile(
        optimizer=_adam(PHASE1_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def compile_phase2(model, base):
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=_adam(PHASE2_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(run_dir, phase):
    checkpoint = ModelCheckpoint(
        filepath=str(run_dir / f"phase{phase}_best.keras"),
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
    return checkpoint, early_stopping


class EpochLog(keras.callbacks.Callback):
    """Record phase, within-phase epoch, global epoch, and validation accuracy.

    on_record, when set, receives each completed row so it can be flushed
    immediately. It is not part of checkpoint selection.
    """

    def __init__(self, phase, global_offset, records, on_record=None):
        super().__init__()
        self.phase = phase
        self.global_offset = global_offset
        self.records = records
        self.on_record = on_record

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {
            "phase": self.phase,
            "epoch_in_phase": int(epoch) + 1,
            "global_epoch": self.global_offset + int(epoch) + 1,
            "loss": float(logs["loss"]),
            "accuracy": float(logs["accuracy"]),
            "val_loss": float(logs["val_loss"]),
            "val_accuracy": float(logs["val_accuracy"]),
        }
        self.records.append(row)
        if self.on_record is not None:
            self.on_record(row)
