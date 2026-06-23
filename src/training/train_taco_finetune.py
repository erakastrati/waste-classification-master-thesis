"""
train_taco_finetune.py  —  EXP-005: Domain Adaptation with TACO

Strategy:
  1. Load EfficientNetB0 + Augmentation (best model so far, 89.70% TrashNet, 50% real-world)
  2. Train on combined dataset: TrashNet (2,527) + TACO crops (3,601) = 6,128 images
  3. Fine-tune with class_weight to handle plastic over-representation (1,989 images)
  4. Very low learning rate (1e-5) to adapt domain without losing ImageNet features

Expected: 70-80% real-world accuracy

Run from project root:
    python -m src.training.train_taco_finetune
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
# Config
# ==========================================

BASE_MODEL_PATH   = "results/efficientnet_augmented/efficientnet_augmented.keras"
TRASHNET_PATH     = "data/trashNet-dataset"
TACO_PATH         = "data/taco-prepared"
OUTPUT_DIR        = "results/efficientnet_taco"
IMAGE_SIZE        = (224, 224)
BATCH_SIZE        = 32
SEED              = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load Datasets
# ==========================================

print("Loading TrashNet dataset...")
trashnet_train = image_dataset_from_directory(
    TRASHNET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Loading TACO dataset (all as training)...")
taco_ds = image_dataset_from_directory(
    TACO_PATH,
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = trashnet_train.class_names
print(f"Classes: {class_names}")

# Validation set: TrashNet only (same as all previous experiments)
print("Loading TrashNet validation set...")
val_ds = image_dataset_from_directory(
    TRASHNET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# Preprocessing
# ==========================================

trashnet_train = trashnet_train.map(lambda x, y: (preprocess_input(x), y))
taco_ds        = taco_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds         = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Combine TrashNet + TACO for training
combined_train = trashnet_train.concatenate(taco_ds)
combined_train = combined_train.prefetch(tf.data.AUTOTUNE)
val_ds         = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ==========================================
# Class Weights (handle plastic imbalance)
# Combined counts: cardboard=669, glass=664, metal=871, paper=801, plastic=2471, trash=652
# ==========================================

combined_counts = {
    0: 669,   # cardboard
    1: 664,   # glass
    2: 871,   # metal
    3: 801,   # paper
    4: 2471,  # plastic (overrepresented)
    5: 652,   # trash
}
total_samples = sum(combined_counts.values())
n_classes     = len(combined_counts)
class_weight  = {
    cls: total_samples / (n_classes * count)
    for cls, count in combined_counts.items()
}

print("\nClass weights:")
for cls_idx, weight in class_weight.items():
    print(f"  {class_names[cls_idx]:12s} → {weight:.3f}")

# ==========================================
# Load Base Model
# ==========================================

print(f"\nLoading base model: {BASE_MODEL_PATH}")
model = load_model(BASE_MODEL_PATH)
model.summary()

# ==========================================
# Callbacks
# ==========================================

checkpoint = ModelCheckpoint(
    filepath=f"{OUTPUT_DIR}/efficientnet_taco_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ==========================================
# Phase 1 — Fine-tune entire model, very low lr
# ==========================================

print("\n" + "=" * 50)
print("PHASE 1: Fine-Tuning on Combined Dataset (lr=1e-5)")
print("=" * 50 + "\n")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    combined_train,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop]
)

# ==========================================
# Load Best Model
# ==========================================

from tensorflow.keras.models import load_model as lm
best_model = lm(f"{OUTPUT_DIR}/efficientnet_taco_best.keras")

# ==========================================
# Evaluate on TrashNet Validation
# ==========================================

print("\n" + "=" * 50)
print("FINAL VALIDATION RESULTS (TrashNet val set)")
print("=" * 50)

val_loss, val_acc = best_model.evaluate(val_ds, verbose=1)
print(f"\nValidation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"\nEfficientNetB0 + Augment (before TACO): 0.8970")
delta = val_acc - 0.8970
print(f"Delta after TACO fine-tune:             {delta:+.4f}")

# ==========================================
# Save Final Model
# ==========================================

best_model.save(f"{OUTPUT_DIR}/efficientnet_taco.keras")
print("\nModel saved.")

# ==========================================
# Training Curves
# ==========================================

epochs = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"],     label="Train Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
plt.title("EfficientNetB0 + TACO — Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["loss"],     label="Train Loss")
plt.plot(epochs, history.history["val_loss"], label="Val Loss")
plt.title("EfficientNetB0 + TACO — Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png")
plt.close()
print("Training curves saved.")
