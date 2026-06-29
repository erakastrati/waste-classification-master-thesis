"""
train_realwaste_finetune.py  —  EXP-007: Domain Adaptation with RealWaste

Strategy:
  1. Load EfficientNetB0 + TACO (best model: 92.87% TrashNet, 52.56% real-world)
  2. Fine-tune on TrashNet train + RealWaste full-scene images (3,905 images)
  3. RealWaste provides diverse real backgrounds (wood, floor, landfill)
  4. Keep real_test_dataset untouched for final evaluation only

Run from project root:
    python -m src.training.train_realwaste_finetune
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

BASE_MODEL_PATH = "results/efficientnet_taco/efficientnet_taco.keras"
TRASHNET_PATH   = "data/trashNet-dataset"
REALWASTE_PATH  = "data/realwaste-prepared"
OUTPUT_DIR      = "results/efficientnet_realwaste"
IMAGE_SIZE      = (224, 224)
BATCH_SIZE      = 32
SEED            = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def count_images_in_dir(root, class_names):
    counts = {}
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            counts[idx] = 0
            continue
        counts[idx] = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
    return counts


# ==========================================
# Load Datasets
# ==========================================

print("Loading TrashNet training set...")
trashnet_train = image_dataset_from_directory(
    TRASHNET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print("Loading RealWaste dataset (all as training)...")
realwaste_ds = image_dataset_from_directory(
    REALWASTE_PATH,
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class_names = trashnet_train.class_names
print(f"Classes: {class_names}")

print("Loading TrashNet validation set...")
val_ds = image_dataset_from_directory(
    TRASHNET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ==========================================
# Preprocessing
# ==========================================

trashnet_train = trashnet_train.map(lambda x, y: (preprocess_input(x), y))
realwaste_ds   = realwaste_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds         = val_ds.map(lambda x, y: (preprocess_input(x), y))

combined_train = trashnet_train.concatenate(realwaste_ds)
combined_train = combined_train.prefetch(tf.data.AUTOTUNE)
val_ds         = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ==========================================
# Class Weights (TrashNet train + RealWaste)
# ==========================================

trashnet_counts  = count_images_in_dir(TRASHNET_PATH, class_names)
realwaste_counts = count_images_in_dir(REALWASTE_PATH, class_names)

# TrashNet: use ~80% of full counts (training split)
combined_counts = {
    idx: int(trashnet_counts[idx] * 0.8) + realwaste_counts[idx]
    for idx in range(len(class_names))
}

total_samples = sum(combined_counts.values())
n_classes     = len(combined_counts)
class_weight  = {
    cls: total_samples / (n_classes * count)
    for cls, count in combined_counts.items()
}

print("\nCombined training counts (TrashNet 80% + RealWaste):")
for idx, cls in enumerate(class_names):
    print(f"  {cls:12s}  {combined_counts[idx]:5d}  (weight: {class_weight[idx]:.3f})")

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
    filepath=f"{OUTPUT_DIR}/efficientnet_realwaste_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# ==========================================
# Fine-Tuning
# ==========================================

print("\n" + "=" * 50)
print("EXP-007: Fine-Tuning on TrashNet + RealWaste (lr=1e-5)")
print("=" * 50 + "\n")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    combined_train,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop],
)

# ==========================================
# Evaluate & Save
# ==========================================

from tensorflow.keras.models import load_model as lm

best_model = lm(f"{OUTPUT_DIR}/efficientnet_realwaste_best.keras")

print("\n" + "=" * 50)
print("FINAL VALIDATION RESULTS (TrashNet val set)")
print("=" * 50)

val_loss, val_acc = best_model.evaluate(val_ds, verbose=1)
print(f"\nValidation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"\nEfficientNetB0 + TACO (before RealWaste): 0.9287")
print(f"Delta after RealWaste fine-tune:            {val_acc - 0.9287:+.4f}")

best_model.save(f"{OUTPUT_DIR}/efficientnet_realwaste.keras")
print("\nModel saved.")

# ==========================================
# Training Curves
# ==========================================

epochs = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"],     label="Train Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
plt.title("EfficientNetB0 + RealWaste — Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["loss"],     label="Train Loss")
plt.plot(epochs, history.history["val_loss"], label="Val Loss")
plt.title("EfficientNetB0 + RealWaste — Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png")
plt.close()
print("Training curves saved.")
