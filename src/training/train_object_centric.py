"""
train_object_centric.py  —  EXP-008: Object-Centric Domain Adaptation

Strategy:
  1. Load EfficientNetB0 + RealWaste (55.13% real-world baseline)
  2. Fine-tune on TrashNet train + TACO object crops + Household real_world
  3. TACO crops = object-focused; Household = home/kitchen real scenes
  4. Skip RealWaste full-scene (teaches background, not object material)
  5. real_test_dataset (78 photos) held out for evaluation only

Run from project root:
    python -m src.training.train_object_centric
"""

import os
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

BASE_MODEL_PATH = "results/efficientnet_realwaste/efficientnet_realwaste.keras"
TRASHNET_PATH   = "data/trashNet-dataset"
TACO_PATH       = "data/taco-prepared"
HOUSEHOLD_PATH  = "data/local_train"
OUTPUT_DIR      = "results/efficientnet_household"
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

print("Loading TACO object crops...")
taco_ds = image_dataset_from_directory(
    TACO_PATH,
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print("Loading Household Waste (real_world, local_train)...")
household_ds = image_dataset_from_directory(
    HOUSEHOLD_PATH,
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
taco_ds        = taco_ds.map(lambda x, y: (preprocess_input(x), y))
household_ds   = household_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds         = val_ds.map(lambda x, y: (preprocess_input(x), y))

combined_train = (
    trashnet_train
    .concatenate(taco_ds)
    .concatenate(household_ds)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ==========================================
# Class Weights
# ==========================================

trashnet_counts  = count_images_in_dir(TRASHNET_PATH, class_names)
taco_counts      = count_images_in_dir(TACO_PATH, class_names)
household_counts = count_images_in_dir(HOUSEHOLD_PATH, class_names)

combined_counts = {
    idx: int(trashnet_counts[idx] * 0.8) + taco_counts[idx] + household_counts[idx]
    for idx in range(len(class_names))
}

total_samples = sum(combined_counts.values())
n_classes     = len(combined_counts)
class_weight  = {
    cls: total_samples / (n_classes * count)
    for cls, count in combined_counts.items()
}

print("\nCombined training counts (TrashNet 80% + TACO + Household):")
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
    filepath=f"{OUTPUT_DIR}/efficientnet_household_best.keras",
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
print("EXP-008: Object-Centric Fine-Tune (TrashNet + TACO + Household)")
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

best_model = lm(f"{OUTPUT_DIR}/efficientnet_household_best.keras")

print("\n" + "=" * 50)
print("FINAL VALIDATION RESULTS (TrashNet val set)")
print("=" * 50)

val_loss, val_acc = best_model.evaluate(val_ds, verbose=1)
print(f"\nValidation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"\nEfficientNetB0 + RealWaste (before EXP-008): 0.9129")
print(f"Delta after Household fine-tune:              {val_acc - 0.9129:+.4f}")

best_model.save(f"{OUTPUT_DIR}/efficientnet_household.keras")
print("\nModel saved.")

# ==========================================
# Training Curves
# ==========================================

epochs = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"],     label="Train Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
plt.title("EXP-008 Object-Centric — Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["loss"],     label="Train Loss")
plt.plot(epochs, history.history["val_loss"], label="Val Loss")
plt.title("EXP-008 Object-Centric — Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png")
plt.close()
print("Training curves saved.")
