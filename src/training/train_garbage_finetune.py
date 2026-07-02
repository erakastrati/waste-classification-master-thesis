"""
train_garbage_finetune.py  —  EXP-011: Fine-tune with Garbage Classification v2

Strategy:
  1. Load EfficientNetB0 + Household (57.69% real-world, best single-pass)
  2. Fine-tune on TrashNet train + TACO + Household + Garbage v2 (~19k extra images)
  3. real_test_dataset (78 photos) held out for evaluation only

Prerequisites:
    python -m src.data.prepare_garbage_v2

Run from project root:
    python -m src.training.train_garbage_finetune
"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

BASE_MODEL_PATH = "results/efficientnet_household/efficientnet_household.keras"
TRASHNET_PATH   = "data/trashNet-dataset"
TACO_PATH       = "data/taco-prepared"
HOUSEHOLD_PATH  = "data/local_train"
GARBAGE_PATH    = "data/garbage-v2-prepared"
OUTPUT_DIR      = "results/efficientnet_garbage"
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


if not os.path.isdir(GARBAGE_PATH):
    raise FileNotFoundError(
        f"Missing {GARBAGE_PATH}. Run: python -m src.data.prepare_garbage_v2"
    )

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

print("Loading Household (local_train)...")
household_ds = image_dataset_from_directory(
    HOUSEHOLD_PATH,
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print("Loading Garbage v2 prepared...")
garbage_ds = image_dataset_from_directory(
    GARBAGE_PATH,
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

trashnet_train = trashnet_train.map(lambda x, y: (preprocess_input(x), y))
taco_ds        = taco_ds.map(lambda x, y: (preprocess_input(x), y))
household_ds   = household_ds.map(lambda x, y: (preprocess_input(x), y))
garbage_ds     = garbage_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds         = val_ds.map(lambda x, y: (preprocess_input(x), y))

combined_train = (
    trashnet_train
    .concatenate(taco_ds)
    .concatenate(household_ds)
    .concatenate(garbage_ds)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

trashnet_counts  = count_images_in_dir(TRASHNET_PATH, class_names)
taco_counts      = count_images_in_dir(TACO_PATH, class_names)
household_counts = count_images_in_dir(HOUSEHOLD_PATH, class_names)
garbage_counts   = count_images_in_dir(GARBAGE_PATH, class_names)

combined_counts = {
    idx: int(trashnet_counts[idx] * 0.8) + taco_counts[idx]
         + household_counts[idx] + garbage_counts[idx]
    for idx in range(len(class_names))
}

total_samples = sum(combined_counts.values())
n_classes     = len(combined_counts)
class_weight  = {
    cls: total_samples / (n_classes * count)
    for cls, count in combined_counts.items()
}

print("\nCombined training counts (TrashNet 80% + TACO + Household + Garbage v2):")
for idx, cls in enumerate(class_names):
    print(f"  {cls:12s}  {combined_counts[idx]:5d}  (weight: {class_weight[idx]:.3f})")

print(f"\nLoading base model: {BASE_MODEL_PATH}")
model = load_model(BASE_MODEL_PATH)

checkpoint = ModelCheckpoint(
    filepath=f"{OUTPUT_DIR}/efficientnet_garbage_best.keras",
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

print("\n" + "=" * 50)
print("EXP-011: Fine-Tune on TrashNet + TACO + Household + Garbage v2")
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

from tensorflow.keras.models import load_model as lm

best_model = lm(f"{OUTPUT_DIR}/efficientnet_garbage_best.keras")
val_loss, val_acc = best_model.evaluate(val_ds, verbose=1)

print("\n" + "=" * 50)
print("FINAL VALIDATION RESULTS (TrashNet val set)")
print("=" * 50)
print(f"Validation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"\nHousehold (before Garbage v2): 0.9050")
print(f"Delta:                         {val_acc - 0.9050:+.4f}")

best_model.save(f"{OUTPUT_DIR}/efficientnet_garbage.keras")
print("\nModel saved.")

epochs = range(1, len(history.history["accuracy"]) + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"], label="Train Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
plt.title("EXP-011 Garbage v2 — Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["loss"], label="Train Loss")
plt.plot(epochs, history.history["val_loss"], label="Val Loss")
plt.title("EXP-011 Garbage v2 — Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png")
plt.close()
print("Training curves saved.")
