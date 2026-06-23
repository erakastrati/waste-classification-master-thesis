import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast
)


# ==========================================
# Create Results Directory
# ==========================================

os.makedirs("results/efficientnet_augmented", exist_ok=True)


# ==========================================
# Dataset Loading
# ==========================================

train_ds = image_dataset_from_directory(
    "data/trashNet-dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    "data/trashNet-dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)


# ==========================================
# Preprocess Images
# ==========================================

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y)
)


# ==========================================
# Build Model with Data Augmentation
# ==========================================

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = Input(shape=(224, 224, 3))

# Data augmentation layers (applied only during training)
x = RandomFlip("horizontal")(inputs)
x = RandomRotation(0.15)(x)
x = RandomZoom(0.15)(x)
x = RandomContrast(0.2)(x)

x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(6, activation="softmax")(x)

model = Model(inputs, outputs)

model.summary()


# ==========================================
# Callbacks
# ==========================================

checkpoint = ModelCheckpoint(
    filepath="results/efficientnet_augmented/efficientnet_augmented_best.keras",
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
# Phase 1: Train Classification Head
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================")
print("PHASE 1: Training Classification Head (Base Frozen, With Augmentation)")
print("==============================\n")

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)


# ==========================================
# Phase 2: Fine-Tuning (lr=1e-5)
# ==========================================

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================")
print("PHASE 2: Fine-Tuning (Top 30 Layers Unfrozen, lr=1e-5, With Augmentation)")
print("==============================\n")

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)


# ==========================================
# Final Evaluation (best model)
# ==========================================

best_model = load_model(
    "results/efficientnet_augmented/efficientnet_augmented_best.keras"
)

loss, accuracy = best_model.evaluate(val_ds, verbose=1)

print("\n==============================")
print("FINAL VALIDATION RESULTS (Best Model — With Augmentation)")
print("==============================")
print(f"Validation Loss:     {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"\nEfficientNetB0 without augmentation: 0.8990")

if accuracy > 0.8990:
    print("Augmentation IMPROVED results.")
else:
    diff = (accuracy - 0.8990) * 100
    print(f"Augmentation result: {diff:+.2f}pp vs no augmentation.")


# ==========================================
# Save Final Model
# ==========================================

best_model.save(
    "results/efficientnet_augmented/efficientnet_augmented.keras"
)


# ==========================================
# Combine History and Plot
# ==========================================

combined_accuracy     = history_phase1.history["accuracy"]     + history_phase2.history["accuracy"]
combined_val_accuracy = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
combined_loss         = history_phase1.history["loss"]         + history_phase2.history["loss"]
combined_val_loss     = history_phase1.history["val_loss"]     + history_phase2.history["val_loss"]

phase1_epochs = len(history_phase1.history["accuracy"])
total_epochs  = len(combined_accuracy)
epochs_range  = range(1, total_epochs + 1)


plt.figure(figsize=(10, 5))
plt.plot(epochs_range, combined_accuracy,     label="Training Accuracy")
plt.plot(epochs_range, combined_val_accuracy, label="Validation Accuracy")
plt.axvline(x=phase1_epochs, color="gray", linestyle="--", label="Fine-Tuning Start")
plt.title("EfficientNetB0 + Augmentation — Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/efficientnet_augmented/accuracy.png")
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(epochs_range, combined_loss,     label="Training Loss")
plt.plot(epochs_range, combined_val_loss, label="Validation Loss")
plt.axvline(x=phase1_epochs, color="gray", linestyle="--", label="Fine-Tuning Start")
plt.title("EfficientNetB0 + Augmentation — Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/efficientnet_augmented/loss.png")
plt.close()


print("\nModel saved successfully.")
print("Accuracy plot saved.")
print("Loss plot saved.")
