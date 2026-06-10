import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.models.mobilenet_model import build_mobilenet_model


# ==========================================
# Create Results Directory
# ==========================================

os.makedirs(
    "results/mobilenet",
    exist_ok=True
)


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
# Preprocess Images (MobileNetV2 scaling)
# ==========================================

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y)
)


# ==========================================
# Build Model
# ==========================================

model, base_model = build_mobilenet_model()

model.summary()


# ==========================================
# Callbacks
# ==========================================

checkpoint = ModelCheckpoint(
    filepath="results/mobilenet/mobilenet_best.keras",
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
print("PHASE 1: Training Classification Head (Base Frozen)")
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
print("PHASE 2: Fine-Tuning (Top 30 Layers Unfrozen, lr=1e-5)")
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

from tensorflow.keras.models import load_model

best_model = load_model("results/mobilenet/mobilenet_best.keras")

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print("\n==============================")
print("FINAL VALIDATION RESULTS (Best Model)")
print("==============================")
print(f"Validation Loss:     {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"\nCNN Baseline Accuracy:        0.5644")
print(f"MobileNetV2 v1 (Phase 1 end): 0.8772")

if accuracy > 0.8772:
    print("MobileNetV2 v2 IMPROVES over previous run.")
elif accuracy > 0.5644:
    print("MobileNetV2 v2 OUTPERFORMS CNN Baseline.")
else:
    print("MobileNetV2 v2 does not outperform CNN Baseline.")


# ==========================================
# Save Final Model
# ==========================================

best_model.save(
    "results/mobilenet/mobilenet.keras"
)


# ==========================================
# Combine History from Both Phases
# ==========================================

combined_accuracy = (
    history_phase1.history["accuracy"] +
    history_phase2.history["accuracy"]
)

combined_val_accuracy = (
    history_phase1.history["val_accuracy"] +
    history_phase2.history["val_accuracy"]
)

combined_loss = (
    history_phase1.history["loss"] +
    history_phase2.history["loss"]
)

combined_val_loss = (
    history_phase1.history["val_loss"] +
    history_phase2.history["val_loss"]
)

total_epochs = len(combined_accuracy)
epochs_range = range(1, total_epochs + 1)


# ==========================================
# Accuracy Plot
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    combined_accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    combined_val_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    x=10,
    color="gray",
    linestyle="--",
    label="Fine-Tuning Start"
)

plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    "results/mobilenet/accuracy.png"
)

plt.close()


# ==========================================
# Loss Plot
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    combined_loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    combined_val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=10,
    color="gray",
    linestyle="--",
    label="Fine-Tuning Start"
)

plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    "results/mobilenet/loss.png"
)

plt.close()


print("\nModel saved successfully.")
print("Accuracy plot saved.")
print("Loss plot saved.")
