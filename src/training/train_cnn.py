import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.models.cnn_model import build_cnn_model


# ==========================================
# Create Results Directory
# ==========================================

os.makedirs(
    "results/cnn_baseline",
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
# Normalize Images
# ==========================================

train_ds = train_ds.map(
    lambda x, y: (x / 255.0, y)
)

val_ds = val_ds.map(
    lambda x, y: (x / 255.0, y)
)


# ==========================================
# Build Model
# ==========================================

model = build_cnn_model()

model.summary()


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# ==========================================
# Final Evaluation
# ==========================================

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n==============================")
print("FINAL VALIDATION RESULTS")
print("==============================")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


# ==========================================
# Save Model
# ==========================================

model.save(
    "results/cnn_baseline/cnn_baseline.keras"
)


# ==========================================
# Accuracy Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Baseline Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    "results/cnn_baseline/accuracy.png"
)

plt.close()


# ==========================================
# Loss Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Baseline Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    "results/cnn_baseline/loss.png"
)

plt.close()


print("\nModel saved successfully.")
print("Accuracy plot saved.")
print("Loss plot saved.")