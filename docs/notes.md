# Dataset Analysis

## Dataset

Source:
TrashNet Dataset (Kaggle)

Classes:
- cardboard
- glass
- metal
- paper
- plastic
- trash

Total images:
2527

## Class Distribution

cardboard: 403
glass: 501
metal: 410
paper: 594
plastic: 482
trash: 137

Observation:
The dataset is imbalanced. The trash class contains significantly fewer samples.

## Image Dimensions

Width: 512
Height: 384

Observation:
All images have consistent dimensions.

## Dataset Split

Training: 2022
Validation: 505

Split ratio:
80/20

## Preprocessing Configuration

Image Resize:
224 x 224 pixels

Batch Size:
32

Validation Split:
20%

Random Seed:
42

Normalization:
To be applied during model training.

Reasoning:
Images were resized from 512x384 to 224x224 to ensure compatibility with CNN, MobileNetV2, and EfficientNetB0 architectures while reducing computational cost.

## Experiment Log

### EXP-001

Model:
CNN Baseline

Architecture:

* Conv2D(32) + MaxPooling
* Conv2D(64) + MaxPooling
* Conv2D(128) + MaxPooling
* GlobalAveragePooling2D
* Dense(128)
* Dropout(0.3)
* Dense(6, Softmax)

Input Shape:
224x224x3

Total Parameters:
110,534

Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metrics:
Accuracy

Epochs:
10

Status:
Completed

Results:

Epochs:
10

Training Accuracy:
~67%

Validation Accuracy:
56.44%

Training Loss:
~0.92

Validation Loss:
1.1258

Observations:

- The model learned meaningful visual patterns from scratch.
- No significant overfitting was observed over 10 epochs.
- Validation accuracy stabilized around 56%, which is expected for a lightweight custom CNN on a relatively small and imbalanced dataset.
- The trash class (137 images) likely contributed to misclassifications due to underrepresentation.
- This result establishes the baseline for comparison with transfer learning architectures.

Artifacts:
- results/cnn_baseline/cnn_baseline.keras
- results/cnn_baseline/accuracy.png
- results/cnn_baseline/loss.png

---

### EXP-002

Model:
MobileNetV2 (Transfer Learning)

Architecture:

* MobileNetV2 base (ImageNet pretrained, frozen in Phase 1)
* GlobalAveragePooling2D
* Dense(128, ReLU)
* Dropout(0.3)
* Dense(6, Softmax)

Input Shape:
224x224x3

Total Parameters:
~2.3M (base) + classification head

Preprocessing:
MobileNetV2 preprocess_input (scales to [-1, 1])

Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metrics:
Accuracy

Training Strategy:

Phase 1 (Epochs 1-10):
Base model frozen. Only classification head trained.

Phase 2 (Epochs 11-20):
Top 30 layers of base unfrozen. Fine-tuning applied.

Status:
Completed (Final Run — v2 with corrected fine-tuning)

Total Parameters:
2,422,726 (9.24 MB)

Trainable (Phase 1):
164,742 (classification head only)

---

Phase 1 Results (Epochs 1–10, Base Frozen, lr=1e-3):

Best Validation Accuracy:
86.73% at Epoch 5

Training Accuracy at best epoch:
93.89%

Validation Loss at best epoch:
0.4611

Note: EarlyStopping triggered at Epoch 10. Weights restored from Epoch 5.

---

Phase 2 Results (Epochs 1–14, Top 30 Layers Unfrozen, lr=1e-5):

Best Validation Accuracy:
87.13% at Phase 2 Epoch 9

Training Accuracy at best epoch:
96.59%

Validation Loss at best epoch:
0.4410

Note: EarlyStopping triggered at Phase 2 Epoch 14. Weights restored from Phase 2 Epoch 9.
Fine-tuning was fully stable — no loss spikes observed.

---

Final Results (Best Model via ModelCheckpoint):

Validation Accuracy:
87.13%

Validation Loss:
0.4410

---

Comparison vs CNN Baseline:

CNN Baseline:           56.44%
MobileNetV2 (final):   87.13%  (+30.69 pp)

---

Observations:

- Fine-tuning with lr=1e-5 was fully stable. No catastrophic forgetting occurred.
  Validation loss remained consistently around 0.44 throughout Phase 2.
- Phase 2 provided a marginal improvement over Phase 1 peak (86.73% → 87.13%),
  confirming that careful fine-tuning with a low learning rate can improve generalization.
- ModelCheckpoint saved the best model across all epochs of both phases automatically.
- EarlyStopping prevented unnecessary computation once the model plateaued.
- MobileNetV2 represents a +30.69 percentage point improvement over the CNN baseline,
  demonstrating the significant advantage of transfer learning on small datasets.
- The model plateau around 87% suggests the dataset size is the main limiting factor.

Artifacts:
- results/mobilenet/mobilenet.keras         (best model)
- results/mobilenet/mobilenet_best.keras    (checkpoint copy)
- results/mobilenet/accuracy.png
- results/mobilenet/loss.png

---

### EXP-003

Model:
EfficientNetB0 (Transfer Learning)

Architecture:

* EfficientNetB0 base (ImageNet pretrained, frozen in Phase 1)
* GlobalAveragePooling2D
* Dense(128, ReLU)
* Dropout(0.3)
* Dense(6, Softmax)

Input Shape:
224x224x3

Total Parameters:
~4.05M (base) + classification head

Preprocessing:
EfficientNetB0 preprocess_input (scales pixel values for EfficientNet internal normalization)

Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metrics:
Accuracy

Training Strategy:

Phase 1 (Epochs 1-10):
Base model frozen. Only classification head trained. lr=1e-3.

Phase 2 (Epochs up to 15):
Top 30 layers of base unfrozen. Fine-tuning with lr=1e-5.
EarlyStopping with patience=5.

Callbacks:
ModelCheckpoint — saves best model by val_accuracy.
EarlyStopping — stops training when val_accuracy plateaus.

Status:
Pending — run: python -m src.training.train_efficientnet

Results:
To be completed after training.

Artifacts (expected):
- results/efficientnet/efficientnet.keras
- results/efficientnet/efficientnet_best.keras
- results/efficientnet/accuracy.png
- results/efficientnet/loss.png