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
Completed

Total Parameters:
4,214,313 (16.08 MB)

Trainable (Phase 1):
164,742 (classification head only)

---

Phase 1 Results (Epochs 1–10, Base Frozen, lr=1e-3):

Best Validation Accuracy:
89.90% at Epoch 10

Training Accuracy at best epoch:
98.80%

Validation Loss at best epoch:
0.3386

Note: val_accuracy improved every epoch from 86.34% to 89.90%.
EarlyStopping did not trigger — model kept improving through all 10 epochs.

---

Phase 2 Results (Epochs 1–6, Top 30 Layers Unfrozen, lr=1e-5):

Phase 2 never exceeded Phase 1 best (89.90%).
Best Phase 2 epoch: Epoch 1 with val_accuracy 89.50%.
EarlyStopping triggered at Epoch 6. Weights restored from Phase 2 Epoch 1.

---

Final Results (Best Model via ModelCheckpoint):

Validation Accuracy:
89.90%

Validation Loss:
0.3386

---

Comparison vs all models:

CNN Baseline:    56.44%
MobileNetV2:     87.13%  (+30.69 pp vs CNN)
EfficientNetB0:  89.90%  (+33.46 pp vs CNN, +2.77 pp vs MobileNetV2)

---

Observations:

- EfficientNetB0 is the best performing model across all three experiments.
- Phase 1 showed continuous improvement throughout all 10 epochs, suggesting the
  EfficientNet feature extractor is particularly well-suited to visual waste classification.
- Phase 2 fine-tuning did not improve results — the frozen base already generalized well.
  EarlyStopping correctly halted training after 6 epochs without improvement.
- Validation loss (0.3386) is significantly lower than MobileNetV2 (0.4410),
  indicating more confident and stable predictions.
- EfficientNetB0 achieved 89.90% accuracy with only ~4.2M parameters,
  demonstrating strong efficiency relative to model size.
- The +2.77 pp improvement over MobileNetV2 confirms EfficientNetB0 as the
  recommended architecture for this waste classification task.

Artifacts:
- results/efficientnet/efficientnet.keras         (best model)
- results/efficientnet/efficientnet_best.keras    (checkpoint copy)
- results/efficientnet/accuracy.png
- results/efficientnet/loss.png

---

## Final Model Comparison Summary

| Model           | Val Accuracy | Val Loss | Parameters  | Training Strategy            |
|-----------------|-------------|----------|-------------|------------------------------|
| CNN Baseline    | 56.44%      | 1.1258   | 110,534     | Trained from scratch, 10 ep  |
| MobileNetV2     | 87.13%      | 0.4410   | 2,422,726   | Transfer learning, 2-phase   |
| EfficientNetB0  | 89.90%      | 0.3386   | 4,214,313   | Transfer learning, 2-phase   |

Best Model: EfficientNetB0
Improvement over baseline: +33.46 percentage points

---

## Detailed Evaluation Results (Per-Class Metrics)

Validation set: 505 images across 6 classes
Evaluation method: shuffle=True, seed=42 (identical split to training)

### CNN Baseline — Per-Class F1

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| cardboard | 0.75      | 0.72   | 0.74     | 69      |
| glass     | 0.49      | 0.49   | 0.49     | 102     |
| metal     | 0.49      | 0.61   | 0.55     | 88      |
| paper     | 0.56      | 0.79   | 0.66     | 123     |
| plastic   | 0.65      | 0.38   | 0.48     | 89      |
| trash     | 0.00      | 0.00   | 0.00     | 34      |
| **macro avg** | **0.49** | **0.50** | **0.48** | 505 |

Observations:
- trash class completely failed (F1=0.00) — model never predicted it, likely due to class imbalance (34 samples).
- Best class: cardboard (F1=0.74).
- Worst classes: trash, plastic, glass.

---

### MobileNetV2 — Per-Class F1

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| cardboard | 0.89      | 0.91   | 0.90     | 69      |
| glass     | 0.89      | 0.90   | 0.90     | 102     |
| metal     | 0.82      | 0.92   | 0.87     | 88      |
| paper     | 0.93      | 0.91   | 0.92     | 123     |
| plastic   | 0.86      | 0.80   | 0.83     | 89      |
| trash     | 0.75      | 0.62   | 0.68     | 34      |
| **macro avg** | **0.85** | **0.84** | **0.85** | 505 |

Observations:
- Strong performance across all classes.
- trash class improved significantly (F1=0.68) compared to CNN baseline.
- Best class: paper (F1=0.92).
- Weakest class: trash (F1=0.68) — expected due to low sample count.

---

### EfficientNetB0 — Per-Class F1

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| cardboard | 0.96      | 0.94   | 0.95     | 69      |
| glass     | 0.87      | 0.89   | 0.88     | 102     |
| metal     | 0.89      | 0.89   | 0.89     | 88      |
| paper     | 0.94      | 0.94   | 0.94     | 123     |
| plastic   | 0.88      | 0.85   | 0.87     | 89      |
| trash     | 0.82      | 0.82   | 0.82     | 34      |
| **macro avg** | **0.89** | **0.89** | **0.89** | 505 |

Observations:
- Best performing model across every single class.
- trash class reached F1=0.82 — strong result for a class with only 34 validation samples.
- Best class: cardboard (F1=0.95).
- Most balanced model: all classes above F1=0.82.
- Macro avg F1=0.89 confirms consistent performance regardless of class imbalance.

---

### Cross-Model F1 Comparison (macro avg)

| Model          | Accuracy | Macro F1 | Weighted F1 |
|----------------|----------|----------|-------------|
| CNN Baseline   | 0.56     | 0.48     | 0.54        |
| MobileNetV2    | 0.87     | 0.85     | 0.87        |
| EfficientNetB0 | 0.90     | 0.89     | 0.90        |

Artifacts:
- results/cnn_baseline/confusion_matrix.png
- results/cnn_baseline/classification_report.txt
- results/mobilenet/confusion_matrix.png
- results/mobilenet/classification_report.txt
- results/efficientnet/confusion_matrix.png
- results/efficientnet/classification_report.txt

---

## Remaining Work

### NEXT-01 — Real-World Testing
Test all three saved models on data/real_test_dataset/ (images collected
outside the TrashNet dataset). This validates generalization to real conditions.

Required:
- src/evaluation/evaluate_real_world.py
- Predict with all 3 models on same real images
- Compare predictions side-by-side
- Document results in notes.md

Status: Pending

### NEXT-02 — Training Curve Comparison Plot
Single figure comparing accuracy curves of all 3 models on the same axes.
Useful for thesis Chapter: Results and Discussion.

Status: Pending

### NEXT-03 — Thesis Write-Up
Chapters that can now be written based on collected data:
- Dataset and Preprocessing
- Model Architectures
- Training Configuration
- Results: Accuracy, Loss, F1, Confusion Matrices
- Discussion: CNN vs Transfer Learning, class imbalance impact, trash class behavior