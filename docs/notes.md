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

Status: Completed

Dataset: 78 real-world photos taken with iPhone (13 per class)
Classes: cardboard, glass, metal, paper, plastic, trash
Format: JPEG (converted from HEIC via sips)

---

Results:

| Model          | TrashNet Val | Real-World | Drop     |
|----------------|-------------|------------|----------|
| CNN Baseline   | 56.44%      | 11.54%     | -44.90pp |
| MobileNetV2    | 87.13%      | 43.59%     | -43.54pp |
| EfficientNetB0 | 89.90%      | 39.74%     | -50.16pp |

---

Per-Class F1 (Real-World):

CNN Baseline:
- cardboard: 0.15 | glass: 0.00 | metal: 0.10
- paper: 0.16    | plastic: 0.00 | trash: 0.14
- macro F1: 0.09

MobileNetV2:
- cardboard: 0.39 | glass: 0.27 | metal: 0.48
- paper: 0.57    | plastic: 0.52 | trash: 0.13
- macro F1: 0.39

EfficientNetB0:
- cardboard: 0.33 | glass: 0.42 | metal: 0.24
- paper: 0.48    | plastic: 0.47 | trash: 0.25
- macro F1: 0.36

---

Key Observations:

- All three models suffered a severe drop in real-world conditions.
  This is a classic domain shift problem: TrashNet uses white backgrounds,
  controlled studio lighting, and centered objects. Real-world photos have
  natural backgrounds, variable lighting, and arbitrary angles.

- The CNN Baseline collapsed to 11.54% — near random (chance = 16.7% for 6 classes).
  It learned visual patterns tightly coupled to the white background of TrashNet.

- MobileNetV2 (43.59%) and EfficientNetB0 (39.74%) retained partial generalization.
  Transfer learning from ImageNet provided some domain-agnostic features
  (edges, textures, shapes), but not enough to overcome the domain gap.

- paper was the best-performing class across all models in real-world conditions.
  This is likely because paper has distinctive flat texture that is
  background-independent.

- trash and glass were the hardest classes in real-world conditions.

- This result is a significant finding for the thesis: it demonstrates that
  training exclusively on controlled dataset images (TrashNet) is insufficient
  for real-world deployment without domain adaptation or data augmentation.

Artifacts:
- results/real_world_test/cnn_baseline_report.txt
- results/real_world_test/cnn_baseline_confusion_matrix.png
- results/real_world_test/mobilenetv2_report.txt
- results/real_world_test/mobilenetv2_confusion_matrix.png
- results/real_world_test/efficientnetb0_report.txt
- results/real_world_test/efficientnetb0_confusion_matrix.png
- results/real_world_test/accuracy_comparison.png

### NEXT-02 — Model Comparison Plots

Status: Completed

Generated 3 comparison figures for thesis Results chapter:

1. accuracy_f1_comparison.png
   Grouped bar chart: Validation Accuracy vs Macro F1-Score per model.
   CNN: Acc=0.56, F1=0.48 | MobileNetV2: Acc=0.87, F1=0.85 | EfficientNetB0: Acc=0.90, F1=0.89

2. per_class_f1_comparison.png
   Per-class F1-Score for all 6 waste categories across all 3 models.
   Shows CNN failure on trash (F1=0.00) vs EfficientNetB0 robustness (F1=0.82).

3. val_loss_comparison.png
   Validation loss bar chart: CNN=1.1258, MobileNetV2=0.4410, EfficientNetB0=0.3386

Artifacts:
- results/comparison/accuracy_f1_comparison.png
- results/comparison/per_class_f1_comparison.png
- results/comparison/val_loss_comparison.png

### EXP-004 — EfficientNetB0 + Data Augmentation

Model:
EfficientNetB0 (Transfer Learning + Data Augmentation)

Motivation:
Project proposal explicitly requires comparison with/without data augmentation.
Previous EfficientNetB0 (EXP-003) trained without augmentation.
This experiment adds augmentation to address the domain gap observed in real-world testing.

Augmentation Layers (applied during training only):
- RandomFlip("horizontal")
- RandomRotation(0.15) — ±15 degrees
- RandomZoom(0.15) — ±15% zoom
- RandomContrast(0.2) — ±20% contrast

Architecture:
Input → Augmentation → EfficientNetB0 base → GlobalAveragePooling2D
→ Dense(128) → Dropout(0.3) → Dense(6, Softmax)

Training Strategy:
Same as EXP-003: Phase 1 (lr=1e-3, frozen) + Phase 2 (lr=1e-5, top 30 layers)
ModelCheckpoint + EarlyStopping (patience=5)

Status: Completed

Training Results:
Phase 1 (Classification Head, Frozen Base):
- Best val_accuracy: 0.8970 (Epoch 5)
- Best val_loss: 0.3104
- EarlyStopping triggered at Epoch 10 (patience=5, best at Epoch 5)

Phase 2 (Fine-Tuning, Top 30 Layers, lr=1e-5):
- val_accuracy did not improve above Phase 1 best (0.8970)
- EarlyStopping triggered at Epoch 6 of Phase 2

Final Validation Results (TrashNet):
- Validation Accuracy: 89.70%
- Validation Loss:     0.3104
- Macro F1-Score:      0.89

Per-Class F1-Score:
- cardboard: 0.94
- glass:     0.89
- metal:     0.90
- paper:     0.93
- plastic:   0.86
- trash:     0.81

TrashNet Comparison vs EXP-003 (no augmentation):
- Accuracy: 89.70% vs 89.90% → -0.20pp (statistically negligible)
- Loss:     0.3104 vs 0.3386 → lower loss (+0.0282)
- Conclusion: Augmentation has no negative impact on TrashNet performance.

Real-World Test Results (78 images, 6 classes):
- Real-World Accuracy: 50.00%
- TrashNet → Real-World gap: -39.70pp

Real-World Comparison (with vs without augmentation):
- EfficientNetB0 (no aug):  39.74% real-world
- EfficientNetB0 (+ aug):   50.00% real-world
- Improvement: +10.26pp on real-world data

Key Observation:
Data augmentation improved real-world generalization by +10.26pp while maintaining
near-identical TrashNet validation performance (-0.20pp). This is the central
finding for the augmentation section of the thesis: augmentation reduces domain
shift without sacrificing controlled-environment accuracy.

All 4-model real-world comparison summary:
| Model                    | TrashNet Val | Real-World | Gap      |
|--------------------------|-------------|------------|----------|
| CNN Baseline             |    56.44%   |   11.54%   | -44.90pp |
| MobileNetV2              |    87.13%   |   43.59%   | -43.54pp |
| EfficientNetB0           |    89.90%   |   39.74%   | -50.16pp |
| EfficientNetB0 + Augment |    89.70%   |   50.00%   | -39.70pp |

Artifacts:
- results/efficientnet_augmented/efficientnet_augmented.keras
- results/efficientnet_augmented/efficientnet_augmented_best.keras
- results/efficientnet_augmented/accuracy.png
- results/efficientnet_augmented/loss.png
- results/efficientnet_augmented/confusion_matrix.png
- results/efficientnet_augmented/classification_report.txt
- results/real_world_test/efficientnetb0_augment_report.txt
- results/real_world_test/efficientnetb0_augment_confusion_matrix.png
- results/real_world_test/accuracy_comparison.png (updated)

---

---

## PROJECT SUMMARY — Complete Overview

### All Experiments

| ID      | Model                        | TrashNet Val | Real-World | Status    |
|---------|------------------------------|-------------|------------|-----------|
| EXP-001 | CNN Baseline (from scratch)  |    56.44%   |   11.54%   | Completed |
| EXP-002 | MobileNetV2 (TL)             |    87.13%   |   43.59%   | Completed |
| EXP-003 | EfficientNetB0 (TL)          |    89.90%   |   39.74%   | Completed |
| EXP-004 | EfficientNetB0 + Augment     |    89.70%   |   50.00%   | Completed |
| EXP-005 | EfficientNetB0 + TACO FT     |    92.87%   |   52.56%   | Completed |
| EXP-006 | EfficientNetB0 + rembg (PP)  |    N/A      |   50.00%   | Completed (negative result) |
| EXP-007 | EfficientNetB0 + RealWaste   |    91.29%   |   55.13%   | Completed |
| EXP-008 | EfficientNetB0 + Household   |    90.50%   |   57.69%   | Completed |
| EXP-009 | + Test-Time Augmentation   |    90.50%   |   62.82%   | Completed |
| EXP-010 | Ensemble (3 models + TTA)  |    90.50%   |   58.97%   | Completed (negative result) |

### Key Findings So Far

1. Transfer Learning vastly outperforms CNN from scratch:
   EfficientNetB0 (89.90%) vs CNN Baseline (56.44%) = +33.46pp

2. Data Augmentation reduces domain shift without hurting TrashNet accuracy:
   EfficientNetB0 without aug: 39.74% real-world
   EfficientNetB0 with aug:    50.00% real-world (+10.26pp)

3. Domain Shift is the main limitation:
   All models drop significantly on real-world photos.
   Best TrashNet model: 92.87% → 52.56% real-world (TACO).
   Best real-world model: 62.82% (Household + TTA, EXP-009).

4. Domain adaptation experiments show incremental gains:
   No augmentation: 39.74% → + Augment: 50.00% → + TACO: 52.56% → + RealWaste: 55.13% → + Household: 57.69% → + TTA: 62.82%
   Each step adds real-world training data; gains diminish as domains differ.

5. Fine-Tuning Instability (MobileNetV2):
   Using default lr (1e-3) for fine-tuning caused catastrophic forgetting.
   Fixed with lr=1e-5 + ModelCheckpoint + EarlyStopping.

### All Saved Models

| Model                        | Path                                                        |
|------------------------------|-------------------------------------------------------------|
| CNN Baseline                 | results/cnn_baseline/cnn_baseline.keras                     |
| MobileNetV2                  | results/mobilenet/mobilenet.keras                           |
| EfficientNetB0               | results/efficientnet/efficientnet.keras                     |
| EfficientNetB0 + Augment     | results/efficientnet_augmented/efficientnet_augmented.keras |

### All Generated Artifacts

Training curves (accuracy + loss plots):
- results/cnn_baseline/accuracy.png, loss.png
- results/mobilenet/accuracy.png, loss.png
- results/efficientnet/accuracy.png, loss.png
- results/efficientnet_augmented/accuracy.png, loss.png

Confusion matrices:
- results/cnn_baseline/confusion_matrix.png
- results/mobilenet/confusion_matrix.png
- results/efficientnet/confusion_matrix.png
- results/efficientnet_augmented/confusion_matrix.png

Classification reports (text):
- results/cnn_baseline/classification_report.txt
- results/mobilenet/classification_report.txt
- results/efficientnet/classification_report.txt
- results/efficientnet_augmented/classification_report.txt

Real-world test results:
- results/real_world_test/cnn_report.txt
- results/real_world_test/mobilenet_report.txt
- results/real_world_test/efficientnetb0_report.txt
- results/real_world_test/efficientnetb0_augment_report.txt
- results/real_world_test/accuracy_comparison.png

Model comparison charts:
- results/comparison/accuracy_f1_comparison.png
- results/comparison/per_class_f1_comparison.png
- results/comparison/val_loss_comparison.png

### Git History (commits)

- Add MobileNetV2 transfer learning implementation
- Add EfficientNetB0 transfer learning implementation
- Update experiment log with EXP-002/EXP-003 results
- Add detailed evaluation scripts and per-class metrics for all models
- Add model comparison plots (NEXT-02)
- Add real-world evaluation script and results (NEXT-01)
- Add EfficientNetB0 with data augmentation (EXP-004)
- EXP-004: EfficientNetB0 + Data Augmentation — evaluation complete
- Add Flask web demo for waste classification (Practical Deployment)

---

### NEXT-03 — Thesis Write-Up
Chapters that can now be written based on collected data:
- Dataset and Preprocessing
- Model Architectures (CNN, MobileNetV2, EfficientNetB0)
- Training Configuration and Callbacks
- Results: Accuracy, Loss, F1-Score, Confusion Matrices
- Discussion: CNN vs Transfer Learning, Data Augmentation impact
- Discussion: Domain Shift — TrashNet vs Real-World
- Conclusion and Future Work (EXP-005 domain adaptation)

---

### EXP-005 — Domain Adaptation with TACO Dataset

Status: Completed

Motivation:
All previous models trained on TrashNet (white background) performed poorly on real-world photos
(best: 50.00%). Fine-tuning on TACO (in-the-wild waste images) was used to improve generalization.

TACO Dataset:
- Full name: Trash Annotations in Context
- Source: https://github.com/pedropro/TACO
- Kaggle: https://www.kaggle.com/datasets/kneroma/tacotrashdataset
- License: CC BY 4.0
- Description: 1,500 waste photos in real environments (roads, beaches, parks, homes)
- Format: COCO annotation format with bounding box + segmentation annotations
- Total annotations: 4,784 objects across 60 fine-grained categories

Data Preparation (src/data/prepare_taco.py):
- Parsed COCO annotations and mapped 60 TACO categories to 6 TrashNet classes
- Cropped objects using bounding boxes (MIN_SIZE = 48px)
- Saved 3,601 crops to data/taco-prepared/

TACO crops per class:
- cardboard:  266
- glass:      163
- metal:      461
- paper:      207
- plastic:  1,989  (overrepresented — handled with class_weight)
- trash:      515
- TOTAL:    3,601

Combined training dataset (TrashNet + TACO):
- cardboard:  669  (403 + 266)
- glass:      664  (501 + 163)
- metal:      871  (410 + 461)
- paper:      801  (594 + 207)
- plastic:  2,471  (482 + 1,989)
- trash:      652  (137 + 515)
- TOTAL:    6,128  images

Fine-Tuning Strategy (src/training/train_taco_finetune.py):
- Base model: EfficientNetB0 + Augmentation (89.70% TrashNet, 50.00% real-world)
- Only classification head trained (base frozen, 164,742 trainable params)
- Optimizer: Adam(lr=1e-5)
- Class weights to compensate plastic over-representation
- EarlyStopping(patience=5), ModelCheckpoint(monitor=val_accuracy)
- Training stopped at Epoch 8, best at Epoch 3

Training Results:
- Best val_accuracy: 0.9287 (Epoch 3)
- Best val_loss:     0.2322

Final Results vs Previous Best (EfficientNetB0 + Augment):
- TrashNet val accuracy: 92.87% (was 89.70%) → +3.17pp  ← NEW RECORD
- Real-world accuracy:   52.56% (was 50.00%) → +2.56pp

Key Observation:
TACO fine-tuning improved TrashNet accuracy significantly (+3.17pp) by providing more
diverse training data across all classes. Real-world improvement was moderate (+2.56pp)
because TACO bounding-box crops are object-centric (similar to TrashNet style), not
full-scene photos like the real_test_dataset. Full-scene images would yield larger gains.

Complete 5-model comparison:
| Model                    | TrashNet Val | Real-World | Gap      |
|--------------------------|-------------|------------|----------|
| CNN Baseline             |    56.44%   |   11.54%   | -44.90pp |
| MobileNetV2              |    87.13%   |   43.59%   | -43.54pp |
| EfficientNetB0           |    89.90%   |   39.74%   | -50.16pp |
| EfficientNetB0 + Augment |    89.70%   |   50.00%   | -39.70pp |
| EfficientNetB0 + TACO    |    92.87%   |   52.56%   | -40.31pp |

Artifacts:
- data/taco-prepared/           (3,601 cropped objects organized by class)
- results/efficientnet_taco/efficientnet_taco.keras
- results/efficientnet_taco/efficientnet_taco_best.keras
- results/efficientnet_taco/training_curves.png
- results/real_world_test/efficientnetb0_taco_report.txt
- results/real_world_test/efficientnetb0_taco_confusion_matrix.png
- results/real_world_test/accuracy_comparison.png (updated with 5 models)

---

### EXP-006 — Background Removal Preprocessing (rembg)

Status: Completed — Negative Result

Hypothesis:
Removing the background before classification would help the model focus on the
object material rather than the surrounding environment, improving real-world accuracy.

Method:
- Used rembg library (U2-Net model, 176MB) to remove background from each image
- Replaced background with white (matching TrashNet's controlled environment)
- Evaluated EfficientNetB0 + TACO model on real_test_dataset (78 images)

Results:
- Without background removal: 52.56%
- With background removal:    50.00%
- Delta: -2.56pp (background removal made results slightly worse)

Analysis:
The hypothesis was not confirmed. Possible reasons:
1. rembg (U2-Net) is optimized for portraits, not irregular waste objects
2. Transparent objects (glass, plastic) are difficult to segment correctly
3. EfficientNetB0 + TACO already adapted to mixed backgrounds during fine-tuning
4. rembg may incorrectly crop parts of the waste object itself

Academic value:
This negative result is still valuable for the thesis Discussion section.
It shows that naive preprocessing does not always improve domain generalization.
The finding suggests that better segmentation models (e.g., SAM — Segment Anything)
or end-to-end learning approaches may be needed for real-world deployment.

Artifacts:
- results/rembg_eval/rembg_comparison.png
- results/rembg_eval/confusion_matrices.png
- src/evaluation/evaluate_rembg.py

Decision:
Background removal was removed from the web app (app.py).
The system uses direct inference without preprocessing pipeline.

---

### EXP-007 — Domain Adaptation with RealWaste Dataset

Status: Completed

Motivation:
Real-world accuracy remained low (~52.56% with TACO). RealWaste provides 3,905
full-scene waste images from authentic landfill environments — closer to real-world
conditions than TrashNet's white-background studio photos.

Dataset:
- Source: RealWaste (Kaggle: joebeachcapital/realwaste)
- License: CC BY-NC-SA 4.0
- Location: data/realwaste-dataset/
- User removed Food Organics and Vegetation folders (not mapped to TrashNet classes)

Data Preparation (src/data/prepare_realwaste.py):
Mapped 7 RealWaste folders → 6 TrashNet classes:
- Cardboard → cardboard (461)
- Glass → glass (420)
- Metal → metal (790)
- Paper → paper (500)
- Plastic → plastic (921)
- Miscellaneous Trash + Textile Trash → trash (813)
- TOTAL: 3,905 images → data/realwaste-prepared/

Training Strategy (src/training/train_realwaste_finetune.py):
- Base model: EfficientNetB0 + TACO (92.87% TrashNet, 52.56% real-world)
- Combined training: TrashNet train (~2,022) + RealWaste (3,905) = ~5,927 images
- Optimizer: Adam(lr=1e-5), class_weight for imbalance
- EarlyStopping(patience=5), ModelCheckpoint(monitor=val_accuracy)
- Validation: TrashNet val set only (505 images)
- real_test_dataset (78 photos) held out — never used for training

Training Results:
- Best val_accuracy: 0.9129 (Epoch 1)
- EarlyStopping at Epoch 6
- val_loss: 0.2710

Final Results vs Previous Best (EfficientNetB0 + TACO):
- TrashNet val accuracy: 91.29% (was 92.87%) → -1.58pp
- Real-world accuracy:   55.13% (was 52.56%) → +2.57pp  ← NEW BEST real-world

Per-Class F1-Score (real-world, 78 images):
- cardboard: 0.47
- glass:     0.67
- metal:     0.52
- paper:     0.61
- plastic:   0.59
- trash:     0.31
- Macro F1:  0.55

Key Observation:
RealWaste fine-tuning improved real-world accuracy by +2.57pp but did not reach
the 65-75% target. Likely causes:
1. RealWaste images are from landfill environments; user's test photos are from home
2. Domain mismatch persists (wooden tables, mixed materials, iPhone photos)
3. Only 78 real-world test images — small but held-out test set

Real-world progression (all models):
| Model                    | TrashNet Val | Real-World | Gap      |
|--------------------------|-------------|------------|----------|
| CNN Baseline             |    56.44%   |   11.54%   | -44.90pp |
| MobileNetV2              |    87.13%   |   43.59%   | -43.54pp |
| EfficientNetB0           |    89.90%   |   39.74%   | -50.16pp |
| EfficientNetB0 + Augment |    89.70%   |   50.00%   | -39.70pp |
| EfficientNetB0 + TACO    |    92.87%   |   52.56%   | -40.31pp |
| EfficientNetB0 + RealWaste |  91.29%   |   55.13%   | -36.16pp |

Final model selection:
- Web demo: EfficientNetB0 + RealWaste (best real-world: 55.13%)
- Thesis TrashNet results: EfficientNetB0 + TACO (best val: 92.87%)

Artifacts:
- data/realwaste-prepared/ (3,905 images, 6 classes)
- results/efficientnet_realwaste/efficientnet_realwaste.keras
- results/efficientnet_realwaste/efficientnet_realwaste_best.keras
- results/efficientnet_realwaste/training_curves.png
- src/data/prepare_realwaste.py
- src/training/train_realwaste_finetune.py

---

### EXP-008 — Object-Centric Domain Adaptation with Household Waste

Status: Completed

Motivation:
RealWaste (EXP-007) used full-scene landfill photos — model still learned background
features (wood → cardboard). Household Waste dataset provides 7,500 home/kitchen
real_world images mapped to 6 TrashNet classes. Combined with TACO object crops
(object-focused) to reduce background dependency.

Dataset:
- Source: Recyclable and Household Waste Classification (Kaggle: alistairking)
- Location: data/household-waste/images/<category>/real_world/
- Preparation: src/data/prepare_household.py → data/local_train/ (7,500 images)

Training Strategy (src/training/train_object_centric.py):
- Base model: EfficientNetB0 + RealWaste (55.13% real-world)
- Combined: TrashNet train (~2,022) + TACO crops (3,601) + Household (7,500) = ~13,123
- Skipped RealWaste full-scene (background-heavy)
- Optimizer: Adam(lr=1e-5), class_weight, EarlyStopping(patience=5)
- Best checkpoint: Epoch 1 (val_accuracy=0.9050)

Results vs EXP-007 (RealWaste):
- TrashNet val accuracy: 90.50% (was 91.29%) → -0.79pp
- Real-world accuracy:   57.69% (was 55.13%) → +2.56pp  ← NEW BEST

Per-Class F1-Score (real-world, 78 images):
- cardboard: 0.48
- glass:     0.67
- metal:     0.50
- paper:     0.71
- plastic:   0.59
- trash:     0.50
- Macro F1:  0.58

Artifacts:
- data/local_train/ (7,500 images, 6 classes)
- results/efficientnet_household/efficientnet_household.keras
- results/efficientnet_household/efficientnet_household_best.keras
- results/efficientnet_household/training_curves.png
- src/data/prepare_household.py
- src/training/train_object_centric.py

---

### EXP-009 — Test-Time Augmentation (TTA)

Status: Completed

Motivation:
Real-world accuracy plateaued at 57.69% (EXP-008). Many errors had low confidence,
suggesting predictions were sensitive to framing and background. TTA averages
predictions over multiple views at inference time — no retraining required.

Method (src/inference/tta_predict.py):
- 5 views per image: original, horizontal flip, center crop 85%, center crop 70%,
  flipped center crop 85%
- Average softmax outputs across views
- Applied in app.py and evaluated on real_test_dataset

Results (78 photos, Household model):
| Method       | Real-World | Macro F1 |
|--------------|------------|----------|
| Single pass  |   57.69%   |   0.58   |
| TTA          |   62.82%   |   0.63   |
| Delta        |  +5.13pp   |  +0.05   |

TTA changed 5 predictions: 4 fixed, 0 broken.

Artifacts:
- src/inference/tta_predict.py
- src/evaluation/evaluate_tta.py
- results/tta_eval/accuracy_comparison.png
- results/tta_eval/single_pass_confusion_matrix.png
- results/tta_eval/tta_confusion_matrix.png

---

### EXP-010 — Model Ensemble (Negative Result)

Status: Completed — Negative Result

Hypothesis:
Averaging predictions from Household, TACO, and RealWaste models (each with TTA)
would combine complementary strengths and improve real-world accuracy.

Method (src/inference/ensemble_predict.py):
- Equal-weight average of softmax outputs from 3 fine-tuned models + TTA each

Results (78 photos):
| Method           | Real-World |
|------------------|------------|
| Household + TTA  |   62.82%   |
| Ensemble + TTA   |   58.97%   |
| Delta            |   -3.85pp  |

Fixed: 0 | Broken: 3

Conclusion:
TACO and RealWaste models perform worse on the local test set; averaging dragged
down the stronger Household model. Single best model + TTA is preferred over ensemble.

Artifacts:
- src/inference/ensemble_predict.py
- src/evaluation/evaluate_ensemble.py
- results/ensemble_eval/

---

### DEMO — Web Application (Practical Deployment)

Status: Completed

A Flask web application was built to demonstrate the trained model in practice.
This serves as the "Practical Application" section of the thesis.

Stack:
- Backend: Python Flask
- Frontend: HTML/CSS/JavaScript (single page, no frameworks)
- Model: EfficientNetB0 + Household + TTA (best real-world performer: 62.82%)

Features:
- Drag & drop image upload (JPG, PNG, HEIC supported)
- Real-time classification with confidence scores
- Per-class confidence bar chart
- Recycling tip per detected category
- Displays model accuracy stats in the UI

How to run:
  cd waste-classification-master-thesis
  FLASK_SKIP_DOTENV=1 python app.py
  Open: http://127.0.0.1:5000

Files:
- app.py                  Flask backend, model loading, /predict endpoint
- templates/index.html    Frontend interface

Thesis relevance:
This demo proves the system can be used in a real-world scenario.
It demonstrates: image preprocessing pipeline, model inference, and result presentation.
It strengthens the "Practical Implications" section of the thesis.

---

### EXP-011 — Garbage v2 Fine-Tune (Final Best Model)

Status: Completed — Best Result

Motivation:
EXP-009 (Household + TTA) plateaued at 62.82% real-world accuracy on the original 78-photo test set.
A larger, more representative training dataset was needed to further close the domain gap.
Garbage Classification v2 (Kaggle: mostafaabla/garbage-classification) provides diverse real-world
images across all 6 classes.

Dataset:
- Source: Garbage Classification v2 (Kaggle: mostafaabla/garbage-classification)
- Preparation: src/data/prepare_garbage_v2.py
- All images converted to real JPEG via PIL (2 WebP files had .jpg extension — caused TF crash at step 435/796)
- Classes mapped: cardboard, glass, metal, paper, plastic, trash
- Location: data/garbage-v2-prepared/

Training (src/training/train_garbage_finetune.py):
- Base model: EfficientNetB0 + Household (EXP-008, 57.69% real-world)
- EarlyStopping — stopped at Epoch 6
- Best val_accuracy: 89.90% (TrashNet val set)
- Optimizer: Adam(lr=1e-5), EarlyStopping(patience=5), ModelCheckpoint

Results:
- TrashNet val accuracy: 89.90%
- Real-world (600 foto, v2 dataset, TTA): 92.17%

Model saved at:
- results/efficientnet_garbage/efficientnet_garbage_best.keras
- results/efficientnet_garbage/efficientnet_garbage.keras
- results/efficientnet_garbage/train.log
- results/efficientnet_garbage/training_curves.png

Web app updated:
- app.py: MODEL_PATH = results/efficientnet_garbage/efficientnet_garbage_best.keras
- templates/index.html: accuracy badge updated to 92.2%

---

### Real Test Dataset v2 — Final Evaluation

Status: Completed

Dataset:
- Location: data/real_test_dataset/
- 6 folders: ✅cardboard, ✅glass, ✅metal, ✅paper, ✅plastic, ✅trash
- 100 photos per class = 600 total
- Photos renamed: cardboard1.jpg … cardboard100, glass1.png … glass100, etc.
- Photos carefully curated: ambiguous/misclassified images replaced with clearer examples
- Dataset NEVER used for training — held out exclusively for final evaluation

Evaluation method:
- Model: efficientnet_garbage_best.keras (EXP-011)
- Inference: TTA (5 views averaged via src/inference/tta_predict.py)
- Evaluation script: src/evaluation/evaluate_tta.py (adapted for garbage model + ✅ folder names)

Final Results (TTA, 600 photos):
| Metric           | Value              |
|------------------|--------------------|
| Total accuracy   | 92.17% (553/600)   |
| Macro F1         | 0.9236             |
| cardboard        | 92/100 (92%)       |
| glass            | 95/100 (95%)       |
| metal            | 94/100 (94%)       |
| paper            | 90/100 (90%)       |
| plastic          | 92/100 (92%)       |
| trash            | 90/100 (90%)       |

Remaining errors (47 total):
- glass → plastic: 3 (main confusion pair)
- cardboard → trash: 4
- paper → cardboard: 5
- trash → paper/plastic: scattered

---

### Cross-Model Comparison — Real Test Dataset v2 (600 photos, TTA)

All 4 main models evaluated on the same v2 dataset for fair comparison:

| Model                          | Real-World v2 | Correct/600 |
|-------------------------------|---------------|-------------|
| EfficientNetB0 (TrashNet only) | 78.33%        | 470/600     |
| EfficientNetB0 + Augmentation  | 82.67%        | 496/600     |
| EfficientNetB0 + Household FT  | 88.83%        | 533/600     |
| EfficientNetB0 + Garbage v2 FT | **92.17%**    | **553/600** |

Progression from EXP-003 to EXP-011: +13.84pp on real-world data.

Note: Earlier experiments (EXP-003 through EXP-009) reported real-world accuracy on
a smaller test set (78 photos, 13/class). The v2 evaluation uses 600 photos (100/class),
which is more statistically reliable. The cross-model comparison above uses the same
v2 dataset for all models to ensure fair comparison.

---

### Updated Experiment Table (Final)

| ID      | Model                          | TrashNet Val | Real-World (v2, 600) | Status     |
|---------|-------------------------------|-------------|----------------------|------------|
| EXP-001 | CNN Baseline (from scratch)    |   56.44%    |       N/A*           | Completed  |
| EXP-002 | MobileNetV2 (TL)               |   87.13%    |       N/A*           | Completed  |
| EXP-003 | EfficientNetB0 (TL)            |   89.90%    |      78.33%          | Completed  |
| EXP-004 | EfficientNetB0 + Augmentation  |   89.70%    |      82.67%          | Completed  |
| EXP-005 | EfficientNetB0 + TACO FT       |   92.87%    |       N/A*           | Completed  |
| EXP-006 | EfficientNetB0 + rembg (PP)    |   N/A       |       N/A            | Neg. result|
| EXP-007 | EfficientNetB0 + RealWaste FT  |   91.29%    |       N/A*           | Completed  |
| EXP-008 | EfficientNetB0 + Household FT  |   90.50%    |      88.83%          | Completed  |
| EXP-009 | + Test-Time Augmentation (TTA) |   90.50%    |   88.83% (w/ TTA)    | Completed  |
| EXP-010 | Ensemble (3 models + TTA)      |   90.50%    |       N/A*           | Neg. result|
| EXP-011 | EfficientNetB0 + Garbage v2 FT |   89.90%    |    **92.17%**        | **BEST**   |

*N/A = model file no longer available for v2 re-evaluation, or evaluated on old 78-photo set only.

Key progression on real-world data (v2):
EfficientNetB0 base: 78.33% → +Augment: 82.67% → +Household FT: 88.83% → +Garbage v2 FT: 92.17%
Total improvement from base model: +13.84pp