# Corrected four-experiment comparison

Status: EXP-001 and EXP-002 pipelines implemented and preflighted. Full training has not started. The EXP-003 and EXP-004 trainer is unchanged.

## Comparison

All four experiments use `results/retraining_2026/manifests/trashnet_seed42_dedup/` (2,016 training images, 501 validation images) and training seeds 42, 123, and 456. That is 12 runs. EXP-003 is not trained twice.

The 501-image figure is validation accuracy. It is also the model-selection set for EXP-002, EXP-003, and EXP-004, and it is the only TrashNet score for the EXP-001 weights after epoch 10. The historical 78-image set is not part of this comparison. The 600-image set in `data/real_test_dataset` stays held out from training, checkpoint selection, hyperparameter tuning, and this report.

## What each experiment keeps

| Experiment | Model and preprocessing | Schedule and selection |
|---|---|---|
| EXP-001 | Original CNN. Pixels divided by 255. | Adam at the string `"adam"` defaults, 10 epochs, no checkpoint and no early stopping. The epoch-10 weights are the result. |
| EXP-002 | Original MobileNetV2, ImageNet, `include_top=False`. `preprocess_input` in TensorFlow mode, `x/127.5 - 1`. | Phase 1: Adam `1e-3`, 10 epochs, base frozen. Phase 2: unfreeze the last 30 layers, Adam `1e-5`, 15 epochs. The 11 batch-norm layers inside those 30 stay trainable: `block_13_project_BN` through `Conv_1_bn`. One `ModelCheckpoint` spans both phases; an equal validation accuracy does not replace the file. Evaluation loads `best.keras`. |
| EXP-003 | Approved EfficientNetB0 protocol. | Separate phase checkpoints. The higher validation accuracy is published, and a tie keeps phase 1. Every base batch-norm layer stays frozen. |
| EXP-004 | Same as EXP-003 plus the approved augmentation. | Same selection and batch-norm rule as EXP-003. |

## Corrections applied to EXP-001 and EXP-002

These are the only deliberate departures from `src/training/train_cnn.py` and `src/training/train_mobilenet.py`:

- The corrected manifest replaces `image_dataset_from_directory(..., validation_split=0.2, seed=42)`.
- `keras.utils.set_random_seed(training_seed)` runs before each model is built. The original scripts set no seed. Seed 42 is a new controlled run, not the historical checkpoint.
- Each epoch shuffles the full training list with `random.Random(training_seed)`. The original loader used a shuffle buffer of 256.
- Validation stays in sorted manifest order. The original loader shuffled it.
- The input pipeline sets `tf.data.Options.deterministic` to True. TensorFlow operation determinism stays off.

The original files, `results/cnn_baseline/`, `results/mobilenet/`, and `results/seed_study/` are not outputs. New runs go to `experiments/EXP-001/runs/seed_{seed}/` and `experiments/EXP-002/runs/seed_{seed}/`.

## Report

After all 12 runs exist, `src/corrected_baselines/report.py` reads their validation files and writes `results/retraining_2026/comparison/summary.json`. For each experiment it reports accuracy and macro F1 as mean ± sample standard deviation (`ddof=1`) across the three seeds, plus per-class precision, recall, and F1 on the same three seeds. Support is the corrected validation count.

EXP-001 and EXP-002 write those per-class metrics during their own evaluation. EXP-003 and EXP-004 already write validation accuracy from their trainer. Their per-class metrics are added later by scoring `published.keras` on the same 501-image manifest. That scoring is not part of this preflight, and it does not edit `src/retraining/`.

A record is refused if it is not labeled validation, if it does not cover 501 images, or if it refers to the historical seed study, the 78-image set, or the 600-image set. Historical numbers stay in `results/seed_study/` and are not inputs to the mean.

## Known differences that remain inside the 12-run table

EXP-002 does not use the EXP-003 batch-norm freeze. Its original graph still calls the base with `training=False`. During `fit`, Keras supplies the training flag itself, so batch-norm layers in the unfrozen tail can update their moving statistics. That is the original script's effective behavior.

EXP-001, EXP-002, and EXP-003/004 also use different checkpoint rules, listed in the table above. The report stores the rule next to each result and does not re-rank the four experiments onto one rule.
