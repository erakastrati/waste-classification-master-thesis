# Proposed training protocol for EXP-003 and EXP-004

**Superseded for checkpoint records and training seeds by `EXP-003_EXP-004_final_protocol.md`.** The model, manifests, batch-norm freeze, and contrast range below remain the agreed design. This file is the earlier proposal.

Status: **proposal**. It is not implemented and it has not been run. It does not change `src/training/train_efficientnet.py`, `src/training/train_efficientnet_augmented.py`, the historical checkpoints, the manifests, or the dataset.

Both experiments use this protocol. The only intended difference is that EXP-004 inserts the four augmentation layers and EXP-003 does not.

## Data

Read membership only from:

- `results/retraining_2026/manifests/trashnet_seed42_dedup/train.txt` (2,016 paths)
- `results/retraining_2026/manifests/trashnet_seed42_dedup/val.txt` (501 paths)

Do not call `image_dataset_from_directory(..., validation_split=0.2)`.

The label is the parent folder. The index is fixed:

| Index | Class |
|---:|---|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

Decode each file, resize to 224×224 with bilinear interpolation, and cast to `float32`. Leave pixel values in the 0–255 range. Call `efficientnet.preprocess_input` only as a documented no-op, or omit it. Do not divide by 255 outside the model. The EfficientNetB0 stem already rescales by 1/255, applies the ImageNet `Normalization` loaded with the weights, and rescales by `1/sqrt([0.229, 0.224, 0.225])`.

Batch size is 32. No class weights.

Training order: at the start of each epoch, shuffle the training paths with `random.Random` seeded from 42 so the sequence of permutations is deterministic. Use the full list, not a shuffle buffer of 256. Validation order is the sorted order of `val.txt`, with no shuffle, so a metric can be tied to `val.txt` SHA-256 `b4b344db8ac6e2cc99199ac12754b8fe951ff8c2cf0d36e14e88bb567a022cb5`.

`metal149.jpg` and `metal9.jpg` are both in `train.txt`. The loader must keep both.

## Model

One builder produces both models.

`keras.utils.set_random_seed(42)` runs once, before the builder, in each process.

EfficientNetB0, `include_top=False`, `weights="imagenet"`, input `(224, 224, 3)`. Head: GlobalAveragePooling2D, Dense(128, relu), Dropout(0.3), Dense(6, softmax).

Do not call the base with a permanent `training=False`. `model.fit` supplies `training=True` on training batches and `training=False` on validation batches. Dropout and the augmentation layers must follow that flag. The historical saved graphs store a concrete `training` value on each node; Keras 3.10 overwrites it during `fit`, and the new builder must not rely on that accident.

EXP-004, and only EXP-004, inserts these layers before the base, each with `seed=42`:

| Layer | Argument | Recorded range |
|---|---|---|
| `RandomFlip` | `"horizontal"` | horizontal only |
| `RandomRotation` | `0.15` | ±54 degrees, fill `reflect`, bilinear |
| `RandomZoom` | `0.15` | scale 0.85 to 1.15 on both axes together, fill `reflect`, bilinear |
| `RandomContrast` | `0.2` | multiplier 1.0 to 1.2, which is 0% to +20% |

That contrast range is the Keras 3.10.0 behavior of the current script. It is not changed to ±20%. Augmentation is inactive when the model is called with `training=False`.

## Phases

Loss `sparse_categorical_crossentropy`. Metric `accuracy`. Adam uses the Keras 3.10.0 defaults except for the learning rate: `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7`, `amsgrad=False`.

Phase 1. Set the base `trainable = False`. Compile Adam at `1e-3`. Fit for at most 10 epochs.

Phase 2. Set the base `trainable = True`. Set `base.layers[:-30]` to `trainable = False`. Then set every `BatchNormalization` layer in the base, and the stem `Normalization` layer, to `trainable = False`. Compile a new Adam at `1e-5`. Fit for at most 15 epochs.

The last 30 layers are the same list as in the historical checkpoint, from `block6d_expand_bn` through `top_activation`. Seven of them are batch-norm layers. This protocol unfreezes the convolutions in that tail and leaves every batch-norm layer frozen, so ImageNet moving statistics, gamma, and beta stay put. The June scripts would have left those seven batch-norm layers trainable. Both corrected experiments use this freeze. It is not an extra difference between them.

## What is kept

Create one `ModelCheckpoint` before phase 1 and pass that same object to both fits. Monitor `val_accuracy`, `mode="max"`, `save_best_only=True`. It must not reset its best value at phase 2. The file can therefore remain the phase-1 weights when phase 2 never beats them.

Create a new `EarlyStopping` for each phase, patience 5, `restore_best_weights=True`, monitoring `val_accuracy`. Its restored weights are not the published model. After phase 2, load the checkpoint file and save that object as the run result. Also write the per-epoch history, including which phase produced the saved checkpoint.

## Where outputs go

Write a new run directory under `experiments/EXP-003/runs/` and `experiments/EXP-004/runs/`. Stop if the chosen directory already exists. Do not write into `results/efficientnet/` or `results/efficientnet_augmented/`. Do not replace `results/efficientnet_augmented/efficientnet_augmented.keras`. EXP-005 still loads that historical file and does not unfreeze its base.

Record in the run directory: the manifest hashes, the seed, the Python and library versions, the augmentation switch, the batch-norm freeze, and the checkpoint epoch.

## Before the first run

1. Accept this protocol, including the batch-norm freeze and the 0% to +20% contrast range.
2. Add a new entry point. Do not edit the June scripts in place.
3. Confirm on one batch that EXP-004 augmentation changes a training forward pass and does not change a `training=False` forward pass, and that EXP-003 has no augmentation layers.
4. Confirm both runs read the 2,016 and 501 manifest paths and no others.

GPU kernels can still make two seeded runs differ. The run records the library versions so a later repeat uses the same stack. That limit is accepted. It is the same for both experiments.
