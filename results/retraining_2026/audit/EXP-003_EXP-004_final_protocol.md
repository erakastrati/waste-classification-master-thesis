# Final protocol for corrected EXP-003 and EXP-004

Status: **implemented, not trained**. The pipeline is `src/retraining/`. Preflight passed and is recorded in `pretraining_validation.json`. Full training has not been started.

This replaces the checkpoint and seed parts of `EXP-003_EXP-004_proposed_protocol.md`. The shared model, the deduplicated manifests, and the 0% to +20% contrast range stay as agreed there.

## 1. Manifest loader

The loader reads `results/retraining_2026/manifests/trashnet_seed42_dedup/train.txt` and `val.txt`. It does not call `image_dataset_from_directory(..., validation_split=0.2)`.

Labels are integers from this fixed table, which is the alphanumeric order the historical loader used:

| Index | Folder |
|---:|---|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

The class is the parent directory of each manifest path. A path whose parent is not in the table stops the run. Labels are sparse integers, matching `sparse_categorical_crossentropy`.

Decoding and resizing follow `image_dataset_from_directory` in Keras 3.10.0 (`image_dataset_utils.py` `load_image`):

1. `tf.io.read_file`
2. `tf.image.decode_image(..., channels=3, expand_animations=False)`
3. `tf.image.resize` to `(224, 224)` with `method="bilinear"`
4. No crop and no pad
5. `efficientnet.preprocess_input`, which returns the tensor unchanged
6. Batches of 32

Pixels stay in the 0–255 range. The EfficientNetB0 stem does the 1/255 rescale, the ImageNet normalization loaded with the weights, and the `1/sqrt([0.229, 0.224, 0.225])` rescale. Nothing outside the model divides by 255.

Validation is read in the sorted order of `val.txt` and is not shuffled. The historical loader also buffer-shuffled validation, which changes order only. Accuracy over all 501 images does not depend on that order. Keeping the manifest order ties the metric to the file hash.

## 2. The only intended difference

Every training seed runs both experiments with the same:

- manifest paths and label table
- `keras.utils.set_random_seed(training_seed)` before the model is built
- EfficientNetB0, `include_top=False`, `weights="imagenet"`, input `(224, 224, 3)`
- head: GlobalAveragePooling2D, Dense(128, relu), Dropout(0.3), Dense(6, softmax)
- Adam defaults `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7`, `amsgrad=False`
- phase 1: base frozen, Adam `1e-3`, at most 10 epochs
- phase 2: last 30 base layers unfrozen, every batch-norm layer frozen, new Adam `1e-5`, at most 15 epochs
- loss, accuracy metric, batch size 32, no class weights
- the checkpoint and early-stopping rules in section 4

EXP-004 alone inserts `RandomFlip("horizontal")`, `RandomRotation(0.15)`, `RandomZoom(0.15)`, and `RandomContrast(0.2)` before the base, each with `seed=training_seed`.

## 3. Batch normalization

Keras updates moving statistics only when both flags are true (`batch_normalization.py` lines 253–265):

```python
if training and self.trainable:
    # update moving_mean and moving_variance
```

`trainable = False` also removes gamma and beta from the optimizer (docstring lines 97–113) and forces inference normalization.

The base is called as `base_model(x)` with no `training` argument. During `fit`, Keras injects `training=True` on training batches and `training=False` on validation batches (`functional.py` `operation_fn`). A `training=False` written into the graph is overwritten by that injection, so it is not the freeze.

Phase 1 sets `base.trainable = False` before `compile`. That freezes every inner layer, including batch norm.

Phase 2, in this order, immediately before its `compile`:

1. `base.trainable = True`
2. `base.layers[:-30].trainable = False`
3. every `BatchNormalization` in the base, including the seven inside the last 30, set to `trainable = False`
4. `compile` Adam at `1e-5`

The stem `Normalization` layer is not a batch-norm layer. It sits outside the last 30, so step 2 leaves it frozen. Step 3 is what stops the seven tail batch-norm layers from updating ImageNet statistics when `fit` passes `training=True`.

## 4. Checkpoints

Each phase gets a new `ModelCheckpoint` and a new `EarlyStopping`. Early stopping monitors `val_accuracy`, patience 5, `restore_best_weights=True`, and its memory is not the published model. Each checkpoint monitors `val_accuracy`, `mode="max"`, `save_best_only=True`. `mode="max"` uses `numpy.greater`, so an equal accuracy does not replace the saved file.

- `phase1_best.keras` is the best validation accuracy inside phase 1.
- `phase2_best.keras` is the best validation accuracy inside phase 2, even when it is worse than phase 1.
- The epoch log records phase, epoch within the phase, global epoch, and `val_accuracy`.
- The published model is whichever of the two files has the higher `val_accuracy`. A tie keeps phase 1.
- `selection.json` records the chosen file, phase, epoch within that phase, global epoch, and `val_accuracy`.

## 5. Seeds

The manifest was built with CPython `random.Random(42)` per class. That seed is not reused as a training seed, and the split is not redrawn.

For a run with training seed `S`:

1. `keras.utils.set_random_seed(S)` runs before the model exists. In Keras 3.10.0 this sets `random.seed`, `numpy.random.seed`, and `tf.random.set_seed` (`rng_utils.py` lines 55–58). Dense initialization and dropout then draw from that seed.
2. Training order uses a separate `random.Random(S)`. Each epoch shuffles the full training list with that generator. Both experiments use this shuffle. It is not the historical buffer of 256.
3. EXP-004 passes `seed=S` into all four augmentation layers. EXP-003 has no augmentation layers. Dropout in both experiments follows the TensorFlow seed from step 1.

`tf.data` deterministic option is set on the input pipeline. TensorFlow operation determinism stays off. A seed does not promise bit-identical GPU repeats. Both experiments share that limit.

## 6. Three training seeds

`src/seed_study/protocol.py` already defines `TRAINING_SEEDS = (42, 123, 456)` for repeated EXP-003 runs. Those runs used the old 2,022/505 split and are not results of this protocol. No committee sentence naming the seeds was found under `docs/`. The three integers already in the repository are the ones this protocol uses.

EXP-003 and EXP-004 each run seeds 42, 123, and 456 on the same corrected manifest. The comparison at each seed is paired. The reported summary is the mean and sample standard deviation of the three published validation accuracies for each experiment. Historical single-run numbers stay outside that summary.

## 7. Augmentation check

The four layers and their Keras 3.10.0 ranges match the saved EXP-004 config and the agreed protocol:

| Layer | Source argument | Actual draw |
|---|---|---|
| `RandomFlip("horizontal")` | horizontal | horizontal flip |
| `RandomRotation(0.15)` | fill `reflect`, bilinear | ±54 degrees |
| `RandomZoom(0.15)` | width factor left empty | scale 0.85 to 1.15, both axes together |
| `RandomContrast(0.2)` | single float | multiplier 1.0 to 1.2 |

Contrast discrepancy: `docs/notes.md` describes ±20%. `RandomContrast._FACTOR_BOUNDS` is `(0, 1)`, so a single float `0.2` is stored as `[0, 0.2]` and the multiplier is drawn from `[1.0, 1.2]`. That is 0% to +20%, not a decrease. The saved checkpoint has factor `[0, 0.2]`. This protocol keeps that implemented range for EXP-004. It does not change the layer to a symmetric ±20% draw.

## 8. Historical files

Runs write only under new directories `experiments/EXP-003/runs/<seed>/` and `experiments/EXP-004/runs/<seed>/`, and stop if the directory exists. They do not write to `results/efficientnet/`, `results/efficientnet_augmented/`, or `results/efficientnet_taco/`.

`src/training/train_taco_finetune.py` keeps `BASE_MODEL_PATH = "results/efficientnet_augmented/efficientnet_augmented.keras"`. That file is not replaced, and EXP-005 is not retrained.

## Remaining decision

None of the eight points above is left open. Implementation waits until this protocol is accepted as the run specification. The contrast range is the implemented 0% to +20% range. Changing it to the ±20% wording in the thesis note would be a different experiment and is not part of this protocol.
