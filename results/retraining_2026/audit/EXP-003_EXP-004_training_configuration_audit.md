# EXP-003 and EXP-004 training-configuration audit

Read-only check of the current training scripts, the Keras 3.10.0 sources in the project virtual environment, and the saved `.keras` configs. No script, manifest, checkpoint, or dataset file was modified. No training was started.

The corrected split this protocol must use is `results/retraining_2026/manifests/trashnet_seed42_dedup/`. The scripts below do not read it.

Status labels: **VERIFIED** from a file opened in this pass, **RECONSTRUCTED** from source logic that was not executed, **UNVERIFIED** where a June run left no numeric history.

## 1. Architecture and ImageNet initialization

Both experiments are EfficientNetB0, `include_top=False`, `weights="imagenet"`, input `(224, 224, 3)`.

EXP-003 builds that model in `src/models/efficientnet_model.py` `build_efficientnet_model`. The head is GlobalAveragePooling2D, Dense(128, relu), Dropout(0.3), Dense(6, softmax). The base is set `trainable = False` before the first fit.

EXP-004 builds the same head inline in `src/training/train_efficientnet_augmented.py` lines 69–92, then inserts four augmentation layers in front of the base. There is no shared builder.

ImageNet weights are loaded inside Keras `EfficientNet` (`keras/src/applications/efficientnet.py` lines 420–434) from `efficientnetb0_notop.h5`. The head is created after that load, so the head starts from the layer defaults, not from ImageNet. Neither script sets a global random seed before that creation. **VERIFIED** in source. The numerical head initialization of the June runs is **UNVERIFIED**.

The saved files `results/efficientnet/efficientnet.keras` and `results/efficientnet_augmented/efficientnet_augmented.keras` both record Keras 3.10.0. Their outer models have Adam at learning rate `0.0010000000474974513`, loss `sparse_categorical_crossentropy`, and metric `accuracy`. The nested `efficientnetb0` has 238 layers and `trainable` false on all 237 non-input layers. **VERIFIED** from `config.json` inside those archives. These files are the phase-1 compile state, not a saved unfrozen fine-tune.

## 2. Input size, preprocessing, normalization

| Step | What the scripts do | Status |
|---|---|---|
| Resize | `image_size=(224, 224)`. Keras default interpolation is `"bilinear"` (`image_dataset_utils.py` line 30). | **VERIFIED** |
| `preprocess_input` | `keras.applications.efficientnet.preprocess_input` returns its input unchanged (lines 830–848). | **VERIFIED** |
| Inside the base | `Rescaling(1/255)`, then a `Normalization` layer, then a second `Rescaling` by `1/sqrt(0.229)`, `1/sqrt(0.224)`, `1/sqrt(0.225)` (lines 325–338 and line 205). | **VERIFIED** in source |
| Normalization mean and variance | Not written as Python constants. They are whatever the ImageNet weight file loads into the `Normalization` layer. | **UNVERIFIED** as numbers in this pass |

No script divides by 255 again outside the model. Doing so would scale the image twice.

## 3. How data are loaded, and how the new manifests are not used

EXP-003 (`train_efficientnet.py` lines 27–43) and EXP-004 (`train_efficientnet_augmented.py` lines 33–49) both call `image_dataset_from_directory` on `data/trashNet-dataset` with `validation_split=0.2`, `seed=42`, batch size 32, and no `shuffle` argument. The default `shuffle` is True (line 26 of `image_dataset_utils.py`). Membership is a global NumPy `RandomState(42)` shuffle of the directory listing, not the corrected manifest (`dataset_utils.py` lines 613–620). After membership is chosen, each subset is shuffled again with buffer `batch_size * 8` = 256 (lines 250–253 and 388–389).

`trashnet_seed42_dedup/train.txt` and `val.txt` are not referenced by either script. Running the current files would ignore the corrected split and rebuild the historical directory split, including the three mislabeled files and the seven redundant copies.

Labels are integers from alphanumeric class-folder order. For the six TrashNet folders that order is cardboard, glass, metal, paper, plastic, trash, indices 0–5. `label_mode` is left at its default `"int"`, which matches `sparse_categorical_crossentropy`. **VERIFIED** as library defaults. The June scripts never pass an explicit `class_names` list.

## 4. Augmentation, and whether it is the only difference

EXP-003 has no augmentation layers.

EXP-004, lines 80–85, adds, in order:

| Layer | Argument in source | Behavior in Keras 3.10.0 |
|---|---|---|
| `RandomFlip("horizontal")` | no seed | horizontal flip only |
| `RandomRotation(0.15)` | default fill `"reflect"`, interpolation `"bilinear"` | factor becomes `[-0.15, 0.15]`, drawn as degrees `[-54, 54]` (`random_rotation.py` lines 203–204) |
| `RandomZoom(0.15)` | `width_factor` left as `None` | height factor becomes `[-0.15, 0.15]`; width zoom is copied from height (`random_zoom.py` lines 332–346), so aspect is preserved. Scale is drawn from 0.85 to 1.15. Fill `"reflect"`, interpolation `"bilinear"`. |
| `RandomContrast(0.2)` | no seed | `_FACTOR_BOUNDS` is `(0, 1)`, so the stored factor is `[0, 0.2]`, and the multiplier is drawn from `[1.0, 1.2]`. That is 0% to +20% contrast, not ±20%. |

The saved EXP-004 config records rotation `[-0.15, 0.15]`, zoom height factor `0.15`, contrast `[0, 0.2]`, and `seed: null` on all four layers. **VERIFIED**. `docs/notes.md` describes contrast as ±20%. The source and the checkpoint disagree with that note.

The training hyperparameters other than those four layers match: same loss, metric, batch size, epoch caps, Adam rates, callbacks, and directory-split arguments. The models are still built by two copies of the head, and they write to two result directories. Those are not data differences, but they are not one shared program with one augmentation switch.

Augmentation layers declare `call(self, data, training=True)`. Dropout declares `call(self, inputs, training=False)`. Keras stores that resolved value on the functional node. The saved EXP-004 graph therefore shows augmentation nodes with `training: true` and dropout with `training: false`. At `fit` time, `Functional.call` overwrites `training` on every layer that accepts it (`functional.py` `operation_fn`, lines 636–648). Training steps pass `training=True` and validation steps pass `training=False`. Validation is not augmented by that override. A caller that invoked the augmentation layers by themselves would see the stored flag instead. The corrected protocol should not depend on someone remembering this override.

## 5. Optimizer, learning rate, batch size, loss, metric

Both scripts:

| Item | Value |
|---|---|
| Optimizer | `Adam(learning_rate=1e-3)` in phase 1, `Adam(learning_rate=1e-5)` in phase 2. A new Adam is created at each `compile`. |
| Unset Adam arguments | Keras 3.10.0 defaults: `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7`, `amsgrad=False` (`adam.py` lines 21–39). |
| Loss | `sparse_categorical_crossentropy` |
| Metric | `accuracy` |
| Batch size | 32 |
| Class weights | none |

No learning-rate schedule is defined. Phase 2 does not continue the phase-1 optimizer state, because `compile` is called again.

## 6. The two phases

Phase 1, both scripts: base frozen, Adam `1e-3`, `epochs=10`, the shared callbacks.

Phase 2, both scripts: `base_model.trainable = True`, then every layer in `base_model.layers[:-30]` is set `trainable = False`. Adam `1e-5`, `epochs=15`, the same callback objects.

The last 30 base layers, taken from the saved 238-layer `efficientnetb0` config, are `block6d_expand_bn` through `top_activation`. **VERIFIED** against `results/efficientnet/efficientnet.keras`. They include `block6d` from its expand batch-norm onward, all of `block7a`, `top_conv`, `top_bn`, and `top_activation`.

The saved checkpoints do not contain this unfrozen state. All 237 non-input base layers are `trainable: false`, and the saved learning rate is `1e-3`. A phase-2 save would have the last 30 base layers trainable and learning rate `1e-5`. The files that exist are phase-1 models. Whether the June phase-2 `fit` returned is a separate question already recorded from the training plots. It is not visible in the checkpoints. **VERIFIED** for the files. Phase-2 weight updates are **UNVERIFIED**.

## 7. Batch normalization during fine-tuning

The scripts never search for `BatchNormalization` and never set those layers to `trainable = False` after the unfreeze.

Seven batch-norm layers sit inside the last 30 and would become `trainable = True` in phase 2: `block6d_expand_bn`, `block6d_bn`, `block6d_project_bn`, `block7a_expand_bn`, `block7a_bn`, `block7a_project_bn`, and `top_bn`. **VERIFIED** from the saved layer list.

The source also calls the base as `base_model(..., training=False)`. That value is stored on the node. During `fit`, the outer `training` flag overwrites it, as described in section 4. A phase-2 training step would therefore call those seven trainable batch-norm layers with `training=True`, which updates their moving mean and variance as well as gamma and beta. Batch-norm layers outside the last 30 stay `trainable = False`, so they keep using their moving statistics.

The saved checkpoints never reached that state. In the files on disk, every base batch-norm layer is non-trainable, and the base node was saved with `training: false`.

## 8. Checkpoints, early stopping, and which weights survive

Both experiments create one `ModelCheckpoint` and one `EarlyStopping` and pass those same objects to both `fit` calls.

`ModelCheckpoint` monitors `val_accuracy`, `save_best_only=True`, `mode="max"`. Its class has no `on_train_begin`, so `self.best` is not reset when phase 2 starts (`model_checkpoint.py` lines 148–165). The file keeps the best validation accuracy across both phases.

`EarlyStopping` uses patience 5 and `restore_best_weights=True`. `on_train_begin` sets `self.best = None` (lines 140–146). Phase 2 therefore judges improvement only against phase 2, and `restore_best_weights` restores the best weights of that phase only. Those in-memory weights can be worse than the phase-1 checkpoint.

After phase 2, both scripts ignore the in-memory model. They `load_model` the checkpoint file and save that object again:

- EXP-003: `results/efficientnet/efficientnet_best.keras`, then `results/efficientnet/efficientnet.keras`
- EXP-004: `results/efficientnet_augmented/efficientnet_augmented_best.keras`, then `results/efficientnet_augmented/efficientnet_augmented.keras`

The retained model is the checkpoint, not the early-stopping restoration. The final file and the best file are not byte-identical. This pass did not compare their weight tensors. The load path in the training script is still the best file, followed by a second save.

## 9. Seeds and reproducibility

Neither training script calls `keras.utils.set_random_seed`, `tf.random.set_seed`, or enables deterministic TensorFlow ops. **VERIFIED** by their absence.

`seed=42` on `image_dataset_from_directory` seeds the membership shuffle and the later buffer shuffle. It does not seed the head initialization, dropout, or the augmentation `SeedGenerator(None)` objects.

The manifest seed 42 is a different generator. It is CPython `random.Random(42)`, reseeded per class, and it only decided which file is in which list. It does not make a training run reproducible by itself.

Two runs of the current scripts can diverge in head initialization, dropout masks, augmentation samples, and GPU reductions even when the file lists match.

## 10. Which checkpoint initializes EXP-005

`src/training/train_taco_finetune.py` line 31 sets

`BASE_MODEL_PATH = "results/efficientnet_augmented/efficientnet_augmented.keras"`

and line 122 loads that path. It does not load `efficientnet_augmented_best.keras`. The training script writes the final path by saving the loaded best file, so the intended content is the best EXP-004 checkpoint after that June run.

After the load, EXP-005 compiles Adam at `1e-5` and fits. It never sets `efficientnetb0.trainable = True`. The loaded base therefore stays frozen. The comment on lines 145–146 says “Fine-tune entire model”. The code does not do that. **VERIFIED**.

EXP-005 validation uses `shuffle=False` (lines 70–78). EXP-003 and EXP-004 use the default `shuffle=True`. Those are different TrashNet validation memberships. EXP-005 also still points at the historical result path. A corrected EXP-004 must not be copied over `results/efficientnet_augmented/efficientnet_augmented.keras`.

## Risks to close before retraining

1. The current scripts cannot consume `trashnet_seed42_dedup`. Running them retrains the old directory split.
2. The two model constructions can drift. One builder has to feed both experiments.
3. Phase 2 would unfreeze seven batch-norm layers. The scripts never say whether ImageNet batch-norm statistics may move. That choice has to be explicit and identical for both experiments.
4. Early stopping and the checkpoint do not select the same weights. The retained file must be defined as the checkpoint with the best `val_accuracy` across both phases.
5. No global seed is set. Head initialization and EXP-004 augmentation are not reproducible from the scripts as written.
6. Contrast is 0% to +20% in code and in the saved layer config. The thesis note says ±20%. The rerun has to keep the code behavior and not silently “correct” it in only one experiment.
7. Historical result directories must not be overwritten. EXP-005 still loads the historical EXP-004 file.
8. GPU deterministic execution is not configured. A seed makes the run controlled. It does not by itself prove bit-identical repeats on a GPU.

The proposed resolution is `EXP-003_EXP-004_proposed_protocol.md` in this directory. It is not a training run.
