# EXP-003 audit

Historical run: EfficientNetB0 transfer learning on TrashNet, without augmentation. Phase 1 read the script, the saved Keras files, the classification report, and the dataset audit. It did not call `fit`, `evaluate`, or `predict`.

## Identity

| Item | Finding | Status |
|---|---|---|
| Training script | `src/training/train_efficientnet.py` | **VERIFIED** |
| Model builder | `build_efficientnet_model` in `src/models/efficientnet_model.py` lines 11–35 | **VERIFIED** |
| Checkpoint written during `fit` | `results/efficientnet/efficientnet_best.keras` line 73 | **VERIFIED** path in source. File exists. |
| File evaluated and then saved as the final model | `load_model` of the best file, lines 141–146, then `save` to `results/efficientnet/efficientnet.keras`, lines 168–169 | **VERIFIED** in source. Both files exist. |
| Parent checkpoint | None. The base is constructed with `weights="imagenet"` (`efficientnet_model.py` lines 13–17). | **VERIFIED** |
| Saved Keras version | `metadata.json` inside both zip files: Keras 3.10.0. Best saved `2026-06-10@22:39:57`. Final saved `2026-06-10@22:52:02`. | **VERIFIED** from the zip metadata |

There is no separate training log and no `history.json`. Epoch metrics in `docs/notes.md` lines 310–340 are **UNVERIFIED**. The only numeric evaluation artifact is `results/efficientnet/classification_report.txt`, written by `src/evaluation/evaluate_efficientnet.py` lines 93–94. Its file timestamp is 11 June 2026, 10:52, the morning after the checkpoint.

## What the script does

The script is one process with two `model.fit` calls.

**Phase 1, lines 88–107.** `base_model.trainable = False` is set in the builder (line 19). `model.compile` uses `Adam(learning_rate=1e-3)`, `sparse_categorical_crossentropy`, and `accuracy` (lines 92–96). `epochs=10`.

**Phase 2, lines 110–134.** The script sets `base_model.trainable = True`, then sets `trainable = False` on `base_model.layers[:-30]` (lines 114–117). It compiles again with `Adam(learning_rate=1e-5)` (lines 119–123) and calls `fit` with `epochs=15` (lines 129–134).

The same `ModelCheckpoint` and `EarlyStopping` objects are passed to both `fit` calls (lines 72–85, 106, 133).

| Callback | Arguments in source | Installed Keras 3.10.0 behaviour |
|---|---|---|
| `ModelCheckpoint` | `monitor="val_accuracy"`, `save_best_only=True`, `mode="max"`, `verbose=1`. No `initial_value_threshold`. | `model_checkpoint.py` lines 158–165: `mode="max"` uses `numpy.greater`, so an equal accuracy does not replace the file. The class has no `on_train_begin`, so `self.best` is kept across the two `fit` calls. |
| `EarlyStopping` | `monitor="val_accuracy"`, `patience=5`, `restore_best_weights=True`, `verbose=1`. `mode` omitted. | `early_stopping.py` lines 140–146: `on_train_begin` resets `wait`, `best`, and `best_weights`. Phase 2 therefore does not inherit Phase 1's patience counter. Restored weights, when stopping happens, are the best weights **of that `fit` call only** (lines 187–194). |

The script then ignores the in-memory weights and loads the checkpoint file (lines 141–146). The number printed as the final validation result is the checkpoint's `evaluate` result, not whatever EarlyStopping restored after Phase 2. That `evaluate` stdout was not saved. **UNVERIFIED** as a recorded number.

## Saved checkpoint versus the two-phase source

**VERIFIED** by reading `config.json` and hashing every tensor in `model.weights.h5` for both files:

- `efficientnet_best.keras` and `efficientnet.keras` have the same 326 tensors, including optimizer slots. Zero tensors differ.
- The nested model named `efficientnetb0` has `trainable: false`. Its 238 config layers contain 0 with `trainable: true` and 237 with `trainable: false`. The remaining layer is the base input, whose `trainable` field is null. The last layers, `block7a_project_conv` through `top_activation`, are all `trainable: false`.
- `compile_config` on both files is Adam, `learning_rate` 0.0010000000474974513, loss `sparse_categorical_crossentropy`, metric `accuracy`.
- Non-optimizer parameters: 4,214,313. This matches `docs/notes.md` line 303.
- Head tensors: `dense` kernel `(1280, 128)` and bias `(128,)`, `dense_1` kernel `(128, 6)` and bias `(6,)`. Those four arrays are 164,742 parameters, which matches the Phase 1 trainable count in `docs/notes.md` line 306.
- The inbound call of `efficientnetb0` is stored with `"training": false`. That matches the builder, `efficientnet_model.py` line 23: `base_model(inputs, training=False)`.

A checkpoint written after lines 114–117 would have the last 30 base layers marked trainable, and a compile written at lines 119–123 would store learning rate `1e-5`. The file on disk has neither. **VERIFIED:** the persisted model is the frozen-base, learning-rate-`1e-3` model.

**UNVERIFIED:** whether Phase 2's `fit` actually ran. The 12 minutes between the best-file timestamp and the final-file timestamp only show that the process continued after the best file was written. The final file is a resave of those same weights, which lines 168–169 do after both phases in source. Continuation is compatible with a Phase 2 that never beat the Phase 1 checkpoint, and also compatible with the process stopping after Phase 1 if the script on disk in June differed from the script read now. No June log is in the repository. `docs/notes.md` lines 326–330 says Phase 2 ran for 6 epochs, peaked at 89.50% on its first epoch, and early-stopped. That paragraph has no accompanying history file. It stays **UNVERIFIED**.

`src/seed_study/` is a later controlled rerun of the EXP-003 configuration. Its metrics are not the June checkpoint and are not used here.

## Architecture and preprocessing

**VERIFIED** from `efficientnet_model.py` lines 13–31 and the saved config:

- EfficientNetB0, `include_top=False`, `input_shape=(224, 224, 3)`, `weights="imagenet"`.
- Then `GlobalAveragePooling2D`, `Dense(128, relu)`, `Dropout(0.3)`, `Dense(6, softmax)`.
- No `RandomFlip`, `RandomRotation`, `RandomZoom`, or `RandomContrast` layer in the script or in the saved config.

`tensorflow.keras.applications.efficientnet.preprocess_input` is applied to both datasets (`train_efficientnet.py` lines 50–56). In the installed Keras 3.10.0 this function returns its input unchanged (`keras/src/applications/efficientnet.py` lines 829–848). The model itself rescales by `1/255` and applies ImageNet normalization (`efficientnet.py` lines 157–162 and 325–338). The checkpoint's `keras_version` is 3.10.0, so this pass-through is the implementation that matches the saved model. **VERIFIED** for the current source and the saved graph's Keras version. The June machine was not re-inspected beyond the version string inside the checkpoint.

Resize uses the loader default `interpolation="bilinear"` (`image_dataset_utils.py` line 30). The script does not override it.

## Data

| Item | Finding | Status |
|---|---|---|
| Directory | `data/trashNet-dataset` | **VERIFIED**, lines 28 and 37 |
| Split | `validation_split=0.2`, `subset` training or validation, `seed=42`, `batch_size=32`, `image_size=(224, 224)` | **VERIFIED**, lines 27–43 |
| Shuffle | Omitted, therefore default `True` | **VERIFIED** as a default. See the dataset mapping audit. |
| Train / validation path overlap | 0 paths in the cached shuffled lists | **RECONSTRUCTED** |
| Exact duplicate content across that split | `glass176.jpg` (validation, label glass) is byte-identical to `plastic152.jpg` (training, label plastic). `plastic332.jpg` (validation, label plastic) is byte-identical to `glass389.jpg` (training, label glass). | **VERIFIED** bytes. **RECONSTRUCTED** membership. Full table in `audit/dataset_mapping_audit.md`. |
| Third exact pair | `glass115.jpg` and `metal91.jpg`, both in the cached training list, opposite labels | **RECONSTRUCTED** membership |
| Class counts | Train 334 / 399 / 322 / 471 / 393 / 103. Validation 69 / 102 / 88 / 123 / 89 / 34. | **RECONSTRUCTED** from the cache. Validation supports **VERIFIED** against the classification report. |
| Class weights | Not used | **VERIFIED**. No `class_weight` argument. |
| Augmentation | None | **VERIFIED** |

## Frozen layers and BatchNormalization

In the saved file the whole EfficientNetB0 base is frozen, including every `BatchNormalization` layer in that base. The head (`global_average_pooling2d`, `dense`, `dropout`, `dense_1`) is trainable.

The builder calls the base with `training=False` (line 23). The saved inbound node repeats `"training": false`. While that call argument stays false, BatchNormalization uses its moving statistics on the forward pass even if a later line sets layer `trainable` flags. Phase 2's flag changes are visible in source (lines 114–117) and are absent from the saved file. No forward pass was run, so the moving-statistic claim is **RECONSTRUCTED** from the call argument and the Keras BatchNormalization contract, not from a probed activation.

Individual base tensors were not aligned to `~/.keras/models/efficientnetb0_notop.h5`, because the checkpoint stores them under anonymous Keras 3 names (`conv2d`, `batch_normalization`, …). Equality of this base with the ImageNet file is **UNVERIFIED**. Equality with the EXP-004 base is **VERIFIED** and is reported in the EXP-004 audit: every base tensor matches, and only the two Dense layers differ.

## The 89.90% figure

`docs/notes.md` lines 312–340 says the best validation accuracy is 89.90% at Phase 1 epoch 10, with training accuracy 98.80% and validation loss 0.3386, and that the checkpoint retained that result after Phase 2.

What exists:

- The checkpoint files exist, and the evaluation script loads `efficientnet.keras` (`evaluate_efficientnet.py` lines 30–32) on the shuffled validation subset (lines 39–46).
- `classification_report.txt` line 12 prints `accuracy 0.90` on 505 images. Sklearn's two-decimal format does not preserve 89.90% versus 89.70% or 90.10%.
- Later source hardcodes `0.8990`: `train_efficientnet_augmented.py` line 181, `src/evaluation/compare_models.py` line 15, `src/evaluation/evaluate_real_world.py` line 192. Those are constants, not a log of `evaluate`.
- `results/real_world_test/efficientnetb0_report.txt` prints real-world accuracy `0.40` on 78 images. `docs/notes.md` line 488 says 39.74%. The only integer count that prints as `0.40` at two decimals and as 39.74% at two percentage decimals is 31/78. That reading is **RECONSTRUCTED**. It is a different dataset from the TrashNet validation claim.

**UNVERIFIED:** the four-decimal 89.90%, the epoch index, the training accuracy 98.80%, and the validation loss 0.3386. They cannot be tied to a saved history row. The per-class F1 table in `docs/notes.md` lines 433–441 matches `classification_report.txt` lines 5–13, including macro F1 0.89. That table is **VERIFIED** as the contents of the report file. It does not restore the third decimal of accuracy.

`results/duplicate_check/verification/trashnet_val_reevaluation.json` contains a later rescore. `verification/experiment_impact_analysis.md` says that rescore used PIL resize rather than the training loader and can move accuracy by about half a point to one and a half points. It is not the June evaluation. It was not used as a substitute for the missing history.

## Contradictions

1. `docs/notes.md` lines 286–293 describes Phase 2 as part of the completed training strategy. The saved checkpoint is the Phase 1 compiled model. The note's Phase 2 metrics are not in a log. The code and the checkpoint disagree about which state was kept; they do not disagree about what the code asked to run.
2. `docs/notes.md` line 321 says validation accuracy rose every epoch from 86.34% to 89.90%. No epoch table was saved. The curves in `results/efficientnet/accuracy.png` and `loss.png` are plots, not a numeric history. They were not digitized.

## Unresolved

- The numeric value of `loss`, `accuracy`, `val_loss`, and `val_accuracy` on each completed epoch. The plots were not digitized.
- Which epoch produced the saved checkpoint, given that `save_best_only` keeps one file.
- Are the anonymous base tensors byte-identical to `efficientnetb0_notop.h5`?
- Did the June loader's file order match `trashnet_split.json` exactly?
- Did any base weight change in memory during the plotted second fit? Those weights were not saved.

No correction was applied.

## Follow-up on 18 November 2026 — training phases

The earlier section left Phase 2 execution unverified because no history file exists. The saved figures supply a narrower fact.

`results/efficientnet/accuracy.png` and `results/efficientnet/loss.png` were both written at 22:52:03 on 10 June 2026, one second after `efficientnet.keras` (`metadata.json` time `2026-06-10@22:52:02`). `src/training/train_efficientnet.py` was last modified at 22:20:38 the same evening, before either file. The current script draws the figures only after both `fit` calls, at lines 197–234: it concatenates `history_phase1` and `history_phase2`, then draws a dashed line labelled `Fine-Tuning Start` at `phase1_epochs`.

Both PNGs show that title and that dashed line. The line sits at epoch 10 and the series continues to epoch 16. That is **RECONSTRUCTED** from the chart, not from a log: a second `fit` returned and contributed about six epoch points after a first run of about ten. It matches the placement of the plotting code. It does not recover the y-axis numbers. `docs/notes.md` lines 310–330 remains an unverified numeric table.

**The saved checkpoint still does not support a claim that fine-tuning of the EfficientNetB0 base was completed.** `efficientnet.keras` and `efficientnet_best.keras` keep every base layer non-trainable and store Adam at learning rate `1e-3`. A second `fit` that unfreezes layers and then fails to beat `ModelCheckpoint` leaves exactly this file on disk. The figures are compatible with that sequence. They are not a saved fine-tuned base.

## Follow-up — comparison with a retrained EXP-004

EXP-003 and EXP-004 request the same TrashNet loader, including the same two opposite-label exact copies and the same train-only exact pair. EXP-004's purpose, in `docs/notes.md` lines 518–521 and in the difference between the two scripts, is the with/without augmentation comparison.

Keeping this historical EXP-003 result as the control for a retrained EXP-004 would mix three changes: the augmentation layers, the corrected file list, and a new unseeded run. The published gap of −0.20 percentage points (`docs/notes.md` lines 573–576) could not be read as an augmentation effect.

The historical checkpoint and the 89.90% note stay as the record of the June run. They are not a valid control arm for a corrected manifest. If that comparison is redone, both experiments need the same corrected train and validation lists, with augmentation as the only intended difference. That retraining was not started. The recommendation is in `decisions/decision_log.md`, entry D6.
