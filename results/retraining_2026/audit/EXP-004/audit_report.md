# EXP-004 audit

Historical run: EfficientNetB0 with four Keras augmentation layers, trained from ImageNet on the same TrashNet loader as EXP-003. Phase 1 did not call `fit`, `evaluate`, or `predict`.

## Purpose, and how it differs from EXP-003

`docs/notes.md` lines 513–521 says EXP-004 exists to add augmentation to the EXP-003 setup, because the proposal required a with/without comparison and because real-world accuracy had dropped. The code agrees that augmentation is the structural change. It does not load the EXP-003 checkpoint.

**VERIFIED** differences between `src/training/train_efficientnet.py` and `src/training/train_efficientnet_augmented.py`:

| Item | EXP-003 | EXP-004 |
|---|---|---|
| Script | `src/training/train_efficientnet.py` | `src/training/train_efficientnet_augmented.py` |
| Builder | `build_efficientnet_model` | Model built inline, lines 69–92. Does not import the EXP-003 builder. |
| Starting weights | `EfficientNetB0(..., weights="imagenet")` | Same, lines 69–73. Not a load of `results/efficientnet/efficientnet.keras`. |
| Graph | Input → base → GAP → Dense(128) → Dropout(0.3) → Dense(6) | Input → RandomFlip → RandomRotation → RandomZoom → RandomContrast → base → the same head, lines 77–90 |
| Output paths | `results/efficientnet/` | `results/efficientnet_augmented/` |
| Dataset arguments | `data/trashNet-dataset`, split 0.2, seed 42, 224, batch 32 | Same arguments, lines 33–49 |
| Phase 1 / Phase 2 schedule | 10 epochs at `1e-3` frozen, then 15 epochs at `1e-5` with the last 30 base layers unfrozen | Same schedule, lines 117–163 |
| Callbacks | `ModelCheckpoint` on `val_accuracy`, `EarlyStopping` patience 5, `restore_best_weights=True` | Same arguments, lines 101–114 |
| `class_weight` | Absent | Absent |

The two scripts therefore share the data loader, the optimizer schedule, the callbacks, and the head. EXP-004 is a new ImageNet initialization plus augmentation, not a fine-tune of the EXP-003 weights.

## Checkpoints

| File | Timestamp inside `metadata.json` | Keras |
|---|---|---|
| `results/efficientnet_augmented/efficientnet_augmented_best.keras` | `2026-06-23@13:46:06` | 3.10.0 |
| `results/efficientnet_augmented/efficientnet_augmented.keras` | `2026-06-23@14:10:49` | 3.10.0 |

**VERIFIED** from the Keras zip files:

- The best file and the final file have the same 326 weight tensors. Zero tensors differ. The final file is the resave at lines 194–196 of the model loaded at lines 170–172.
- `efficientnetb0` is `trainable: false`. Of its 238 config layers, 0 are `trainable: true` and 237 are `trainable: false`. The remaining layer is the base input, whose `trainable` field is null. `block7a_project_conv`, `block7a_project_bn`, `top_conv`, `top_bn`, and `top_activation` are all `trainable: false`.
- `compile_config` learning rate is 0.0010000000474974513, loss `sparse_categorical_crossentropy`, metric `accuracy`.
- The call into `efficientnetb0` is stored with `"training": false` and its input history is `random_contrast`. That matches line 85: `base_model(x, training=False)`.
- Non-optimizer parameter count was measured on the EXP-003 file as 4,214,313. EXP-004's augmentation layers have no weights. The only tensor differences versus EXP-003 are `layers/dense/vars/0`, `layers/dense/vars/1`, `layers/dense_1/vars/0`, `layers/dense_1/vars/1`, and nine optimizer slots. Every base tensor matches.

A file saved after lines 143–151 would mark the last 30 base layers trainable and would store learning rate `1e-5`. This file does not. **VERIFIED:** the persisted EXP-004 model is the Phase 1 model: frozen ImageNet base, trainable classification head, Adam at `1e-3`.

**UNVERIFIED:** whether Phase 2's `fit` ran. `docs/notes.md` lines 555–558 says it ran six epochs and never passed the Phase 1 best. There is no history file. The 25 minutes between the two timestamps show that the process continued after the best file was written. Lines 194–196 write the final file only after both `fit` calls and after `evaluate`. Continuation is not an epoch log.

EarlyStopping and ModelCheckpoint are the same objects across both `fit` calls, with the same Keras 3.10.0 reset behaviour documented in the EXP-003 report. The script evaluates the loaded checkpoint, not the in-memory weights left by `restore_best_weights`.

## Augmentation that was actually saved

Saved layer configs, which match the constructor calls at lines 80–83 except where Keras normalizes the factor:

| Layer | Saved config | Meaning in Keras 3.10.0 |
|---|---|---|
| `random_flip` | `mode: horizontal`, `seed: null` | Horizontal flip only. `docs/notes.md` line 524. |
| `random_rotation` | `factor: [-0.15, 0.15]`, `fill_mode: reflect`, `interpolation: bilinear`, `seed: null` | `random_rotation.py` lines 38–50: a factor of 0.15 is ±15% of 360°, which is ±54°. The note at `docs/notes.md` lines 525–529 matches this and matches the saved `fill_mode`. |
| `random_zoom` | `height_factor: 0.15`, `width_factor: null`, `fill_mode: reflect`, `interpolation: bilinear` | `random_zoom.py` lines 158–161 turns a single float into `[-factor, factor]`. `width_factor=None` keeps aspect ratio (docstring lines 56–58). The note's "±15% zoom, aspect ratio preserved" matches. |
| `random_contrast` | `factor: [0, 0.2]`, `value_range: [0, 255]`, `seed: null` | See the contradiction below. |

`BaseImagePreprocessingLayer._set_factor` (`base_image_preprocessing_layer.py` lines 46–53) converts a single float `0.2` into `[max(-0.2, 0), 0.2]` because `RandomContrast._FACTOR_BOUNDS` is `(0, 1)`. `get_random_transformation` (`random_contrast.py` lines 84–87) then draws the multiplier from `[1.0 - 0, 1.0 + 0.2] = [1.0, 1.2]`.

**Contradiction.** `docs/notes.md` line 527 says `RandomContrast(0.2)` is ±20% contrast. The layer docstring lines 36–40 also says a single float uses `lower = upper`. The saved config and `_set_factor` do not do that. The serialized range is 0% to +20%, not ±20%. The docstring and the note disagree with the saved config and with `_set_factor`.

No vertical flip, brightness, translation, or shear layer is in the script or the saved config. Seeds on all four layers are null, so their random streams are not fixed by the layer. **VERIFIED.**

The layer classes apply the transform only when the call is in training mode (`random_rotation.py` lines 112–120; `random_contrast.py` lines 78–79 returns a zero factor when `training` is false; `random_zoom.py` lines 176–180). `evaluate` and `predict` use `training=False` unless a caller overrides it. **RECONSTRUCTED** that the published validation and real-world reports did not apply these transforms, because the evaluation scripts call `model.predict` without `training=True` (`evaluate_efficientnet_augmented.py` line 73). That call was not repeated in this phase.

## Data, optimization, and training length

Dataset path, split, seed, image size, batch size, shuffle default, class counts, and the exact duplicate files are the same as EXP-003. Detail and evidence grades are in `audit/dataset_mapping_audit.md`.

| Item | Source | Status |
|---|---|---|
| Optimizer, loss, metric | Lines 121–125 and the saved `compile_config` | **VERIFIED** Adam, sparse categorical crossentropy, accuracy. Saved learning rate is `1e-3`, not the Phase 2 `1e-5`. |
| Batch size | Line 39, value 32 | **VERIFIED** |
| Maximum epochs | Line 135 value 10, line 161 value 15 | **VERIFIED** as limits |
| Completed epochs | `docs/notes.md` lines 550–558: Phase 1 stopped at epoch 10 with the best at epoch 5; Phase 2 stopped at epoch 6 | **UNVERIFIED**. No history file. |
| Class weights | No `class_weight` in `fit` | **VERIFIED** |
| Global seed | Not set | **VERIFIED** as an absence |

`docs/notes.md` lines 550–562 says Phase 1 best validation accuracy 0.8970 at epoch 5, validation loss 0.3104, and Phase 2 did not improve on it.

The classification report `results/efficientnet_augmented/classification_report.txt` (written 23 June 2026, 14:17, by `evaluate_efficientnet_augmented.py` lines 92–94) prints accuracy `0.90` on 505 images and the per-class F1 values that `docs/notes.md` lines 565–571 repeats (cardboard 0.94, glass 0.89, metal 0.90, paper 0.93, plastic 0.86, trash 0.81, macro 0.89). Those F1 values are **VERIFIED** as the contents of the report. The report's two-decimal accuracy does not identify 89.70%.

Later code hardcodes `0.8970` (`train_taco_finetune.py` line 184, `evaluate_real_world.py` line 193). That is a constant. **UNVERIFIED** as a measured `evaluate` output. The real-world file `results/real_world_test/efficientnetb0_+_augment_report.txt` prints accuracy `0.50` on 78 images, which is exactly 50% and matches `docs/notes.md` line 579.

## Is EXP-004 the parent of EXP-005?

**VERIFIED** from `src/training/train_taco_finetune.py`:

- Line 31 sets `BASE_MODEL_PATH = "results/efficientnet_augmented/efficientnet_augmented.keras"`.
- Line 122 is `model = load_model(BASE_MODEL_PATH)`.
- The file comment at lines 4–5 calls this "the best model so far, 89.70% TrashNet".

No other training script loads that path. The later chain does not reload EXP-004 directly:

- `train_realwaste_finetune.py` line 29 loads `results/efficientnet_taco/efficientnet_taco.keras`.
- `train_object_centric.py` line 29 loads `results/efficientnet_realwaste/efficientnet_realwaste.keras`.
- `train_garbage_finetune.py` line 26 loads `results/efficientnet_household/efficientnet_household.keras`.

So EXP-005's source names the EXP-004 final file as its parent. EXP-007, EXP-008, and EXP-011 name the previous experiment's own output. This audit did not open those later checkpoints and does not claim their weights still contain the EXP-004 tensors.

**Contradiction inside EXP-005, recorded because it changes what "reuse" would mean.** Lines 145–156 say "Fine-Tuning" of the loaded model at `1e-5` and do not set `efficientnetb0.trainable = True`. The loaded EXP-004 file has that base frozen. Loading it and compiling it leaves the frozen flags as saved. The comment describes a full-model fine-tune. The loaded config plus the absence of an unfreeze is a frozen base plus a trainable head, with the augmentation layers still in the graph. This was not executed.

EXP-005 validation is also not the EXP-004 validation. Lines 70–78 call `image_dataset_from_directory` with `subset="validation"` and `shuffle=False`. The cached unshuffled validation list has 395 paths that are also in the shuffled training list (`audit/dataset_mapping_audit.md`). That defect belongs to EXP-005's script. It remains even if the parent weights are replaced.

## The two images

Full hashes, byte sizes, and the third same-split pair are in `audit/dataset_mapping_audit.md`. The original rows are in `results/duplicate_check/supplementary_trashnet_candidates.csv`:

| Validation file | Label | Training file | Label | Relation |
|---|---|---|---|---|
| `data/trashNet-dataset/glass/glass176.jpg` | glass | `data/trashNet-dataset/plastic/plastic152.jpg` | plastic | Same SHA-256, same pixel hash, 16,192 bytes, 512×384 |
| `data/trashNet-dataset/plastic/plastic332.jpg` | plastic | `data/trashNet-dataset/glass/glass389.jpg` | glass | Same SHA-256, same pixel hash, 15,420 bytes, 512×384 |

They are exact duplicates with opposite folder labels. They are not merely similar photographs. Path overlap of the shuffled lists is 0, because the copies have different names. Content overlap is these two validation files.

**RECONSTRUCTED** membership: both pairs sit on the shuffled split that EXP-003 and EXP-004 request. `plastic332.jpg` is also on the unshuffled list that EXP-005 uses as validation. `glass176.jpg` is on the shuffled validation list only.

Both copies were therefore in the datasets passed to EXP-004's `fit`: one copy in `train_ds`, the other in `val_ds`. The same is true of EXP-003. A third exact opposite-label pair, `glass115.jpg` and `metal91.jpg`, is inside the training list only and also entered both experiments.

The checkpoint zip is internally consistent: best and final weights match, the archive opens, and the config parses. The two images do not corrupt that file. They are part of the data that produced the head weights. Because the saved base matches EXP-003's base tensor-for-tensor, the contradictory labels did not move the saved base weights. They could only have affected the head and the optimizer state, which do differ from EXP-003.

This phase did not recompute accuracy with the two validation files removed. Inference was out of scope. The missing delta is not evidence that the contradiction is small.

There are also 136 further confirmed perceptual pairs across the same split (5 close, 131 at the looser confirmation rule). They are not the two-image claim. They were not reviewed visually here, and they are an additional reason the shuffled split is not a clean holdout. See the dataset mapping audit.

## Conclusion on reuse

**RETRAIN EXP-004.**

The parent that EXP-005 loads was trained and validated on two byte-identical images carrying opposite labels, plus one more opposite-label exact pair inside training. Reusing the checkpoint keeps that training inside the starting head. Deleting or ignoring the four files only in a later dataset does not take them out of the saved weights. The count being two validation images is not a reason to keep the checkpoint: the labels are direct contradictions on identical pixels, and the published 89.70% has no saved epoch row that could be corrected by hand.

This is a recommendation. It is recorded in `decisions/decision_log.md` as **REQUIRES DECISION**. No retraining was started.

## Unresolved

- The numeric per-epoch metrics, including whether validation accuracy was 0.8970 and validation loss was 0.3104. The plots were not digitized.
- Which Phase 1 epoch was the one `ModelCheckpoint` kept.
- Byte alignment of the anonymous base tensors to `efficientnetb0_notop.h5`.
- Exact filename identity between the June loader and `trashnet_split.json`.
- The accuracy change on 503 validation images after dropping `glass176.jpg` and `plastic332.jpg`. Not measured in this phase.
- Visual identity of the 5 tight paper pairs and the 29 non-exact opposite-class pairs. See `audit/EXP-004/cross_split_pair_review.md`.
- Whether a corrected EXP-004, if approved, should also drop Pair C before training.

## Follow-up on 18 November 2026 — the 138 pairs

The 138 count is 138 unordered pairs from the automatic pHash/dHash/ZNCC rule, covering 110 distinct images (73 training, 37 validation). Two of the pairs are the exact opposite-label copies. Five more are same-class paper pairs with ZNCC from 0.9886 to 0.9953 and are not byte-identical. The other 131, including a 64-image component that is mostly paper, are only above the loose automatic cutoff. No human review and no SIFT score exists for these pairs. Detail, the method, and `cross_split_pairs.csv` are in `audit/EXP-004/cross_split_pair_review.md`.

This does not change the reuse conclusion. The two exact pairs are enough to refuse the historical checkpoint as a parent. The other 136 are not, on current evidence, a list of duplicates to delete.

## Follow-up — training phases

`results/efficientnet_augmented/accuracy.png` and `loss.png` were written at 14:10:50 on 23 June 2026. `efficientnet_augmented.keras` records `2026-06-23@14:10:49`. `src/training/train_efficientnet_augmented.py` was last modified at 13:26:43 the same day, before the run finished. The plotting block at lines 200–233 runs only after both `fit` calls and draws `Fine-Tuning Start` at `phase1_epochs`.

Both PNGs show that line at epoch 10 and a series that continues to epoch 16. As with EXP-003, that is **RECONSTRUCTED** from the chart: the second `fit` returned and added about six points. It is not a numeric history, and it is not a fine-tuned base. The saved file still has `efficientnetb0` frozen and Adam at `1e-3`. The checkpoint does not support a claim that fine-tuning of the base was completed and retained.

## Follow-up — shared manifest with EXP-003

A corrected EXP-004 compared with the historical EXP-003 result would not isolate augmentation. Both experiments need the same corrected manifests if that comparison is repeated. The historical EXP-004 checkpoint is still not a parent to reuse. See decision D6.

## Follow-up on 18 November 2026 — visual review material

Contact sheets for all 138 cross-split pairs, and for the within-training exact pair, are under `audit/EXP-004/visual_review/`. There are 75 sheets. No source image was missing, moved, or relabeled. The register is `audit/EXP-004/visual_review_register.csv`: 139 rows, every `visual_review_status`, `proposed_action`, and `action_requirement` set to `PENDING`, and every `reviewer_decision` left empty.

The proposed shared split is `audit/EXP-004/split_strategy_proposal.md`. It is not implemented. The corrected split is not ready, because the register has no reviewer decisions and D2 and D8 are still open.
