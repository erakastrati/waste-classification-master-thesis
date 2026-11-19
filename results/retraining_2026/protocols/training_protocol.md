# Training protocol record

This file records the protocol that the EXP-003 and EXP-004 sources implement, and the points that are not yet decided. It does not authorize a run. No run directory has been created.

A future run, after a decision in `decisions/decision_log.md`, writes only under `experiments/EXP-00X/runs/<UTC timestamp>_<label>/` and stops if that directory already exists.

## Historical protocol that the source implements

Status is for the June checkpoints of EXP-003 and EXP-004, not for a proposed rerun.

| Setting | Value in source | Status for the saved checkpoint |
|---|---|---|
| Data directory | `data/trashNet-dataset` | **VERIFIED** |
| Model | EfficientNetB0, `include_top=False`, ImageNet weights, input 224×224×3 | **VERIFIED** in source and saved config |
| Head | GlobalAveragePooling2D, Dense(128, relu), Dropout(0.3), Dense(6, softmax) | **VERIFIED** |
| Preprocessing | `efficientnet.preprocess_input`, which in Keras 3.10.0 returns the tensor unchanged. Rescaling and ImageNet normalization are inside the base. Resize interpolation default bilinear. | **VERIFIED** for Keras 3.10.0 |
| Split | `validation_split=0.2`, `seed=42`, default `shuffle=True`. 2,022 / 505 on 2,527 files. | **VERIFIED** as arguments. Membership **RECONSTRUCTED** from `results/duplicate_check/cache/trashnet_split.json`. |
| Batch size | 32 | **VERIFIED** |
| Loss / metric | `sparse_categorical_crossentropy` / `accuracy` | **VERIFIED** in source and `compile_config` |
| Phase 1 | Adam `1e-3`, base frozen, up to 10 epochs | Limits **VERIFIED**. The saved compile config is this phase. Completed epoch count **UNVERIFIED**. |
| Phase 2 | Unfreeze last 30 base layers, Adam `1e-5`, up to 15 epochs | Present in source. **Absent from the saved checkpoint.** Execution **UNVERIFIED**. |
| Checkpoint | One `ModelCheckpoint`, `val_accuracy`, `save_best_only`, `mode=max`, shared by both phases | **VERIFIED** |
| Early stopping | Patience 5, `restore_best_weights=True`, monitor `val_accuracy`. The object resets at each `fit`. The script then loads the checkpoint file anyway. | **VERIFIED** in source |
| Class weights | None | **VERIFIED** |
| Global random seed | None | **VERIFIED** as an absence |
| Augmentation | EXP-003: none. EXP-004: horizontal flip, rotation factor 0.15, zoom height factor 0.15 with width factor null, contrast factor 0.2, fill mode reflect, layer seeds null. | **VERIFIED** in the EXP-004 checkpoint. Contrast range is 0% to +20%, which contradicts the "±20%" wording in `docs/notes.md` line 527. |
| Batch normalization | Base is called with `training=False`. Saved base layers are all non-trainable. | **VERIFIED** for the saved file |

## What is not part of this protocol

- A corrected file list with the exact opposite-label duplicates removed.
- A decision to rerun Phase 2, to skip Phase 2, or to unfreeze a different set of layers.
- A new seed policy. `src/seed_study/` is a separate later study and is not the historical protocol.
- Any use of the unshuffled validation subset. That subset is what `train_taco_finetune.py` lines 70–78 requests, and it overlaps the shuffled training list in 395 paths.

## Open points before a new run

Each of these is **REQUIRES DECISION**. None is chosen here.

1. Whether EXP-004 is retrained before any later experiment loads it. The audit recommendation is to retrain. See the EXP-004 report.
2. Whether Pair C (`glass115.jpg`, `metal91.jpg`) is removed from training as well as Pair A and Pair B.
3. Whether the 136 non-exact cross-split perceptual pairs are excluded, reviewed, or left in.
4. Which checkpoint a later experiment is allowed to load, and under what filename inside a new run directory.
5. Whether Phase 2 is part of the corrected protocol, given that the historical saved models never left Phase 1.
6. How epoch history, the split manifest, and the evaluation predictions will be written so that a four-decimal accuracy can be traced to one file.

Until those are decided, `experiments/` stays free of runs.
