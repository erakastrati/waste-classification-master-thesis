# Retraining 2026

This directory is the record for the correction and retraining process that follows the thesis committee review. Phase 1 only established what EXP-003 and EXP-004 actually did. It did not train, evaluate, or change any historical file.

Status labels used in every report:

- **VERIFIED** — read from the current source file, a saved checkpoint, or a saved measurement file during this audit.
- **RECONSTRUCTED** — inferred by combining those sources, without a new run that would reproduce the number.
- **UNVERIFIED** — stated in an earlier note or required by the questions, but not present in a log, history file, or checkpoint.
- **REQUIRES DECISION** — a choice that changes the next experiment. Nothing in this directory approves that choice.

## Directories

| Path | Purpose |
|---|---|
| `audit/EXP-003/` | What the historical EfficientNetB0 run without augmentation did, and which claims are not backed by a saved history. |
| `audit/EXP-004/` | What the augmented EfficientNetB0 run did, how it differs from EXP-003, and whether its checkpoint is the parent loaded by EXP-005. The pair review is `audit/EXP-004/cross_split_pair_review.md`. Sheets and the pending register are `audit/EXP-004/visual_review/` and `audit/EXP-004/visual_review_register.csv`. The split procedure is `audit/EXP-004/split_strategy_proposal.md`. |
| `audit/dataset_mapping_audit.md` | TrashNet file membership, class counts, and the exact duplicate files that cross the train/validation boundary. |
| `protocols/training_protocol.md` | The historical training settings that are evidenced, and the settings that are still open. This is not an approved protocol for a new run. |
| `decisions/decision_log.md` | Decisions that are required before any retraining. Phase 1 records the recommendation and does not execute it. |
| `experiments/EXP-004/` | Reserved for a future corrected EXP-004 run, only after an explicit decision to retrain. |
| `experiments/EXP-005/` | Reserved for a future EXP-005 run that would load a parent named in the decision log. |
| `experiments/EXP-007/` | Reserved for a future EXP-007 run. |
| `experiments/EXP-008/` | Reserved for a future EXP-008 run. |
| `experiments/EXP-011/` | Reserved for a future EXP-011 run. |

`experiments/EXP-006`, `EXP-009`, `EXP-010`, and `EXP-012` are outside this Phase 1 structure.

## Run folders

Each experiment directory starts empty of results. A future run must create a new directory:

`experiments/EXP-00X/runs/<UTC timestamp>_<short label>/`

Rules, also written in each experiment `README.md`:

1. The run directory is chosen before any artifact is written.
2. If that directory already exists, the run stops. Files inside it are not replaced.
3. Artifacts are not written into the experiment directory itself, into `results/efficientnet/`, or into `results/efficientnet_augmented/`.
4. No run directory was created in Phase 1.

## What Phase 1 did not do

No training, fine-tuning, inference, or metric recomputation was run. Existing code, datasets, checkpoints, logs, and results were not modified. Missing epoch histories were not filled in.
