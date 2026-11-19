# Decision log

Phase 1 records recommendations. It does not approve them and it does not start a run. An entry becomes a decision only when it is explicitly accepted in a later note in this file.

## 2026-11-18 — Phase 1 audit complete, no action taken

Evidence: `audit/EXP-003/audit_report.md`, `audit/EXP-004/audit_report.md`, `audit/dataset_mapping_audit.md`.

Findings that drive the open decisions:

- The saved EXP-003 and EXP-004 checkpoints are Phase 1 models. The EfficientNetB0 base is frozen, the stored learning rate is `1e-3`, and the two bases have identical tensors. Phase 2 is in the scripts and is not in the saved files. Epoch histories were not saved.
- EXP-005's script loads `results/efficientnet_augmented/efficientnet_augmented.keras`. Later scripts load their own preceding outputs, not EXP-004 directly.
- Two validation files are byte-identical to two training files and carry the opposite label. A third exact pair sits inside training with opposite labels. Both EXP-003 and EXP-004 used this shuffled split.

### D1. Reuse of the historical EXP-004 checkpoint

Recommendation from the audit: **RETRAIN EXP-004**.

The checkpoint file is readable and internally consistent. It is not a safe parent for EXP-005, because its head was trained on exact images that also appear in validation under the other label. This recommendation does not treat the small count as a reason to keep the weights.

Status: **REQUIRES DECISION**. Not accepted. No run directory was created.

### D2. Exact duplicates to exclude if EXP-004 is retrained

Pairs identified in the dataset mapping audit:

- `glass176.jpg` (validation, glass) and `plastic152.jpg` (training, plastic)
- `plastic332.jpg` (validation, plastic) and `glass389.jpg` (training, glass)
- `glass115.jpg` (training, glass) and `metal91.jpg` (training, metal)

Which copies to drop, and whether one copy may remain under a single chosen label, is **REQUIRES DECISION**.

### D3. Non-exact cross-split pairs

138 confirmed perceptual pairs cross the shuffled split. Two of them are D2. The other 136 were not given a visual verdict in this phase. Whether they block a retrained EXP-004 is **REQUIRES DECISION**. Leaving them undecided does not weaken D1: D1 stands on the exact opposite-label copies alone.

### D4. EXP-005 validation split

Even after a new EXP-004 parent, `train_taco_finetune.py` lines 70–78 builds validation with `shuffle=False`. The cached unshuffled list shares 395 paths with the shuffled training list. Changing that script is **REQUIRES DECISION** and was not done.

### D5. Phase 2 of the corrected protocol

The historical saved models never stored an unfrozen base. Whether a corrected run includes the 30-layer unfreeze is **REQUIRES DECISION**.

## Explicitly not decided

- No historical result was withdrawn or rewritten.
- No checkpoint was deleted, moved, or replaced.
- EXP-005, EXP-007, EXP-008, and EXP-011 were not retrained and were not fully audited.

## 2026-11-18 — Follow-up before Phase 2

Evidence added: `audit/EXP-004/cross_split_pair_review.md`, `audit/EXP-004/cross_split_pairs.csv`, and the follow-up sections of the two experiment audits. No training, no evaluation, and no change under `data/` or historical `results/`.

### D3 revised

The 138 figure is 138 unordered pairs from the automatic rule in `check_duplicates.py` lines 97–99 and 373–376, not 138 images and not 138 human-confirmed duplicates. The pairs cover 110 images. Two pairs are the exact opposite-label copies already in D2. Five paper pairs are tighter automatic matches and are still unverified by eye. The other 131 stay above a loose cutoff, including one connected component of 64 images. They are not an exclusion list.

D1 is unchanged: those 131 pairs are not required for the decision to retrain EXP-004, and they are not a reason to delete files before a visual review.

### D6. EXP-003 as the control for EXP-004

Recommendation: **retrain EXP-003 and EXP-004 on one shared corrected manifest** if the augmentation comparison is repeated. Keep both historical checkpoints and the published notes as the June record. Do not compare a new EXP-004 with the historical 89.90%.

Reason: both runs used the same defective shuffled split. Replacing only EXP-004 mixes the augmentation change with the data correction and with a new unseeded optimization. The historical EXP-003 file is not a parent of EXP-005. It is the control arm of the augmentation claim in `docs/notes.md` lines 573–576.

Status: **REQUIRES DECISION**. Not accepted. No run was created.

### D7. Reuse of either saved checkpoint

Recommendation: **do not reuse either checkpoint** as the weights of a corrected EXP-003 or EXP-004, and do not reuse `results/efficientnet_augmented/efficientnet_augmented.keras` as the parent of a corrected EXP-005.

The files remain the historical artifacts. Each saved model is the frozen-base, learning-rate-`1e-3` compile. The accuracy and loss figures show a second plotted segment after a marker at epoch 10, so a second `fit` is the likely source of those figures. That does not put fine-tuned base weights in the checkpoint. A claim that base fine-tuning was completed and retained is not supported.

Status: **REQUIRES DECISION**. Not accepted.

### D8. What has to be settled before a corrected TrashNet manifest

1. D2: for each exact pair, drop both files or keep one file under one label. No relabeling has been done.
2. Visual review, in the order in `cross_split_pair_review.md`, of the 5 tight paper pairs and the 29 non-exact opposite-class pairs. A pair is excluded only after a `same_file` or `same_object_different_photo` verdict.
3. The 96 edges inside the large similarity components stay in the dataset unless that review says otherwise. They are not bulk-deleted.
4. One train list and one validation list, used by both corrected experiments. Augmentation is the only intended difference between them.
5. D5: whether the corrected protocol includes the 30-layer unfreeze. The historical checkpoints did not retain it.
6. D4 remains open for EXP-005 and is not part of the TrashNet manifest itself.

Until D8 is accepted, no manifest is written and Phase 2 does not start.

## 2026-11-18 — Visual review material prepared

Sheets: `audit/EXP-004/visual_review/` (75 PNG files). Index: `audit/EXP-004/visual_review/sheet_index.csv`. Register: `audit/EXP-004/visual_review_register.csv`. Split procedure: `audit/EXP-004/split_strategy_proposal.md`.

All 138 cross-split pairs are on a sheet. P139 adds the within-training exact pair `glass115.jpg` / `metal91.jpg`. No file was missing. No reviewer decision was entered. Every register row is `PENDING`.

### D9. Corrected split is not ready

The sheet procedure in `split_strategy_proposal.md` is the recommended way to build one manifest for EXP-003 and EXP-004: per-class shuffle with `random.Random(42)`, validation size `int(0.2 * units)`, explicit path lists, and a hard stop when a SHA-256 or a confirmed same-object group would cross the split.

It must not be run yet. Blocking approvals:

1. Visual decisions in the register. Priority order is the three exact pairs, the five paper pairs, then the 29 opposite-class pairs. The 102 component pairs are also on sheets and stay `PENDING`.
2. D2: exclude all six exact-duplicate files, or keep one copy under one chosen label. The proposal recommends exclusion so that no folder label is invented. That recommendation is not accepted.
3. Near-duplicate groups are created only from later register values `same_file` or `same_object_different_photo`. Automatic scores do not define a group.
4. D5, the 30-layer unfreeze, is still open. It does not change membership, and it should be decided with the split so both corrected experiments have one specification.

Status: **REQUIRES DECISION**. No manifest was written. No training was started.

## 2026-11-18 — Human review recorded, split not written

The reviewer decisions that were stated explicitly are in `audit/EXP-004/human_review_decision_summary.md` and in `audit/EXP-004/visual_review_register.csv`.

D2 is superseded for the three exact pairs. The recorded action is keep `plastic152.jpg`, `glass389.jpg`, and `glass115.jpg`, and exclude `glass176.jpg`, `plastic332.jpg`, and `metal91.jpg` from a future corrected inventory. The files on disk are unchanged. The proposal to exclude all six files is not the decision.

P003–P007, P037, and P039 are `visual_duplicate`. P038 is `same_object_different_photo`. P010, P046, P053, P126, P132, and P133 are `unrelated`. The other 122 pair IDs are `UNRESOLVED` because they were not named. They are listed in `audit/EXP-004/unresolved_pair_ids.csv`.

No corrected split was created. `audit/EXP-004/split_readiness.md` states that the split is not ready for approval. No training was started.

## 2026-11-18 — Completed review and corrected split

Every pair ID P001–P139 now has an individual `HUMAN_CONFIRMED` decision in `audit/EXP-004/visual_review_register.csv`. The 122 IDs that were `UNRESOLVED` are `unrelated`. P008, P009, and P106 record the explicit recheck. Earlier decisions for P001–P007, P010, P037–P039, P046, P053, P126, P132, P133, and P139 were preserved.

The shared manifests are `manifests/trashnet_seed42/train.txt` and `val.txt`: 2,021 training files and 503 validation files. Eight confirmed groups are each entirely on one side. The three excluded files remain on disk and are in neither list. Checks are in `manifests/trashnet_seed42/validation_report.md`.

Status of the file lists: **ready for approval**. Training was not started. D5, the 30-layer unfreeze, remains open and does not change membership.

## 2026-11-19 — Visual-copy deduplication proposal, not applied

Before the `trashnet_seed42` lists are approved, `audit/EXP-004/visual_dedup_proposal.md` proposes omitting the lexicographically larger path from each of P003–P007, P037, and P039. P038 keeps both photographs in one unit. Expected lists would be 2,016 training files and 501 validation files. The current manifests were not overwritten, the dataset was not changed, and training was not started.

Status: **REQUIRES DECISION**. Approval of `trashnet_seed42` waits on this proposal.

## 2026-11-19 — Deduplicated manifests written

The visual-deduplication policy was approved and applied in `manifests/trashnet_seed42_dedup/`. Inventory 2,517, training 2,016, validation 501. All recorded checks passed. No remaining image changed side relative to `trashnet_seed42`. That earlier directory, the review register, the dataset, the training scripts, and the checkpoints were not modified. Training was not started.

`train.txt` SHA-256 `bc768a67584e2be3b758a3390de3d90ffb3c4bb90b27e158e422dcf1d468e549`. `val.txt` SHA-256 `b4b344db8ac6e2cc99199ac12754b8fe951ff8c2cf0d36e14e88bb567a022cb5`.

Status of these lists: **validated**. D5 remains open.

## 2026-11-19 — Final corrected training protocol, not implemented

`audit/EXP-003_EXP-004_final_protocol.md` records the loader, the single augmentation difference, the batch-norm freeze, separate phase checkpoints plus an overall selection, and training seeds 42, 123, and 456 for both EXP-003 and EXP-004. Contrast stays at the implemented 0% to +20% range. Historical checkpoints and the EXP-005 load path are unchanged. No code was edited and training was not started.

Status: **awaiting acceptance before implementation**.

## 2026-11-19 — Pipeline implemented, training not started

`src/retraining/` loads `manifests/trashnet_seed42_dedup/` and implements the shared EfficientNetB0 protocol. `audit/pretraining_validation.json` records a passing preflight, including a forward pass and no call to `fit`. Run directories were not created. Historical scripts, checkpoints, and the dataset were not modified.

## 2026-11-19 — Twelve-run comparison approved; EXP-001 and EXP-002 preflighted

The controlled comparison is EXP-001, EXP-002, EXP-003, and EXP-004, each with seeds 42, 123, and 456, all on `trashnet_seed42_dedup`. That is 12 runs. Historical seed-study numbers stay out of the mean. The 78-image set is excluded. The 600-image set stays held out and has not been scored.

`src/corrected_baselines/` implements EXP-001 and EXP-002 without editing `src/retraining/`. `audit/EXP-001_EXP-002_preflight.json` records 37 passing checks and no call to `fit`. Run directories were not created. EXP-002 keeps its original checkpoint rule and leaves batch-norm trainable inside the unfrozen last 30 layers. Training remains unapproved.
