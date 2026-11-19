# Evidence coverage after the save-step fix

Status: **GO for training authorization. Training has not been started.**

The split, seeds, architectures, preprocessing, augmentation, optimizer, learning rates, batch size, epoch budgets, batch-norm rules, EarlyStopping, and checkpoint rules are unchanged. No class weights were added. EXP-005 through EXP-012 were not modified. The 78-image set and the 600-image set were not read.

`src/corrected_baselines/evidence_audit.py` ran 11 synthetic and static checks. All passed. The report is `results/retraining_2026/audit/evidence_audit.json`.

## What every run now writes

| Evidence | Result |
|---|---|
| 1. Complete configuration | PASS. `config.json` records architecture, initialization, preprocessing, augmentation or `null`, Adam settings, loss, batch size 32, phase budgets and learning rates, batch-norm behavior, EarlyStopping or `null`, and the existing checkpoint rule. `class_weight` is `null`. |
| 2. Dataset provenance | PASS. The same file records both manifest paths, both SHA-256 hashes, the class counts, manifest seed 42, and the training seed. The hash is checked against the approved manifest before training. |
| 3. Epoch history, including a partial phase | PASS. `history.json` is replaced atomically after each completed epoch. A failure during a later phase keeps the completed epochs and an empty list for a phase that started but finished no epoch. |
| 4. Selected checkpoint, phase, and epoch | PASS. EXP-001 still saves epoch-10 weights in `model.keras`. EXP-002 still uses one shared `best.keras`. EXP-003 and EXP-004 still keep separate phase checkpoints and publish the higher validation accuracy, with a tie keeping phase 1. `selection.json` records the phase and epoch. |
| 5. Per-image validation predictions | PASS. `val_predictions.csv` follows `val.txt` and has the path, true label, predicted label, and `prob_cardboard` through `prob_trash`. |
| 6. Confusion matrix and per-class metrics | PASS. All four experiments write the same `evaluation.json` fields. |
| 7. Validation accuracy and macro-F1 | PASS. Both are in `evaluation.json`, labeled validation on 501 images. |
| 8. Status, versions, warnings, failures | PASS. `environment.json`, `warnings.txt`, and `status.json` (`running`, then `completed` or `failed` with the exception and traceback). |

The synthetic test confirmed val.txt order, the twelve prediction columns, a perfect-score confusion matrix, sample standard deviation `0.6000 ± 0.1000` for inputs 0.5, 0.6, and 0.7, and McNemar’s exact two-sided probability on discordant pairs (`mcnemar_exact_p(0, 6) == 0.03125`). A simulated phase-2 failure left the phase-1 epoch on disk and recorded the exception.

## Remaining limits

- An epoch that crashes before it ends is not written. Completed epochs remain.
- `warnings.txt` records Python warnings. TensorFlow messages that never enter the Python warning system are not copied there.
- If prediction writing fails after `fit`, the checkpoint and history stay, and `status.json` says `failed`. The weights do not need to be trained again to rebuild the prediction table, but that run is not complete until the table exists.
- Operation determinism stays off. A seed does not promise bit-identical GPU results.
- The 501-image scores remain validation accuracy. They are not independent test accuracy.

## Sixteen committee comments

| Comment | After this change |
|---|---|
| 9. Stratified split, optimistic validation, three seeds | Split evidence already exists. The new runs will save the three seeds. The 501-image figure stays validation accuracy. The old seed study stays historical. |
| 10. TACO, Garbage v2, biological waste, EXP-008 | NOT VERIFIED. Separate from these 12 runs. Whether `biological` stays in `trash` is unresolved. |
| 12. Duplicates across subsets | TrashNet exclusions are already recorded. Overlap with the fine-tuning sets and the 600-image set is separate work. |
| 13. Hyperparameter table | EXP-001–004 rows can be filled from `config.json`. EXP-005–011 stay on their historical settings. |
| 14. Class weights | These four runs record `class_weight: null`. A with/without comparison would change the protocol and was not added. Later fine-tunes keep their historical weights as separate evidence. |
| 15. EXP-012 | Already studied on the original split. Not regenerated here. Repeating it on the deduplicated split is a separate decision. |
| 18. CNN conditions, trash recall, curves | The new EXP-001 history and per-class metrics will supply this. The 78-image result stays out of the thesis. |
| 19. v1 versus v2 | Excluded with the 78-image set. Domain shift is a later 600-image evaluation. |
| 21. Score EXP-005 and EXP-007 on 600 images | Separate inference. Not this training. |
| 22. McNemar | The saved validation predictions can be paired. EXP-003 versus EXP-004 on the 501 images is a validation-set test. EXP-008, EXP-011, and TTA belong to the later 600-image evaluation. |
| 23. TTA chosen during development | Not re-read in the thesis chapter. These runs do not use the 600-image set to choose TTA. |
| 24. EXP-006 and EXP-010 on 600 images | Not re-scored here. Separate from EXP-001–004. |
| 25. Three-seed mean and the EfficientNet margin | The summary writer can read the shared `evaluation.json` schema. Old percentages stay historical. |
| 28. Complete the 600-image column | Separate evaluation after selection. |
| 29. Training curves | New curves can be drawn from `history.json`. Existing EXP-008 and EXP-011 curves stay historical. |
| 35. The 600-image set was not used for selection | The trainers do not open it. |

## Recommendation

**GO.** The evidence gap that blocked the 12 runs is closed. Training is still waiting for explicit authorization.
