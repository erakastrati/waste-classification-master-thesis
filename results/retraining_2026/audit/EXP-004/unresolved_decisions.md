# Unresolved decisions

No pair ID is unresolved. All 139 rows in `visual_review_register.csv` are `HUMAN_CONFIRMED`.

| Decision | Pairs |
|---|---:|
| `keep_one_exclude_other` | 3 |
| `visual_duplicate` | 7 |
| `same_object_different_photo` | 1 |
| `unrelated` | 128 |

The corrected lists are `results/retraining_2026/manifests/trashnet_seed42/`. They are ready for approval. No source image, historical result, training script, or checkpoint was changed. Retraining has not started.

## Still open

Phase 2 unfreeze, decision D5 in `decisions/decision_log.md`. It does not change the file lists. It has to be settled before a corrected training run.

`unresolved_pair_ids.csv` lists the 122 IDs that were unnamed in the earlier partial review. The completed review recorded each of them as `unrelated`.
