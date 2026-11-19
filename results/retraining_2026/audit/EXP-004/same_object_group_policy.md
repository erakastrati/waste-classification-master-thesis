# Same-object groups in the corrected split

**Applied on 18 November 2026** in `results/retraining_2026/manifests/trashnet_seed42/`. Keep-together units are the register decisions `visual_duplicate` and `same_object_different_photo`. The three exact pairs were not kept as two-file units: one incorrectly labeled file was omitted from each, as recorded in `human_review_decision_summary.md`. The text below is the earlier policy.

## What counts as a confirmed group

A confirmed same-object edge is only one of these:

1. A SHA-256 identity. Today that is P001, P002, and P139.
2. A later `reviewer_decision` of `same_file` or `same_object_different_photo` in `visual_review_register.csv`.

Connected components are taken over those edges only. An edge that is still `PENDING`, and an edge decided `different`, is not part of a group. The perceptual components C01 (64 images) and C02 (14 images) are not confirmed groups.

## Conflicting labels

P001, P002, and P139 each join two folder names. A group with two labels cannot be given a class and cannot be placed in training or validation.

The current proposal is to exclude all six member files before the inventory is shuffled. See `exact_duplicate_exclusion_proposal.md`. That exclusion is waiting for approval. If approval instead keeps one file, the approval must name the surviving path and the single label. This policy will not choose the label.

## Same-label groups

After exclusions, a confirmed group whose remaining files share one class becomes one atomic unit.

- The unit is shuffled with the other units of that class.
- `random.Random(42)` shuffles units, not individual files. The procedure is the one in `split_strategy_proposal.md`: per class, validation receives the last `int(0.2 * number_of_units)` units.
- Every path in the unit is written to the same manifest. A group is never divided between `train.txt` and `val.txt`.

## Pending pairs that touch a confirmed file

The split generator stops if a path belongs to a confirmed group and also belongs to a pair that is still `PENDING`. That pair has to be decided, or explicitly marked as not the same object, before manifests are written. Otherwise the group boundary is incomplete.

## Checks before the manifests are saved

- No path is in both lists.
- No SHA-256 is in both lists.
- No confirmed group has members in both lists.
- No excluded path appears in either list.
- No pair that this policy treats as blocking is still `PENDING`.

EXP-003 and EXP-004 read the same two lists. The historical Keras `validation_split` is not used.
