# Human review decision summary

Recorded from the reviewer’s statements on 18 November 2026. Original files under `data/trashNet-dataset/` were not modified, moved, or relabeled. The register is `visual_review_register.csv`.

The earlier proposal to exclude all six SHA-256 files is superseded. See the banner in `exact_duplicate_exclusion_proposal.md`.

## Exact duplicates with conflicting folder labels

These three pairs are SHA-256 identical. The reviewer also named the object. The kept file already sits in the folder that matches that name, so no label is written onto disk. The excluded file stays on disk and is omitted only from a future corrected inventory.

| Pair | Reviewer decision | Keep | Exclude |
|---|---|---|---|
| P001 | Both images depict plastic | `plastic/plastic152.jpg` | `glass/glass176.jpg` |
| P002 | Both images depict glass | `glass/glass389.jpg` | `plastic/plastic332.jpg` |
| P139 | Both images depict glass | `glass/glass115.jpg` | `metal/metal91.jpg` |

`reviewer_decision` for these rows is `keep_one_exclude_other`.

## Visually identical images

Not SHA-256 duplicates. `exact_duplicate` remains `no`. `reviewer_decision` is `visual_duplicate`. Both files are kept and must be placed in the same split.

| Pair | Files |
|---|---|
| P003 | `paper/paper185.jpg`, `paper/paper390.jpg` |
| P004 | `paper/paper101.jpg`, `paper/paper108.jpg` |
| P005 | `paper/paper180.jpg`, `paper/paper284.jpg` |
| P006 | `paper/paper345.jpg`, `paper/paper451.jpg` |
| P007 | `paper/paper147.jpg`, `paper/paper99.jpg` |
| P037 | `paper/paper2.jpg`, `paper/paper40.jpg` |
| P039 | `metal/metal122.jpg`, `metal/metal57.jpg` |

## Same object, different photograph

| Pair | Decision | Files |
|---|---|---|
| P038 | `same_object_different_photo` | `metal/metal149.jpg`, `metal/metal9.jpg` |

Both files are kept and must be placed in the same split.

## Reviewed pairs with no duplication and no wrong label

`reviewer_decision` is `unrelated`. The files are not grouped.

| Pair | Files |
|---|---|
| P010 | `glass/glass344.jpg`, `metal/metal272.jpg` |
| P046 | `glass/glass183.jpg`, `glass/glass5.jpg` |
| P053 | `glass/glass3.jpg`, `glass/glass4.jpg` |
| P126 | `paper/paper52.jpg`, `paper/paper567.jpg` |
| P132 | `paper/paper110.jpg`, `paper/paper87.jpg` |
| P133 | `glass/glass187.jpg`, `glass/glass3.jpg` |

## All remaining pair IDs

The completed review states that every remaining pair has no problematic visual duplication and no incorrect label. That covers P008–P036 and P040–P138, except the pairs already classified above. Each of those IDs is recorded individually as `unrelated`.

P008, P009, and P106 were explicitly rechecked and are recorded as different images with correct labels. P010, P046, P053, P126, P132, and P133 keep the earlier `unrelated` wording. `unresolved_pair_ids.csv` is the earlier list of IDs that had not yet been named. Those IDs are now `unrelated` in the register.
