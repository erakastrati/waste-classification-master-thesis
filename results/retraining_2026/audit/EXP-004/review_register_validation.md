# Review register validation

Checked against `visual_review_register.csv` after every pair ID was given an individual decision. No source image was changed.

## Coverage

| Check | Result |
|---|---|
| Pair IDs present | Pass. P001–P139, 139 rows, in order, none missing. |
| Status | Pass. All 139 are `HUMAN_CONFIRMED`. None are `UNRESOLVED`. |
| Decisions | `keep_one_exclude_other` 3, `visual_duplicate` 7, `same_object_different_photo` 1, `unrelated` 128. |
| Preserved decisions | Pass. P001, P002, P139, P003–P007, P037, P038, P039, P010, P046, P053, P126, P132, and P133 still have the decisions recorded before this completion. |
| Explicit rechecks | P008, P009, and P106 are `unrelated`, with the recheck named in the rationale. |
| Filenames | Pass. Kept and excluded paths match the review, and the kept files already sit in the stated class folders. |
| SHA-256 versus visual | Pass. Only P001, P002, and P139 are `exact_duplicate` yes. The visual pairs are not byte-identical. |
| Perceptual components | Not used as groups. |

## Repeated paths

The eight keep-together groups do not share a file with each other or with any other pair. No path has two grouping decisions. No `unrelated` pair joins two files that are also in the same group.

An `unrelated` decision still applies only to that pair. Files such as `paper110.jpg` and `glass3.jpg` appear in several pairs, and every one of those pairs is now `unrelated`. That is consistent: those files are not grouped with any of those partners.

## Split

The manifests were written after this check. See `results/retraining_2026/manifests/trashnet_seed42/validation_report.md`.
