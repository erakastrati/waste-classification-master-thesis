# Pending pair review table

`pending_pairs_review_table.csv` lists the 136 pairs whose visual decision is still `PENDING`. P001, P002, and P139 are not in this table. They are the SHA-256 proposal in `exact_duplicate_exclusion_proposal.md`.

P003–P007 are in this table and also in `paper_pairs_P003-P007_decision_proposal.md`. Their scores are not a verdict.

The PDF page is the page of `visual_review/trashnet_visual_review.pdf`. The `sheet` column is the contact-sheet PNG. Both show the pair ID.

| Priority | Pairs | PDF pages | What the reviewer is being asked |
|---:|---:|---|---|
| 1. Paper high-similarity | 5 (P003–P007) | 3–5 | Decide using the paper proposal. Still pending. |
| 2. Opposite class | 29 (P008–P036) | 6–20 | Different folder labels. Automatic score only. |
| 3. Isolated same-class | 6 (P037–P042) | 21–26 | One pair each, no shared image with another pair. |
| 4. Large similarity components | 96 (C01 has 77, C02 has 19) | 27–75 | C01 on pages 27–65, C02 on pages 66–75. Many images repeat. Use `repeated_images.csv` before writing a decision. |

Columns in the CSV: pair ID, PDF page, sheet path, both image paths, class, raw split, train or validation membership, same-class flag, pHash, dHash, minimum hash distance, ZNCC, exact-duplicate flag (all `no` in this table), component, whether either image appears in another pair, those other pair IDs, and the pending status.

`visual_review_status` and `proposed_action` are `PENDING` on every row. Nothing in this table was inferred from the similarity score.
