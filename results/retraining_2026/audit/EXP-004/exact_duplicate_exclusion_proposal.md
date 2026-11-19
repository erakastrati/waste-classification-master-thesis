# Proposal — exclude six SHA-256 duplicate files

**Superseded on 18 November 2026.** The reviewer did not accept exclusion of all six files. The recorded decision is in `human_review_decision_summary.md`: keep `plastic152.jpg`, `glass389.jpg`, and `glass115.jpg`; exclude only `glass176.jpg`, `plastic332.jpg`, and `metal91.jpg`. This page is the earlier proposal and is not the decision.

Status of this page: **superseded proposal, not executed**. No file under `data/trashNet-dataset/` has been removed, moved, or relabeled. When this page was written, the register had `proposed_action` `PROPOSE_EXCLUDE` and an empty `reviewer_decision`. The register now records the keep-one decisions above. This page remains the earlier byte-identity proposal.

The hashes were recomputed during the dataset mapping audit and are recorded in `dataset_mapping_audit.md`.

| Pair | PDF page | Files | Folder labels | Split | SHA-256 | Bytes |
|---|---:|---|---|---|---|---:|
| P001 | 1 | `data/trashNet-dataset/glass/glass176.jpg` | glass | validation | `c41b99aec8a3257c8668a2144ebbaf8042018f0a85d88e22bf4e413b43693dd8` | 16,192 |
| P001 | 1 | `data/trashNet-dataset/plastic/plastic152.jpg` | plastic | training | same | 16,192 |
| P002 | 1 | `data/trashNet-dataset/glass/glass389.jpg` | glass | training | `e971f4f8f50e960e454ea724cec922fc2988bbf4027aa48b553e139bb5890968` | 15,420 |
| P002 | 1 | `data/trashNet-dataset/plastic/plastic332.jpg` | plastic | validation; also unshuffled fine-tune validation | same | 15,420 |
| P139 | 2 | `data/trashNet-dataset/glass/glass115.jpg` | glass | training | `81546d1362d75fc60718e5852593c58f66f813e8c1e9d1eae6871fedb02e2868` | 8,672 |
| P139 | 2 | `data/trashNet-dataset/metal/metal91.jpg` | metal | training | same | 8,672 |

Each of these six paths appears in exactly one pair. Excluding a path does not collide with a second pair.

## Proposed action

Exclude all six files from the corrected TrashNet inventory used by EXP-003 and EXP-004.

Keeping one copy would require choosing glass, plastic, or metal for that copy. That would be a new label. Exclusion removes the conflict without assigning a label.

## What approval must say

An approval has to name the six paths and state that they are left out of both `train.txt` and `val.txt`. Until that sentence is written into `unresolved_decisions.md` as accepted, the files stay in `data/trashNet-dataset/` and in the historical split.
