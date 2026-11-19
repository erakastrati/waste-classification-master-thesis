# EXP-005 run directory

Reserved for a future TACO fine-tune. Phase 1 did not create a run. The historical parent relationship is documented in `audit/EXP-004/audit_report.md`: `src/training/train_taco_finetune.py` loads `results/efficientnet_augmented/efficientnet_augmented.keras`. Whether that parent may be reused is an open decision.

## Unique run folders

Create one new directory per attempt:

`runs/<UTC timestamp>_<short label>/`

If the chosen directory already exists, stop. Do not delete, rename, or replace files inside an existing run directory. Do not write checkpoints, logs, or reports into this directory itself.
