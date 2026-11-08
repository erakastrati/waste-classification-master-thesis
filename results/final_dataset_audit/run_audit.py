"""Run the existing duplicate audit against the NEW data/real_test_dataset.

Training-set hash and embedding caches are copies. The old
results/duplicate_check/ directory is not written.
"""

import os
import csv
import json
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import src.data_audit.check_duplicates as C

C.OUT = "results/final_dataset_audit"
C.CACHE = os.path.join(C.OUT, "cache")
C.SETS["test"] = ("data/real_test_dataset", None)
os.makedirs(C.CACHE, exist_ok=True)

for stage in ("index", "hash", "exact", "perceptual", "embed", "sift", "verify", "control"):
    print(f"\n===== STAGE {stage} =====", flush=True)
    C.STAGES[stage]()

# Decision from the same evidence files the original report uses.
# Model predictions are intentionally not read: they belong to the old dataset.
idx = C.load_index()
tests = idx["test"]
tpos = {p: i for i, p in enumerate(tests)}
confirmed = defaultdict(list)
suspected = defaultdict(list)

with open(os.path.join(C.OUT, "exact_matches.csv")) as f:
    for r in csv.DictReader(f):
        if r["query_set"] != "test" or r["matched_set"] in ("test(internal)",):
            continue
        ds = C.DATASET_OF.get(r["matched_set"])
        if not ds:
            continue
        confirmed[r["query_path"]].append(
            {"dataset": ds, "evidence": r["hash_type"], "match": r["matched_path"]}
        )

with open(os.path.join(C.OUT, "perceptual_verified.csv")) as f:
    for r in csv.DictReader(f):
        if r["compared_with"] == "test(internal)":
            continue
        ds = C.DATASET_OF[r["compared_with"]]
        tier = r["tier"]
        item = {"dataset": ds, "evidence": tier, "match": r["matched_image"],
                "zncc": r["zncc"], "sift_inliers": r["sift_inliers"]}
        if tier.startswith("A") or tier.startswith("B"):
            confirmed[r["test_image"]].append(item)
        elif tier.startswith("C") or tier.startswith("D"):
            suspected[r["test_image"]].append(item)

with open(os.path.join(C.OUT, "embedding_sift_candidates.csv")) as f:
    for r in csv.DictReader(f):
        if r["compared_with"] == "test(internal)":
            continue
        inl = int(r["sift_ransac_inliers"])
        if inl < C.SIFT_STRONG:
            continue
        ds = C.DATASET_OF[r["compared_with"]]
        suspected[r["test_image"]].append(
            {"dataset": ds, "evidence": "sift_geometric", "match": r["candidate_image"],
             "sift_inliers": inl, "cosine": r["cosine_similarity"]}
        )

with open(os.path.join(C.OUT, "sift_exhaustive_hits.csv")) as f:
    for r in csv.DictReader(f):
        if r["compared_with"] != "taco_full":
            continue
        inl = int(r["sift_ransac_inliers"])
        if inl < C.SIFT_STRONG:
            continue
        suspected[r["test_image"]].append(
            {"dataset": "TACO", "evidence": "sift_exhaustive", "match": r["matched_image"],
             "sift_inliers": inl}
        )

# internal exact groups, reported but not a training overlap
internal = []
with open(os.path.join(C.OUT, "exact_matches.csv")) as f:
    for r in csv.DictReader(f):
        if r["query_set"] == "test" and r["matched_set"] == "test(internal)":
            internal.append({"hash_type": r["hash_type"], "image": r["query_path"], "others": r["matched_path"]})

decision = {
    "n_test": len(tests),
    "confirmed_training_overlap_images": len(confirmed),
    "suspected_training_overlap_images": len(suspected),
    "confirmed": confirmed,
    "suspected": suspected,
    "internal_exact_groups": internal,
    "block_evaluation": bool(confirmed or suspected),
}
with open(os.path.join(C.OUT, "overlap_decision.json"), "w") as f:
    json.dump(decision, f, indent=2)
print("AUDIT_DONE", flush=True)
print("confirmed", len(confirmed), "suspected", len(suspected), "block", decision["block_evaluation"], flush=True)
