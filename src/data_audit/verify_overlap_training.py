"""
verify_overlap_training.py  --  Per-image verification of the test/training overlap (READ-ONLY)

Builds, from the outputs of check_duplicates.py and from the training code, a per-image table that
answers: for each Real Test v2 image with a confirmed overlap, WHICH file(s) it matches, HOW it was
confirmed, and whether the matching file was actually part of the training / fine-tuning data of the
final model (EXP-011, results/efficientnet_garbage/efficientnet_garbage_best.keras).

Nothing under data/, results/ (other than results/duplicate_check/verification/), models, predictions or docs
is modified.

Categories (per Real Test v2 image)
  A  CONFIRMED TRAINING/FINE-TUNING LEAKAGE   >=1 matching file was read by a training stage in the lineage of EXP-011
  B  OVERLAP WITH VALIDATION ONLY             matches only TrashNet files of the validation split (never in a train subset)
  C  EXISTS IN DATASET, NOT PROVEN SEEN       matches only files outside the folders read by the training scripts
  D  CLEAN                                    no confirmed match

Training lineage of EXP-011 (from the training scripts; each stage loads the previous model)
  EXP-003/004  TrashNet train subset                                    (train_efficientnet.py, train_efficientnet_augmented.py)
  EXP-005      + data/taco-prepared             base: efficientnet_augmented   (train_taco_finetune.py)
  EXP-007      + data/realwaste-prepared        base: efficientnet_taco        (train_realwaste_finetune.py)
  EXP-008      + data/taco-prepared + data/local_train   base: efficientnet_realwaste (train_object_centric.py)
  EXP-011      + data/taco-prepared + data/local_train + data/garbage-v2-prepared
               base: efficientnet_household (train_garbage_finetune.py); deployed checkpoint = epoch 1

Run from project root:
    ./venv/bin/python -m src.data_audit.verify_overlap_training
"""

import os
import csv
import json
import glob
import hashlib
import random
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data_audit import check_duplicates as C
from src.data import prepare_garbage_v2 as PG

SRC = C.OUT
OUT = os.path.join(C.OUT, "verification")
SHEETS = os.path.join(OUT, "contact_sheets")
DS = {"trashnet": "TrashNet", "taco_crops": "TACO", "taco_full": "TACO", "realwaste_raw": "RealWaste",
      "realwaste_prepared": "RealWaste", "household_raw": "Household", "household_prepared": "Household",
      "garbage_original": "GarbageV2", "garbage_prepared": "GarbageV2"}
PREPARED = ["trashnet", "taco_crops", "realwaste_prepared", "household_prepared", "garbage_prepared"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(name):
    with open(os.path.join(SRC, name)) as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def class_of(p):
    return p.split("/")[2].replace("\u2705", "")


def main():
    os.makedirs(SHEETS, exist_ok=True)
    idx = C.load_index()
    H = {n: C.load_hash(n) for n in idx}
    tests = idx["test"]
    info = {n: {p: i for i, p in enumerate(idx[n])} for n in idx}

    def size_of(setname, path):
        i = info[setname][path]
        return int(H[setname]["w"][i]), int(H[setname]["h"][i])

    # ---------------- TrashNet split membership (re-created as in the training scripts)
    with open(os.path.join(C.CACHE, "trashnet_split.json")) as f:
        sp = json.load(f)
    tn_train = set(sp["train_shuffled"])
    tn_val = set(sp["val_shuffled"])
    tn_ftval = set(sp["val_unshuffled_finetune_scripts"])

    # ---------------- Garbage v2: original -> prepared mapping (replicates prepare_garbage_v2.py exactly)
    class_root = PG.find_class_root()
    counters = {c: 0 for c in PG.TRASHNET_CLASSES}
    g_map, g_inv = {}, {}
    for entry in sorted(os.listdir(class_root)):
        key = entry.lower()
        if key not in PG.CLASS_MAP:
            continue
        dst_class = PG.CLASS_MAP[key]
        src_dir = os.path.join(class_root, entry)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(PG.IMAGE_EXTS):
                continue
            src = os.path.join(class_root, entry, fname)
            dst = os.path.join(PG.OUTPUT_ROOT, dst_class, f"garbage_{key}_{counters[dst_class]:04d}.jpg")
            counters[dst_class] += 1
            g_map[src] = dst
            g_inv[dst] = src
    ok_map = sum(1 for s_, d_ in g_map.items() if s_ in info["garbage_original"] and d_ in info["garbage_prepared"])
    ph = H["garbage_original"]["ph"][:, 0]
    php = H["garbage_prepared"]["ph"][:, 0]
    dist = [int(C.ham(np.array([ph[info["garbage_original"][s_]]], dtype=np.uint64),
                      np.array([php[info["garbage_prepared"][d_]]], dtype=np.uint64))[0, 0]) for s_, d_ in g_map.items()]
    map_check = {"pairs": len(g_map), "both_files_exist": ok_map, "max_phash_dist_original_vs_prepared": max(dist),
                 "pairs_with_dist_le_8": sum(1 for d_ in dist if d_ <= 8)}

    # ---------------- gather matches per test image
    matches = defaultdict(dict)       # test -> key(training copy path) -> record

    def training_status(set_, train_copy):
        if set_ == "trashnet":
            if train_copy in tn_train:
                return "TRAINED", "TrashNet 80% train subset (all stages EXP-003..EXP-011); " \
                                   + ("also in validation list of fine-tuning scripts" if train_copy in tn_ftval else ""), "LOG+CODE"
            return "VALIDATION_ONLY", "TrashNet validation split (not in any train subset)", "CODE"
        if set_ == "garbage_prepared":
            return "TRAINED", "EXP-011 stage (read from data/garbage-v2-prepared; epoch 1 = deployed checkpoint)", "LOG+CODE"
        if set_ == "realwaste_prepared":
            return "TRAINED", "EXP-007 stage (data/realwaste-prepared); inherited by EXP-011 via base-model chain", "CODE+CHAIN"
        if set_ == "household_prepared":
            return "TRAINED", "EXP-008 and EXP-011 stages (data/local_train)", "LOG+CODE"
        if set_ == "taco_crops":
            return "TRAINED", "EXP-005/008/011 stages (data/taco-prepared)", "LOG+CODE"
        return "NOT_PROVEN", "folder not read by any training script", ""

    def add(t, set_, matched, mtype, method, exact):
        # training copy: the file the training scripts really read
        if set_ == "garbage_original":
            tcopy_set, tcopy = "garbage_prepared", g_map[matched]
            rel = "re-encoded JPEG q95 copy made by prepare_garbage_v2.py"
        elif set_ in ("realwaste_raw", "household_raw"):
            tcopy_set = set_.replace("_raw", "_prepared")
            sha = H[set_]["sha"][info[set_][matched]]
            cands = [p for p, s_ in zip(idx[tcopy_set], H[tcopy_set]["sha"]) if s_ == sha]
            tcopy = cands[0] if cands else matched
            rel = "byte-identical copy made by prepare script"
        else:
            tcopy_set, tcopy, rel = set_, matched, "same file"
        key = tcopy
        st, why, proof = training_status(tcopy_set, tcopy)
        rec = matches[t].get(key)
        new = {"dataset": DS[tcopy_set], "set": tcopy_set, "matched_path": matched, "matched_set": set_,
               "training_copy_path": tcopy, "training_copy_relation": rel, "type": mtype, "method": method,
               "status": st, "status_reason": why, "proof": proof,
               "trashnet_split": ("train" if tcopy in tn_train else "val") if tcopy_set == "trashnet" else ""}
        if rec is None or (mtype == "EXACT" and rec["type"] != "EXACT"):
            matches[t][key] = new

    # exact (SHA-256) matches
    for r in read_csv("exact_matches.csv"):
        if r["hash_type"] != "sha" or r["query_set"] != "test" or r["matched_set"] == "test(internal)":
            continue
        if r["matched_set"] in ("realwaste_prepared", "household_prepared", "trashnet", "garbage_original",
                                "taco_crops", "garbage_prepared"):
            add(r["query_path"], r["matched_set"], r["matched_path"], "EXACT",
                "SHA-256 of file bytes identical (pixel SHA-256 identical)", True)

    # near-identical pixels (pHash/dHash + ZNCC + SIFT) -- tier B
    trans = {(r["compared_with"], r["test_image"], r["matched_image"]): int(r["dihedral_transform_idx"])
             for r in read_csv("perceptual_candidates.csv")}
    for r in read_csv("perceptual_verified.csv"):
        if not r["tier"].startswith("B") or r["compared_with"] not in PREPARED:
            continue
        t, m, s_ = r["test_image"], r["matched_image"], r["compared_with"]
        tr = trans.get((s_, t, m), 0)
        ws, hs = H["test"]["w"][info["test"][t]], H["test"]["h"][info["test"][t]]
        wm, hm = size_of(s_, m)
        aspect = abs((ws / hs) / (wm / hm) - 1) if hs and hm else 0
        # aspect-ratio change with pHash/ZNCC ~ identical on the WHOLE image = anisotropic resize (same content), not a crop
        mtype = "CROPPED-MIRRORED" if tr != 0 else "RE-ENCODED-RESIZED"
        method = (f"pHash={r['phash_dist']} dHash={r['dhash_dist']} ZNCC={r['zncc']} "
                  f"(dihedral transform {tr}); SIFT+RANSAC inliers={r['sift_inliers']}"
                  + (f"; size test {ws}x{hs} vs match {wm}x{hm} (aspect changed {aspect:.0%}, full content kept)" if aspect > 0.02 else ""))
        add(t, s_, m, mtype, method, False)

    # manually verified (visual) SIFT-only matches
    emb = read_csv("embedding_sift_candidates.csv")
    for r in read_csv("manual_review.csv"):
        if r["kind"] == "dataset" and r["verdict"] == "duplicate":
            best = max((e for e in emb if e["test_image"] == r["test_image"] and DS.get(e["compared_with"]) == r["dataset_or_image2"]),
                       key=lambda e: int(e["sift_ransac_inliers"]))
            add(r["test_image"], best["compared_with"], best["candidate_image"], "CROPPED-MIRRORED",
                f"SIFT+RANSAC inliers={best['sift_ransac_inliers']} + manual visual review ({r['note']})", False)

    # ---------------- classify all 600
    prob = {r["test_image"]: r for r in read_csv("test_overlap_per_image.csv")}
    rows600, wide, long_rows = [], [], []
    cat_of = {}
    for t in tests:
        ms = list(matches.get(t, {}).values())
        confirmed = bool(prob[t]["confirmed_overlap_datasets"])
        if confirmed and not ms:
            raise RuntimeError(f"confirmed image without a recorded match: {t}")
        if not ms:
            cat = "D"
        elif any(m["status"] == "TRAINED" for m in ms):
            cat = "A"
        elif any(m["status"] == "VALIDATION_ONLY" for m in ms):
            cat = "B"
        else:
            cat = "C"
        cat_of[t] = cat
        ms.sort(key=lambda m: (m["status"] != "TRAINED", m["dataset"], m["type"] != "EXACT", m["matched_path"]))
        for m in ms:
            long_rows.append([os.path.basename(t), t, class_of(t), cat, m["dataset"], m["matched_set"],
                              os.path.basename(m["matched_path"]), m["matched_path"], m["type"], m["method"],
                              m["status"], m["status_reason"], m["proof"], os.path.basename(m["training_copy_path"]),
                              m["training_copy_path"], m["training_copy_relation"], m["trashnet_split"]])
        if ms:
            wide.append([os.path.basename(t), t, class_of(t), cat,
                         ";".join(sorted({m["dataset"] for m in ms})),
                         " | ".join(f"{m['dataset']}: {m['matched_path']}" for m in ms),
                         " | ".join(f"{m['dataset']}: {m['type']}" for m in ms),
                         " | ".join(f"{m['dataset']}: {m['method']}" for m in ms),
                         " | ".join(f"{m['dataset']}: {m['status']}" for m in ms),
                         " | ".join(f"{m['dataset']}: {m['training_copy_path']}" for m in ms),
                         len(ms)])
        flag = ""
        if cat == "D" and prob[t]["weak_unconfirmed_datasets"]:
            flag = "possible duplicate not confirmed: " + prob[t]["weak_unconfirmed_datasets"]
        rows600.append([os.path.basename(t), t, class_of(t), cat, flag])

    hdr_wide = ["test_file", "test_path", "class", "category", "datasets_with_match", "matched_files",
                "match_types", "confirmation_methods", "training_status", "training_copy_files", "n_matches"]
    hdr_long = ["test_file", "test_path", "class", "category", "dataset", "matched_set", "matched_file",
                "matched_path", "match_type", "confirmation_method", "training_status", "training_status_reason",
                "proof_level", "training_copy_file", "training_copy_path", "training_copy_relation", "trashnet_split"]
    order = {"A": 0, "B": 1, "C": 2}
    wide.sort(key=lambda r: (order[r[3]], r[2], r[1]))
    long_rows.sort(key=lambda r: (order.get(r[3], 3), r[2], r[1]))
    write_csv(os.path.join(OUT, "overlap_388_per_image.csv"), hdr_wide, [w for w in wide if w[3] in "ABC"])
    write_csv(os.path.join(OUT, "overlap_388_matches_long.csv"), hdr_long, long_rows)
    write_csv(os.path.join(OUT, "classification_all_600.csv"),
              ["test_file", "test_path", "class", "category", "note"], rows600)

    counts = Counter(cat_of.values())
    sub = Counter()
    for t, c in cat_of.items():
        if c == "A":
            ms = matches[t].values()
            sets = {m["dataset"] for m in ms if m["status"] == "TRAINED"}
            proofs = {m["proof"] for m in ms if m["status"] == "TRAINED"}
            sub["A_proof_LOG+CODE" if "LOG+CODE" in proofs else "A_proof_CODE+CHAIN_only"] += 1
            sub["A_trained_datasets=" + "+".join(sorted(sets))] += 1
    types = Counter()
    for t in tests:
        if cat_of[t] in "ABC":
            types[min((m["type"] for m in matches[t].values()),
                      key=lambda x: ["EXACT", "RE-ENCODED-RESIZED", "CROPPED-MIRRORED"].index(x))] += 1
    status_detail = {
        "TrashNet_matches_in_train": sum(1 for t in tests for m in matches[t].values() if m["set"] == "trashnet" and m["status"] == "TRAINED"),
        "TrashNet_matches_validation_only": sum(1 for t in tests for m in matches[t].values() if m["set"] == "trashnet" and m["status"] == "VALIDATION_ONLY")}

    # ---------------- re-verify EXACT claims by re-hashing the files now
    sha_rows, bad = [], 0
    exact_pairs = []
    for t in tests:
        for m in matches[t].values():
            if m["type"] == "EXACT":
                exact_pairs.append((t, m))
    n_exact_imgs = len({t for t, _ in exact_pairs})
    for t, m in exact_pairs:
        a, b = sha256_file(t), sha256_file(m["matched_path"])
        if a != b:
            bad += 1
        prep_dist = ""
        if m["matched_set"] == "garbage_original":
            prep_dist = int(C.ham(np.array([ph[info["garbage_original"][m["matched_path"]]]], dtype=np.uint64),
                                  np.array([php[info["garbage_prepared"][m["training_copy_path"]]]], dtype=np.uint64))[0, 0])
        sha_rows.append([os.path.basename(t), t, m["matched_path"], a, b, "IDENTICAL" if a == b else "DIFFERENT",
                         m["dataset"], m["training_copy_path"], m["training_copy_relation"], prep_dist])
    write_csv(os.path.join(OUT, "sha256_verification_exact.csv"),
              ["test_file", "test_path", "matched_path", "sha256_test", "sha256_matched", "result", "dataset",
               "training_copy_path", "training_copy_relation", "phash_dist_original_vs_training_copy"], sha_rows)

    # ---------------- training provenance checks
    prov = {"garbage_original_to_prepared_mapping": map_check,
            "trashnet_split_check": {
                "train_shuffled": len(tn_train), "val_shuffled": len(tn_val), "overlap_train_val_shuffled": len(tn_train & tn_val),
                "val_shuffled_class_counts": dict(Counter(p.split("/")[2] for p in tn_val)),
                "support_in_results/efficientnet/classification_report.txt": {"cardboard": 69, "glass": 102, "metal": 88,
                                                                              "paper": 123, "plastic": 89, "trash": 34}},
            "final_stage_log_counts": {"results/efficientnet_garbage/train.log": "TrashNet 2022 train + 505 val, TACO 3601, Household 7500, Garbage v2 12259; 796 steps/epoch x 32 = 25,4k samples; best checkpoint = epoch 1"},
            "directory_counts_now": {n: len(idx[n]) for n in ("trashnet", "taco_crops", "realwaste_prepared", "household_prepared", "garbage_prepared")}}
    # step count check: 2022+3601+7500+12259
    prov["final_stage_expected_samples"] = 2022 + 3601 + 7500 + 12259
    prov["final_stage_log_steps_x32"] = 796 * 32

    summary = {"categories": {"A_confirmed_training_or_finetuning_leakage": counts["A"],
                              "B_overlap_with_validation_only": counts["B"],
                              "C_exists_in_dataset_not_proven_seen": counts["C"],
                              "D_clean": counts["D"]},
               "total": sum(counts.values()), "confirmed_overlap_images": counts["A"] + counts["B"] + counts["C"],
               "A_breakdown": dict(sub), "best_match_type_per_image": dict(types), "status_detail": status_detail,
               "exact": {"images_with_exact_match": n_exact_imgs, "exact_pairs": len(exact_pairs),
                         "pairs_with_different_sha_after_rehash": bad},
               "provenance_checks": prov}
    with open(os.path.join(OUT, "category_counts.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))

    # ---------------- 20 examples + 10 SHA examples
    rnd = random.Random(7)
    def pick(pred, k):
        pool = [(t, m) for t in tests for m in matches[t].values() if pred(t, m)]
        pool.sort(key=lambda x: x[0])
        return rnd.sample(pool, k)
    ex = []
    ex += [("TrashNet", *x) for x in pick(lambda t, m: m["set"] == "trashnet" and m["type"] == "EXACT" and m["status"] == "TRAINED", 5)]
    ex += [("RealWaste", *x) for x in pick(lambda t, m: m["set"] == "realwaste_prepared" and m["type"] == "EXACT", 5)]
    ex += [("GarbageV2", *x) for x in pick(lambda t, m: m["set"] == "garbage_prepared" and m["type"] == "EXACT", 5)]
    near = []
    near += pick(lambda t, m: m["set"] == "trashnet" and m["type"] == "RE-ENCODED-RESIZED", 1)
    near += pick(lambda t, m: m["set"] == "realwaste_prepared" and m["type"] == "RE-ENCODED-RESIZED", 1)
    near += pick(lambda t, m: m["set"] == "garbage_prepared" and m["type"] == "RE-ENCODED-RESIZED", 1)
    near += pick(lambda t, m: "manual visual" in m["method"], 2)
    ex += [("near-duplicate", *x) for x in near]
    lines = ["# 20 shembuj konkretë (LEFT = Real Test v2, RIGHT = dataseti tjetër)\n"]
    for k, (grp, t, m) in enumerate(ex, 1):
        lines += [f"## {k}. [{grp}] {os.path.basename(t)} -> {m['dataset']} / {m['type']}",
                  f"- TEST : `{os.path.abspath(t)}`",
                  f"- MATCH: `{os.path.abspath(m['matched_path'])}`",
                  (f"- Skedari që e ka lexuar trajnimi: `{os.path.abspath(m['training_copy_path'])}` ({m['training_copy_relation']})"
                   if m["training_copy_path"] != m["matched_path"] else "- Skedari i trajnimit = skedari i përputhjes"),
                  f"- Metoda: {m['method']}", f"- Statusi: {m['status']} - {m['status_reason']}\n"]
    with open(os.path.join(OUT, "examples_20.md"), "w") as f:
        f.write("\n".join(lines))

    sha_ex = []
    def exact_pick(pred, k):
        pool = [r for r in sha_rows if pred(r)]
        return rnd.sample(pool, k)
    sha_ex += exact_pick(lambda r: r[6] == "TrashNet", 4)
    sha_ex += exact_pick(lambda r: r[6] == "RealWaste", 3)
    sha_ex += exact_pick(lambda r: r[6] == "GarbageV2", 3)
    lines = ["# 10 shembuj me SHA-256 (rillogaritur gjatë këtij verifikimi)\n"]
    for k, r in enumerate(sha_ex, 1):
        lines += [f"## {k}. {r[6]}", f"- `{os.path.abspath(r[1])}`", f"  SHA-256 = `{r[3]}`",
                  f"- `{os.path.abspath(r[2])}`", f"  SHA-256 = `{r[4]}`", f"- Rezultati: **{r[5]}**"]
        if r[6] == "GarbageV2":
            lines.append(f"- Skedari që e ka lexuar trajnimi: `{os.path.abspath(r[7])}` ({r[8]}; pHash distance nga origjinali = {r[9]})")
        lines.append("")
    with open(os.path.join(OUT, "examples_10_sha256.md"), "w") as f:
        f.write("\n".join(lines))

    # ---------------- contact sheets (one cell per overlapping test image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    cases = [t for t in tests if cat_of[t] in "ABC"]
    cases.sort(key=lambda t: (order[cat_of[t]], sorted({m["dataset"] for m in matches[t].values()}),
                              class_of(t), t))
    TH, CW, PER_PAGE, COLS = 190, 250, 8, 2
    cell_w, cell_h = CW * 3 + 12, TH + 66
    pages = 0
    index_rows = []
    pil_pages = []
    for s0 in range(0, len(cases), PER_PAGE):
        chunk = cases[s0:s0 + PER_PAGE]
        page = Image.new("RGB", (cell_w * COLS, 22 + cell_h * ((len(chunk) + COLS - 1) // COLS)), "white")
        d = ImageDraw.Draw(page)
        d.text((6, 4), f"Real Test v2 overlap - faqja {pages + 1} (rastet {s0 + 1}-{s0 + len(chunk)} nga {len(cases)}); "
                       f"LEFT = test, RIGHT = dataseti tjetër", fill="black", font=font)
        for k, t in enumerate(chunk):
            x0, y0 = (k % COLS) * cell_w, 22 + (k // COLS) * cell_h
            ms = sorted(matches[t].values(), key=lambda m: (m["status"] != "TRAINED", m["dataset"], m["type"] != "EXACT"))
            shown = [(t, f"TEST {os.path.basename(t)}")] + [(m["matched_path"], m) for m in ms[:2]]
            for j, (pth, lab) in enumerate(shown):
                try:
                    im = Image.open(pth).convert("RGB")
                    im.thumbnail((CW - 6, TH))
                    page.paste(im, (x0 + j * CW + 3, y0 + 2))
                except Exception:
                    pass
            d.rectangle([x0, y0, x0 + cell_w - 6, y0 + cell_h - 6], outline="#bbbbbb")
            ty = y0 + TH + 4
            d.text((x0 + 4, ty), f"#{s0 + k + 1}  LEFT = REAL TEST v2  [{cat_of[t]}]", fill="#aa0000", font=font)
            d.text((x0 + 4, ty + 14), os.path.basename(t)[:38], fill="black", font=font)
            d.text((x0 + 4, ty + 28), f"class: {class_of(t)}", fill="black", font=font)
            for j, m in enumerate(ms[:2]):
                cx = x0 + (j + 1) * CW + 4
                d.text((cx, ty), f"RIGHT {j + 1} = {m['dataset']}", fill="#003399", font=font)
                d.text((cx, ty + 14), os.path.basename(m["matched_path"])[:38], fill="black", font=font)
                d.text((cx, ty + 28), f"{m['type']}", fill="black", font=font)
                d.text((cx, ty + 42), f"{m['status']}", fill=("#aa0000" if m["status"] == "TRAINED" else "#cc7700"), font=font)
            if len(ms) > 2:
                d.text((x0 + 4, ty + 42), f"(+{len(ms) - 2} more matches, see CSV)", fill="#aa0000", font=font)
            index_rows.append([s0 + k + 1, pages + 1, os.path.basename(t), cat_of[t], len(ms)])
        fn = os.path.join(SHEETS, f"page_{pages + 1:03d}.png")
        page.save(fn)
        pil_pages.append(page.copy())
        pages += 1
    write_csv(os.path.join(OUT, "contact_sheets_index.csv"), ["case_no", "page", "test_file", "category", "n_matches"], index_rows)
    pil_pages[0].save(os.path.join(OUT, "contact_sheets_all.pdf"), save_all=True, append_images=pil_pages[1:], quality=80)
    print("contact sheet pages:", pages)


if __name__ == "__main__":
    main()
