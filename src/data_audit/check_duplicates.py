"""
check_duplicates.py  --  Duplicate / near-duplicate audit (READ-ONLY on datasets)

Purpose
-------
Checks whether images of the final held-out test set (data/real_test_dataset,
600 photos) also occur
  (a) inside the test set itself, and
  (b) in any dataset the model saw during training / fine-tuning:
      TrashNet, TACO (crops + full source photos), RealWaste, Household Waste
      (real_world only), Garbage Classification v2.

Nothing under data/ is modified. All outputs go to results/duplicate_check/.

Detection layers (each one catches what the previous cannot)
------------------------------------------------------------
 L1  exact byte duplicate     SHA-256 of the file bytes
 L2  exact pixel duplicate    SHA-256 of decoded RGB pixels (same image, different file
                              container / metadata)
 L3  perceptual hash          pHash (32x32 DCT, 8x8 low freq, 64 bit) and dHash (9x8, 64 bit)
                              Hamming distance. The test image is hashed under all 8
                              dihedral transforms (flips / 90 deg rotations) so that
                              mirrored or rotated copies are found. Candidates
                              (pHash<=12 or dHash<=12) are verified with the zero-mean
                              normalised cross-correlation (ZNCC) of 32x32 gray thumbnails.
                              Catches re-encoding, resizing, compression, brightness changes.
 L4  deep-feature retrieval   ImageNet EfficientNetB0 pooled features (cosine), top-5
                              neighbours per test image per dataset.
 L5  local-feature matching   SIFT + ratio test + RANSAC similarity transform. Counts
                              geometrically consistent inliers. Catches crops, partial
                              views, strong rescaling. Run (i) exhaustively test vs
                              TACO full photos and test vs test, (ii) on the L4 top-5
                              candidates of every other dataset.

The pipeline is validated with a synthetic positive control (known modified copies
of test images) and a negative control (random unrelated pairs).

Usage (project root)
--------------------
    ./venv/bin/python -m src.data_audit.check_duplicates --stage index
    ./venv/bin/python -m src.data_audit.check_duplicates --stage hash
    ./venv/bin/python -m src.data_audit.check_duplicates --stage split
    ./venv/bin/python -m src.data_audit.check_duplicates --stage exact
    ./venv/bin/python -m src.data_audit.check_duplicates --stage perceptual
    ./venv/bin/python -m src.data_audit.check_duplicates --stage embed
    ./venv/bin/python -m src.data_audit.check_duplicates --stage sift
    ./venv/bin/python -m src.data_audit.check_duplicates --stage control
    ./venv/bin/python -m src.data_audit.check_duplicates --stage report
"""

import os
import io
import csv
import sys
import json
import time
import hashlib
import argparse
import random
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance
from scipy.fft import dctn

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

OUT = "results/duplicate_check"
CACHE = os.path.join(OUT, "cache")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# set name -> (root folder, required path substring or None)
SETS = {
    "test":               ("data/real_test_dataset", None),
    "trashnet":           ("data/trashNet-dataset", None),
    "taco_crops":         ("data/taco-prepared", None),
    "taco_full":          ("data/taco-dataset/data", None),
    "realwaste_raw":      ("data/realwaste-dataset", None),
    "realwaste_prepared": ("data/realwaste-prepared", None),
    "household_raw":      ("data/household-waste/images", "/real_world/"),
    "household_prepared": ("data/local_train", None),
    "garbage_original":   ("data/garbage-v2/original", None),
    "garbage_prepared":   ("data/garbage-v2-prepared", None),
}
# Datasets (folders) that were actually read by the training scripts.
SEEN_BY_MODEL = ["trashnet", "taco_crops", "realwaste_prepared",
                 "household_prepared", "garbage_prepared"]
# Raw/source folders the seen sets were derived from.
SOURCE_SETS = ["taco_full", "realwaste_raw", "household_raw", "garbage_original"]
COMPARE_SETS = SEEN_BY_MODEL + SOURCE_SETS            # everything the test set is compared with
EMBED_SETS = SEEN_BY_MODEL + ["taco_full"]            # sets embedded for L4

HASH_T = 12            # candidate threshold on Hamming distance (pHash or dHash)
CONFIRM_HASH_T = 10    # confirmed if min(pHash,dHash) <= this ...
CONFIRM_ZNCC = 0.90    # ... and ZNCC >= this
REVIEW_ZNCC = 0.75
SIFT_STRONG = 15       # unique RANSAC inliers considered a strong geometric match
SIFT_WEAK = 8          # below this: noise (calibrated by the negative control)
SIFT_NFEAT = 400
SIFT_MAX_DIM = 512
TOPK = 5

TRANSFORMS = [None,
              Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
              Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
              Image.Transpose.ROTATE_270, Image.Transpose.TRANSPOSE,
              Image.Transpose.TRANSVERSE]

POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------

def list_images(root, must_contain):
    paths = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith("."))
        for fn in sorted(fns):
            if fn.startswith(".") or not fn.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(dp, fn)
            if must_contain and must_contain not in p:
                continue
            paths.append(p)
    return paths


def stage_index():
    os.makedirs(CACHE, exist_ok=True)
    idx = {}
    for name, (root, sub) in SETS.items():
        idx[name] = list_images(root, sub)
        log(f"{name:20s} {root:40s} {len(idx[name]):6d} images")
    with open(os.path.join(CACHE, "index.json"), "w") as f:
        json.dump({"sets": {k: {"root": SETS[k][0], "filter": SETS[k][1], "n": len(v)}
                            for k, v in idx.items()},
                   "paths": idx}, f)


def load_index():
    with open(os.path.join(CACHE, "index.json")) as f:
        return json.load(f)["paths"]


# ----------------------------------------------------------------------------
# Hashing primitives
# ----------------------------------------------------------------------------

def _pack(bits):
    return np.packbits(bits.astype(np.uint8)).view(">u8")[0].astype(np.uint64)


def phash_dhash_thumb(gray):
    """gray: PIL 'L' image. Returns (pHash uint64, dHash uint64, 32x32 uint8 thumb)."""
    g32 = gray.resize((32, 32), Image.LANCZOS, reducing_gap=2.0)
    arr = np.asarray(g32, dtype=np.float64)
    low = dctn(arr, norm="ortho")[:8, :8].flatten()
    med = np.median(low[1:])
    ph = _pack(low > med)
    g98 = np.asarray(gray.resize((9, 8), Image.LANCZOS, reducing_gap=2.0), dtype=np.float64)
    dh = _pack((g98[:, 1:] > g98[:, :-1]).flatten())
    return ph, dh, np.asarray(g32, dtype=np.uint8).flatten()


def image_features(rgb, all_transforms):
    gray = rgb.convert("L")
    ts = TRANSFORMS if all_transforms else TRANSFORMS[:1]
    phs, dhs, ths = [], [], []
    for t in ts:
        g = gray if t is None else gray.transpose(t)
        p, d, th = phash_dhash_thumb(g)
        phs.append(p); dhs.append(d); ths.append(th)
    return np.array(phs, dtype=np.uint64), np.array(dhs, dtype=np.uint64), np.stack(ths)


def hash_worker(args):
    path, k8 = args
    try:
        with open(path, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        pix = hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()
        ph, dh, th = image_features(rgb, k8)
        return sha, pix, rgb.size[0], rgb.size[1], ph, dh, th, ""
    except Exception as e:  # unreadable file
        k = 8 if k8 else 1
        return "", "", 0, 0, np.zeros(k, np.uint64), np.zeros(k, np.uint64), \
            np.zeros((k, 1024), np.uint8), f"{type(e).__name__}: {e}"


def stage_hash():
    idx = load_index()
    ctx = mp.get_context("fork")
    for name, paths in idx.items():
        outp = os.path.join(CACHE, f"hash_{name}.npz")
        if os.path.exists(outp):
            log(f"{name}: cached"); continue
        t0 = time.time()
        k8 = name == "test"
        with ctx.Pool(11) as pool:
            res = pool.map(hash_worker, [(p, k8) for p in paths], chunksize=16)
        sha = np.array([r[0] for r in res]); pix = np.array([r[1] for r in res])
        w = np.array([r[2] for r in res]); h = np.array([r[3] for r in res])
        ph = np.stack([r[4] for r in res]); dh = np.stack([r[5] for r in res])
        th = np.stack([r[6] for r in res]); err = np.array([r[7] for r in res])
        np.savez_compressed(outp, sha=sha, pix=pix, w=w, h=h, ph=ph, dh=dh, th=th, err=err)
        log(f"{name:20s} n={len(paths):6d} errors={int((err != '').sum())} {time.time()-t0:.0f}s")


def load_hash(name):
    z = np.load(os.path.join(CACHE, f"hash_{name}.npz"))
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------
# TrashNet train/validation membership (re-created exactly as in the training scripts)
# ----------------------------------------------------------------------------

def stage_split():
    """Re-create the TrashNet train/validation membership exactly as the scripts do.

    Keras (validation_split=0.2, seed=42) takes the LAST 20% of the file list as
    validation. With shuffle=True the list is shuffled (seed 42) first; with
    shuffle=False it stays in sorted directory order. The project uses both:
      * shuffled training subset                    -> all training scripts (2022 files)
      * shuffled validation subset (complement)     -> EXP-001..004 and all evaluate_*.py
      * UNshuffled validation subset (last 505 files in sorted order)
                                                    -> val_ds of the four fine-tuning scripts
                                                       (train_taco / realwaste / object_centric / garbage)
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    kw = dict(validation_split=0.2, seed=42, image_size=(224, 224), batch_size=32)
    d = "data/trashNet-dataset"
    out = {
        "train_shuffled": list(image_dataset_from_directory(d, subset="training", shuffle=True, **kw).file_paths),
        "val_shuffled": list(image_dataset_from_directory(d, subset="validation", shuffle=True, **kw).file_paths),
        "val_unshuffled_finetune_scripts": list(
            image_dataset_from_directory(d, subset="validation", shuffle=False, **kw).file_paths),
    }
    with open(os.path.join(CACHE, "trashnet_split.json"), "w") as f:
        json.dump(out, f)
    tr, va, fv = (set(out[k]) for k in ("train_shuffled", "val_shuffled", "val_unshuffled_finetune_scripts"))
    log("train", len(tr), "val_shuffled", len(va), "overlap(train,val_shuffled)", len(tr & va),
        "| val_unshuffled", len(fv), "overlap(train,val_unshuffled)", len(tr & fv))


def trashnet_membership(idx):
    """Label per TrashNet file: 'train' / 'val' (canonical shuffled split) + '+ftval' if the file is
    in the validation set that the fine-tuning scripts actually used."""
    with open(os.path.join(CACHE, "trashnet_split.json")) as f:
        sp = json.load(f)
    tr, va, fv = set(sp["train_shuffled"]), set(sp["val_shuffled"]), set(sp["val_unshuffled_finetune_scripts"])
    m = []
    for p in idx["trashnet"]:
        lab = "train" if p in tr else ("val" if p in va else "unknown")
        if p in fv:
            lab += "+ftval"
        m.append(lab)
    return np.array(m)


# ----------------------------------------------------------------------------
# L1 / L2 exact duplicates
# ----------------------------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def stage_exact():
    idx = load_index()
    H = {n: load_hash(n) for n in idx}
    summary = {"cross_test": {}, "intra_set": {}, "raw_vs_prepared": {}, "trashnet_supplementary": {}}
    rows = []

    # index sha / pix -> list of (set, i)
    for kind in ("sha", "pix"):
        lut = defaultdict(list)
        for n in idx:
            for i, s in enumerate(H[n][kind]):
                if s:
                    lut[s].append((n, i))
        # test vs every other set
        for other in COMPARE_SETS:
            hit = set()
            for i, s in enumerate(H["test"][kind]):
                for (n, j) in lut.get(s, []):
                    if n == other:
                        hit.add(i)
                        rows.append([kind, "test", idx["test"][i], other, idx[other][j]])
            summary["cross_test"].setdefault(other, {})[kind] = len(hit)
        # test vs test
        groups = [v for s, v in lut.items() if s and sum(1 for (n, _) in v if n == "test") > 1]
        test_groups = [[idx["test"][i] for (n, i) in g if n == "test"] for g in groups]
        summary["cross_test"].setdefault("test_internal", {})[kind] = {
            "groups": len(test_groups), "files_in_groups": sum(len(g) for g in test_groups),
            "examples": test_groups[:10]}
        for g in test_groups:
            rows.append([kind, "test", g[0], "test(internal)", "; ".join(g[1:])])
        # intra-set (extra duplicate files inside each dataset)
        for n in idx:
            vals = [s for s in H[n][kind] if s]
            summary["intra_set"].setdefault(n, {})[kind] = len(vals) - len(set(vals))

    # raw vs prepared equivalence (bytes)
    for a, b in (("realwaste_raw", "realwaste_prepared"), ("household_raw", "household_prepared")):
        sa, sb = sorted(H[a]["sha"]), sorted(H[b]["sha"])
        summary["raw_vs_prepared"][f"{a}__{b}"] = {
            "n_raw": len(sa), "n_prepared": len(sb), "byte_identical_multiset": sa == sb}

    # supplementary: TrashNet vs the fine-tuning sets (bytes)
    mem = trashnet_membership(idx)
    for kind in ("sha", "pix"):
        lut = defaultdict(list)
        for n in ("taco_crops", "realwaste_prepared", "household_prepared", "garbage_prepared",
                  "taco_full", "realwaste_raw", "household_raw", "garbage_original"):
            for i, s in enumerate(H[n][kind]):
                if s:
                    lut[s].append((n, i))
        for i, s in enumerate(H["trashnet"][kind]):
            for (n, j) in lut.get(s, []):
                rows.append([kind, f"trashnet[{mem[i]}]", idx["trashnet"][i], n, idx[n][j]])
                summary["trashnet_supplementary"].setdefault(f"{kind}:{mem[i]}->{n}", 0)
                summary["trashnet_supplementary"][f"{kind}:{mem[i]}->{n}"] += 1

    write_csv(os.path.join(OUT, "exact_matches.csv"),
              ["hash_type", "query_set", "query_path", "matched_set", "matched_path"], rows)
    with open(os.path.join(OUT, "exact_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(json.dumps(summary["cross_test"], indent=1, ensure_ascii=False)[:3000])
    log("raw_vs_prepared:", summary["raw_vs_prepared"])
    log("intra_set:", summary["intra_set"])
    log("trashnet_supplementary:", summary["trashnet_supplementary"])


# ----------------------------------------------------------------------------
# L3 perceptual hashing
# ----------------------------------------------------------------------------

def ham(a, B):
    """a: (k,) uint64, B: (n,) uint64 -> (k,n) Hamming distances."""
    x = np.bitwise_xor(a[:, None], B[None, :])
    x = np.ascontiguousarray(x)
    return POP8[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1, dtype=np.int16)


def znorm(th):
    t = th.astype(np.float32)
    t -= t.mean(-1, keepdims=True)
    n = np.linalg.norm(t, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return t / n


def verdict(pm, dm, z):
    if min(pm, dm) <= CONFIRM_HASH_T and z >= CONFIRM_ZNCC:
        return "confirmed"
    if z >= REVIEW_ZNCC:
        return "review"
    return "rejected"


def hash_search(q, t, same_set=False):
    """q: dict ph (nq,k) dh th ; t: dict ph (nt,) ... Returns (candidates, nearest)."""
    qph, qdh, qth = q["ph"], q["dh"], znorm(q["th"])
    tph, tdh, tth = t["ph"][:, 0], t["dh"][:, 0], znorm(t["th"][:, 0])
    cands, nearest = [], []
    for i in range(len(qph)):
        pd, dd = ham(qph[i], tph), ham(qdh[i], tdh)
        pm, dm = pd.min(0), dd.min(0)
        if same_set:
            pm = pm.copy(); dm = dm.copy(); pm[: i + 1] = 99; dm[: i + 1] = 99   # only j>i
        j0 = int(pm.argmin())
        nearest.append((j0, int(pm[j0]), int(dm.min())))
        for j in np.nonzero((pm <= HASH_T) | (dm <= HASH_T))[0]:
            z = float((qth[i] @ tth[j]).max())
            cands.append((i, int(j), int(pm[j]), int(dm[j]), int(pd[:, j].argmin()), z,
                          verdict(int(pm[j]), int(dm[j]), z)))
    return cands, nearest


def stage_perceptual():
    idx = load_index()
    H = {n: load_hash(n) for n in idx}
    cand_rows, near_rows, summary = [], [], {}
    for other in COMPARE_SETS + ["test"]:
        same = other == "test"
        c, nr = hash_search(H["test"], H[other], same_set=same)
        label = "test(internal)" if same else other
        n_img = len({x[0] for x in c})
        nconf = len({x[0] for x in c if x[6] == "confirmed"})
        nrev = len({x[0] for x in c if x[6] == "review"})
        dists = np.array([x[1] for x in nr])
        summary[label] = {
            "candidates_pairs": len(c), "test_imgs_with_candidate": n_img,
            "test_imgs_confirmed": nconf, "test_imgs_review": nrev,
            "min_phash_distance_over_test_imgs": int(dists.min()),
            "phash_dist_histogram": {"0-4": int((dists <= 4).sum()), "5-10": int(((dists > 4) & (dists <= 10)).sum()),
                                     "11-12": int(((dists > 10) & (dists <= 12)).sum()),
                                     "13-16": int(((dists > 12) & (dists <= 16)).sum()),
                                     ">16": int((dists > 16).sum())},
            "median_nearest_phash_distance": float(np.median(dists)),
        }
        for (i, j, pm, dm, t, z, v) in c:
            cand_rows.append([label, idx["test"][i], idx[other][j], pm, dm, t, f"{z:.4f}", v])
        for i, (j, pm, dm) in enumerate(nr):
            near_rows.append([label, idx["test"][i], idx[other][j] if j < len(idx[other]) else "", pm, dm])
        log(f"test vs {label:20s} cand_pairs={len(c):4d} imgs={n_img:3d} confirmed={nconf:3d} review={nrev:3d} "
            f"min_pHash={int(dists.min())}")

    # raw -> prepared equivalence for Garbage v2 (re-encoded copies)
    ge, gp = H["garbage_original"], H["garbage_prepared"]
    best = np.zeros(len(gp["ph"]), dtype=np.int16)
    for s in range(0, len(gp["ph"]), 256):
        # ham(a(k,), B(n,)) -> (k, n): here k = chunk of prepared images, n = originals
        best[s:s + 256] = ham(gp["ph"][s:s + 256, 0], ge["ph"][:, 0]).min(1)
    summary["garbage_prepared_vs_original"] = {
        "n_prepared": int(len(best)), "n_with_orig_pHash<=4": int((best <= 4).sum()),
        "max_min_dist": int(best.max())}
    log("garbage prepared->original", summary["garbage_prepared_vs_original"])

    write_csv(os.path.join(OUT, "perceptual_candidates.csv"),
              ["compared_with", "test_image", "matched_image", "phash_dist", "dhash_dist",
               "dihedral_transform_idx", "zncc", "verdict"], cand_rows)
    write_csv(os.path.join(OUT, "perceptual_nearest.csv"),
              ["compared_with", "test_image", "nearest_image_by_phash", "phash_dist", "min_dhash_dist"], near_rows)
    with open(os.path.join(OUT, "perceptual_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # supplementary: TrashNet (all 2527) vs fine-tuning sets and vs itself
    mem = trashnet_membership(idx)
    tn = {"ph": H["trashnet"]["ph"], "dh": H["trashnet"]["dh"], "th": H["trashnet"]["th"]}
    sup_rows, sup_summary = [], {}
    for other in ["taco_crops", "realwaste_prepared", "household_prepared", "garbage_prepared", "trashnet"]:
        same = other == "trashnet"
        c, nr = hash_search(tn, H[other], same_set=same)
        conf = [x for x in c if x[6] == "confirmed"]
        sup_summary[other + ("(internal i<j)" if same else "")] = {
            "candidate_pairs": len(c), "confirmed_pairs": len(conf),
            "confirmed_trashnet_imgs": len({x[0] for x in conf}),
            "confirmed_val_imgs": len({x[0] for x in conf if mem[x[0]] == "val"}),
            "confirmed_train_imgs": len({x[0] for x in conf if mem[x[0]] == "train"})}
        for (i, j, pm, dm, t, z, v) in c:
            if v == "rejected":
                continue
            sup_rows.append([other, idx["trashnet"][i], mem[i], idx[other][j],
                             mem[j] if same else "", pm, dm, f"{z:.4f}", v])
        log("TrashNet supplementary vs", other, sup_summary[other + ("(internal i<j)" if same else "")])
    write_csv(os.path.join(OUT, "supplementary_trashnet_candidates.csv"),
              ["compared_with", "trashnet_image", "trashnet_split", "matched_image", "matched_split",
               "phash_dist", "dhash_dist", "zncc", "verdict"], sup_rows)
    with open(os.path.join(OUT, "supplementary_trashnet_summary.json"), "w") as f:
        json.dump(sup_summary, f, indent=2)


# ----------------------------------------------------------------------------
# L4 deep-feature embeddings
# ----------------------------------------------------------------------------

def load224(path):
    try:
        with Image.open(path) as im:
            if im.format == "JPEG":
                im.draft("RGB", (448, 448))
            im = im.convert("RGB").resize((224, 224), Image.BILINEAR, reducing_gap=2.0)
            return np.asarray(im, dtype=np.float32), True
    except Exception:
        return np.zeros((224, 224, 3), np.float32), False


def get_backbone():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.applications import EfficientNetB0
    return EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))


def embed_arrays(model, arrays):
    f = model.predict_on_batch(np.stack(arrays)).astype(np.float32)
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    return f


def stage_embed():
    idx = load_index()
    model = get_backbone()
    ex = ThreadPoolExecutor(8)
    for name in ["test"] + EMBED_SETS:
        outp = os.path.join(CACHE, f"emb_{name}.npy")
        if os.path.exists(outp):
            log(name, "cached"); continue
        paths = idx[name]
        E = np.zeros((len(paths), 1280), np.float16)
        bad, t0 = 0, time.time()
        for s in range(0, len(paths), 128):
            res = list(ex.map(load224, paths[s:s + 128]))
            bad += sum(1 for r in res if not r[1])
            E[s:s + 128] = embed_arrays(model, [r[0] for r in res]).astype(np.float16)
            if (s // 128) % 20 == 0:
                log(f"  {name} {s}/{len(paths)} {time.time()-t0:.0f}s")
        np.save(outp + ".tmp.npy", E)
        os.replace(outp + ".tmp.npy", outp)      # atomic: never leave a truncated cache file
        log(f"{name:20s} embedded {len(paths)} (unreadable={bad}) {time.time()-t0:.0f}s")


# ----------------------------------------------------------------------------
# L5 SIFT + RANSAC
# ----------------------------------------------------------------------------

def sift_from_gray(gray):
    import cv2
    h, w = gray.shape
    s = SIFT_MAX_DIM / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(nfeatures=SIFT_NFEAT).detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    return np.float32([k.pt for k in kp]), np.clip(des, 0, 255).astype(np.uint8)


def gray_for_sift(path_or_img):
    if isinstance(path_or_img, str):
        with Image.open(path_or_img) as im:
            if im.format == "JPEG":
                im.draft("L", (SIFT_MAX_DIM * 2, SIFT_MAX_DIM * 2))
            g = np.asarray(im.convert("L"))
    else:
        g = np.asarray(path_or_img.convert("L"))
    return g


def inliers(p1, d1, p2, d2):
    """Number of geometrically consistent, UNIQUE keypoint correspondences between two images.

    SIFT descriptors -> Lowe ratio test (0.75) -> keep only the best match per target keypoint
    (one-to-one; without this several query keypoints can map to the same target keypoint and
    inflate the count) -> RANSAC similarity transform (translation, rotation, uniform scale;
    reprojection threshold 3 px) -> reject implausible scales (<0.15 or >6) -> count unique
    inlier locations. Calibrated with a negative control (random unrelated pairs) and a
    positive control (synthetic crops / resizes / compression of test images).
    """
    import cv2
    if len(d1) < 10 or len(d2) < 10:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    best = {}
    for x in m:
        if len(x) == 2 and x[0].distance < 0.75 * x[1].distance:
            g = x[0]
            if g.trainIdx not in best or g.distance < best[g.trainIdx].distance:
                best[g.trainIdx] = g
    good = list(best.values())
    if len(good) < 6:
        return 0
    src = p1[[g.queryIdx for g in good]]
    dst = p2[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                         maxIters=2000, confidence=0.995)
    if M is None or inl is None:
        return 0
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (0.15 < scale < 6.0):
        return 0
    inl = inl.ravel().astype(bool)
    return int(len(np.unique(np.round(src[inl]).astype(int), axis=0)))


# globals filled before fork
G = {}


def _extract_worker(path):
    import cv2
    cv2.setNumThreads(1)
    try:
        g = gray_for_sift(path)
        p, d = sift_from_gray(g)
        pf, df = sift_from_gray(np.ascontiguousarray(g[:, ::-1]))
        return p, d, pf, df
    except Exception:
        z = (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
        return z + z


def _extract_plain_worker(path):
    import cv2
    cv2.setNumThreads(1)
    try:
        return sift_from_gray(gray_for_sift(path))
    except Exception:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)


def _match_test_vs_taco(i):
    import cv2
    cv2.setNumThreads(1)
    best = []
    pa, da, pf, df = G["test"][i]
    for j, (pb, db) in enumerate(G["taco"]):
        a = inliers(pa, da, pb, db)
        b = inliers(pf, df, pb, db)
        best.append(max(a, b))
    return i, np.array(best, dtype=np.int16)


def _match_test_vs_test(i):
    import cv2
    cv2.setNumThreads(1)
    pa, da, _, _ = G["test"][i]
    out = []
    for j in range(i + 1, len(G["test"])):
        pb, db, pbf, dbf = G["test"][j]
        out.append((j, max(inliers(pa, da, pb, db), inliers(pa, da, pbf, dbf))))
    return i, out


def _verify_pair(args):
    import cv2
    cv2.setNumThreads(1)
    ti, path = args
    pb, db = _extract_plain_worker(path)
    pa, da, pf, df = G["test"][ti]
    return max(inliers(pa, da, pb, db), inliers(pf, df, pb, db))


def stage_sift():
    import cv2
    cv2.setNumThreads(1)
    idx = load_index()
    ctx = mp.get_context("fork")
    # --- descriptors
    log("extracting SIFT for test (orig + mirrored) and TACO full photos")
    with ctx.Pool(11) as pool:
        G["test"] = pool.map(_extract_worker, idx["test"], chunksize=8)
        G["taco"] = pool.map(_extract_plain_worker, idx["taco_full"], chunksize=8)
    log("mean #kp test:", np.mean([len(x[0]) for x in G["test"]]),
        " taco:", np.mean([len(x[0]) for x in G["taco"]]))
    # --- exhaustive: test vs TACO full photos
    t0 = time.time()
    with ctx.Pool(11) as pool:
        res = dict(pool.imap_unordered(_match_test_vs_taco, range(len(idx["test"])), chunksize=4))
    M = np.stack([res[i] for i in range(len(idx["test"]))])
    np.save(os.path.join(CACHE, "sift_test_vs_tacofull.npy"), M)
    log(f"test vs TACO full: done {time.time()-t0:.0f}s  max inliers over all pairs = {int(M.max())}, "
        f"pairs>={SIFT_STRONG}: {int((M >= SIFT_STRONG).sum())}, >={SIFT_WEAK}: {int((M >= SIFT_WEAK).sum())}")
    # --- exhaustive: test vs test
    t0 = time.time()
    with ctx.Pool(11) as pool:
        tt = dict(pool.imap_unordered(_match_test_vs_test, range(len(idx["test"])), chunksize=4))
    rows = [(i, j, v) for i in tt for (j, v) in tt[i] if v >= SIFT_WEAK]
    np.save(os.path.join(CACHE, "sift_test_vs_test_hits.npy"), np.array(rows, dtype=np.int32).reshape(-1, 3))
    allv = np.array([v for i in tt for (_, v) in tt[i]])
    log(f"test vs test: done {time.time()-t0:.0f}s  max inliers={int(allv.max())}  pairs>={SIFT_STRONG}: "
        f"{int((allv >= SIFT_STRONG).sum())}  >={SIFT_WEAK}: {int((allv >= SIFT_WEAK).sum())}")

    # --- L4 top-k embedding candidates -> SIFT verification
    Et = np.load(os.path.join(CACHE, "emb_test.npy")).astype(np.float32)
    pairs, meta = [], []
    for name in EMBED_SETS:
        E = np.load(os.path.join(CACHE, f"emb_{name}.npy")).astype(np.float32)
        S = Et @ E.T
        top = np.argpartition(-S, TOPK, axis=1)[:, :TOPK]
        for i in range(len(Et)):
            for j in top[i]:
                pairs.append((i, idx[name][int(j)]))
                meta.append((name, i, int(j), float(S[i, j])))
    # embeddings test vs test
    S = Et @ Et.T
    np.fill_diagonal(S, -1)
    top = np.argpartition(-S, TOPK, axis=1)[:, :TOPK]
    emb_tt_max = float(S.max())
    for i in range(len(Et)):
        for j in top[i]:
            pairs.append((i, idx["test"][int(j)]))
            meta.append(("test(internal)", i, int(j), float(S[i, j])))
    log("verifying", len(pairs), "embedding-candidate pairs with SIFT+RANSAC")
    t0 = time.time()
    with ctx.Pool(11) as pool:
        inl = pool.map(_verify_pair, pairs, chunksize=16)
    log(f"done {time.time()-t0:.0f}s")
    rows = [[m[0], idx["test"][m[1]], idx[m[0]][m[2]] if m[0] != "test(internal)" else idx["test"][m[2]],
             f"{m[3]:.4f}", v] for m, v in zip(meta, inl)]
    write_csv(os.path.join(OUT, "embedding_sift_candidates.csv"),
              ["compared_with", "test_image", "candidate_image", "cosine_similarity", "sift_ransac_inliers"], rows)

    # --- negative control: random unrelated (test, train) pairs
    rng = random.Random(123)
    neg_pairs = []
    for _ in range(3000):
        n = rng.choice(EMBED_SETS)
        neg_pairs.append((rng.randrange(len(idx["test"])), rng.choice(idx[n])))
    with ctx.Pool(11) as pool:
        neg = np.array(pool.map(_verify_pair, neg_pairs, chunksize=16))
    nc = {"n_pairs": len(neg), "mean": float(neg.mean()), "p95": float(np.percentile(neg, 95)),
          "p99": float(np.percentile(neg, 99)), "max": int(neg.max()),
          "n_ge_WEAK": int((neg >= SIFT_WEAK).sum()), "n_ge_STRONG": int((neg >= SIFT_STRONG).sum())}
    log("negative control:", nc)

    # --- summaries
    s = {"negative_control": nc, "test_vs_taco_full": {
            "pairs": int(M.size), "max_inliers": int(M.max()),
            "test_imgs_with_any_ge_STRONG": int((M.max(1) >= SIFT_STRONG).sum()),
            "test_imgs_with_any_ge_WEAK": int((M.max(1) >= SIFT_WEAK).sum())},
         "test_vs_test": {"pairs": int(allv.size), "max_inliers": int(allv.max()),
                          "pairs_ge_STRONG": int((allv >= SIFT_STRONG).sum()),
                          "pairs_ge_WEAK": int((allv >= SIFT_WEAK).sum()),
                          "max_embedding_cosine_between_test_images": emb_tt_max},
         "embedding_candidates": {}}
    arr = defaultdict(list)
    for (m, v) in zip(meta, inl):
        arr[m[0]].append((m[1], v, m[3]))
    for name, lst in arr.items():
        per_img = defaultdict(int)
        for (i, v, c) in lst:
            per_img[i] = max(per_img[i], v)
        vals = np.array(list(per_img.values()))
        s["embedding_candidates"][name] = {
            "pairs_verified": len(lst), "max_inliers": int(vals.max()),
            "test_imgs_ge_STRONG": int((vals >= SIFT_STRONG).sum()),
            "test_imgs_ge_WEAK": int((vals >= SIFT_WEAK).sum()),
            "max_cosine": float(max(c for (_, _, c) in lst))}
    with open(os.path.join(OUT, "sift_summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    log(json.dumps(s, indent=1))

    # save TACO-full hits and test-test hits as readable CSV
    rows = []
    ii, jj = np.nonzero(M >= SIFT_WEAK)
    for i, j in zip(ii, jj):
        rows.append(["taco_full", idx["test"][i], idx["taco_full"][j], int(M[i, j])])
    for (i, j, v) in np.load(os.path.join(CACHE, "sift_test_vs_test_hits.npy")):
        rows.append(["test(internal)", idx["test"][i], idx["test"][j], int(v)])
    write_csv(os.path.join(OUT, "sift_exhaustive_hits.csv"),
              ["compared_with", "test_image", "matched_image", "sift_ransac_inliers"], rows)


# ----------------------------------------------------------------------------
# Verification of perceptual-hash candidates (non byte-identical) with SIFT + RANSAC
# ----------------------------------------------------------------------------

def _verify_pair_paths(args):
    import cv2
    cv2.setNumThreads(1)
    ti, path = args
    pb, db = _extract_plain_worker(path)
    pa, da, pf, df = G["test"][ti]
    return max(inliers(pa, da, pb, db), inliers(pf, df, pb, db))


def tier_of(byte_identical, pm, dm, z, sift):
    if byte_identical:
        return "A_exact_bytes"
    if min(pm, dm) <= 4 and z >= 0.98:
        return "B_near_identical_pixels"
    if sift >= SIFT_STRONG and z >= REVIEW_ZNCC:
        return "C_sift_verified"
    if verdict(pm, dm, z) == "confirmed":
        return "D_uncertain_needs_visual_review"
    return "E_rejected"


def stage_verify():
    import cv2
    cv2.setNumThreads(1)
    idx = load_index()
    H = {n: load_hash(n) for n in idx}
    sha_of = {}
    for n in idx:
        for p, s_ in zip(idx[n], H[n]["sha"]):
            sha_of[p] = s_
    ctx = mp.get_context("fork")
    with ctx.Pool(11) as pool:
        G["test"] = pool.map(_extract_worker, idx["test"], chunksize=8)
    test_pos = {p: i for i, p in enumerate(idx["test"])}
    rows_in = []
    with open(os.path.join(OUT, "perceptual_candidates.csv")) as f:
        for r in csv.DictReader(f):
            if r["compared_with"] in ("trashnet", "taco_crops", "taco_full", "realwaste_prepared",
                                      "household_prepared", "garbage_prepared", "test(internal)") \
                    and r["verdict"] != "rejected":
                rows_in.append(r)
    log("pairs to verify with SIFT:", len(rows_in))
    pairs = [(test_pos[r["test_image"]], r["matched_image"]) for r in rows_in]
    with ctx.Pool(11) as pool:
        inl = pool.map(_verify_pair_paths, pairs, chunksize=16)
    out = []
    for r, v in zip(rows_in, inl):
        bi = sha_of[r["test_image"]] == sha_of[r["matched_image"]]
        pm, dm, z = int(r["phash_dist"]), int(r["dhash_dist"]), float(r["zncc"])
        out.append([r["compared_with"], r["test_image"], r["matched_image"], pm, dm, f"{z:.4f}", v,
                    int(bi), tier_of(bi, pm, dm, z, v)])
    write_csv(os.path.join(OUT, "perceptual_verified.csv"),
              ["compared_with", "test_image", "matched_image", "phash_dist", "dhash_dist", "zncc",
               "sift_inliers", "byte_identical", "tier"], out)
    c = defaultdict(lambda: defaultdict(set))
    for o in out:
        c[o[0]][o[8]].add(o[1])
    for k, v in c.items():
        log(k, {t: len(s_) for t, s_ in sorted(v.items())})


# ----------------------------------------------------------------------------
# Positive control: known, synthetically modified copies of test images
# ----------------------------------------------------------------------------

def make_variants(img, rng):
    w, h = img.size
    out = {}
    buf = io.BytesIO()
    img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR).save(buf, "JPEG", quality=30)
    out["resize50_jpeg30"] = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    cw, ch = int(w * 0.8), int(h * 0.8)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    out["center_crop80"] = img.crop((x0, y0, x0 + cw, y0 + ch))
    cw, ch = int(w * 0.6), int(h * 0.6)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    out["center_crop60_resized"] = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    out["hflip"] = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    out["rotate90"] = img.transpose(Image.Transpose.ROTATE_90)
    e = ImageEnhance.Brightness(img).enhance(1.3)
    e = ImageEnhance.Contrast(e).enhance(0.8)
    out["brightness_contrast"] = ImageEnhance.Color(e).enhance(1.2)
    buf = io.BytesIO()
    img.resize((224, 224), Image.BILINEAR).save(buf, "JPEG", quality=85)
    out["squash224_jpeg85"] = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out


def stage_control():
    import cv2
    cv2.setNumThreads(4)
    idx = load_index()
    Ht = load_hash("test")
    tdict = {"ph": Ht["ph"][:, :1], "dh": Ht["dh"][:, :1], "th": Ht["th"][:, :1]}
    rng = random.Random(0)
    chosen = sorted(rng.sample(range(len(idx["test"])), 60))
    variants = defaultdict(list)    # variant -> list of (orig_idx, image)
    for i in chosen:
        img = Image.open(idx["test"][i]).convert("RGB")
        for k, v in make_variants(img, rng).items():
            variants[k].append((i, v))

    # retrieval index for the control = test originals + ALL embedded training/source images
    all_paths = list(idx["test"])
    Eall = [np.load(os.path.join(CACHE, "emb_test.npy")).astype(np.float32)]
    for name in EMBED_SETS:
        all_paths += idx[name]
        Eall.append(np.load(os.path.join(CACHE, f"emb_{name}.npy")).astype(np.float32))
    Et = np.concatenate(Eall)
    model = get_backbone()
    result = {}
    for vname, lst in variants.items():
        q_ph, q_dh, q_th = [], [], []
        arrs = []
        for (i, im) in lst:
            ph, dh, th = image_features(im, True)
            q_ph.append(ph); q_dh.append(dh); q_th.append(th)
            arrs.append(np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.float32))
        q = {"ph": np.stack(q_ph), "dh": np.stack(q_dh), "th": np.stack(q_th)}
        cands, _ = hash_search(q, tdict)
        hit_hash = set()
        for (qi, j, pm, dm, t, z, v) in cands:
            if j == lst[qi][0] and v == "confirmed":
                hit_hash.add(qi)
        Eq = embed_arrays(model, arrs)
        S = Eq @ Et.T
        top = np.argsort(-S, axis=1)[:, :TOPK]
        hit_emb, hit_sift = set(), set()
        for qi, (i, im) in enumerate(lst):
            if i in top[qi]:
                hit_emb.add(qi)
            pq, dq = sift_from_gray(np.asarray(im.convert("L")))
            pqf, dqf = sift_from_gray(np.ascontiguousarray(np.asarray(im.convert("L"))[:, ::-1]))
            best = 0
            for j in top[qi]:
                po, do = sift_from_gray(gray_for_sift(all_paths[int(j)]))
                best_j = max(inliers(pq, dq, po, do), inliers(pqf, dqf, po, do))
                if int(j) == i:
                    best = best_j
            if best >= SIFT_STRONG:
                hit_sift.add(qi)
        union = hit_hash | hit_sift
        n = len(lst)
        result[vname] = {"n": n, "found_by_hash": len(hit_hash), "original_in_embedding_top5": len(hit_emb),
                         "found_by_embedding+sift": len(hit_sift), "found_by_any_layer": len(union)}
        log(vname, result[vname])
    tot = sum(r["n"] for r in result.values())
    result["_overall"] = {"n": tot, "found_by_any_layer": sum(r["found_by_any_layer"] for r in result.values())}
    with open(os.path.join(OUT, "positive_control.json"), "w") as f:
        json.dump(result, f, indent=2)
    log("overall", result["_overall"])


# ----------------------------------------------------------------------------
# Report: consolidate every layer into one per-image table + summary + contact sheets
# ----------------------------------------------------------------------------

DATASET_OF = {"trashnet": "TrashNet", "taco_crops": "TACO", "taco_full": "TACO",
              "realwaste_raw": "RealWaste", "realwaste_prepared": "RealWaste",
              "household_raw": "Household", "household_prepared": "Household",
              "garbage_original": "GarbageV2", "garbage_prepared": "GarbageV2"}
DATASETS = ["TrashNet", "TACO", "RealWaste", "Household", "GarbageV2"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def contact_sheet(pairs, outpath, title, per_sheet=8, h=190):
    """pairs: list of (left_path, right_path, caption). Writes one or more PNG sheets."""
    from PIL import ImageDraw
    n_sheets = 0
    for s0 in range(0, len(pairs), per_sheet):
        chunk = pairs[s0:s0 + per_sheet]
        W = 2 * 300 + 20
        sheet = Image.new("RGB", (W * 2, (h + 34) * ((len(chunk) + 1) // 2)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (a, b, cap) in enumerate(chunk):
            x0 = (k % 2) * W
            y0 = (k // 2) * (h + 34)
            for m, pth in enumerate((a, b)):
                try:
                    im = Image.open(pth).convert("RGB")
                    im.thumbnail((290, h))
                    sheet.paste(im, (x0 + m * 300 + 5, y0 + 2))
                except Exception:
                    pass
            d.text((x0 + 5, y0 + h + 4), cap[:95], fill="black")
            d.text((x0 + 5, y0 + h + 17), os.path.basename(a)[:28] + "  |  " + os.path.basename(b)[:34], fill="gray")
        fn = f"{outpath}_{n_sheets + 1:02d}.png"
        sheet.save(fn)
        n_sheets += 1
    return n_sheets


def read_manual_review():
    """Optional manual visual verdicts (results/duplicate_check/manual_review.csv) for pairs that
    only have weak evidence (SIFT-only or uncertain). Columns: kind, test_image, dataset_or_image2, verdict"""
    path = os.path.join(OUT, "manual_review.csv")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                out[(r["kind"], r["test_image"], r["dataset_or_image2"])] = r["verdict"]
    return out


def stage_report():
    idx = load_index()
    tests = idx["test"]
    tpos = {p: i for i, p in enumerate(tests)}
    n = len(tests)
    STRICT = {"exact_bytes", "near_identical_pixels"}
    ev = [defaultdict(set) for _ in tests]       # test image -> dataset -> evidence labels
    detail = [defaultdict(list) for _ in tests]  # test image -> dataset -> matched paths (best first)
    uncertain = defaultdict(list)
    ipairs = defaultdict(lambda: {"ev": set(), "sift": 0})

    def ikey(a_, b_):
        return tuple(sorted((a_, b_)))

    # --- exact bytes
    with open(os.path.join(OUT, "exact_matches.csv")) as f:
        for r in csv.DictReader(f):
            if r["query_set"] != "test" or r["hash_type"] != "sha":
                continue
            if r["matched_set"] == "test(internal)":
                for other in r["matched_path"].split("; "):
                    ipairs[ikey(r["query_path"], other)]["ev"].add("exact_bytes")
                continue
            ds = DATASET_OF.get(r["matched_set"])
            if ds:
                ev[tpos[r["query_path"]]][ds].add("exact_bytes")
                detail[tpos[r["query_path"]]][ds].append(r["matched_path"])

    # --- perceptual candidates verified with SIFT
    with open(os.path.join(OUT, "perceptual_verified.csv")) as f:
        for r in csv.DictReader(f):
            i, tier = tpos[r["test_image"]], r["tier"]
            if r["compared_with"] == "test(internal)":
                if tier[0] == "B":
                    ipairs[ikey(r["test_image"], r["matched_image"])]["ev"].add("near_identical_pixels")
                elif tier[0] in "CD":
                    ipairs[ikey(r["test_image"], r["matched_image"])]["ev"].add("weak_perceptual_sift")
                continue
            ds = DATASET_OF[r["compared_with"]]
            if tier[0] == "B":
                ev[i][ds].add("near_identical_pixels"); detail[i][ds].append(r["matched_image"])
            elif tier[0] == "C":
                ev[i][ds].add("sift_verified"); detail[i][ds].append(r["matched_image"])
            elif tier[0] == "D":
                uncertain[(i, ds)].append(r["matched_image"])

    # --- SIFT-only evidence (crops / partial views / same scene)
    with open(os.path.join(OUT, "sift_exhaustive_hits.csv")) as f:
        for r in csv.DictReader(f):
            v = int(r["sift_ransac_inliers"])
            if v < SIFT_STRONG:
                continue
            if r["compared_with"] == "taco_full":
                i = tpos[r["test_image"]]
                ev[i]["TACO"].add("sift_geometric"); detail[i]["TACO"].append(r["matched_image"])
            else:
                k = ikey(r["test_image"], r["matched_image"])
                ipairs[k]["ev"].add("sift_geometric"); ipairs[k]["sift"] = max(ipairs[k]["sift"], v)
    with open(os.path.join(OUT, "embedding_sift_candidates.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["sift_ransac_inliers"]) < SIFT_STRONG or r["compared_with"] == "test(internal)":
                continue
            i = tpos[r["test_image"]]
            ds = DATASET_OF[r["compared_with"]]
            ev[i][ds].add("sift_geometric"); detail[i][ds].append(r["candidate_image"])

    manual = read_manual_review()

    # --- final per-image decision
    def final_overlap(i, d):
        e = ev[i][d]
        if e & STRICT:
            return "confirmed"
        if e:
            v = manual.get(("dataset", tests[i], d))
            if v == "duplicate":
                return "confirmed_manual"
            return "weak_unconfirmed" if v != "different_image" else "rejected_manual"
        return ""

    pred = {}
    with open("results/final_model_eval/all_predictions_confidence.csv") as f:
        for r in csv.DictReader(f):
            pred[r["path"]] = int(r["correct"])

    rows, overlap_set, weak_set = [], set(), set()
    for i, p in enumerate(tests):
        st = {d: final_overlap(i, d) for d in DATASETS}
        ok = [d for d in DATASETS if st[d] in ("confirmed", "confirmed_manual")]
        wk = [d for d in DATASETS if st[d] in ("weak_unconfirmed",)]
        if ok:
            overlap_set.add(i)
        elif wk:
            weak_set.add(i)
        rows.append([p, pred.get(p, ""), *[(st[d] + ":" + "+".join(sorted(ev[i][d]))) if st[d] else "" for d in DATASETS],
                     ";".join(ok), ";".join(wk)])
    write_csv(os.path.join(OUT, "test_overlap_per_image.csv"),
              ["test_image", "model_correct", *DATASETS, "confirmed_overlap_datasets", "weak_unconfirmed_datasets"], rows)

    # --- internal duplicates inside the test set
    parent = {p: p for p in tests}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    irows = []
    for (a_, b_), info in sorted(ipairs.items()):
        strict = bool(info["ev"] & STRICT)
        verdict_m = manual.get(("internal", a_, b_), "")
        if strict or verdict_m == "duplicate":
            parent[find(a_)] = find(b_)
        irows.append([a_, b_, "+".join(sorted(info["ev"])), info["sift"],
                      "strict" if strict else ("manual_duplicate" if verdict_m == "duplicate" else
                                                "weak" if verdict_m == "" else verdict_m)])
    write_csv(os.path.join(OUT, "test_internal_duplicate_pairs.csv"),
              ["image_a", "image_b", "evidence", "sift_inliers", "decision"], irows)
    comps = defaultdict(list)
    for p in tests:
        comps[find(p)].append(p)
    dup_groups = [g for g in comps.values() if len(g) > 1]
    internal = {
        "pairs_total_with_any_evidence": len(irows),
        "pairs_strict_exact_or_near_identical": sum(1 for r in irows if r[4] == "strict"),
        "pairs_weak_only_unreviewed": sum(1 for r in irows if r[4] == "weak"),
        "duplicate_groups": len(dup_groups),
        "images_in_groups": sum(len(g) for g in dup_groups),
        "redundant_images_if_deduplicated": sum(len(g) - 1 for g in dup_groups),
        "unique_images_after_dedup": len(comps),
        "group_size_distribution": dict(sorted(defaultdict(int, {k: sum(1 for g in dup_groups if len(g) == k)
                                                                    for k in {len(g) for g in dup_groups}}).items())),
        "groups_crossing_class_folders": sum(1 for g in dup_groups
                                             if len({x.split("/")[2] for x in g}) > 1)}

    # --- summary
    S = {"n_test": n, "datasets_compared": DATASETS, "per_dataset": {}, "internal": internal}
    for d in DATASETS:
        st = [final_overlap(i, d) for i in range(n)]
        S["per_dataset"][d] = {
            "confirmed_overlap_test_imgs": sum(1 for x in st if x in ("confirmed", "confirmed_manual")),
            "of_which_exact_bytes": sum(1 for i in range(n) if "exact_bytes" in ev[i][d]),
            "of_which_near_identical_pixels_only": sum(1 for i in range(n) if "near_identical_pixels" in ev[i][d]
                                                      and "exact_bytes" not in ev[i][d]),
            "confirmed_by_manual_visual_review": sum(1 for x in st if x == "confirmed_manual"),
            "weak_sift_only_unconfirmed": sum(1 for x in st if x == "weak_unconfirmed"),
            "rejected_after_manual_review": sum(1 for x in st if x == "rejected_manual")}
    S["test_imgs_with_confirmed_overlap_any_dataset"] = len(overlap_set)
    S["test_imgs_with_only_weak_unconfirmed_evidence"] = len(weak_set)
    clean = [i for i in range(n) if i not in overlap_set]

    def acc(sel):
        sel = list(sel)
        k = sum(pred.get(tests[i], 0) for i in sel)
        lo, hi = wilson(k, len(sel))
        return {"n": len(sel), "correct": k, "accuracy": round(k / len(sel), 4) if sel else None,
                "wilson95": [round(lo, 4), round(hi, 4)]}
    S["model_accuracy_existing_predictions"] = {
        "all_600": acc(range(n)),
        "confirmed_overlap_with_training_data": acc(sorted(overlap_set)),
        "no_confirmed_overlap": acc(clean)}
    cls = defaultdict(lambda: [0, 0])
    for i in range(n):
        c = tests[i].split("/")[2].replace("\u2705", "")
        cls[c][0] += 1
        cls[c][1] += int(i in overlap_set)
    S["confirmed_overlap_by_class"] = {k: {"n": v[0], "overlapping": v[1]} for k, v in sorted(cls.items())}
    with open(os.path.join(OUT, "overlap_summary.json"), "w") as f:
        json.dump(S, f, indent=2, ensure_ascii=False)
    log(json.dumps(S, indent=1, ensure_ascii=False))

    # --- contact sheets for visual verification
    sheets = os.path.join(OUT, "review_sheets")
    os.makedirs(sheets, exist_ok=True)
    for fn in os.listdir(sheets):
        os.remove(os.path.join(sheets, fn))
    groups = defaultdict(list)
    for i, p in enumerate(tests):
        for d in DATASETS:
            if ev[i][d] and not (ev[i][d] & STRICT):
                groups["weak_only_" + d].append((p, detail[i][d][0] if detail[i][d] else "", f"{d} / {'+'.join(sorted(ev[i][d]))}"))
            elif "near_identical_pixels" in ev[i][d] and "exact_bytes" not in ev[i][d] and detail[i][d]:
                groups["near_identical_" + d].append((p, detail[i][d][0], f"{d} / near_identical_pixels"))
            if not ev[i][d] and (i, d) in uncertain:
                groups["uncertain_" + d].append((p, uncertain[(i, d)][0], f"{d} / UNCERTAIN"))
    groups["internal_non_exact"] = [(a_, b_, f"{'+'.join(sorted(info['ev']))} sift={info['sift']}")
                                    for (a_, b_), info in ipairs.items() if not (info["ev"] & {"exact_bytes"})]
    rnd = random.Random(1)
    ex_pairs = [(tests[i], detail[i][d][0], f"{d} / exact_bytes") for i in range(n) for d in DATASETS
                if "exact_bytes" in ev[i][d] and detail[i][d]]
    groups["examples_exact_bytes"] = rnd.sample(ex_pairs, min(16, len(ex_pairs)))
    ns = {}
    for k, lst in sorted(groups.items()):
        if k.startswith("near_identical_") or k.startswith("internal_non_exact"):
            lst = lst[:40]
        ns[k] = (len(lst), contact_sheet(lst[:80], os.path.join(sheets, k), k))
    log("contact sheets (pairs, files):", ns)


def stage_supplementary():
    """TrashNet (train / validation) vs the fine-tuning datasets, from existing outputs only.
    A TrashNet image counts as 'also in dataset X' if it is byte-identical to a file of X or a
    confirmed perceptual match (pHash/dHash <= 10 and ZNCC >= 0.90)."""
    idx = load_index()
    mem = trashnet_membership(idx)
    pos = {p: i for i, p in enumerate(idx["trashnet"])}
    hit = {d: set() for d in ("TACO", "RealWaste", "Household", "GarbageV2")}
    with open(os.path.join(OUT, "exact_matches.csv")) as f:
        for r in csv.DictReader(f):
            if r["hash_type"] == "sha" and r["query_set"].startswith("trashnet["):
                d = DATASET_OF.get(r["matched_set"])
                if d:
                    hit[d].add(pos[r["query_path"]])
    with open(os.path.join(OUT, "supplementary_trashnet_candidates.csv")) as f:
        for r in csv.DictReader(f):
            if r["verdict"] == "confirmed" and r["compared_with"] != "trashnet":
                hit[DATASET_OF[r["compared_with"]]].add(pos[r["trashnet_image"]])
    out = {"trashnet_total": len(mem),
           "split_sizes": {"train_shuffled": int(sum(m.startswith("train") for m in mem)),
                           "val_shuffled": int(sum(m.startswith("val") for m in mem)),
                           "val_unshuffled_used_by_finetune_scripts": int(sum("ftval" in m for m in mem))}}
    for d, h in hit.items():
        out[f"also_in_{d}"] = {"total": len(h),
                               "train_shuffled": sum(1 for i in h if mem[i].startswith("train")),
                               "val_shuffled": sum(1 for i in h if mem[i].startswith("val")),
                               "val_unshuffled_finetune": sum(1 for i in h if "ftval" in mem[i])}
    # internal TrashNet duplicates (confirmed pairs i<j) between train and val
    cross = 0
    n_pairs = 0
    with open(os.path.join(OUT, "supplementary_trashnet_candidates.csv")) as f:
        for r in csv.DictReader(f):
            if r["compared_with"] == "trashnet" and r["verdict"] == "confirmed":
                n_pairs += 1
                a, b = r["trashnet_split"], r["matched_split"]
                if a.startswith("train") != b.startswith("train"):
                    cross += 1
    out["trashnet_internal_confirmed_pairs"] = n_pairs
    out["trashnet_internal_confirmed_pairs_crossing_train_val"] = cross
    with open(os.path.join(OUT, "supplementary_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

STAGES = {"index": stage_index, "hash": stage_hash, "split": stage_split, "exact": stage_exact,
          "perceptual": stage_perceptual, "embed": stage_embed, "sift": stage_sift,
          "control": stage_control, "verify": stage_verify,
          "report": stage_report, "supplementary": stage_supplementary}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    STAGES[a.stage]()
