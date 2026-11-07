"""Shared building blocks for training and export (plain Python with OpenCV/NumPy, optionally torch).

IMPORTANT: letterbox and die crop must work EXACTLY like in the firmware:
- Letterbox: keep the aspect ratio, scale to imgsz (bilinear), border gray 114, centred, RGB, /255.
- Crop: square around the box centre, side = max(w, h) * (1 + 2*margin), bilinear to N×N, outside gray 114, RGB.
"""
import glob
import json
import os
import zlib

import cv2
import numpy as np

GRAY = 114


# ------------------------------------------------------------------ project
def load_project(p):
    with open(os.path.join(p, "project.json"), encoding="utf-8") as f:
        return json.load(f)


def classes(d):
    k = (d.get("symbols") or {}).get("classes") or []
    if not k:
        raise SystemExit("No class list in project.json (confirm step 3).")
    return k


def sizes(d, imgsz=None, crop=None):
    """Detector input and crop size: start parameter > step 1 (project.json) > default."""
    cs = d.get("camera_setup") or {}
    i = imgsz or cs.get("detector_imgsz") or 224
    a = crop or cs.get("crop_px") or 64
    return int(i), int(a)


# ------------------------------------------------------------------ images
def read_rgb(p):
    b = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if b is None:
        raise ValueError(f"not readable: {p}")
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def letterbox_params(w, h, imgsz):
    s = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    return s, nw, nh, (imgsz - nw) // 2, (imgsz - nh) // 2


def letterbox(rgb, imgsz):
    """-> (uint8 imgsz×imgsz×3, (scale, dx, dy))"""
    h, w = rgb.shape[:2]
    s, nw, nh, dx, dy = letterbox_params(w, h, imgsz)
    out = np.full((imgsz, imgsz, 3), GRAY, np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (s, dx, dy)


def crop(rgb, box, size, margin=0.25, factor=1.0):
    """Square crop around the box centre (x0,y0,x1,y1 in pixels), bilinear, outside gray 114.
    factor > 1: extra margin (for random shifting during training)."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin) * factor
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = side / size
    # mapping target pixel (u,v) -> source: x = cx + (u + 0.5 - size/2) * s - 0.5
    M = np.array([[s, 0, cx - (size / 2.0) * s + 0.5 * s - 0.5],
                  [0, s, cy - (size / 2.0) * s + 0.5 * s - 0.5]], np.float32)
    return cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(GRAY, GRAY, GRAY))


# ------------------------------------------------------------------ data from renders and real images
def chunk_gaps(chunk):
    """Image names from meta.jsonl of a chunk that are missing in the folder images/ (Drive occasionally
    loses single files when copying). Empty list = complete."""
    try:
        present = set(os.listdir(os.path.join(chunk, "images")))
    except OSError:
        present = set()
    missing = []
    with open(os.path.join(chunk, "meta.jsonl"), encoding="utf-8") as m:
        for line in m:
            if line.strip():
                b = json.loads(line)["image"]
                if b not in present:
                    missing.append(b)
    return missing


def render_entries(renders, warn=True):
    """List (image_path, meta) from all finished chunks. Images missing on Drive are skipped
    (with a warning) - a single missing image should not abort a long run."""
    result = []
    gaps = []
    for f in sorted(glob.glob(os.path.join(renders, "chunk_*", "done.json"))):
        ch = os.path.dirname(f)
        try:
            present = set(os.listdir(os.path.join(ch, "images")))
        except OSError:
            present = set()
        with open(os.path.join(ch, "meta.jsonl"), encoding="utf-8") as m:
            for line in m:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["image"] in present:
                    result.append((os.path.join(ch, "images", d["image"]), d))
                else:
                    gaps.append(f"{os.path.basename(ch)}/images/{d['image']}")
    if gaps and warn:
        print(f"⚠ {len(gaps)} render image(s) missing on Drive and skipped: "
              + ", ".join(gaps[:10]) + (" …" if len(gaps) > 10 else ""), flush=True)
    return result


def real_entries(project):
    """List (image_path, label) from test_images/*.json (step 9). Box x,y,w,h -> box x0,y0,x1,y1."""
    result = []
    for f in sorted(glob.glob(os.path.join(project, "test_images", "*.json"))):
        with open(f, encoding="utf-8") as h:
            d = json.load(h)
        p = os.path.join(project, "test_images", d["image"])
        if os.path.isfile(p):
            dice = [{"box": [w["x"], w["y"], w["x"] + w["w"], w["y"] + w["h"]], "class": w["class"],
                     "class_name": w["symbol"]} for w in d["dice"]]
            result.append((p, {"image": d["image"], "dice": dice}))
    return result


def is_val(name, share=0.1):
    """Fixed split (same in every run): ~share of the images for validation."""
    return (zlib.crc32(name.encode()) % 1000) < share * 1000


def save_npz_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
