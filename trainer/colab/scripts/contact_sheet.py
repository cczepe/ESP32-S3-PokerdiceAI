"""Contact sheet: grid of sample images to look at (plain Python with OpenCV, not Blender).

Two kinds:
  renders  - renders with drawn labels (box + symbol; "?" orange)
  check    - the 8 rendered corner views of the 3D die above your corner photos

Examples:
  python contact_sheet.py renders --folder .../renders_preview --output /content/sheet_preview.jpg
  python contact_sheet.py renders --folder .../renders --count 24 --seed 3 --output /content/sheet.jpg
  python contact_sheet.py check --scene .../scene --project .../projects/My_Dice --output /content/check.jpg
  python contact_sheet.py --help
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np

VIEWS = ["top_front-right", "top_back-right", "top_back-left", "top_front-left",
         "bottom_front-right", "bottom_back-right", "bottom_back-left", "bottom_front-left"]


def read(p, max_side=None):
    try:
        from PIL import Image, ImageOps
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side:
                im.thumbnail((max_side, max_side))
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except ImportError:
        return cv2.imread(p)


def write(p, img):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    cv2.imencode(os.path.splitext(p)[1] or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(p)


def text(b, t, org, size=0.5, color=(255, 255, 255)):
    cv2.putText(b, t, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(b, t, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def grid(images, columns):
    h, w = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    out = np.full((rows * h, columns * w, 3), 30, np.uint8)
    for i, b in enumerate(images):
        r, c = divmod(i, columns)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(b, (w, h))
    return out


def renders(args):
    entries = []
    for meta in sorted(glob.glob(os.path.join(args.folder, "**", "meta.jsonl"), recursive=True)):
        chunk = os.path.dirname(meta)
        with open(meta, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                entries.append((os.path.join(chunk, "images", d["image"]), d))
    if not entries:
        raise SystemExit(f"No renders found in {args.folder}.")
    random.Random(args.seed).shuffle(entries)
    chosen = entries[:args.count]
    images = []
    for path, d in chosen:
        b = cv2.imread(path)
        if b is None:
            continue
        for w in d["dice"]:
            x0, y0, x1, y1 = [int(round(v)) for v in w["box"]]
            unclear = w["class_name"] == "?"
            color = (0, 140, 255) if unclear else (60, 220, 60)
            cv2.rectangle(b, (x0, y0), (x1, y1), color, 2)
            text(b, w["class_name"] + (f" ({w['symbol']})" if unclear else ""), (x0, max(14, y0 - 4)), 0.5, color)
        k = d["camera"]
        text(b, f"{k['elevation_deg']:.0f} deg, {k['distance_cm']:.0f} cm, {k['light_k']} K, {d.get('throw_type', '')}",
             (6, b.shape[0] - 10), 0.5)
        images.append(b)
    out = grid(images, args.columns)
    write(args.output, out)
    # small statistics
    classes, tilted, count = {}, 0, 0
    types = {}
    for _, d in entries:
        types[d.get("throw_type", "normal")] = types.get(d.get("throw_type", "normal"), 0) + 1
        for w in d["dice"]:
            classes[w["class_name"]] = classes.get(w["class_name"], 0) + 1
            tilted += int(w["tilted"])
            count += 1
    print(f"✔ contact sheet: {args.output} ({len(images)} of {len(entries)} images)")
    print(f"  die labels in total: {count} · per class: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))
    print(f"  of which tilted (above the limit): {tilted}")
    print("  throw types: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items())))


def check(args):
    c_folder = os.path.join(args.scene, "check")
    renders_ = []
    for n in VIEWS:
        b = cv2.imread(os.path.join(c_folder, n + ".png"))
        if b is None:
            raise SystemExit(f"{n}.png missing - run check_corners.py first.")
        b = cv2.resize(b, (300, 300))
        text(b, n.replace("_", " "), (6, 18), 0.5)
        renders_.append(b)
    rows = [np.hstack(renders_[:4]), np.hstack(renders_[4:])]
    corners = sorted(glob.glob(os.path.join(args.project, "photos", "corners", "*.jpg")))
    if corners:
        photos = []
        for p in corners[:4]:
            b = read(p, 900)
            h, w = b.shape[:2]
            s = min(h, w)
            b = cv2.resize(b[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (300, 300))
            text(b, "photo " + os.path.basename(p), (6, 18), 0.5, (0, 255, 255))
            photos.append(b)
        while len(photos) < 4:
            photos.append(np.full((300, 300, 3), 40, np.uint8))
        rows.append(np.hstack(photos))
    out = np.vstack(rows)
    write(args.output, out)
    print(f"✔ check image: {args.output}")
    print("  Top/middle: 3D die from 8 corners. Bottom: your corner photos. Find the matching view for each photo -")
    print("  do the symbols AND their rotation match? If not: correct step 4 in the trainer, new Colab package.")


def main():
    ap = argparse.ArgumentParser(description="Contact sheet for renders or the corner check.")
    sub = ap.add_subparsers(dest="kind", required=True)
    r = sub.add_parser("renders", help="renders with labels")
    r.add_argument("--folder", required=True, help="folder with chunks (looks for meta.jsonl recursively)")
    r.add_argument("--output", required=True, help="output file (.jpg)")
    r.add_argument("--count", type=int, default=16)
    r.add_argument("--columns", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("check", help="corner check")
    k.add_argument("--scene", required=True, help="folder with scene.json and check/")
    k.add_argument("--project", required=True, help="project folder (for the corner photos)")
    k.add_argument("--output", required=True)
    args = ap.parse_args()
    renders(args) if args.kind == "renders" else check(args)


if __name__ == "__main__":
    main()
