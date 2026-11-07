"""Create the die crops for the classifier in advance (CPU heavy -> once, then stored on Drive).

Crop EXACTLY as in the firmware: square around the box centre, side = max(w,h)·(1+2·margin), bilinear, RGB.
It is stored with 20 % extra margin (size round(N·1.2)), so that the training can shift/scale the crop a little
at random (just as the boxes of the detector never sit exactly later). Validation and real images are cut back
exactly centred to N during training = firmware rule.

Output <output>/:
  crops_000.npz …  x (n,G,G,3) uint8, y (n) class, image (n) image number (for "per throw"), val (n) bool,
                   n_images: from how many renders (if new renders are added, a last file that is not full is
                   rebuilt, full ones stay, new files are added at the end)
                   labels_hash: checksum of the labels of these images. If the labels change, the file is rebuilt -
                   otherwise the training would silently use old labels.
  real.npz         the same from test_images/*.json (step 9), if present
  info.json        size N, margin, classes, factor

Examples:
  python cls_crops.py --project .../projects/My_Dice --output .../projects/My_Dice/classifier/crops
  python cls_crops.py --project ... --output ... --size 64 --margin 0.25
  python cls_crops.py --help
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_tools  # noqa: E402

FACTOR = 1.2


def main():
    ap = argparse.ArgumentParser(description="Create die crops for the classifier.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--renders", help="default: <project>/renders")
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=0, help="crop N (default: from step 1, otherwise 64)")
    ap.add_argument("--margin", type=float, default=0.25, help="margin all around (share of the die size), default 0.25")
    ap.add_argument("--images-per-file", type=int, default=4000, help="renders per file (finished files are skipped)")
    ap.add_argument("--parallel", type=int, default=32,
                    help="images read at the same time (default 32; Drive reads single files slowly, in parallel ~30x faster)")
    args = ap.parse_args()

    d = ml_tools.load_project(args.project)
    kl = ml_tools.classes(d)
    _, N = ml_tools.sizes(d, crop=args.size or None)
    G = int(round(N * FACTOR))
    renders = args.renders or os.path.join(args.project, "renders")
    os.makedirs(args.output, exist_ok=True)
    info_p = os.path.join(args.output, "info.json")
    if os.path.isfile(info_p):
        with open(info_p, encoding="utf-8") as f:
            old = json.load(f)
        if old["size"] != N or old["margin"] != args.margin or old["classes"] != kl:
            sys.exit(f"{args.output} holds crops with other settings ({old}). Choose another --output.")
    entries = ml_tools.render_entries(renders)
    print(f"{len(entries)} renders, crop {N}×{N} (stored {G}×{G}), margin {args.margin}", flush=True)

    def labels_hash(part):
        """Checksum over image paths, boxes and classes - changes with every change of the labels."""
        h = hashlib.sha1()
        for path, meta in part:
            h.update(os.path.relpath(path, renders).replace(os.sep, "/").encode())
            for w in meta["dice"]:
                h.update(json.dumps([[round(v, 1) for v in w["box"]], int(w["class"])]).encode())
        return h.hexdigest()

    def write(name, items, n_images, lh=""):
        x, y, b, v = zip(*items)
        ml_tools.save_npz_atomic(os.path.join(args.output, name), x=np.stack(x), y=np.array(y, np.int16),
                                 image=np.array(b, np.int32), val=np.array(v, bool), n_images=np.array(n_images, np.int32),
                                 labels_hash=np.array(lh))

    reason = {}

    def complete(name, part):
        """Is the file there, built from exactly these images AND with the current labels?"""
        p = os.path.join(args.output, name)
        if not os.path.isfile(p):
            return False
        with np.load(p) as d:
            if int(d["n_images"]) != len(part):
                reason[name] = "incomplete (new renders were added)"
                return False
            stored = str(d["labels_hash"]) if "labels_hash" in d.files else ""
        if stored != labels_hash(part):
            reason[name] = "the labels have changed"
            return False
        return True

    total = 0
    B = args.images_per_file
    n_files = (len(entries) + B - 1) // B
    open_ = sum(len(entries[n * B:(n + 1) * B]) for n in range(n_files)
                if not complete(f"crops_{n:03d}.npz", entries[n * B:(n + 1) * B]))
    if open_:
        print(f"{open_} images still to crop ({args.parallel} read from Drive at a time) …", flush=True)
    done, t_start = 0, time.time()   # for speed and remaining time over ALL open images
    for file_nr in range(n_files):
        name = f"crops_{file_nr:03d}.npz"
        part = entries[file_nr * B:(file_nr + 1) * B]
        if complete(name, part):
            print(f"  {name}: present already - skipped", flush=True)
            continue
        if os.path.isfile(os.path.join(args.output, name)):
            print(f"  {name}: {reason.get(name, 'outdated')} - rebuilt", flush=True)

        def crops(ie):   # in a thread: read the image (Drive) + crop all dice -> only small crops come back
            i, (path, meta) = ie
            rgb = ml_tools.read_rgb(path)
            val = ml_tools.is_val(os.path.relpath(path, renders))
            return [(ml_tools.crop(rgb, w["box"], G, args.margin, FACTOR), w["class"], file_nr * B + i, val)
                    for w in meta["dice"]]

        buf = []
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            for k, items in enumerate(ex.map(crops, enumerate(part)), 1):
                buf.extend(items)
                done += 1
                if k % 1000 == 0:
                    s = time.time() - t_start
                    print(f"  {name}: {k}/{len(part)} · in total {done}/{open_} images · {done / s:.1f}/s · "
                          f"about {(open_ - done) * s / done / 60:.0f} min left (all)", flush=True)
        if buf:
            write(name, buf, len(part), labels_hash(part))
            total += len(buf)
            print(f"  {name}: {len(buf)} crops from {len(part)} images", flush=True)
    real = []
    real_entries = ml_tools.real_entries(args.project)
    for i, (path, lab) in enumerate(real_entries):
        rgb = ml_tools.read_rgb(path)
        for w in lab["dice"]:
            real.append((ml_tools.crop(rgb, w["box"], G, args.margin, FACTOR), w["class"], 1_000_000 + i, True))
    if real:
        write("real.npz", real, len(real_entries))
    with open(info_p, "w", encoding="utf-8") as f:
        json.dump({"size": N, "stored": G, "factor": FACTOR, "margin": args.margin, "classes": kl}, f, indent=1)
    print(f"✔ {total} synthetic + {len(real)} real crops in {args.output}")


if __name__ == "__main__":
    main()
