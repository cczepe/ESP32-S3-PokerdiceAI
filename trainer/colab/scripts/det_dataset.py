"""Build the dataset for the detector (Ultralytics format) - on the local Colab disk (reading from Drive is slow).

  train/val:  renders (fixed split, ~10 % validation)
  test_real:  real camera board images from step 9 (test_images/*.json), if present
  data.yaml:  one class "die"

Examples:
  python det_dataset.py --project /content/drive/MyDrive/Pokerdice/projects/My_Dice --output /content/det_data
  python det_dataset.py --project ... --output ... --max-images 2000     (small trial dataset)
  python det_dataset.py --help
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_tools  # noqa: E402


def jpeg_size(path):
    """(width, height) from the JPEG header, without decoding the image."""
    with open(path, "rb") as f:
        data = f.read(65536)
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    raise ValueError(f"image size not readable: {path}")


def yolo_lines(dice, w, h):
    lines = []
    for d in dice:
        x0, y0, x1, y1 = d["box"]
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    ap = argparse.ArgumentParser(description="Build the detector dataset.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--renders", help="default: <project>/renders")
    ap.add_argument("--output", required=True, help="local folder, e.g. /content/det_data")
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--max-images", type=int, default=0, help="only this many renders (0 = all)")
    ap.add_argument("--parallel", type=int, default=32,
                    help="simultaneous copies (default 32; Drive reads single files slowly, in parallel ~30x faster)")
    args = ap.parse_args()

    renders = args.renders or os.path.join(args.project, "renders")
    entries = ml_tools.render_entries(renders)
    if args.max_images:
        entries = entries[:args.max_images]
    if not entries:
        sys.exit(f"No finished renders in {renders}.")
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    cnt = {"train": 0, "val": 0, "test_real": 0}
    for part in ("train", "val"):
        for u in ("images", "labels"):
            os.makedirs(os.path.join(args.output, part, u), exist_ok=True)

    def copy(entry):
        path, meta = entry
        part = "val" if ml_tools.is_val(os.path.relpath(path, renders), args.val_share) else "train"
        name = os.path.basename(os.path.dirname(os.path.dirname(path))) + "_" + meta["image"][:-4]
        try:
            shutil.copyfile(path, os.path.join(args.output, part, "images", name + ".jpg"))
        except OSError as e:
            print(f"⚠ skipped (not readable): {path} - {e}", flush=True)
            return None
        with open(os.path.join(args.output, part, "labels", name + ".txt"), "w") as f:
            f.write(yolo_lines(meta["dice"], *(meta.get("size") or jpeg_size(path))))
        return part

    n = len(entries)
    print(f"copying {n} renders from Drive ({args.parallel} at a time) …", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        for i, part in enumerate(ex.map(copy, entries), 1):
            if part:
                cnt[part] += 1
            if i % 1000 == 0 or i == n:
                s = time.time() - t0
                print(f"  {i}/{n} copied · {i / s:.0f} images/s · about {(n - i) * s / i / 60:.1f} min left", flush=True)
    for path, lab in ml_tools.real_entries(args.project):
        for u in ("images", "labels"):
            os.makedirs(os.path.join(args.output, "test_real", u), exist_ok=True)
        name = lab["image"][:-4]
        shutil.copyfile(path, os.path.join(args.output, "test_real", "images", name + ".jpg"))
        with open(os.path.join(args.output, "test_real", "labels", name + ".txt"), "w") as f:
            f.write(yolo_lines(lab["dice"], *jpeg_size(path)))
        cnt["test_real"] += 1
    with open(os.path.join(args.output, "data.yaml"), "w") as f:
        f.write(f"path: {args.output}\ntrain: train/images\nval: val/images\n")
        if cnt["test_real"]:
            f.write("test: test_real/images\n")
        f.write("names:\n  0: die\n")
    print(f"✔ Dataset: {cnt['train']} training, {cnt['val']} validation (synthetic), {cnt['test_real']} real test images")
    if not cnt["test_real"]:
        print("  ⚠ No real test images yet (steps 8-9) - for now testing happens only on synthetic images.")


if __name__ == "__main__":
    main()
