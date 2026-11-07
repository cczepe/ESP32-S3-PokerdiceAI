"""PC test: pre- and post-processing of the firmware (firmware/main/preprocess.hpp) against the Python reference.

Compares
  - letterbox and die crop with trainer/colab/scripts/ml_tools.py (OpenCV),
  - crops from strips (JPEG block mode of the firmware) with crops from the whole image (bit-exact),
  - the decoding of the detector outputs with trainer/convert/decode.py,
  - the poker hands.
Needs g++ (Linux, macOS, WSL or MSYS2 on Windows) plus numpy and opencv-python (both in the trainer environment).

Examples (from the repository folder):
  python firmware/tools/test_pc/compare.py --help
  python firmware/tools/test_pc/compare.py                                  synthetic test images
  python firmware/tools/test_pc/compare.py --images workspace/projects/My_Dice/test_images --count 10
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MAIN = os.path.join(REPO, "firmware", "main")
sys.path.insert(0, os.path.join(REPO, "trainer", "colab", "scripts"))
sys.path.insert(0, os.path.join(REPO, "trainer", "convert"))
import ml_tools  # noqa: E402
import decode  # noqa: E402


def build(tmp):
    if not shutil.which("g++"):
        sys.exit("g++ not found - on Windows use WSL or MSYS2.")
    exe = {}
    for name in ("compare", "strips"):
        exe[name] = os.path.join(tmp, name)
        subprocess.check_call(["g++", "-O2", "-std=c++17", "-I", MAIN, os.path.join(HERE, name + ".cpp"), "-o", exe[name]])
    return exe


def test_images(args):
    if args.images:
        paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
        if not paths:
            sys.exit(f"no images in {args.images}")
        for p in paths[:args.count]:
            yield os.path.basename(p), np.ascontiguousarray(ml_tools.read_rgb(p))
        return
    rng = np.random.default_rng(1)
    for i in range(args.count):
        w, h = (640, 480) if i % 2 else (1024, 768)
        y, x = np.mgrid[0:h, 0:w]
        rgb = np.stack([(x * 255 // (w - 1)), (y * 255 // (h - 1)), ((x + y) % 256)], -1).astype(np.uint8)
        rgb = np.clip(rgb.astype(int) + rng.integers(-40, 40, rgb.shape), 0, 255).astype(np.uint8)
        yield f"synthetic_{i} ({w}x{h})", rgb


def main():
    ap = argparse.ArgumentParser(description="Compare the pre- and post-processing of the firmware with the trainer (Python).",
                                 epilog="Example: python firmware/tools/test_pc/compare.py --images workspace/projects/My_Dice/test_images",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder with real images (optional; otherwise synthetic ones)")
    ap.add_argument("--count", type=int, default=5, help="number of images (default 5)")
    ap.add_argument("--imgsz", type=int, default=224, help="detector input (default 224)")
    ap.add_argument("--size", type=int, default=64, help="classifier crop (default 64)")
    args = ap.parse_args()

    errors = 0
    with tempfile.TemporaryDirectory() as tmp:
        exe = build(tmp)
        rng = np.random.default_rng(7)
        lb_max, cr_max, lb_mean, cr_mean = 0, 0, [], []
        for name, rgb in test_images(args):
            h, w = rgb.shape[:2]
            inp = os.path.join(tmp, "in.rgb")
            rgb.tofile(inp)
            out = os.path.join(tmp, "out.rgb")
            subprocess.check_call([exe["compare"], "letterbox", inp, str(w), str(h), str(args.imgsz), out])
            c = np.fromfile(out, np.uint8).reshape(args.imgsz, args.imgsz, 3).astype(int)
            p, _ = ml_tools.letterbox(rgb, args.imgsz)
            d = np.abs(c - p.astype(int))
            lb_max, lb_mean = max(lb_max, d.max()), lb_mean + [d.mean()]
            for _ in range(10):   # crops at random positions (also at the border)
                bw = rng.uniform(25, 90) * w / 640
                x0, y0 = rng.uniform(-20, w - bw + 20), rng.uniform(-20, h - bw + 20)
                r = [float(f"{v:.4f}") for v in (x0, y0, x0 + bw * rng.uniform(0.8, 1.2), y0 + bw)]
                subprocess.check_call([exe["compare"], "crop", inp, str(w), str(h), "1.0", *[f"{v:.4f}" for v in r],
                                       str(args.size), "0.25", out])
                c = np.fromfile(out, np.uint8).reshape(args.size, args.size, 3).astype(int)
                p = ml_tools.crop(rgb, r, args.size, 0.25).astype(int)
                d = np.abs(c - p)
                cr_max, cr_mean = max(cr_max, d.max()), cr_mean + [d.mean()]
        print(f"letterbox: largest difference {lb_max} gray levels, mean {np.mean(lb_mean):.3f}")
        print(f"crop:      largest difference {cr_max} gray levels, mean {np.mean(cr_mean):.3f}")
        # OpenCV computes in fixed point (1/32 pixel, 11/15-bit weights) -> differences of 1..2 are normal
        if lb_max > 2 or cr_max > 3:
            print("  ✘ difference too large")
            errors += 1
        else:
            print("  ✔ ok (rounding differences of OpenCV)")

        r = subprocess.run([exe["strips"]], capture_output=True, text=True)
        print("strips:    " + r.stdout.strip())
        if r.returncode:
            print("  ✘ crops from strips differ from crops of the whole image")
            errors += 1
        else:
            print("  ✔ bit-exact")

        imgsz, bad = args.imgsz, 0
        for attempt in range(20):
            W, H = (1024, 768) if attempt % 2 else (640, 480)
            cargs, floats = [], []
            for e, s in enumerate((8, 16, 32)):
                hh = ww = imgsz // s
                be, se = -4, -3
                box = rng.integers(0, 60, (hh, ww, 16)).astype(np.int8)
                score = rng.integers(-128, -40, (hh, ww, 16)).astype(np.int8)
                for _ in range(rng.integers(0, 8)):
                    score[rng.integers(hh), rng.integers(ww), 0] = rng.integers(-10, 127)
                fb, fs = os.path.join(tmp, f"b{e}.bin"), os.path.join(tmp, f"s{e}.bin")
                box.tofile(fb)
                score.tofile(fs)
                cargs += [str(hh), str(ww), "16", str(s), str(be), str(se), fb, fs]
                floats += [box[None].astype(np.float32) * 2.0 ** be, score[None].astype(np.float32) * 2.0 ** se]
            res = subprocess.check_output([exe["compare"], "decode", str(imgsz), str(W), str(H), "0.35", "0.5", *cargs],
                                          text=True).split("\n")
            n = int(res[0].split()[0])
            c = [tuple(map(float, z.split())) for z in res[1:1 + n]]
            p = decode.decode(floats, imgsz, 0.35, 0.5, orig=(W, H), nhwc=True)
            if len(c) != len(p) or any(max(abs(x - y) for x, y in zip(a, b)) > 0.05 for a, b in zip(c, p)):
                bad += 1
        print(f"decode:    {20 - bad}/20 attempts identical to decode.py")
        if bad:
            errors += 1

        # poker hands (ranks 0..5 = 9 10 J Q K A, -1 = unclear)
        cases = [((5, 5, 5, 5, 5), "Five of a Kind"), ((4, 4, 4, 4, 1), "Four of a Kind"), ((3, 3, 3, 0, 0), "Full House"),
                 ((2, 2, 2, 5, 1), "Three of a Kind"), ((5, 5, 4, 4, 0), "Two Pairs"), ((1, 1, 2, 3, 5), "One Pair"),
                 ((0, 1, 2, 3, 4), "Small Straight"), ((1, 2, 3, 4, 5), "Large Straight"), ((0, 1, 2, 3, 5), "Nothing"),
                 ((0, 1, 2, 3, -1), "")]
        ok = 0
        for r, name in cases:
            res = subprocess.check_output([exe["compare"], "hand", "6", *map(str, r)], text=True).strip("\n").split("|")
            ok += res[1] == name
            if res[1] != name:
                print(f"  ✘ {r}: {res[1]!r} instead of {name!r}")
        print(f"hands:     {ok}/{len(cases)} correct")
        if ok != len(cases):
            errors += 1
    print("ALL OK" if not errors else f"{errors} part(s) with errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
