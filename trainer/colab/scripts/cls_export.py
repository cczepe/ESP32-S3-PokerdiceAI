"""Prepare the classifier for the computer: .pth -> ONNX (padded to 16 outputs) + fixed image sets.

Output in <project>/export/:
  dice_cls.onnx    input "input" [1,3,N,N] (RGB, /255), output "output" [1,16]
  cls_calib.npz    512 fixed calibration crops (N×N, uint8 RGB, firmware rule)
  cls_test.npz     validation crops (synthetic) + real ones, with class and image number
  cls_info.json    N, margin, classes, variant

Examples:
  python cls_export.py --model .../classifier/models/s025.pth --data .../classifier/crops --project ...
  python cls_export.py --help
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cls_model  # noqa: E402
import cls_train as ct  # noqa: E402
import ml_tools  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Export the classifier to ONNX (padded) + image sets.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--output", help="default: <project>/export")
    ap.add_argument("--max-test", type=int, default=3000)
    args = ap.parse_args()

    out = args.output or os.path.join(args.project, "export")
    os.makedirs(out, exist_ok=True)
    model, d = cls_model.load(args.model)
    N, kl = d["size"], d["classes"]
    pad = cls_model.padded(model, 16)
    dummy = torch.zeros(1, 3, N, N)
    target = os.path.join(out, "dice_cls.onnx")
    kw = dict(opset_version=13, input_names=["input"], output_names=["output"], do_constant_folding=True)
    try:
        torch.onnx.export(pad, dummy, target, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(pad, dummy, target, **kw)
    try:
        import onnx
        import onnxsim
        m, _ = onnxsim.simplify(onnx.load(target))
        onnx.save(m, target)
    except ImportError:
        pass
    print(f"✔ {target}  ({cls_model.parameters(model):,} parameters, input {N}×{N}, output 16 - {len(kl)} of them real)")

    info, xt, yt, xv, yv, bv, real = ct.load_data(args.data, "cpu")
    factor = info["factor"]
    g = torch.Generator().manual_seed(12345)
    idx = torch.randperm(len(xt), generator=g)[:512]
    to_u8 = lambda x: (ct.prepare(x, N, factor, False) * 255).round().byte().permute(0, 2, 3, 1).numpy()
    ml_tools.save_npz_atomic(os.path.join(out, "cls_calib.npz"), images=to_u8(xt[idx]))
    ti = torch.randperm(len(xv), generator=g)[:args.max_test]
    x, y, b, q = [to_u8(xv[ti])], [yv[ti].numpy()], [bv[ti.numpy()]], ["synthetic"] * len(ti)
    if real is not None:
        x.append(to_u8(real[0])), y.append(real[1].numpy()), b.append(real[2])
        q += ["real"] * len(real[1])
    ml_tools.save_npz_atomic(os.path.join(out, "cls_test.npz"), images=np.concatenate(x), **{"class": np.concatenate(y)},
                             image=np.concatenate(b), source=np.array(q))
    with open(os.path.join(out, "cls_info.json"), "w", encoding="utf-8") as f:
        json.dump({"size": N, "margin": d["margin"], "classes": kl, "outputs": 16, "architecture": d["architecture"],
                   "model": os.path.basename(args.model)}, f, indent=1, ensure_ascii=False)
    print(f"✔ calibration crops: {len(idx)} · test crops: {len(q)} → {out}")


if __name__ == "__main__":
    main()
