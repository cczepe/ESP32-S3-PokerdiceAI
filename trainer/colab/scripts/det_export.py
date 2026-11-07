"""Prepare the detector for the computer: best.pt -> ONNX (6 raw outputs, padded to 16 channels) + fixed image sets.

Output in <project>/export/:
  dice_det.onnx      input "images" [1,3,imgsz,imgsz], outputs box0,score0,box1,score1,box2,score2
                     (box 4->16 channels with weights 0/bias 0, score nc->16 with bias -20 => sigmoid ≈ 0)
  det_calib.npz      256 fixed calibration images (letterbox, uint8, RGB) for esp-ppq
  det_test.npz       up to 100 test images (synthetic) + real ones, with boxes, for the comparison ONNX ↔ ESP32
  det_info.json      imgsz, threshold, strides

MOST IMPORTANT TRAP: always pad the output channels to multiples of 16, otherwise ESP-DL computes wrongly.

Examples:
  python det_export.py --weights .../detector/runs/run1/weights/best.pt --project .../projects/My_Dice
  python det_export.py --help
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import espdet_patches  # noqa: E402
import ml_tools  # noqa: E402


def pad(conv, channels, bias_value):
    import torch
    import torch.nn as nn
    old = conv.out_channels
    if old >= channels:
        return conv
    new = nn.Conv2d(conv.in_channels, channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:old] = conv.weight
        new.bias.fill_(bias_value)
        new.bias[:old] = conv.bias if conv.bias is not None else 0
    return new


def main():
    ap = argparse.ArgumentParser(description="Export the detector to ONNX (padded) + image sets.")
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--project", required=True)
    ap.add_argument("--renders", help="default: <project>/renders")
    ap.add_argument("--esp-detection", default=os.environ.get("ESP_DETECTION", "/content/esp-detection"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--output", help="default: <project>/export")
    args = ap.parse_args()

    espdet_patches.patches(args.esp_detection)
    import torch
    from ultralytics import YOLO
    from nn.modules import ESPDetect

    out = args.output or os.path.join(args.project, "export")
    os.makedirs(out, exist_ok=True)
    model = YOLO(args.weights).model.float().eval()
    head = next(m for m in model.modules() if isinstance(m, ESPDetect))
    head.forward = ESPDetect.export_onnx_forward.__get__(head)
    for i in range(len(head.cv2)):
        head.cv2[i][-1] = pad(head.cv2[i][-1], 16, 0.0)
        head.cv3[i][-1] = pad(head.cv3[i][-1], 16, -20.0)
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out_t = model(dummy)
    names = ["box0", "score0", "box1", "score1", "box2", "score2"]
    target = os.path.join(out, "dice_det.onnx")
    export_kw = dict(opset_version=13, input_names=["images"], output_names=names, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, target, dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(model, dummy, target, **export_kw)
    import onnx
    m = onnx.load(target)
    try:
        import onnxsim
        m, ok = onnxsim.simplify(m)
    except ImportError:
        print("⚠ onnxsim missing - without simplification")
    onnx.save(m, target)
    shapes = [tuple(t.shape) for t in out_t]
    print(f"✔ {target}  outputs: " + ", ".join(f"{n}{list(s)}" for n, s in zip(names, shapes)))

    # fixed image sets
    renders = args.renders or os.path.join(args.project, "renders")
    entries = ml_tools.render_entries(renders)
    rng = random.Random(12345)
    train = [e for e in entries if not ml_tools.is_val(os.path.relpath(e[0], renders))]
    val = [e for e in entries if ml_tools.is_val(os.path.relpath(e[0], renders))]
    calib = rng.sample(train, min(256, len(train)))
    ml_tools.save_npz_atomic(os.path.join(out, "det_calib.npz"),
                             images=np.stack([ml_tools.letterbox(ml_tools.read_rgb(p), args.imgsz)[0] for p, _ in calib]))
    test = rng.sample(val, min(100, len(val))) if val else rng.sample(train, min(100, len(train)))
    real = ml_tools.real_entries(args.project)
    images, boxes, source = [], [], []
    for p, d in test + real:
        images.append(ml_tools.read_rgb(p))
        boxes.append(np.array([w["box"] for w in d["dice"]], np.float32).reshape(-1, 4))
        source.append("real" if (p, d) in real else "synthetic")
    sizes = sorted({b.shape[:2] for b in images})
    if len(sizes) > 1:
        sys.exit(f"Test images have different sizes {sizes} - renders and real photos must have the same "
                 "camera resolution (step 1). Remove old renders or photos of another size.")
    ml_tools.save_npz_atomic(os.path.join(out, "det_test.npz"), images=np.stack(images),
                             boxes=np.array(boxes, dtype=object), source=np.array(source))
    info = {"imgsz": args.imgsz, "threshold": args.threshold, "nms_iou": 0.5, "strides": [8, 16, 32],
            "channels": 16, "reg_max": 1, "letterbox_gray": 114, "weights": os.path.basename(args.weights)}
    with open(os.path.join(out, "det_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(f"✔ calibration images: {len(calib)} · test images: {len(test)} synthetic + {len(real)} real → {out}")


if __name__ == "__main__":
    main()
