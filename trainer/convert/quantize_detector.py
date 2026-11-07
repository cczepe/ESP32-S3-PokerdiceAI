"""Convert the detector to the ESP32 format (esp-ppq, Python 3.11) and check it.

Input: the folder "export" from Colab (dice_det.onnx, det_calib.npz, det_test.npz, det_info.json).
Output: <dest>/dice_det.espdl (+ .info/.json from esp-ppq) and det_report.json.

Check: the same test images through the ONNX model (PC, float) and the quantized model (like on the ESP32, int8),
the same decoding as in the firmware -> recall/precision of both. Fixed calibration images (otherwise the values vary a lot).
TQT (more exact, takes longer) is switched on; for a quick try use --no-tqt.

Examples (from the repository folder):
  .venv\\Scripts\\python.exe trainer\\convert\\quantize_detector.py --help
  .venv\\Scripts\\python.exe trainer\\convert\\quantize_detector.py --export workspace\\projects\\My_Dice\\export --dest workspace\\projects\\My_Dice\\models
  .venv\\Scripts\\python.exe trainer\\convert\\quantize_detector.py --export ... --dest ... --no-tqt
  (Linux/macOS: .venv/bin/python trainer/convert/quantize_detector.py ...)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode as dec  # noqa: E402


def letterbox(rgb, imgsz):
    import cv2
    h, w = rgb.shape[:2]
    s, nw, nh, dx, dy = dec.letterbox_params(w, h, imgsz)
    out = np.full((imgsz, imgsz, 3), dec.GRAY, np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def as_tensor(u8):
    import torch
    return torch.from_numpy(u8.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()


def evaluate(name, outputs_list, truth_list, info, orig=(640, 480)):
    """orig = size of the test images (camera resolution of the project), so the boxes are converted back correctly."""
    tp = fp = fn = 0
    for out, truth in zip(outputs_list, truth_list):
        v = dec.decode(out, info["imgsz"], info["threshold"], info["nms_iou"], orig=orig)
        a, b, c = dec.matches(v, [tuple(r) for r in truth])
        tp, fp, fn = tp + a, fp + b, fn + c
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    print(f"  {name:24s} recall {recall:.3f} · precision {precision:.3f}  ({tp} correct, {fp} too many, {fn} missed)")
    return {"recall": round(recall, 4), "precision": round(precision, 4), "correct": tp, "too_many": fp, "missed": fn}


def main():
    ap = argparse.ArgumentParser(description="Detector -> .espdl (esp-ppq) and comparison float/int8.",
                                 epilog="Example: .venv\\Scripts\\python.exe trainer\\convert\\quantize_detector.py "
                                        "--export workspace\\projects\\My_Dice\\export --dest workspace\\projects\\My_Dice\\models")
    ap.add_argument("--export", required=True, help="folder “export” from Colab")
    ap.add_argument("--dest", required=True, help="output folder, e.g. workspace\\projects\\My_Dice\\models")
    ap.add_argument("--no-tqt", action="store_true", help="faster, a little less exact")
    ap.add_argument("--tqt-steps", type=int, default=50, help="more = a little more exact, takes proportionally longer")
    args = ap.parse_args()

    import onnxruntime as ort
    from torch.utils.data import DataLoader
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor

    with open(os.path.join(args.export, "det_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    onnx_p = os.path.join(args.export, "dice_det.onnx")
    calib = as_tensor(np.load(os.path.join(args.export, "det_calib.npz"))["images"])
    test = np.load(os.path.join(args.export, "det_test.npz"), allow_pickle=True)
    os.makedirs(args.dest, exist_ok=True)
    target = os.path.join(args.dest, "dice_det.espdl")
    dec.backup_old(args.dest, ["dice_det.espdl", "dice_det.info", "dice_det.json", "det_report.json"])
    I = info["imgsz"]

    setting = QuantizationSettingFactory.espdl_setting()
    if not args.no_tqt:
        setting.tqt_optimization = True
        st = setting.tqt_optimization_setting
        st.lr, st.steps, st.block_size, st.collecting_device = 1e-5, args.tqt_steps, 4, "cpu"
    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=onnx_p, espdl_export_file=target,
        calib_dataloader=DataLoader(list(calib), batch_size=32, shuffle=False),
        calib_steps=int(np.ceil(len(calib) / 32)), input_shape=[1, 3, I, I], inputs=[calib[:1]],
        target="esp32s3", quant_type="w8a8", setting=setting, device="cpu", error_report=False,
        export_test_values=True, verbose=0)
    print(f"✔ {target}  ({os.path.getsize(target) / 1024:.0f} KB, {time.time() - t0:.0f} s)")

    # comparison float (ONNX) against quantized (int8 simulation)
    sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])
    exe = TorchExecutor(graph=graph, device="cpu")
    float_out, int8_out = [], []
    for rgb in test["images"]:
        x = as_tensor(letterbox(rgb, I)[None])
        float_out.append(sess.run(None, {"images": x.numpy()}))
        int8_out.append([t.detach().numpy() for t in exe.forward(inputs=x)])
    truth = list(test["boxes"])
    source = list(test["source"])
    orig = (int(test["images"].shape[2]), int(test["images"].shape[1]))   # (width, height) of the test images
    report = {"file": os.path.basename(target), "kb": round(os.path.getsize(target) / 1024, 1), "tqt": not args.no_tqt}
    for q in ("synthetic", "real"):
        idx = [i for i, s in enumerate(source) if s == q]
        if not idx:
            continue
        print(f"Test images {q} ({len(idx)}):")
        report[q] = {"pc_float": evaluate("PC (float)", [float_out[i] for i in idx], [truth[i] for i in idx], info, orig),
                     "esp32_int8": evaluate("ESP32 (int8, simulated)", [int8_out[i] for i in idx], [truth[i] for i in idx], info, orig)}
    with open(os.path.join(args.dest, "det_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("Good if ESP32 and PC are close to each other (a few percentage points).")


if __name__ == "__main__":
    main()
