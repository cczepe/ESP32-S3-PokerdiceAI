"""Convert the classifier to the ESP32 format (esp-ppq, Python 3.11) and check it.

Input: the folder "export" from Colab (dice_cls.onnx, cls_calib.npz, cls_test.npz, cls_info.json).
Output: <dest>/dice_cls.espdl and cls_report.json (accuracy PC float ↔ ESP32 int8, per die and per throw,
most frequent confusions, agreement).
Split convolutions lose more with int8 - then try --w8a16.

Examples (from the repository folder):
  .venv\\Scripts\\python.exe trainer\\convert\\quantize_classifier.py --help
  .venv\\Scripts\\python.exe trainer\\convert\\quantize_classifier.py --export workspace\\projects\\My_Dice\\export --dest workspace\\projects\\My_Dice\\models
  .venv\\Scripts\\python.exe trainer\\convert\\quantize_classifier.py --export ... --dest ... --w8a16
  (Linux/macOS: .venv/bin/python trainer/convert/quantize_classifier.py ...)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode as dec  # noqa: E402


def key_figures(pred, y, image, classes):
    right = pred == y
    throws = {}
    for r, b in zip(right, image):
        throws.setdefault(int(b), []).append(r)
    conf = {}
    for a, b in zip(y[~right], pred[~right]):
        k = f"{classes[a]}→{classes[b]}"
        conf[k] = conf.get(k, 0) + 1
    return {"per_die": round(float(right.mean()), 4), "per_throw": round(float(np.mean([all(v) for v in throws.values()])), 4),
            "confusions": dict(sorted(conf.items(), key=lambda kv: -kv[1])[:6])}


def main():
    ap = argparse.ArgumentParser(description="Classifier -> .espdl (esp-ppq) and comparison float/int8.",
                                 epilog="Example: .venv\\Scripts\\python.exe trainer\\convert\\quantize_classifier.py "
                                        "--export workspace\\projects\\My_Dice\\export --dest workspace\\projects\\My_Dice\\models")
    ap.add_argument("--export", required=True, help="folder “export” from Colab")
    ap.add_argument("--dest", required=True, help="output folder, e.g. workspace\\projects\\My_Dice\\models")
    ap.add_argument("--w8a16", action="store_true", help="activations 16 bit (more exact, slower)")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor

    with open(os.path.join(args.export, "cls_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    N, kl = info["size"], info["classes"]
    t = lambda u8: torch.from_numpy(u8.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    calib = t(np.load(os.path.join(args.export, "cls_calib.npz"))["images"])
    test = np.load(os.path.join(args.export, "cls_test.npz"))
    os.makedirs(args.dest, exist_ok=True)
    onnx_p = os.path.join(args.export, "dice_cls.onnx")
    target = os.path.join(args.dest, "dice_cls.espdl")
    dec.backup_old(args.dest, ["dice_cls.espdl", "dice_cls.info", "dice_cls.json", "cls_report.json"])
    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=onnx_p, espdl_export_file=target,
        calib_dataloader=DataLoader(list(calib), batch_size=32, shuffle=False),
        calib_steps=int(np.ceil(len(calib) / 32)), input_shape=[1, 3, N, N], inputs=[calib[:1]],
        target="esp32s3", quant_type="w8a16" if args.w8a16 else "w8a8", device="cpu", error_report=False,
        export_test_values=True, verbose=0)
    print(f"✔ {target}  ({os.path.getsize(target) / 1024:.0f} KB, {time.time() - t0:.0f} s)")

    sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])
    exe = TorchExecutor(graph=graph, device="cpu")
    x = t(test["images"])
    pf, pq = [], []
    for i in range(len(x)):
        pf.append(int(np.argmax(sess.run(None, {"input": x[i:i + 1].numpy()})[0][0][:len(kl)])))
        pq.append(int(np.argmax(exe.forward(inputs=x[i:i + 1])[0].detach().numpy()[0][:len(kl)])))
    pf, pq = np.array(pf), np.array(pq)
    y, image, source = test["class"].astype(int), test["image"], test["source"]
    report = {"file": os.path.basename(target), "kb": round(os.path.getsize(target) / 1024, 1), "quant": "w8a16" if args.w8a16 else "w8a8"}
    for q in ("synthetic", "real"):
        m = source == q
        if not m.any():
            continue
        f = key_figures(pf[m], y[m], image[m], kl)
        e = key_figures(pq[m], y[m], image[m], kl)
        same = float((pf[m] == pq[m]).mean())
        report[q] = {"pc_float": f, "esp32_int8": e, "agreement": round(same, 4), "count": int(m.sum())}
        print(f"Test crops {q} ({int(m.sum())}):")
        print(f"  PC (float):               per die {f['per_die']:.1%} · per throw {f['per_throw']:.1%}")
        print(f"  ESP32 (int8, simulated):  per die {e['per_die']:.1%} · per throw {e['per_throw']:.1%} · "
              f"same answer as PC: {same:.1%}")
        if e["confusions"]:
            print("  most frequent confusions ESP32: " + ", ".join(f"{k} {n}×" for k, n in e["confusions"].items()))
    with open(os.path.join(args.dest, "cls_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
