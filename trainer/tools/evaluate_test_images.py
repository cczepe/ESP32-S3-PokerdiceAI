"""Evaluate the test images: detector and classifier (from <project>/export, PC/float) on all real test images
(<project>/test_images, step 9) - per image and per die, not only as a sum.

For each missed die the cause is given:
  suppressed by NMS   The detector had a matching box, but it was dropped when duplicate boxes were sorted out
                      (NMS: of two boxes overlapping by more than nms_iou only the more confident one stays).
                      Typical for rows of dice "away from the camera", where the boxes overlap a lot.
  box shifted         A box sits on the die but overlaps the test box only by 30-50 %.
  not found           No box nearby.
Also: the same count with other NMS values (default 0.5 = firmware; 0.6 and 0.7 for comparison), the symbol per
die (classifier on your test box) and "fully correct" (found AND symbol right, as on the device).
In addition "count": how many dice were found per image - independent of how well the boxes overlap.
For rows, the detector's box covers the WHOLE die (also the hidden part, that is how it is trained), a box
drawn by hand usually only the visible part; then a found die counts as "shifted".

Output in <project>/evaluation_<time>/ (nothing is overwritten):
  <image>.jpg   green = your box + symbol, orange = found (symbol, confidence), red = missed
  report.json   all numbers
Computes with the float models on the PC (like step 9); on the board (int8) the values are usually a little worse.

Needs the virtual environment with onnxruntime (trainer/requirements.txt).

Examples (from the repository folder):
  .venv\\Scripts\\python.exe trainer\\tools\\evaluate_test_images.py --help
  .venv\\Scripts\\python.exe trainer\\tools\\evaluate_test_images.py --project workspace\\projects\\My_Dice
  .venv\\Scripts\\python.exe trainer\\tools\\evaluate_test_images.py --project workspace\\projects\\My_Dice --nms 0.5 0.65 0.8
  (Linux/macOS: .venv/bin/python trainer/tools/evaluate_test_images.py --project workspace/projects/My_Dice)
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "assistant"), os.path.join(HERE, "..", "convert"), os.path.join(HERE, "..", "colab", "scripts")):
    sys.path.insert(0, os.path.abspath(_p))


def assign(predicted, truth, threshold=0.5):
    """Greedy assignment like decode.matches, but with pairs: {index_truth: index_prediction}."""
    import decode as dec
    pairs, free = {}, list(range(len(truth)))
    for vi in sorted(range(len(predicted)), key=lambda i: -predicted[i][4]):
        best, bi = 0.0, -1
        for i in free:
            u = dec.iou(predicted[vi], truth[i])
            if u > best:
                best, bi = u, i
        if best >= threshold:
            pairs[bi] = vi
            free.remove(bi)
    return pairs


def read_image(path):
    import cv2
    data = np.fromfile(path, np.uint8)   # also paths with special characters (Windows)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, bgr):
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(path)


def main():
    ap = argparse.ArgumentParser(description="Evaluate detector + classifier per test image (missed: NMS / shifted / not found).",
                                 epilog="Example: .venv\\Scripts\\python.exe trainer\\tools\\evaluate_test_images.py "
                                        "--project workspace\\projects\\My_Dice")
    ap.add_argument("--project", required=True, help="project folder, e.g. workspace\\projects\\My_Dice (with export/ and test_images/)")
    ap.add_argument("--nms", type=float, nargs="+", default=[0.5, 0.6, 0.7],
                    help="NMS thresholds to compare; the first applies to images and causes (default 0.5 = firmware)")
    ap.add_argument("--threshold", type=float, help="detection threshold (default: from det_info.json, as in the firmware)")
    ap.add_argument("--output", help="output folder (default: <project>/evaluation_<time>)")
    args = ap.parse_args()

    import cv2
    import decode as dec
    import local_model as lm
    import ml_tools

    project = os.path.abspath(args.project)
    try:
        m = lm._load(project)
    except lm.ModelError as e:
        sys.exit(f"✘ {e}")
    labels = sorted(glob.glob(os.path.join(project, "test_images", "*.json")))
    if not labels:
        sys.exit(f"✘ no test images in {os.path.join(project, 'test_images')} (confirm step 9)")
    out_dir = args.output or os.path.join(project, "evaluation_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(out_dir) and os.listdir(out_dir):
        sys.exit(f"✘ {out_dir} exists already and is not empty - nothing is overwritten")
    os.makedirs(out_dir, exist_ok=True)

    imgsz = int(m.det["imgsz"])
    threshold = args.threshold if args.threshold is not None else float(m.det.get("threshold", 0.35))
    strides = tuple(m.det.get("strides", (8, 16, 32)))
    kl = list(m.cls["classes"])
    N, margin = int(m.cls["size"]), float(m.cls.get("margin", 0.25))
    nms_main = args.nms[0]

    def symbol(rgb, r):
        a = ml_tools.crop(rgb, r[:4], N, margin)
        y = m.s_cls.run(None, {m.cls_in: (a.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})[0]
        p = lm._softmax(np.asarray(y).reshape(-1)[:len(kl)])
        k = int(np.argmax(p))
        return kl[k], float(p[k])

    total_nms = {t: {"correct": 0, "too_many": 0, "missed": 0} for t in args.nms}
    causes = {"suppressed by NMS": 0, "box shifted": 0, "not found": 0}
    sym_ok = sym_n = fully_ok = 0
    count = {"right": 0, "too few": 0, "too many": 0}
    confusions = {}
    images = []
    print(f"Models: {lm.version(project)} · threshold {threshold} · NMS {', '.join(str(t) for t in args.nms)} · {len(labels)} test images\n")
    print(f"{'Image':34s} {'Dice':>6s} {'Count':>6s} {'found':>5s} {'too many':>8s}  missed (cause) · symbol wrong")
    for lp in labels:
        with open(lp, encoding="utf-8") as f:
            lab = json.load(f)
        bgr = read_image(os.path.join(project, "test_images", lab["image"]))
        if bgr is None:
            print(f"⚠ {lab['image']}: image not readable - skipped")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        truth = [(d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]) for d in lab["dice"]]
        truth_sym = [d["symbol"] for d in lab["dice"]]
        lb, _ = ml_tools.letterbox(rgb, imgsz)
        out = m.s_det.run(None, {m.det_in: (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})
        out = [out[i] for i in m.order]
        raw = dec.decode(out, imgsz, threshold, 1.01, strides, orig=(w, h))   # 1.01 = without NMS: all candidates
        per_nms = {}
        for t in args.nms:
            v = dec.nms(raw, t)
            a, b, c = dec.matches(v, truth)
            per_nms[t] = {"correct": a, "too_many": b, "missed": c}
            for k, x in per_nms[t].items():
                total_nms[t][k] += x
        found = dec.nms(raw, nms_main)
        pairs = assign(found, truth)
        found_sym = [symbol(rgb, r) for r in found]
        # causes of the missed ones
        missed = []
        for i, r in enumerate(truth):
            if i in pairs:
                continue
            if any(dec.iou(k, r) >= 0.5 for k in raw):
                u = "suppressed by NMS"
            elif any(dec.iou(k, r) >= 0.3 for k in found):
                u = "box shifted"
            else:
                u = "not found"
            causes[u] += 1
            missed.append({"nr": i + 1, "symbol": truth_sym[i], "cause": u})
        # symbol on your box (classifier alone) and "fully correct" (found + symbol)
        wrong = []
        for i, r in enumerate(truth):
            s, p = symbol(rgb, r)
            sym_n += 1
            if s == truth_sym[i]:
                sym_ok += 1
            else:
                wrong.append(f"{truth_sym[i]}→{s}")
                confusions[f"{truth_sym[i]}→{s}"] = confusions.get(f"{truth_sym[i]}→{s}", 0) + 1
            if i in pairs and found_sym[pairs[i]][0] == truth_sym[i]:
                fully_ok += 1
        pair_iou = max([dec.iou(truth[i], truth[j]) for i in range(len(truth)) for j in range(i + 1, len(truth))] or [0.0])
        # draw the image
        img = bgr.copy()
        for i, r in enumerate(truth):
            color = (0, 200, 0) if i in pairs else (0, 0, 255)
            cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 1 if i in pairs else 3)
            cv2.putText(img, f"{i + 1}:{truth_sym[i]}", (int(r[0]) + 2, int(r[3]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for (x0, y0, x1, y1, sc), (s, p) in zip(found, found_sym):
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 165, 255), 2)
            cv2.putText(img, f"{s} {p:.0%} ({sc:.2f})", (int(x0), max(12, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(img, "green/red = test box (red = missed), orange = found: symbol confidence (detector)",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        write_image(os.path.join(out_dir, lab["image"]), img)
        e = per_nms[nms_main]
        n_found = len(found)
        count["right" if n_found == len(truth) else ("too few" if n_found < len(truth) else "too many")] += 1
        text = ", ".join(f"{v['nr']}:{v['symbol']} ({v['cause']})" for v in missed) or "–"
        print(f"{lab['image'][:34]:34s} {len(truth):6d} {n_found:6d} {e['correct']:5d} {e['too_many']:8d}  {text}"
              + (f" · {', '.join(wrong)}" if wrong else ""))
        images.append({"image": lab["image"], "dice": len(truth), "per_nms": {str(t): x for t, x in per_nms.items()},
                       "count_found": n_found, "missed": missed, "symbol_wrong": wrong, "candidates_without_nms": len(raw),
                       "largest_overlap_test_boxes": round(pair_iou, 2)})

    total = sum(b["dice"] for b in images)
    print(f"\nDetector ({total} dice):")
    for t in args.nms:
        s = total_nms[t]
        print(f"  NMS {t:<4}  {s['correct']} found, {s['too_many']} too many, {s['missed']} missed"
              + ("   ← as in the firmware" if abs(t - float(m.det.get('nms_iou', 0.5))) < 1e-6 else ""))
    print(f"Count per image (NMS {nms_main}): right in {count['right']} of {len(images)} images"
          + (f", too few in {count['too few']}" if count["too few"] else "") + (f", too many in {count['too many']}" if count["too many"] else ""))
    print(f"Causes of the missed ones (NMS {nms_main}): " + ", ".join(f"{k} {v}" for k, v in causes.items()))
    print(f"Classifier on your boxes: {sym_ok}/{sym_n} right"
          + (" · " + ", ".join(f"{k} {v}×" for k, v in sorted(confusions.items(), key=lambda x: -x[1])) if confusions else ""))
    print(f"Fully correct (found AND symbol right, NMS {nms_main}): {fully_ok}/{total}")
    report = {"models": lm.version(project), "threshold": threshold, "nms": args.nms,
              "total": {str(t): x for t, x in total_nms.items()}, "causes": causes,
              "classifier": {"right": sym_ok, "total": sym_n, "confusions": confusions},
              "fully_correct": fully_ok, "count_per_image": count, "dice_total": total, "images": images}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\n✔ Images and report.json in {out_dir}")


if __name__ == "__main__":
    main()
