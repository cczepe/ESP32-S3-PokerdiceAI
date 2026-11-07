"""Suggestions from your own models trained in Colab (step 9: label real throws).

Uses <project>/export/ (the folder you fetch from Google Drive for converting anyway):
  dice_det.onnx + det_info.json   detector (boxes)
  dice_cls.onnx + cls_info.json   classifier (symbol)
Runs on the laptop with onnxruntime (part of trainer/requirements.txt).
Pre- and post-processing exactly as in training and in the firmware:
  letterbox (OpenCV, bilinear, border gray 114) -> detector -> decode + NMS (trainer/convert/decode.py)
  -> per box a square crop (trainer/colab/scripts/ml_tools.crop) -> classifier -> softmax.

Important: the suggestions save typing, not looking. The labelled real images are the honest
test of the models - suggestions taken over unchecked would "confirm" their own errors.
"""
import datetime
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "convert"), os.path.join(_HERE, "..", "colab", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

UNSURE_BELOW = 0.80   # symbol probability below this -> "look closely"
FILES = ("dice_det.onnx", "det_info.json", "dice_cls.onnx", "cls_info.json")

_cache = {}   # export folder -> (version, models)


class ModelError(Exception):
    pass


def export_folder(project_path):
    return os.path.join(project_path, "export")


def version(project_path):
    """Modification time of the models as text (changes when a new export folder is fetched), None = no models."""
    e = export_folder(project_path)
    if not all(os.path.isfile(os.path.join(e, n)) for n in FILES):
        return None
    t = max(os.path.getmtime(os.path.join(e, n)) for n in FILES)
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def available(project_path):
    """(True, version) or (False, reason as text)."""
    v = version(project_path)
    if not v:
        missing = [n for n in FILES if not os.path.isfile(os.path.join(export_folder(project_path), n))]
        return False, "export folder from Colab missing or incomplete (" + ", ".join(missing) + ")"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "onnxruntime is missing (run the setup script again or: pip install -r trainer/requirements.txt)"
    return True, v


class _Models:
    def __init__(self, e):
        import onnxruntime as ort
        with open(os.path.join(e, "det_info.json"), encoding="utf-8") as f:
            self.det = json.load(f)
        with open(os.path.join(e, "cls_info.json"), encoding="utf-8") as f:
            self.cls = json.load(f)
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        self.s_det = ort.InferenceSession(os.path.join(e, "dice_det.onnx"), opt, providers=["CPUExecutionProvider"])
        self.s_cls = ort.InferenceSession(os.path.join(e, "dice_cls.onnx"), opt, providers=["CPUExecutionProvider"])
        self.det_in = self.s_det.get_inputs()[0].name
        self.det_out = [o.name for o in self.s_det.get_outputs()]
        self.cls_in = self.s_cls.get_inputs()[0].name
        names = ["box0", "score0", "box1", "score1", "box2", "score2"]
        self.order = [self.det_out.index(n) for n in names] if all(n in self.det_out for n in names) \
            else list(range(len(self.det_out)))


def _load(project_path):
    ok, v = available(project_path)
    if not ok:
        raise ModelError(v)
    e = export_folder(project_path)
    old = _cache.get(e)
    if old and old[0] == v:
        return old[1]
    try:
        m = _Models(e)
    except Exception as ex:   # broken/old file -> understandable message instead of a crash
        raise ModelError(f"Models in {e} cannot be loaded: {ex}")
    _cache[e] = (v, m)
    return m


def _softmax(x):
    x = np.asarray(x, np.float64)
    x = x - x.max()
    p = np.exp(x)
    return p / p.sum()


def suggest(project_path, image_bgr, classes=None):
    """Boxes + symbol from your own models.
    Returns {"boxes": [{nr, x, y, w, h, det_score, model_symbol, model_p, model_unsure}], "version", "classes", "notes"}."""
    import cv2
    import decode as dec
    import ml_tools
    m = _load(project_path)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    imgsz = int(m.det["imgsz"])
    lb, _ = ml_tools.letterbox(rgb, imgsz)
    x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = m.s_det.run(None, {m.det_in: x})
    out = [out[i] for i in m.order]
    cand = dec.decode(out, imgsz, float(m.det.get("threshold", 0.35)), float(m.det.get("nms_iou", 0.5)),
                      tuple(m.det.get("strides", (8, 16, 32))), orig=(w, h))
    model_classes = list(m.cls["classes"])
    notes = []
    if classes and list(classes) != model_classes:
        notes.append(f"Classes in the model ({', '.join(model_classes)}) differ from step 3 ({', '.join(classes)}) - "
                     "treat the symbol suggestions with care.")
    n_cls = len(model_classes)
    N, margin = int(m.cls["size"]), float(m.cls.get("margin", 0.25))
    boxes = []
    for nr, (x0, y0, x1, y1, sc) in enumerate(sorted(cand, key=lambda k: (k[0] + k[2]) / 2), 1):
        a = ml_tools.crop(rgb, (x0, y0, x1, y1), N, margin)
        y = m.s_cls.run(None, {m.cls_in: (a.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})[0]
        p = _softmax(np.asarray(y).reshape(-1)[:n_cls])
        k = int(np.argmax(p))
        xi0, yi0 = int(round(max(0.0, x0))), int(round(max(0.0, y0)))
        xi1, yi1 = int(round(min(float(w), x1))), int(round(min(float(h), y1)))
        if xi1 - xi0 < 4 or yi1 - yi0 < 4:
            continue
        boxes.append({"nr": nr, "x": xi0, "y": yi0, "w": xi1 - xi0, "h": yi1 - yi0, "det_score": round(float(sc), 3),
                      "model_symbol": model_classes[k], "model_p": round(float(p[k]), 3),
                      "model_unsure": bool(p[k] < UNSURE_BELOW or sc < 0.5)})
    return {"boxes": boxes, "version": version(project_path), "classes": model_classes, "notes": notes}
