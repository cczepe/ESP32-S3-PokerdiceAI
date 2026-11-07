"""Pre- and post-processing in Python - reference for the detection in the firmware.

Detector (ESPDet-Pico, reg_max = 1):
  input: letterbox to imgsz (border gray 114), RGB, /255.
  outputs per level (stride 8/16/32): box [4 used of 16] = distances left, top, right, bottom in cells,
  score [1 used of 16] -> sigmoid. Cell (x,y): centre cx=(x+0.5)*stride, cy=(y+0.5)*stride,
  box = (cx-l*s, cy-t*s, cx+r*s, cy+b*s). Threshold, NMS (IoU 0.5), back into the original image.
"""
import numpy as np

GRAY = 114


def letterbox_params(w, h, imgsz):
    s = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    return s, nw, nh, (imgsz - nw) // 2, (imgsz - nh) // 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))   # clip: no overflow warning for very low scores


def decode(outputs, imgsz, threshold=0.35, nms_iou=0.5, strides=(8, 16, 32), orig=(640, 480), nhwc=False):
    """outputs: [box0, score0, box1, score1, box2, score2] (each 1×16×H×W or NHWC). -> list (x0,y0,x1,y1,score)."""
    s, nw, nh, dx, dy = letterbox_params(orig[0], orig[1], imgsz)
    candidates = []
    for e, stride in enumerate(strides):
        box, score = np.asarray(outputs[2 * e]), np.asarray(outputs[2 * e + 1])
        if nhwc:
            box, score = box.transpose(0, 3, 1, 2), score.transpose(0, 3, 1, 2)
        sc = sigmoid(score[0, 0])
        ys, xs = np.where(sc > threshold)
        for y, x in zip(ys, xs):
            l, t, r, b = box[0, 0:4, y, x]
            cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
            x0, y0, x1, y1 = cx - l * stride, cy - t * stride, cx + r * stride, cy + b * stride
            candidates.append(((x0 - dx) / s, (y0 - dy) / s, (x1 - dx) / s, (y1 - dy) / s, float(sc[y, x])))
    return nms(candidates, nms_iou)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(candidates, threshold):
    result = []
    for k in sorted(candidates, key=lambda k: -k[4]):
        if all(iou(k, e) < threshold for e in result):
            result.append(k)
    return result


def matches(predicted, truth, threshold=0.5):
    """(correct, false_positive, missed) at IoU >= threshold (greedy assignment)."""
    free = list(range(len(truth)))
    tp = 0
    for v in predicted:
        best, bi = 0.0, -1
        for i in free:
            u = iou(v, truth[i])
            if u > best:
                best, bi = u, i
        if best >= threshold:
            tp += 1
            free.remove(bi)
    return tp, len(predicted) - tp, len(free)


def backup_old(folder, names):
    """Move existing files (names) from folder to folder/backup_<time>/ - nothing is overwritten."""
    import datetime
    import os
    import shutil
    present = [n for n in names if os.path.exists(os.path.join(folder, n))]
    if not present:
        return None
    old = os.path.join(folder, "backup_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(old, exist_ok=True)
    for n in present:
        shutil.move(os.path.join(folder, n), os.path.join(old, n))
    print(f"Older files saved in {old}")
    return old
