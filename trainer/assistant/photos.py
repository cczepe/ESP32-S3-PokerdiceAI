"""Photo steps (phase A): saving, checking and cropping the phone photos.

Shared by steps 3 (faces), 4 (corners), 5 (tray), 6 (felt), 7 (surroundings).
- Photos are never overwritten: numbered (0001.jpg ...) or fixed names (tray: top.jpg, side.jpg).
- Phone photos often store the rotation only as EXIF -> it is applied when reading.
- The checks (sharpness, brightness, resolution) are rules of thumb, not hard limits.
"""
import io
import os
import re

import cv2
import numpy as np

import projects

# Camera per step: "device" | "phone" | "none"
CAMERA = {
    "1_camera_setup": "device",
    "2_measure": "none",
    "3_faces": "phone",
    "4_corners": "phone",
    "5_tray": "phone",
    "6_felt": "phone",
    "7_surroundings": "phone",
    "8_throws": "device",
    "9_label": "none",
}
CAMERA_TEXT = {
    "device": ("📷", "with the camera board", "Deliberately with the camera board, not with the phone - different lens, colours and sharpness."),
    "phone": ("📱", "with the phone", "With the phone camera at full resolution (no zoom, no flash). Easiest via the QR code directly into the trainer."),
    "none": ("📏", "no camera", "You need no camera for this step."),
}

# Photo steps: folder, count, fixed names
PHOTO_STEPS = {
    "3_faces": {"folder": os.path.join("photos", "faces"), "min": 6, "max": 6, "names": None,
                "title": "Face", "short": "6 faces"},
    "4_corners": {"folder": os.path.join("photos", "corners"), "min": 4, "max": 4, "names": None,
                  "title": "Corner", "short": "4 corners"},
    "5_tray": {"folder": os.path.join("photos", "tray"), "min": 2, "max": 2, "names": ["top", "side"],
               "title": "Tray", "short": "Tray"},
    "6_felt": {"folder": os.path.join("photos", "felt"), "min": 1, "max": 5, "names": None,
               "title": "Felt", "short": "Felt"},
    "7_surroundings": {"folder": os.path.join("photos", "surroundings"), "min": 10, "max": 30, "names": None,
                       "title": "Surroundings", "short": "Surroundings"},
}
NAMES_TEXT = {"top": "Tray from above", "side": "Tray from the side"}
FILE_NAME = re.compile(r"^(\d{4,}|top|side)\.jpg$")

SHARPNESS_WARN = 40.0     # Laplace variance at 1000 px width: below this probably blurred
RESOLUTION_WARN = 1500    # shorter image side in pixels


# ------------------------------------------------------------------ loading (with EXIF rotation)
def load_image(path, max_side=None):
    """BGR image; the EXIF rotation (phone) is applied. max_side: optionally downscale."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except ImportError:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and max_side and max(img.shape[:2]) > max_side:
            f = max_side / float(max(img.shape[:2]))
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if img is None:
        raise ValueError(f"Image not readable: {path}")
    return img


def original_size(path):
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).size
    except Exception:
        b = load_image(path)
        return b.shape[1], b.shape[0]


def jpeg(img, quality=88):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


# ------------------------------------------------------------------ folder / list / save
def folder(workspace, project, step):
    return projects.path(workspace, project, PHOTO_STEPS[step]["folder"])


def list_photos(workspace, project, step):
    o = folder(workspace, project, step)
    if not os.path.isdir(o):
        return []
    names = [n for n in os.listdir(o) if FILE_NAME.match(n)]
    fixed = PHOTO_STEPS[step]["names"]
    if fixed:
        return [n for n in (f + ".jpg" for f in fixed) if n in names]
    return sorted(names)


def free_name(workspace, project, step, wish=None):
    """Next free file name or error text."""
    cfg = PHOTO_STEPS[step]
    present = list_photos(workspace, project, step)
    if cfg["names"]:
        if wish not in cfg["names"]:
            free = [n for n in cfg["names"] if n + ".jpg" not in present]
            if not free:
                return None, "All photos of this step are there. To replace one, delete the old one first."
            wish = free[0]
        if wish + ".jpg" in present:
            return None, f"“{NAMES_TEXT.get(wish, wish)}” is already there. To replace it, delete the old one first."
        return wish + ".jpg", None
    if len(present) >= cfg["max"]:
        return None, f"Already {cfg['max']} photos - this step needs no more. To replace one, delete it first."
    return projects.next_file_name(folder(workspace, project, step), ".jpg"), None


def save(workspace, project, step, data, wish=None):
    """Saves the bytes of an uploaded photo. Returns (name, error)."""
    if data[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c") or b"ftypheic" in data[:32] or b"ftypmif1" in data[:32]:
        return None, ("HEIC photo (iPhone) cannot be read. On the iPhone: Settings → Camera → Formats → "
                      "“Most Compatible”, or take the photo via the phone page of the trainer.")
    probe = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if probe is None:
        return None, "File is not a readable image (JPEG/PNG)."
    name, error = free_name(workspace, project, step, wish)
    if error:
        return None, error
    if not data.startswith(b"\xff\xd8"):   # PNG etc. -> save as JPEG
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        data = jpeg(img, 95)
    os.makedirs(folder(workspace, project, step), exist_ok=True)
    projects.write_new(os.path.join(folder(workspace, project, step), name), data)
    return name, None


def delete(workspace, project, step, name):
    if not FILE_NAME.match(name or ""):
        raise ValueError("invalid name")
    p = os.path.join(folder(workspace, project, step), name)
    if os.path.isfile(p):
        os.remove(p)


# ------------------------------------------------------------------ checks
def check(path):
    """Sharpness, brightness, resolution + understandable hints."""
    w, h = original_size(path)
    small = load_image(path, max_side=1000)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2].mean()) / 2.55
    hints = []
    if min(w, h) < RESOLUTION_WARN:
        hints.append(f"Resolution only {w}×{h} - please full camera resolution, no screenshot/zoom.")
    if sharpness < SHARPNESS_WARN:
        hints.append("Looks blurred - hold the phone still, tap on the subject to focus.")
    if bright < 25:
        hints.append("Quite dark - more light (daylight/lamp), no flash.")
    elif bright > 92:
        hints.append("Very bright/overexposed - less direct light.")
    return {"width": w, "height": h, "sharpness": round(sharpness, 1), "brightness_percent": round(bright, 1),
            "ok": not hints, "hints": hints}


# ------------------------------------------------------------------ find the die on a face photo
def _die_bright(small):
    """Die clearly brighter than the background (green felt, dark surface): bright, roughly square
    area that does not touch the image border. -> (x, y, w, h) in pixels of "small" or None."""
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    med, top = float(np.median(L)), float(np.percentile(L, 99.5))
    if top - med < 60:
        return None   # no clearly brighter die -> rather white paper
    mask = (L > med + 0.6 * (top - med)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = L.shape
    best = None
    for k in contours:
        x, y, bw, bh = cv2.boundingRect(k)
        if bw * bh < 0.003 * w * h or x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        if not (0.7 <= bw / float(bh) <= 1.4):
            continue
        fill = cv2.contourArea(cv2.convexHull(k)) / float(bw * bh)
        if fill < 0.6:
            continue
        # the direct surroundings (ring around the spot) must be clearly darker - on white paper they are not
        m = int(0.4 * max(bw, bh))
        ring = L[max(0, y - m):y + bh + m, max(0, x - m):x + bw + m].copy()
        ring[min(m, y):min(m, y) + bh, min(m, x):min(m, x) + bw] = np.nan
        if float(np.nanmedian(L[y:y + bh, x:x + bw])) - float(np.nanmedian(ring)) < 40:
            continue
        if best is None or bw * bh > best[2] * best[3]:
            best = (x, y, bw, bh)
    return best


def find_die(path, edge_mm=None):
    """Looks for the die on a face photo. Returns {x,y,s} (square crop in original pixels) or None.
    1. Die on green felt/dark surface: brightest roughly square area.
    2. Die on white paper: everything that differs from the paper (edges + shadows), then the
       largest roughly square shape (the ruler is long and narrow)."""
    big_w, big_h = original_size(path)
    small = load_image(path, max_side=900)
    f = max(big_w, big_h) / float(max(small.shape[:2]))
    bright = _die_bright(small)
    if bright:
        x, y, bw, bh = bright
        s = max(bw, bh) * 1.08          # a little margin all around (prepare_textures.py finds the exact edge later)
        cx, cy = x + bw / 2.0, y + bh / 2.0
        result = {"x": int(round((cx - s / 2) * f)), "y": int(round((cy - s / 2) * f)), "s": int(round(s * f)), "auto": True}
        if edge_mm:
            result["px_per_mm"] = round(result["s"] / 1.08 / float(edge_mm), 1)
        return result
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = None
    for k in contours:
        x, y, bw, bh = cv2.boundingRect(k)
        area = bw * bh
        if area < 0.01 * w * h or area > 0.7 * w * h:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue   # touches the border -> rather the paper edge/ruler
        ratio = bw / float(bh)
        if not (0.75 <= ratio <= 1.33):
            continue
        fill = cv2.contourArea(cv2.convexHull(k)) / float(area)
        points = area * fill
        if best is None or points > best[0]:
            best = (points, x, y, bw, bh)
    if not best:
        return None
    _, x, y, bw, bh = best
    s = max(bw, bh)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    result = {"x": int(round((cx - s / 2) * f)), "y": int(round((cy - s / 2) * f)), "s": int(round(s * f)), "auto": True}
    if edge_mm:
        result["px_per_mm"] = round(result["s"] / float(edge_mm), 1)
    return result


def face_crop(path, z, size=256):
    """Load a square crop (original pixels) and scale it to size."""
    img = load_image(path)
    x, y, s = int(z["x"]), int(z["y"]), int(z["s"])
    pad = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_REPLICATE)
    a = pad[y + s:y + 2 * s, x + s:x + 2 * s]
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)


def preview(path, max_side=900):
    return jpeg(load_image(path, max_side=max_side), 85)


# ------------------------------------------------------------------ network address for the phone
def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.1", 9))   # nothing is sent; only to find the matching interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def qr_svg(text):
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8")
