"""Prepares the photos of a project for Blender (runs in Colab with plain Python, not in Blender).

What happens:
- 6 face photos: cut out the die (from the mark in step 3), find the TOP face and straighten it
  (also for slightly tilted photos; visible side faces at the edge are cut off) -> 512x512 pixels each.
  Search in this order: via the brightness (green felt/dark surface - most reliable), via the colour
  (white paper: die a little more yellowish, shadow gray), old method (edges), finally the mark itself.
  White balance via the die colour (flat face -> body colour from step 2), independent of the background.
- Texture atlas (3x2 cells) + assignment face -> symbol + rotation from the die net (step 4).
- Felt texture, colour of the tray wall, surroundings photos (scaled down).
- Rounded edges: measure the die colour on the flat face, fill the corner gussets (paper/shadow) with this colour.
- scene.json: all sizes and random ranges the Blender scripts need.

Output: <project folder>/scene/  (rewritten at every run; the photos are never changed)

Examples:
  python prepare_textures.py --project /content/drive/MyDrive/Pokerdice/projects/My_Dice
  python prepare_textures.py --project ... --output /content/scene_test
  python prepare_textures.py --help
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

FIELDS = ["top", "bottom", "front", "back", "left", "right"]
TEXTURE = 512


# ------------------------------------------------------------------ load images
def load_image(path, max_side=None):
    """BGR, EXIF rotation (phone) applied."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except ImportError:
        b = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if b is None:
            raise ValueError(f"not readable: {path}")
        if max_side and max(b.shape[:2]) > max_side:
            f = max_side / float(max(b.shape[:2]))
            b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return b


def save(path, img, quality=95):
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    buf.tofile(path)


# ------------------------------------------------------------------ face photos
def die_square(crop):
    """Looks for the die square (4 corners) in the crop - difference to the paper + edges.
    Returns 4 points (float32) or None."""
    h, w = crop.shape[:2]
    border = max(2, int(0.04 * min(h, w)))
    frame = np.concatenate([crop[:border].reshape(-1, 3), crop[-border:].reshape(-1, 3),
                            crop[:, :border].reshape(-1, 3), crop[:, -border:].reshape(-1, 3)])
    paper = np.median(frame, axis=0)
    diff = np.linalg.norm(crop.astype(np.float32) - paper[None, None, :], axis=2)
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 30, 90)
    mask = ((diff > max(18.0, np.percentile(diff, 40))) | (edges > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    k = max(contours, key=cv2.contourArea)
    if cv2.contourArea(k) < 0.15 * w * h:
        return None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(k)
    if rw <= 0 or rh <= 0 or not (0.75 <= rw / rh <= 1.33):
        return None
    # the shadow makes the rectangle a little larger -> slightly inwards
    s = min(rw, rh) * 0.97
    box = cv2.boxPoints(((cx, cy), (s, s), angle))
    return box.astype(np.float32), paper


def _border_and_centre(lab):
    h, w = lab.shape[:2]
    r = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([lab[:r].reshape(-1, 3), lab[-r:].reshape(-1, 3), lab[:, :r].reshape(-1, 3), lab[:, -r:].reshape(-1, 3)])
    centre = (slice(h // 2 - h // 6, h // 2 + h // 6), slice(w // 2 - w // 6, w // 2 + w // 6))
    return np.median(border, axis=0), centre


def _top_from_mask(mask, L, background_L):
    """Die mask -> 4 corners of the top face. Holes (symbols) are filled; in the rotated outline darker
    edge strips (visible side faces on a slightly tilted photo) are cut off. The top face may be a
    rectangle in the image (oblique view) - it is rectified to a square later. -> (corners, text) or (None, reason)"""
    h, w = mask.shape
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return None, "nothing found"
    k0 = max(cont, key=cv2.contourArea)
    if cv2.contourArea(k0) < 0.12 * w * h:
        return None, "die area too small"
    full = np.zeros((h, w), np.uint8)
    cv2.drawContours(full, [k0], -1, 255, -1)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(k0)
    while angle > 45:
        angle -= 90
        rw, rh = rh, rw
    while angle < -45:
        angle += 90
        rw, rh = rh, rw
    if max(rw, rh) / max(1.0, min(rw, rh)) > 1.45:
        return None, f"outline not roughly square ({rw:.0f}×{rh:.0f})"
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    gL = cv2.warpAffine(L, M, (w, h), flags=cv2.INTER_LINEAR)
    gm = cv2.warpAffine(full, M, (w, h), flags=cv2.INTER_NEAREST) > 0
    x0, x1 = max(0, int(round(cx - rw / 2))), min(w, int(round(cx + rw / 2)))
    y0, y1 = max(0, int(round(cy - rh / 2))), min(h, int(round(cy + rh / 2)))
    part, m = gL[y0:y1, x0:x1], gm[y0:y1, x0:x1]

    def profile(axis):
        p = []
        for j in range(part.shape[axis]):
            v = part[j][m[j]] if axis == 0 else part[:, j][m[:, j]]
            p.append(np.percentile(v, 75) if v.size > 3 else 0.0)   # 75 % value: symbols do not disturb
        return np.array(p)

    def cut(p):
        n = len(p)
        mid = np.median(p[n // 4:3 * n // 4])
        limit = mid - max(12.0, 0.25 * abs(mid - background_L))     # side face: clearly darker than the top
        lo, hi = 0, n
        while lo < n * 0.2 and p[lo] < limit:
            lo += 1
        while hi > n * 0.8 and p[hi - 1] < limit:
            hi -= 1
        return lo, hi

    ty0, ty1 = cut(profile(0))
    tx0, tx1 = cut(profile(1))
    X0, X1, Y0, Y1 = x0 + tx0, x0 + tx1, y0 + ty0, y0 + ty1
    bw, bh = X1 - X0, Y1 - Y0
    if bw < 0.12 * w or bh < 0.12 * h or max(bw, bh) / max(1, min(bw, bh)) > 1.4:
        return None, f"top face implausible ({bw}×{bh})"
    corners = np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], np.float32)
    corners = cv2.transform(corners[None], cv2.invertAffineTransform(M))[0].astype(np.float32)
    off = [ty0, y1 - y0 - ty1, tx0, x1 - x0 - tx1]
    return corners, f"{bw}×{bh} px" + (f", side faces cut off (t/b/l/r {off[0]}/{off[1]}/{off[2]}/{off[3]} px)" if any(off) else "")


def top_bright(crop):
    """Die clearly brighter than the background (green felt, dark surface). The shadow is darker -> drops out."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    bg, centre = _border_and_centre(lab)
    wL = float(np.percentile(L[centre], 70))            # die face (symbols are darker)
    if wL - bg[0] < 30:
        return None, f"die hardly brighter than the background (L {wL:.0f}/{bg[0]:.0f})"
    mask = (L > bg[0] + 0.5 * (wL - bg[0])).astype(np.uint8) * 255
    k = max(3, int(min(crop.shape[:2]) * 0.01)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return _top_from_mask(mask, L, bg[0])


def top_color(crop):
    """White paper: die (cream) a little more yellowish than the paper; shadow gray/cooler -> drops out."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    bg, centre = _border_and_centre(lab)
    bright = (L[centre] > bg[0] - 25) & (np.hypot(A[centre] - bg[1], B[centre] - bg[2]) < 20)   # without symbols
    if not bright.any():
        return None, "no bright die face"
    db = float(np.median(B[centre][bright])) - bg[2]
    if db < 3:
        return None, f"die hardly distinguishable from the paper by colour ({db:.1f})"
    colourful = np.hypot(A - bg[1], B - bg[2]) > 18
    black = (L < bg[0] - 70) & (B > bg[2] - 3)
    mask = ((B > bg[2] + 0.4 * db) | colourful | black).astype(np.uint8) * 255
    k = max(3, int(min(crop.shape[:2]) * 0.015)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))
    return _top_from_mask(mask, L, bg[0])


def white_balance(tex, target_bgr, round_share):
    """Die colour (flat face) -> target colour (body colour from step 2, slightly darkened so nothing burns out)."""
    colour = body_color([tex], round_share)
    f = np.clip(np.array(target_bgr, np.float32) / np.maximum(colour, 1.0), 0.5, 2.0)
    return np.clip(tex.astype(np.float32) * f[None, None, :], 0, 255).astype(np.uint8), f


def sort_corners(p):
    """top-left, top-right, bottom-right, bottom-left (image coordinates)."""
    p = np.array(p, np.float32)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], np.float32)


def prepare_face(path, face_crop, size=TEXTURE, round_share=0.1, target_bgr=(226, 235, 235)):
    """Returns (texture_bgr, info). info["method"]: bright | color | refined | mark."""
    img = load_image(path)
    H, W = img.shape[:2]
    x, y, s = int(face_crop["x"]), int(face_crop["y"]), int(face_crop["s"])
    m = int(0.2 * s)
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + s + m), min(H, y + s + m)
    crop = img[y0:y1, x0:x1]
    # search scaled down (faster), then convert back
    f = 900.0 / max(crop.shape[:2]) if max(crop.shape[:2]) > 900 else 1.0
    small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else crop
    info = {"photo": os.path.basename(path)}
    corners = None
    reasons = []
    for name, search in (("bright", top_bright), ("color", top_color)):
        e, text = search(small)
        if e is not None:
            corners = sort_corners(e / f + np.array([x0, y0], np.float32))
            info["method"], info["text"] = name, text
            break
        reasons.append(f"{name}: {text}")
    if corners is None:
        res = die_square(small)
        if res is not None:
            corners = sort_corners(res[0] / f + np.array([x0, y0], np.float32))
            info["method"] = "refined"
        else:
            inset = 0.04 * s
            corners = np.array([[x + inset, y + inset], [x + s - inset, y + inset], [x + s - inset, y + s - inset],
                                [x + inset, y + s - inset]], np.float32)
            info["method"] = "mark"
        info["text"] = "; ".join(reasons)
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners, target)
    tex = cv2.warpPerspective(img, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    # white balance via the die colour (colour cast of the phone, e.g. reddish on green felt)
    tex, factors = white_balance(tex, target_bgr, round_share)
    info["white_balance"] = [round(float(v), 3) for v in factors]
    info["edge_px"] = round(float(np.linalg.norm(corners[1] - corners[0])), 1)
    return tex, info


def body_color(textures, round_share=0.0):
    """Body colour of the die: median of a narrow ring on the FLAT face, just inside the
    rounded edge (hardly any symbols there; the rounding itself is darker shaded in the photo)."""
    values = []
    for t in textures:
        g = t.shape[0]
        r = int(max(0.07, round_share + 0.02) * g)
        b = max(2, int(0.07 * g))
        if r + b >= g // 2:
            r, b = int(0.07 * g), max(2, int(0.07 * g))
        ring = np.concatenate([t[r:r + b, r:-r].reshape(-1, 3), t[-r - b:-r, r:-r].reshape(-1, 3),
                               t[r:-r, r:r + b].reshape(-1, 3), t[r:-r, -r - b:-r].reshape(-1, 3)])
        values.append(ring)
    return np.median(np.concatenate(values), axis=0)


def fill_corners(tex, round_share, colour):
    """Photographed from above, a die with rounded edges is not square at the corners: in the corners of the
    crop there is paper or shadow. These gussets (outside the quarter circle with the edge radius) are filled
    with the die colour, with a soft transition (2 px). Returns (texture, share of filled pixels)."""
    g = tex.shape[0]
    R = round_share * g
    if R < 2:
        return tex, 0.0
    yy, xx = np.mgrid[0:g, 0:g].astype(np.float32) + 0.5
    # distance to the nearest corner circle centre, only in the corner areas
    cx = np.where(xx < R, R, np.where(xx > g - R, g - R, xx))
    cy = np.where(yy < R, R, np.where(yy > g - R, g - R, yy))
    d = np.hypot(xx - cx, yy - cy)
    weight = np.clip((d - R) / 2.0 + 0.5, 0.0, 1.0)[..., None]   # 0 = die, 1 = gusset
    new = tex.astype(np.float32) * (1 - weight) + np.array(colour, np.float32)[None, None, :] * weight
    return np.clip(new + 0.5, 0, 255).astype(np.uint8), float((weight > 0.5).mean())


# ------------------------------------------------------------------ tray / felt
def prepare_felt(path, size=1024):
    b = load_image(path, max_side=2000)
    h, w = b.shape[:2]
    s = int(min(h, w) * 0.8)
    y0, x0 = (h - s) // 2, (w - s) // 2
    quad = cv2.resize(b[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    # seamlessly tileable: blend in a mirrored copy halfway (simple, enough for fine felt)
    mirrored = quad[::-1, ::-1]
    xx = np.linspace(-1, 1, size)
    weight = np.clip(np.maximum(np.abs(xx)[None, :], np.abs(xx)[:, None]) ** 4, 0, 1)[..., None]
    mixed = (quad * (1 - weight) + mirrored * weight).astype(np.uint8)
    colour = np.median(quad.reshape(-1, 3), axis=0)
    return mixed, colour


def wall_color(tray_top, felt_color):
    """Colour of the tray wall: ring directly outside the felt-coloured area. None if not found."""
    try:
        b = load_image(tray_top, max_side=1000)
    except Exception:
        return None
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    fh = cv2.cvtColor(np.uint8([[felt_color]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    lo = ((fh[0] - 12) % 180, max(20, fh[1] * 0.4), max(15, fh[2] * 0.35))
    hi = ((fh[0] + 12) % 180, 255, 255)
    if lo[0] <= hi[0]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    else:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array((179, 255, 255), np.uint8)) | \
            cv2.inRange(hsv, np.array((0, lo[1], lo[2]), np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    k = max(contours, key=cv2.contourArea)
    if cv2.contourArea(k) < 0.05 * b.shape[0] * b.shape[1]:
        return None
    hull = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(k), 255)
    width = max(5, int(0.03 * max(b.shape[:2])))
    ring = cv2.dilate(hull, np.ones((width, width), np.uint8)) & ~cv2.dilate(hull, np.ones((5, 5), np.uint8))
    px = b[ring > 0]
    return np.median(px, axis=0) if len(px) > 50 else None


# ------------------------------------------------------------------ main program
def bgr_to_lin_rgb(c):
    """sRGB (0-255, BGR) -> linear RGB (0-1) for Blender."""
    rgb = np.array(c[::-1], np.float64) / 255.0
    return [round(float(v), 4) for v in np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)]


def value_range(values, margin_low, margin_high, limit_low, limit_high, default):
    if not values:
        return list(default), "default (step 1 without result)"
    lo = max(limit_low, values["min"] - margin_low)
    hi = min(limit_high, values["max"] + margin_high)
    if hi - lo < 5:
        lo, hi = max(limit_low, lo - 5), min(limit_high, hi + 5)
    return [round(lo, 1), round(hi, 1)], "from step 1 (" + str(values.get("source", "tray")) + ")"


def main():
    ap = argparse.ArgumentParser(description="Prepare the photos of a project for Blender.",
                                 epilog="Example: python prepare_textures.py --project /content/drive/MyDrive/Pokerdice/projects/My_Dice")
    ap.add_argument("--project", required=True, help="project folder (with project.json)")
    ap.add_argument("--output", help="output folder (default: <project>/scene)")
    ap.add_argument("--fov", type=float, default=66.0, help="diagonal field of view of the camera in degrees (default 66)")
    ap.add_argument("--tilt-deg", type=float, default=20.0, help="from this tilt a die counts as “?” (default 20)")
    ap.add_argument("--min-visible", type=float, default=0.5, help="dice with a smaller visible share get no label (default 0.5)")
    ap.add_argument("--resolution", help="render size, e.g. 1024x768 (default: camera setting of the project from step 1, otherwise 640x480)")
    args = ap.parse_args()

    pj = os.path.join(args.project, "project.json")
    if not os.path.isfile(pj):
        sys.exit(f"ERROR: {pj} not found. Are DRIVE_FOLDER and PROJECT right (upper/lower case)?")
    with open(pj, encoding="utf-8") as f:
        d = json.load(f)
    out = args.output or os.path.join(args.project, "scene")
    os.makedirs(os.path.join(out, "surroundings"), exist_ok=True)
    problems, hints = [], []

    # --- render size = camera resolution of the project (step 1), so the die crops look like on the camera board
    res = args.resolution or (d.get("camera") or {}).get("resolution") or "640x480"
    try:
        render_w, render_h = [int(v) for v in res.lower().split("x")]
    except ValueError:
        sys.exit(f"ERROR: resolution {res} not readable (example 1024x768).")
    if not d.get("camera") and not args.resolution:
        hints.append("No camera setting in the project (step 1 “Apply”) - renders at 640×480.")

    # --- symbols + net
    sym = d.get("symbols") or {}
    classes = sym.get("classes") or []
    if not sym.get("confirmed") or len(classes) < 2:
        problems.append("Symbols (step 3) not confirmed.")
    lay = d.get("layout") or {}
    net = lay.get("net") or {}
    if not lay.get("confirmed"):
        problems.append("Layout (step 4) not confirmed.")
    faces_by_name = {e["name"]: e for e in sym.get("faces", [])}
    face_crops = sym.get("face_crops") or {}

    # die size and target colour for the white balance (body colour from step 2, slightly darkened: nothing burns out)
    w = d.get("die") or {}
    edge = float(w.get("edge_mm") or 16.0)
    rounding = float(w.get("corner_radius_mm") or 1.5)
    round_share = min(0.3, max(0.0, rounding / edge))
    try:
        hx = (w.get("body_color") or "#f0f0ea").lstrip("#")
        target_bgr = [int(hx[4:6], 16) * 0.94, int(hx[2:4], 16) * 0.94, int(hx[0:2], 16) * 0.94]
    except ValueError:
        target_bgr = [220.0, 225.0, 225.0]

    textures, infos, faces = {}, {}, {}
    for field in FIELDS:
        e = net.get(field) or {}
        s = e.get("symbol")
        if not s or s not in faces_by_name:
            problems.append(f"Face “{field}” has no valid symbol.")
            continue
        photo = faces_by_name[s]["photo"]
        p = os.path.join(args.project, "photos", "faces", photo)
        z = face_crops.get(photo)
        if not os.path.isfile(p) or not z:
            problems.append(f"Face photo {photo} or its mark is missing.")
            continue
        if s not in textures:
            textures[s], infos[s] = prepare_face(p, z, TEXTURE, round_share, target_bgr)
        faces[field] = {"symbol": s, "rotation": int(e.get("rotation", 0)) % 360, "class": classes.index(s) if s in classes else -1}

    if problems:
        print("Something is still missing:")
        for x in problems:
            print("  ✘", x)
        sys.exit(1)

    # rounded corners: measure the die colour on the flat face, then fill the gussets in the corners with it
    bc = body_color(list(textures.values()), round_share)
    gusset = 0.0
    for s_ in list(textures):
        textures[s_], share = fill_corners(textures[s_], round_share, bc)
        gusset = max(gusset, share)
    if gusset > 0:
        hints.append(f"Corners filled with the die colour (corner radius {rounding:g} mm: {gusset:.1%} of each texture).")

    # atlas 3x2: cells in a fixed order, each upright (Blender does the rotation via the UVs)
    atlas = np.zeros((2 * TEXTURE, 3 * TEXTURE, 3), np.uint8)
    for i, field in enumerate(FIELDS):
        c, r = i % 3, i // 3
        atlas[r * TEXTURE:(r + 1) * TEXTURE, c * TEXTURE:(c + 1) * TEXTURE] = textures[faces[field]["symbol"]]
        faces[field]["cell"] = [c, r]
    save(os.path.join(out, "atlas.png"), atlas)
    for s, t in textures.items():
        save(os.path.join(out, f"face_{s}.png"), t)

    if not w.get("body_color"):
        hints.append("Body colour missing (step 2) - white balance to a neutral light gray.")
    if not w.get("edge_mm"):
        hints.append("Edge length missing - default 16 mm.")

    # --- tray / felt
    t = d.get("tray") or {}
    diameter = float(t.get("diameter_mm") or 300.0)
    wall_height = float(t.get("wall_height_mm") or 25.0)
    felt_dir = os.path.join(args.project, "photos", "felt")
    felt_photos = [f for f in (sorted(os.listdir(felt_dir)) if os.path.isdir(felt_dir) else []) if f.lower().endswith(".jpg")]
    if felt_photos:
        felt, felt_color = prepare_felt(os.path.join(felt_dir, felt_photos[0]))
        save(os.path.join(out, "felt.jpg"), felt)
    else:
        felt_color = np.array([40, 110, 30], np.float64)
        save(os.path.join(out, "felt.jpg"), np.full((64, 64, 3), felt_color, np.uint8))
        hints.append("No felt photo - plain felt.")
    wc = wall_color(os.path.join(args.project, "photos", "tray", "top.jpg"), felt_color)
    if wc is None:
        wc = np.array([45, 45, 50], np.float64)
        hints.append("Colour of the tray wall not recognised - dark gray.")

    # --- surroundings
    sur_dir = os.path.join(args.project, "photos", "surroundings")
    sur = []
    for n in sorted(os.listdir(sur_dir)) if os.path.isdir(sur_dir) else []:
        if n.lower().endswith(".jpg"):
            b = load_image(os.path.join(sur_dir, n), max_side=1024)
            save(os.path.join(out, "surroundings", n), b, 90)
            sur.append("surroundings/" + n)
    if not sur:
        hints.append("No surroundings photos - plain background.")

    # --- random ranges from step 1 (a little more generous than measured)
    cs = d.get("camera_setup") or {}
    elevation, esrc = value_range(cs.get("elevation_deg"), 5, 15, 30, 88, (35, 85))
    distance, dsrc = value_range(cs.get("distance_cm"), 4, 8, 10, 120, (20, 45))
    if cs.get("distance_cm"):
        distance = [round(max(10, cs["distance_cm"]["min"] * 0.85), 1), round(cs["distance_cm"]["max"] * 1.15, 1)]

    scene = {
        "version": 1,
        "project": d.get("name"),
        "classes": classes,
        "die": {"edge_mm": edge, "corner_radius_mm": rounding, "body_color_lin": bgr_to_lin_rgb(bc)},
        "faces": faces,
        "atlas": "atlas.png",
        "tray": {"diameter_mm": diameter, "wall_height_mm": wall_height, "wall_width_mm": 12.0,
                 "felt": "felt.jpg", "felt_tile_mm": 120.0, "felt_color_lin": bgr_to_lin_rgb(felt_color),
                 "wall_color_lin": bgr_to_lin_rgb(wc)},
        "surroundings": sur,
        "camera": {"fov_diag_deg": args.fov, "resolution": [render_w, render_h],
                   "elevation_deg": elevation, "elevation_source": esrc,
                   "distance_cm": distance, "distance_source": dsrc, "roll_deg": 15.0},
        "labels": {"tilt_deg": args.tilt_deg, "min_visible": args.min_visible},
        "infos": infos,
        "hints": hints,
    }
    with open(os.path.join(out, "scene.json"), "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=1, ensure_ascii=False)

    print(f"✔ prepared: {out}")
    print(f"  die {edge} mm, rounding {rounding} mm · tray {diameter:.0f} mm, wall {wall_height:.0f} mm")
    print(f"  elevation {elevation[0]}-{elevation[1]}° ({esrc}) · distance {distance[0]}-{distance[1]} cm ({dsrc})")
    print(f"  {len(sur)} surroundings photos · classes: {', '.join(classes)}")
    for s, i in infos.items():
        kind = {"bright": "top face found via brightness", "color": "top face found via colour",
                "refined": "old method (edges)", "mark": "only your mark (nothing found)"}[i["method"]]
        print(f"  face {s}: {kind}, edge {i['edge_px']} px" + (f" - {i['text']}" if i.get("text") else ""))
    for h in hints:
        print("  ⚠", h)


if __name__ == "__main__":
    main()
