"""Step 1 "Camera setup": evaluation of the sample photos from the camera board (without AI).

Per image:
1. Find the felt: all pixels with a green hue (HSV) -> largest connected area.
2. Inner area of the tray = convex hull of the felt (the dice lie in it as "holes").
3. Dice = everything non-green and bright inside the hull, connected spots.
4. Elevation: the round tray appears as an ellipse. Ratio short/long axis ≈ sin(elevation).
   (Rough approximation - the perspective distorts a little.)
5. Distance (only if the tray diameter is known): from the ellipse width and the field of view of the camera.
6. Tray by hand: if at least 5 points on the tray edge were clicked, the ellipse computed from them
   is used (instead of the automatically found felt area).

Images may have any resolution (640×480 to 1600×1200). Sizes for the detector are converted to
a 640×480 image; die sizes in "real" camera pixels are listed per resolution in the summary.

All values are estimates for planning, not measurements to the millimetre.
"""
import math

import cv2
import numpy as np

DEFAULT_PARAM = {
    "h_min": 35,      # hue green (OpenCV scale 0..180)
    "h_max": 95,
    "s_min": 45,      # minimum saturation for "green"
    "v_min": 25,      # minimum brightness for "green"
    "v_die": 70,      # dice must be at least this bright (so shadows do not become dice)
    "min_area_share": 0.00012,  # smallest die spot relative to the image area (640x480 -> ~37 px)
}

DETECTOR_SIZES = [224, 256, 288, 320]
# Rules of thumb (no laws of nature - checked later against real results):
DET_MIN_GOOD = 14     # smallest die side in the detector input from which it is relaxed
DET_MIN_LIMIT = 10    # below this it gets very tight for the finest grid (stride 8)
CROP_MARGIN = 0.25    # margin all around the classifier crop (share of the die size)
REFERENCE_WIDTH = 640  # sizes for the detector are given for a 640x480 image
OUTLIERS = (0.5, 2.0)  # distances outside 0.5…2× of the median count as outliers

RATINGS = ("readable", "unsure", "no")   # your judgement per die; "wrong" = no die / several


# ------------------------------------------------------------------ load / save (Windows safe)
def load_image(path):
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not readable: {path}")
    return img


def image_from_bytes(b):
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image data not readable")
    return img


def jpeg_bytes(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _odd(x):
    x = max(1, int(round(x)))
    return x if x % 2 else x + 1


def _kernel(n, w):
    """Morphology kernel that grows with the image size (at 640 px width: n×n)."""
    k = _odd(n * max(1.0, w / 640.0))
    return np.ones((k, k), np.uint8)


def width_43(w, h):
    """Width of the full 4:3 sensor image in pixels of this capture.
    1280×1024 (5:4) cuts a little off left/right, the height stays full -> h·4/3.
    Wider images (e.g. 16:9 from a phone) cut top/bottom -> w."""
    return h * 4.0 / 3.0 if w / float(h) < 4.0 / 3.0 - 0.01 else float(w)


def to_640(px, w, h):
    """Die size converted to a 640×480 image (same field of view assumed)."""
    return px * REFERENCE_WIDTH / width_43(w, h)


# ------------------------------------------------------------------ masks
def _masks(bgr, p):
    """Returns (hsv, surface_raw, surface_smoothed). "green" means the colour of the surface."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_min, h_max = int(p["h_min"]) % 180, int(p["h_max"]) % 180
    if h_min <= h_max:
        green = cv2.inRange(hsv, (h_min, p["s_min"], p["v_min"]), (h_max, 255, 255))
    else:  # range wraps around 180/0 (colour circle)
        green = cv2.bitwise_or(cv2.inRange(hsv, (h_min, p["s_min"], p["v_min"]), (179, 255, 255)),
                               cv2.inRange(hsv, (0, p["s_min"], p["v_min"]), (h_max, 255, 255)))
    w = bgr.shape[1]
    green_smooth = cv2.morphologyEx(green, cv2.MORPH_OPEN, _kernel(3, w))
    green_smooth = cv2.morphologyEx(green_smooth, cv2.MORPH_CLOSE, _kernel(5, w))
    return hsv, green, green_smooth


def surface_from_click(bgr, x, y, radius=None):
    """Measures the colour of the surface around the clicked point and builds the "sieve" from it.
    Tolerances: hue ±14 (of 180), saturation from 40 % of the measured value, brightness from 35 %."""
    h, w = bgr.shape[:2]
    if radius is None:
        radius = int(round(8 * max(1.0, w / 640.0)))   # same area on the cloth as at 640x480
    x, y = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    field = hsv[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].reshape(-1, 3).astype(float)
    # the hue is an angle -> average over the circle
    angle = field[:, 0] * (2 * math.pi / 180.0)
    fh = (math.degrees(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) / 2.0) % 180
    fs, fv = float(np.median(field[:, 1])), float(np.median(field[:, 2]))
    tol = 14
    return {
        "color_hsv": [round(fh, 1), round(fs, 1), round(fv, 1)],
        "color_bgr": [int(c) for c in np.median(bgr[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].reshape(-1, 3), axis=0)],
        "h_min": int(round(fh - tol)) % 180,
        "h_max": int(round(fh + tol)) % 180,
        "s_min": int(max(20, fs * 0.4)),
        "v_min": int(max(12, fv * 0.35)),
        # dice must be clearly brighter than the surface
        "v_die": int(min(130, max(50, fv + 35))),
        "clicked": {"x": x, "y": y},
    }


OVEREXPOSED_LIMIT = 250   # all three colour channels from this value = burnt out (drawing lost)


def _quality(bgr, hsv, spots, v_die=70):
    """Brightness (0-100 %), sharpness (Laplace variance: the larger, the sharper; only for comparing) and
    "overexposed": share of the bright die pixels where all three colours are ≥ 250 (no drawing left there)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    q = {"brightness_percent": round(float(hsv[:, :, 2].mean()) / 2.55, 1),
         "sharpness_image": round(float(lap.var()), 1), "sharpness_dice": None, "brightness_dice_percent": None,
         "overexposed_percent": None, "overexposed_max_percent": None}
    values, bright, over = [], [], []
    burnt = (bgr.min(axis=2) >= OVEREXPOSED_LIMIT)
    for f in spots:
        if f.get("several"):
            continue
        roi = lap[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"]]
        if roi.size >= 25:
            values.append(float(roi.var()))
            bright.append(float(hsv[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"], 2].mean()) / 2.55)
        # only count the bright pixels in the box (die face), not the felt in the corners
        area = hsv[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"], 2] >= v_die
        if area.sum() >= 10:
            share = float(burnt[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"]][area].mean())
            f["overexposed"] = round(share, 3)
            over.append(share)
    if values:
        q["sharpness_dice"] = round(float(np.median(values)), 1)
        q["brightness_dice_percent"] = round(float(np.median(bright)), 1)
    if over:
        q["overexposed_percent"] = round(100 * float(np.median(over)), 1)
        q["overexposed_max_percent"] = round(100 * max(over), 1)
    return q


def _spot_values(f, elevation):
    f["smallest_side"] = min(f["w"], f["h"])
    f["largest_side"] = max(f["w"], f["h"])
    if elevation:
        f["top_height_px"] = round(f["w"] * math.sin(math.radians(elevation)), 1)
    return f


# Touching dice are recognised by their shape, not by their size (dice in front are clearly larger because of the
# perspective, half hidden ones at the tray wall smaller). Measured on 10 photos (1024×768):
# merged spots fill 0.37-0.52, single dice 0.60-0.86.
SEVERAL_FILL = 0.56       # below: indentations -> several dice
SEVERAL_LONG = 1.6        # aspect ratio: two dice side by side ...
SEVERAL_LONG_AREA = 1.4   # ... but only if the spot is also clearly larger than a typical one (half hidden ones are small)


def _several(f, spots):
    if f.get("fill", 1.0) < SEVERAL_FILL:
        return True
    long = max(f["w"], f["h"]) / float(max(1, min(f["w"], f["h"])))
    if long >= SEVERAL_LONG and len(spots) >= 2:
        typical = float(np.median([g["area"] for g in spots]))
        return f["area"] > SEVERAL_LONG_AREA * typical
    return False


def _f_px(w, h, fov_diag_deg):
    """Focal length in pixels from the diagonal field of view (pinhole approximation).
    The field of view applies to the full 4:3 image; for 5:4 (1280×1024) the diagonal of the full image counts."""
    b = width_43(w, h)
    return (math.hypot(b, b * 0.75) / 2.0) / math.tan(math.radians(fov_diag_deg) / 2.0)


def _ellipse_data(ell, w, h, tray_mm, fov_diag_deg):
    (cx, cy), (a1, a2), angle = ell
    big, small = max(a1, a2), min(a1, a2)
    if big <= 0:
        return None, None, None
    elev = round(math.degrees(math.asin(min(1.0, small / big))), 1)
    e = {"center": [round(cx, 1), round(cy, 1)], "axes": [round(a1, 1), round(a2, 1)],
         "angle": round(angle, 1), "long_px": round(big, 1), "short_px": round(small, 1)}
    dist = round(_f_px(w, h, fov_diag_deg) * float(tray_mm) / big / 10.0, 1) if tray_mm else None
    return e, elev, dist


def analyse(bgr, tray_mm=None, fov_diag_deg=66.0, param=None, manual=None, die_mm=None, tray_points=None):
    """manual: list of boxes {x,y,w,h} you drew by hand (they get numbers from 101).
    tray_points: list [[x,y], ...] on the tray edge (at least 5) - then this ellipse is used instead of the automatic one."""
    p = dict(DEFAULT_PARAM)
    if param:
        p.update({k: v for k, v in param.items() if k in DEFAULT_PARAM})
    h, w = bgr.shape[:2]
    res = {"ok": False, "image": [w, h], "dice": [], "hints": [], "tray": None,
           "elevation_deg": None, "distance_cm": None, "distance_die_cm": None}
    hand = None
    if tray_points and len(tray_points) >= 5:
        pts = np.array([[float(x), float(y)] for x, y in tray_points], np.float32)
        try:
            hand = cv2.fitEllipse(pts)
        except cv2.error:
            hand = None
        if hand is not None and (not all(math.isfinite(v) for v in (*hand[0], *hand[1])) or min(hand[1]) < 4):
            hand = None
        if hand is None:
            res["hints"].append("No ellipse could be computed from the marked tray points - automatic detection used.")

    hsv, green_raw, green = _masks(bgr, p)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    felt_share = 0.0
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        felt = (lab == i).astype(np.uint8) * 255
        felt_share = stats[i, cv2.CC_STAT_AREA] / float(w * h)
    if hand is None:
        if n <= 1:
            res["error"] = ("No surface in the set colour found. Tip: “Pick surface colour” and click on the cloth/felt, "
                            "or “Mark tray”.")
            return res
        if felt_share < 0.03:
            res["error"] = (f"Area in surface colour too small ({felt_share:.1%} of the image). Is the tray in the picture? "
                            "Otherwise “Pick surface colour” and click on the cloth/felt, or “Mark tray”.")
            return res

    tray = np.zeros((h, w), np.uint8)
    if hand is not None:
        # tray by hand: filled ellipse = tray area; the edge was set on purpose -> not "cut off"
        cv2.ellipse(tray, hand, 255, -1)
        cut_off = False
        ellipse, elev, dist = _ellipse_data(hand, w, h, tray_mm, fov_diag_deg)
        ellipse["points"] = [[round(float(x), 1), round(float(y), 1)] for x, y in tray_points]
        source = "hand"
    else:
        contours, _ = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(tray, hull, 255)
        # tray cut off? (hull touches the image border)
        hp = hull.reshape(-1, 2)
        border = max(3, int(round(3 * w / 640.0)))
        cut_off = bool((hp[:, 0] <= border).any() or (hp[:, 0] >= w - 1 - border).any()
                       or (hp[:, 1] <= border).any() or (hp[:, 1] >= h - 1 - border).any())
        ellipse, elev, dist = (_ellipse_data(cv2.fitEllipse(hull), w, h, tray_mm, fov_diag_deg)
                               if len(hp) >= 5 else (None, None, None))
        source = "auto"
    tray_inner = cv2.erode(tray, _kernel(5, w))
    res["elevation_deg"], res["distance_cm"] = elev, dist
    res["tray"] = {"felt_share": round(felt_share, 3), "cut_off": cut_off, "ellipse": ellipse, "source": source}
    if cut_off:
        # without the complete tray edge the ellipse is a guess -> show no number instead of a wrong one
        res["elevation_deg"] = None
        res["distance_cm"] = None
        res["hints"].append("Tray/surface reaches the image border - elevation and distance cannot be estimated "
                            "(or “Mark tray” if the edge is in the picture after all).")

    # Dice: not green, bright enough, inside the tray.
    # The brightness limit adapts to darker images (e.g. exposure -2): at most 1.5× the felt brightness
    # (at least felt + 15). For normally exposed images the set value stays.
    if felt_share > 0:
        v_felt = float(np.median(hsv[:, :, 2][felt > 0]))
        p["v_die"] = int(min(p["v_die"], max(30, v_felt * 1.5, v_felt + 15)))
    res["v_die"] = p["v_die"]
    bright = (hsv[:, :, 2] >= p["v_die"]).astype(np.uint8) * 255
    not_green = cv2.bitwise_and(cv2.bitwise_not(green_raw), tray_inner)
    cand = cv2.bitwise_and(not_green, bright)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _kernel(3, w))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _kernel(5, w))  # fill dark symbols
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    min_area = max(20, int(p["min_area_share"] * w * h))
    spots = []
    for j in range(1, n2):
        x, y, bw, bh, ar = [int(v) for v in st2[j]]
        if ar < min_area or ar > 0.15 * w * h:   # crumbs or huge areas (hand, lamp) are no dice
            continue
        # shape: fill (area/box) and compactness (area/convex hull) - touching dice have indentations
        mask_j = (lab2[y:y + bh, x:x + bw] == j).astype(np.uint8)
        cont, _ = cv2.findContours(mask_j, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(max(cont, key=cv2.contourArea))) if cont else 0
        spots.append({"x": x, "y": y, "w": bw, "h": bh, "area": ar,
                      "fill": round(ar / float(bw * bh), 3),
                      "compact": round(min(1.0, ar / hull_area), 3) if hull_area > 0 else 1.0})
    # discard crumbs: the measure are the 5 largest spots (5 dice lie in the tray);
    # what is smaller than 1/5 of that is a false hit (edge, reflection, dust).
    discarded = 0
    if len(spots) >= 2:
        largest = sorted((f["area"] for f in spots), reverse=True)[:5]
        measure = float(np.median(largest))
        before = len(spots)
        spots = [f for f in spots if f["area"] >= 0.2 * measure]
        discarded = before - len(spots)
    spots.sort(key=lambda f: (f["x"] + f["w"] / 2.0))
    for k, f in enumerate(spots, 1):
        f["nr"] = k
        f["several"] = _several(f, spots)
        _spot_values(f, res["elevation_deg"])
    for k, m in enumerate(manual or [], 101):
        f = {"x": int(m["x"]), "y": int(m["y"]), "w": int(m["w"]), "h": int(m["h"]),
             "area": int(m["w"]) * int(m["h"]), "nr": k, "several": False, "manual": True}
        spots.append(_spot_values(f, res["elevation_deg"]))
    res["dice"] = spots
    res["quality"] = _quality(bgr, hsv, spots, p["v_die"])
    # distance via the die width: a die turned at an angle is up to ~1.4× wider in the image than its edge,
    # so the narrower dice count (25 % value of the widths). Rougher than via the tray.
    widths = [f["w"] for f in spots if not f.get("several")]
    if die_mm and widths:
        b25 = float(np.percentile(widths, 25))
        res["distance_die_cm"] = round(_f_px(w, h, fov_diag_deg) * float(die_mm) / b25 / 10.0, 1)
        res["px_per_mm"] = round(b25 / float(die_mm), 2)
    if discarded:
        res["hints"].append(f"{discarded} small spot(s) discarded as crumbs.")
    if any(f["several"] for f in spots):
        res["hints"].append("At least one spot is conspicuously large (orange) - probably dice touch there. "
                            "Tip: set the spot to “no die / several” and draw the dice one by one.")
    if len(spots) != 5:
        res["hints"].append(f"{len(spots)} spots found (5 dice expected). Please check in the image.")
    res["ok"] = True
    return res


# ------------------------------------------------------------------ images for display
def _text(img, text, org, size=0.5, color=(255, 255, 255)):
    d = max(1, int(round(size * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), d + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, d, cv2.LINE_AA)


def marked(bgr, res):
    b = bgr.copy()
    t = res.get("tray") or {}
    e = t.get("ellipse")
    s = max(1.0, b.shape[1] / 640.0)          # lines and text grow with the image size
    d = max(1, int(round(s)))
    if e:
        hand = t.get("source") == "hand"
        color = (255, 0, 255) if hand else (0, 220, 255)   # hand: magenta, automatic: yellow
        cv2.ellipse(b, ((e["center"][0], e["center"][1]), (e["axes"][0], e["axes"][1]), e["angle"]),
                    color, d, cv2.LINE_AA)
        for x, y in e.get("points") or []:
            cv2.circle(b, (int(round(x)), int(round(y))), 3 * d, color, -1, cv2.LINE_AA)
    for f in res.get("dice", []):
        color = (255, 220, 0) if f.get("manual") else ((0, 140, 255) if f.get("several") else (60, 60, 255))
        cv2.rectangle(b, (f["x"], f["y"]), (f["x"] + f["w"] - 1, f["y"] + f["h"] - 1), color, d)
        _text(b, str(f["nr"]), (f["x"], max(int(12 * s), f["y"] - 3 * d)), 0.5 * s, (255, 255, 255))
    return b


def mask_image(bgr, param=None, tray_points=None):
    """Shows what was recognised as felt (green) and as dice (red) - for troubleshooting."""
    p = dict(DEFAULT_PARAM)
    if param:
        p.update({k: v for k, v in param.items() if k in DEFAULT_PARAM})
    res = analyse(bgr, param=p, tray_points=tray_points)
    hsv, green_raw, green = _masks(bgr, p)
    b = (bgr * 0.45).astype(np.uint8)
    b[green > 0] = (b[green > 0] * 0.5 + np.array([0, 160, 0]) * 0.5).astype(np.uint8)
    for f in res.get("dice", []):
        roi = b[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"]]
        roi[:] = (roi * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    return b


def crop_box(f, img_w, img_h, margin=CROP_MARGIN):
    """Square crop around the die with a margin all around (as later for the classifier)."""
    side = max(f["w"], f["h"]) * (1.0 + 2.0 * margin)
    cx, cy = f["x"] + f["w"] / 2.0, f["y"] + f["h"] / 2.0
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    return x0, y0, s


def crop(bgr, f, margin=CROP_MARGIN):
    h, w = bgr.shape[:2]
    x0, y0, s = crop_box(f, w, h, margin)
    # fill the margin outside the image with gray 114 (like letterbox)
    pad = cv2.copyMakeBorder(bgr, s, s, s, s, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return pad[y0 + s:y0 + 2 * s, x0 + s:x0 + 2 * s]


def mosaic(bgr, res, tile=180, pixelated=True, cls=None):
    """Enlarged die crops side by side, numbered.
    pixelated=True: nearest neighbour - you see the real camera pixels (honest impression of the resolution).
    cls=64 etc.: "this is what the classifier sees" - crop first scaled bilinearly to cls×cls (like the firmware),
    then enlarged pixelated."""
    ws = res.get("dice", [])
    if not ws:
        b = np.full((60, 360, 3), 40, np.uint8)
        _text(b, "no dice found", (10, 35), 0.6)
        return b
    columns = min(5, len(ws))
    rows = int(math.ceil(len(ws) / float(columns)))
    foot = 22
    b = np.full((rows * (tile + foot), columns * tile, 3), 30, np.uint8)
    for k, f in enumerate(ws):
        a = crop(bgr, f)
        side = a.shape[0]
        if cls:
            a = cv2.resize(a, (cls, cls), interpolation=cv2.INTER_LINEAR)   # bilinear like the firmware
            a = cv2.resize(a, (tile, tile), interpolation=cv2.INTER_NEAREST)
        else:
            a = cv2.resize(a, (tile, tile), interpolation=cv2.INTER_NEAREST if pixelated else cv2.INTER_CUBIC)
        z, s = divmod(k, columns)
        y0, x0 = z * (tile + foot), s * tile
        b[y0:y0 + tile, x0:x0 + tile] = a
        cv2.rectangle(b, (x0, y0), (x0 + tile - 1, y0 + tile - 1), (80, 80, 80), 1)
        _text(b, str(f["nr"]), (x0 + 6, y0 + 22), 0.7, (0, 255, 255))
        # "W12%" = 12 % of the die face overexposed (pure white)
        if cls:
            below = f"{side}->{cls} px"
        else:
            below = f"{f['w']}x{f['h']} px" + (f" W{100 * f['overexposed']:.0f}%" if (f.get("overexposed") or 0) >= 0.005 else "")
        _text(b, below, (x0 + 6, y0 + tile + 16), 0.45 if tile >= 140 else 0.4)
    return b


# ------------------------------------------------------------------ summary over all images
def _perc(values, q):
    return float(np.percentile(np.array(values, float), q)) if values else None


def _to16(x):
    return int(16 * round(x / 16.0))


def _robust(values):
    """Median and range without outliers (outside 0.5…2× of the median). None if empty."""
    if not values:
        return None
    m = float(np.median(values))
    good = [v for v in values if OUTLIERS[0] * m <= v <= OUTLIERS[1] * m] or list(values)
    return {"min": min(good), "max": max(good), "median": round(float(np.median(good)), 1),
            "count": len(good), "outliers": len(values) - len(good)}


def _res_text(w, h):
    return f"{int(w)}×{int(h)}"


def exposure_text(b):
    """Short text and key for a saved exposure (from the X-Exposure header)."""
    if not b:
        return "unknown", "unknown"
    if b.get("mode") == "manual":
        return f"fixed aec {b.get('aec')}, gain {b.get('gain')}", f"m{b.get('aec')}_{b.get('gain')}"
    lv = int(b.get("level", 0))
    return (f"auto {lv:+d}".replace("-", "−") if lv else "auto 0"), f"a{lv}"


def _crop_px(largest_median):
    return max(32, min(96, _to16(largest_median * (1 + 2 * CROP_MARGIN))))


def summary(entries):
    """entries: list of dicts {name, analysis, rating{nr:value}, use, exposure, ai}.
    Returns key figures + recommendation (English texts).
    Detector values are converted to a 640×480 image (independent of the capture resolution);
    "per_resolution" shows the real camera pixels per capture resolution."""
    used = [e for e in entries if e.get("use", True) and e.get("analysis", {}).get("ok")]
    smallest, largest, angles, distance = [], [], [], []   # smallest/largest: converted to 640
    sharpness, brightness, distance_d = [], [], []
    ai_rat, ai_angles, ai_count = [], [], []
    rated = []  # (smallest_side in real pixels, value)
    over_all = []  # share of overexposed die pixels per die
    per = {}       # "WxH" -> collection
    pe = {}        # exposure -> collection
    for e in used:
        a = e["analysis"]
        bw, bh = (a.get("image") or [640, 480])[:2]
        key = f"{int(bw)}x{int(bh)}"
        r = per.setdefault(key, {"resolution": _res_text(bw, bh), "width": int(bw), "height": int(bh), "images": 0,
                                 "smallest": [], "largest": [], "rated": [], "ai": []})
        r["images"] += 1
        btext, bkey = exposure_text(e.get("exposure"))
        rb = pe.setdefault(bkey, {"exposure": btext, "images": 0, "over": [], "bright": [], "sharp": [],
                                  "rated": [], "ai": [], "actual_aec": [], "actual_gain": [],
                                  "order": (0 if bkey.startswith("a") else 1 if bkey.startswith("m") else 2,
                                            int(bkey[1:]) if bkey.startswith("a") else 0)})
        rb["images"] += 1
        eb = e.get("exposure") or {}
        if isinstance(eb.get("actual_aec"), (int, float)) and eb["actual_aec"] >= 0:
            rb["actual_aec"].append(eb["actual_aec"])
            rb["actual_gain"].append(eb.get("actual_gain") or 0)
        if a.get("elevation_deg") and not (a.get("tray") or {}).get("cut_off"):
            angles.append(a["elevation_deg"])
        if a.get("distance_cm") and not (a.get("tray") or {}).get("cut_off"):
            distance.append(a["distance_cm"])
        if a.get("distance_die_cm"):
            distance_d.append(a["distance_die_cm"])
        k = e.get("ai") or {}
        valid = {f["nr"] for f in a.get("dice", [])
                 if (e.get("rating") or {}).get(str(f["nr"])) != "wrong" and not f.get("several")}
        if k:
            if isinstance(k.get("elevation_deg"), (int, float)):
                ai_angles.append(float(k["elevation_deg"]))
            if isinstance(k.get("dice_in_image"), int):
                ai_count.append((e["name"], k["dice_in_image"], len(valid)))
            for kw in k.get("dice") or []:
                try:
                    nr = int(kw.get("nr"))
                except (TypeError, ValueError):
                    continue
                if nr in valid and kw.get("is_die", True) and kw.get("readability") in ("good", "partly", "no"):
                    ai_rat.append(kw["readability"])
                    r["ai"].append(kw["readability"])
                    rb["ai"].append(kw["readability"])
        q = a.get("quality") or {}
        if q.get("sharpness_dice") is not None:
            sharpness.append((q["sharpness_dice"], e["name"]))
            rb["sharp"].append(q["sharpness_dice"])
        if q.get("brightness_percent") is not None:
            brightness.append(q["brightness_percent"])
        if q.get("brightness_dice_percent") is not None:
            rb["bright"].append(q["brightness_dice_percent"])
        for f in a.get("dice", []):
            value = (e.get("rating") or {}).get(str(f["nr"]))
            if value == "wrong" or f.get("several"):
                continue  # no die or several together
            smallest.append(to_640(f["smallest_side"], bw, bh))
            largest.append(to_640(f["largest_side"], bw, bh))
            r["smallest"].append(f["smallest_side"])
            r["largest"].append(f["largest_side"])
            if f.get("overexposed") is not None:
                rb["over"].append(f["overexposed"])
                over_all.append(f["overexposed"])
            if value in RATINGS:
                rated.append((f["smallest_side"], value))
                r["rated"].append(value)
                rb["rated"].append(value)

    z = {"images_total": len(entries), "images_used": len(used), "dice_count": len(smallest)}
    if not smallest:
        z["status"] = "empty"
        z["recommendation"] = ["No usable dice yet. Please take photos (or leave images on “use”)."]
        return z

    # die size converted to 640×480
    z["dice_px"] = {"min": round(min(smallest), 1), "p10": round(_perc(smallest, 10), 1),
                    "median": round(_perc(smallest, 50), 1), "max": round(max(smallest), 1)}
    p10 = z["dice_px"]["p10"]
    z["detector"] = []
    for g in DETECTOR_SIZES:
        d = p10 * g / float(REFERENCE_WIDTH)
        level = "good" if d >= DET_MIN_GOOD else ("tight" if d >= DET_MIN_LIMIT else "too small")
        z["detector"].append({"imgsz": g, "die_px": round(d, 1), "level": level,
                              "compute_factor": round((g / 224.0) ** 2, 2)})

    # per capture resolution (real camera pixels)
    z["per_resolution"] = []
    for key, r in sorted(per.items(), key=lambda kv: kv[1]["width"] * kv[1]["height"]):
        if not r["smallest"]:
            z["per_resolution"].append({"resolution": r["resolution"], "key": key, "images": r["images"], "dice": 0})
            continue
        rr = r["rated"]
        kp10 = _perc(r["smallest"], 10)
        z["per_resolution"].append({
            "resolution": r["resolution"], "key": key, "images": r["images"], "dice": len(r["smallest"]),
            "px_median": round(_perc(r["smallest"], 50), 1), "px_p10": round(kp10, 1),
            "detector224_px": round(to_640(kp10, r["width"], r["height"]) * 224 / float(REFERENCE_WIDTH), 1),
            "crop_camera_px": int(round(_perc(r["largest"], 50) * (1 + 2 * CROP_MARGIN))),
            "rated": len(rr),
            "readable_share": round(rr.count("readable") / float(len(rr)), 3) if rr else None,
            "ai_rated": len(r["ai"]),
            "ai_share": round(r["ai"].count("good") / float(len(r["ai"])), 3) if r["ai"] else None,
        })

    # per exposure
    z["per_exposure"] = []
    for key, rb in sorted(pe.items(), key=lambda kv: kv[1]["order"]):
        rr = rb["rated"]
        z["per_exposure"].append({
            "exposure": rb["exposure"], "key": key, "images": rb["images"], "dice": len(rb["over"]),
            "overexposed_percent": round(100 * float(np.median(rb["over"])), 1) if rb["over"] else None,
            "overexposed_p90_percent": round(100 * _perc(rb["over"], 90), 1) if rb["over"] else None,
            "brightness_dice_percent": round(float(np.median(rb["bright"])), 1) if rb["bright"] else None,
            "sharpness_dice": round(float(np.median(rb["sharp"])), 1) if rb["sharp"] else None,
            "actual_aec": int(np.median(rb["actual_aec"])) if rb["actual_aec"] else None,
            "actual_gain": round(float(np.median(rb["actual_gain"])), 2) if rb["actual_gain"] else None,
            "rated": len(rr),
            "readable_share": round(rr.count("readable") / float(len(rr)), 3) if rr else None,
            "ai_rated": len(rb["ai"]),
            "ai_share": round(rb["ai"].count("good") / float(len(rb["ai"])), 3) if rb["ai"] else None,
        })
    z["overexposed_percent"] = round(100 * float(np.median(over_all)), 1) if over_all else None

    if angles:
        z["elevation_deg"] = {"min": min(angles), "max": max(angles), "source": "tray"}
    elif ai_angles:   # last resort: estimate of the AI
        z["elevation_deg"] = {"min": min(ai_angles), "max": max(ai_angles), "source": "AI estimate"}
    else:
        z["elevation_deg"] = None
    rt, rw = _robust(distance), _robust(distance_d)
    z["distance_cm"] = dict(rt, source="tray") if rt else None
    z["distance_die_cm"] = rw
    if not z["distance_cm"] and z["distance_die_cm"]:
        z["distance_cm"] = dict(z["distance_die_cm"], source="die width (rough)")
    # AI only for comparison (decides nothing)
    z["ai_rated"] = len(ai_rat)
    z["ai_determinable_share"] = round(ai_rat.count("good") / float(len(ai_rat)), 3) if ai_rat else None
    z["ai_count_mismatch"] = [{"image": b, "ai": kz, "found": g} for b, kz, g in ai_count if kz != g]
    if sharpness:
        sharpness.sort()
        z["sharpness_dice"] = {"min": sharpness[0][0], "min_image": sharpness[0][1],
                               "max": sharpness[-1][0], "max_image": sharpness[-1][1]}
    z["brightness_percent"] = {"min": min(brightness), "max": max(brightness)} if brightness else None

    # determinability (your judgement counts): readable / unsure / no
    z["rated"] = len(rated)
    if rated:
        cnt = {k: sum(1 for _, w in rated if w == k) for k in RATINGS}
        z["readability"] = cnt
        z["readable_share"] = round(cnt["readable"] / float(len(rated)), 3)
        # from which die size (real camera pixels) was everything readable?
        sizes = sorted(set(s for s, _ in rated))
        readable_from = None
        for s in sizes:
            if all(w == "readable" for g2, w in rated if g2 >= s):
                readable_from = s
                break
        z["readable_from_px"] = readable_from
    else:
        z["readable_share"] = None
        z["readable_from_px"] = None

    # recommendations
    rec = []
    good = [d for d in z["detector"] if d["level"] == "good"]
    tight = [d for d in z["detector"] if d["level"] == "tight"]
    if good:
        z["detector_imgsz"] = good[0]["imgsz"]
        rec.append(f"Detector input {good[0]['imgsz']}×{good[0]['imgsz']}: the smaller dice (10 % value) "
                   f"are about {good[0]['die_px']} px there - sufficient.")
    elif tight:
        z["detector_imgsz"] = tight[-1]["imgsz"]
        rec.append(f"Dice are tight in the detector input: best {tight[-1]['imgsz']}×{tight[-1]['imgsz']} "
                   f"(about {tight[-1]['die_px']} px, compute time ×{tight[-1]['compute_factor']}). "
                   "Better: hold the camera a little closer.")
    else:
        z["detector_imgsz"] = None
        rec.append("Dice are too small for the detector - please hold the camera closer and capture again.")
    if len(per) > 1:
        rec.append("The detector values do not depend on the camera resolution: the detector always gets the image scaled down to "
                   f"{DETECTOR_SIZES[0]}-{DETECTOR_SIZES[-1]} px. The resolution only decides how sharp the "
                   "die crop for the classifier is.")

    mg = _perc(largest, 50)
    z["crop_px"] = _crop_px(mg)
    higher = [r for r in z["per_resolution"] if r.get("dice") and r["key"] != "640x480"]
    rec.append(f"Classifier input: {z['crop_px']}×{z['crop_px']} px (die converted to 640×480 about {mg:.0f} px "
               "+ margin all around, rounded to multiples of 16)."
               + (" At a higher camera resolution the crop is larger in the camera image and is scaled down to this."
                  if higher else ""))

    # recommended camera resolution: the smallest where you rated ≥ 90 % as determinable (at least 5 dice rated)
    z["resolution_recommended"] = None
    rated_res = [r for r in z["per_resolution"] if r.get("rated", 0) >= 5]
    if rated_res:
        ok = [r for r in rated_res if r["readable_share"] >= 0.9]
        choice = ok[0] if ok else max(rated_res, key=lambda r: r["readable_share"])
        z["resolution_recommended"] = choice["resolution"]
        if len(rated_res) > 1:
            if ok:
                rec.append(f"Camera resolution: from {choice['resolution']} you rated {choice['readable_share']:.0%} of the dice as determinable "
                           f"(crop there about {choice['crop_camera_px']} px in the camera image).")
            else:
                rec.append(f"Camera resolution: best was {choice['resolution']} with {choice['readable_share']:.0%} determinable - "
                           "a higher resolution alone is not enough here, also check light/sharpness/distance.")
        if choice["resolution"] != "640×480" and len(rated_res) > 1:
            rec.append("Note: a higher resolution means more work per image on the board - "
                       "choose the smallest one that is good enough.")
    elif len(per) > 1:
        rec.append("For the resolution comparison please rate at least 5 dice per resolution as readable/unsure/no.")

    # recommended exposure: highest share "determinable" (at least 5 rated dice), on a tie less overexposed
    z["exposure_recommended"] = None
    exp_rated = [r for r in z["per_exposure"] if r["rated"] >= 5 and r["key"] != "unknown"]
    if exp_rated:
        choice = max(exp_rated, key=lambda r: (r["readable_share"], -(r["overexposed_percent"] or 0)))
        z["exposure_recommended"] = {k: choice[k] for k in ("exposure", "key", "actual_aec", "actual_gain",
                                                             "readable_share", "overexposed_percent")}
        if len(exp_rated) > 1:
            rec.append(f"Exposure: best determinable was “{choice['exposure']}” ({choice['readable_share']:.0%} determinable"
                       + (f", {choice['overexposed_percent']:.0f} % of the die face overexposed" if choice["overexposed_percent"] is not None else "")
                       + ").")
    elif len([r for r in z["per_exposure"] if r["key"] != "unknown"]) > 1:
        rec.append("For the exposure comparison please rate at least 5 dice per exposure (easiest with an exposure series).")
    if z["overexposed_percent"] is not None and z["overexposed_percent"] >= 5:
        rec.append(f"In the median {z['overexposed_percent']:.0f} % of the die face is overexposed (pure white, no drawing left). "
                   "Tip: take an exposure series; if −2 does not help, try a fixed exposure or make the light softer "
                   "(lamp indirect/with a shade, not in mirror direction to the camera).")

    if z["readable_share"] is None:
        rec.append("Please still tell for the dice whether you can determine the top symbol unambiguously (colour, pips/figure, drawing) - "
                   "this is the most important criterion.")
        z["status"] = "unrated"
    elif z["readable_share"] >= 0.9:
        rec.append(f"You rated {z['readable_share']:.0%} of the dice as unambiguously determinable - a good basis for the training.")
    elif z["readable_share"] >= 0.6:
        rec.append(f"Only {z['readable_share']:.0%} unambiguously determinable. What a human cannot determine for sure on the photo, "
                   "no model can either.")
    else:
        rec.append(f"Only {z['readable_share']:.0%} unambiguously determinable - first sort out light, sharpness or distance "
                   "before rendering and training.")
    if z.get("ai_determinable_share") is not None and z.get("readable_share") is not None:
        diff = z["readable_share"] - z["ai_determinable_share"]
        if diff >= 0.3:
            rec.append(f"You: {z['readable_share']:.0%} determinable, AI: {z['ai_determinable_share']:.0%}. Big difference - "
                       "probably you know it because you know the dice, but it is hardly visible. For the model what is visible counts "
                       "(more light/sharpness helps).")
    for name, r in (("tray", rt), ("die width", rw)):
        if r and r["outliers"]:
            rec.append(f"Distance via {name}: {r['outliers']} outlier image(s) not counted (outside 0.5…2× of the median).")
    if rt and rw:
        t, wv = rt["median"], rw["median"]
        if t > 0 and abs(t - wv) / t > 0.35:   # medians without outliers -> only real contradictions
            rec.append(f"Distance via tray (median ≈{t:.0f} cm) and via die width (median ≈{wv:.0f} cm) contradict each other - "
                       "please check the tray diameter and die width, or mark the tray by hand in one image.")
    if z.get("readable_from_px"):
        rec.append(f"From about {z['readable_from_px']} px die size (real camera pixels) everything was determinable - "
                   "this is the lower limit for the camera position.")

    if "status" not in z:
        det_ok = z["detector_imgsz"] is not None
        if det_ok and good and z["readable_share"] >= 0.9:
            z["status"] = "good"
        elif det_ok and z["readable_share"] >= 0.6:
            z["status"] = "borderline"
        else:
            z["status"] = "too small"
    z["recommendation"] = rec
    return z
