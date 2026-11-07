"""Pokerdice Trainer - local web program (operated in the browser).

Guides you through everything that has to come from you: projects, setup, camera setup (feasibility check),
photos, capturing and labelling. The repository folder is found from the location of this file
(trainer/assistant/app.py -> two levels up); projects live in <repository>/workspace/projects/. No fixed paths in the code.

Example calls (from the repository folder):
  Windows:  .venv\\Scripts\\python.exe trainer\\assistant\\app.py
            .venv\\Scripts\\python.exe trainer\\assistant\\app.py --port 5050
            .venv\\Scripts\\python.exe trainer\\assistant\\app.py --no-browser
            .venv\\Scripts\\python.exe trainer\\assistant\\app.py --local-only     (not reachable from the phone)
            .venv\\Scripts\\python.exe trainer\\assistant\\app.py --workspace D:\\other\\workspace
  Linux/macOS: .venv/bin/python trainer/assistant/app.py
Or simply start.bat (Windows) / ./start.sh (Linux, macOS) in the repository folder.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import sys
import threading
import webbrowser

ASSISTANT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASSISTANT)

from flask import Flask, abort, jsonify, redirect, render_template, request, Response, url_for  # noqa: E402

import ai  # noqa: E402
import camera_setup as cs  # noqa: E402
import deploy  # noqa: E402
import device  # noqa: E402
import photos  # noqa: E402
import projects  # noqa: E402
import settings  # noqa: E402
import steps_dice  # noqa: E402
import steps_throws  # noqa: E402

VERSION = "1.0.0"   # trainer; the firmware has its own number (firmware/main/version.h)
IMAGE_NAME = re.compile(r"^\d{4,}\.jpg$")

TRAINER = os.path.dirname(ASSISTANT)
REPO = os.path.dirname(TRAINER)

app = Flask(__name__, template_folder=os.path.join(ASSISTANT, "templates"),
            static_folder=os.path.join(ASSISTANT, "static"))
app.config["WORKSPACE"] = os.path.join(REPO, "workspace")
app.config["TRAINER"] = TRAINER
app.config["REPO"] = REPO
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


def workspace():
    return app.config["WORKSPACE"]


def project_or_404(folder):
    if not re.match(r"^[A-Za-z0-9_\-]+$", folder or "") or not os.path.isfile(projects.path(workspace(), folder, "project.json")):
        abort(404, "Project not found")
    return projects.load(workspace(), folder)


@app.template_filter("path")
def join_path(base, *parts):
    """Path with the separator of this computer: {{ REPO|path("firmware") }} -> C:\\...\\firmware or /home/.../firmware."""
    return os.path.join(base, *[p for part in parts for p in part.split("/")])


@app.context_processor
def global_data():
    key, source = settings.api_key()
    e = settings.load()
    return {"VERSION": VERSION, "WORKSPACE": workspace(), "REPO": REPO, "sett": e,
            "CAMERA": photos.CAMERA, "CAMERA_TEXT": photos.CAMERA_TEXT, "api_key_present": bool(key),
            "api_key_source": source, "api_key_masked": settings.masked(key),
            "STEPS": projects.STEPS}


# ====================================================================== start / projects
@app.route("/")
def index():
    return render_template("index.html", projects=projects.list_projects(workspace()), parts=projects.PARTS,
                           error=request.args.get("error"), message=request.args.get("message"))


@app.route("/new", methods=["POST"])
def new():
    name = request.form.get("name", "")
    source = request.form.get("source") or None
    parts = request.form.getlist("parts") if source else []
    try:
        folder = projects.create(workspace(), name, source, parts)
    except ValueError as e:
        return redirect(url_for("index", error=str(e)))
    return redirect(url_for("project", folder=folder))


@app.route("/p/<folder>/")
def project(folder):
    d = project_or_404(folder)
    return render_template("project.html", p=d, folder=folder, active="overview",
                           project_path=projects.path(workspace(), folder), error=request.args.get("error"))


@app.route("/p/<folder>/step/<int:nr>")
def step(folder, nr):
    project_or_404(folder)
    if nr == 1:
        return redirect(url_for("camera_setup_page", folder=folder))
    if 2 <= nr <= 7:
        return redirect(url_for("step_a", folder=folder, nr=nr))
    if nr == 8:
        return redirect(url_for("throws_page", folder=folder))
    if nr == 9:
        return redirect(url_for("label_page", folder=folder))
    abort(404)


# ====================================================================== step 0: setup
@app.route("/setup", methods=["GET", "POST"])
def setup():
    message = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "key":
            k = request.form.get("api_key", "").strip()
            if k:
                settings.api_key_save(k)
                message = "Key saved."
        elif action == "key_delete":
            settings.api_key_delete()
            message = "Saved key deleted."
        elif action == "settings":
            data = {k: request.form.get(k, "").strip() for k in ("model", "device_address", "drive_folder")}
            try:
                data["fov_diag_deg"] = float(request.form.get("fov_diag_deg", "66").replace(",", "."))
            except ValueError:
                data["fov_diag_deg"] = 66.0
            if "//" in data["device_address"]:
                data["device_address"] = data["device_address"].split("//", 1)[1].strip("/")
            settings.save(data)
            message = "Settings saved."
    return render_template("setup.html", message=message, active="setup",
                           config_folder=settings.config_folder(), back=request.args.get("back"))


@app.route("/setup/api_test", methods=["POST"])
def api_test():
    try:
        return jsonify(ok=True, **ai.connection_test(settings.load()["model"]))
    except ai.AIError as e:
        return jsonify(ok=False, error=str(e))


@app.route("/setup/device_test", methods=["POST"])
def device_test():
    try:
        return jsonify(ok=True, info=device.info(settings.load()["device_address"]))
    except device.DeviceError as e:
        return jsonify(ok=False, error=str(e))


# ====================================================================== guides
@app.route("/guide/<name>")
def guide(name):
    if name not in ("setup", "firmware"):
        abort(404)
    return render_template(f"guide_{name}.html", active="guide_" + name)


# ====================================================================== step 1: camera setup
def _cs_folder(folder):
    return projects.path(workspace(), folder, "camera_setup")


def _evaluation_load(folder):
    p = os.path.join(_cs_folder(folder), "evaluation.json")
    try:
        with open(p, encoding="utf-8") as f:
            a = json.load(f)
    except (OSError, ValueError):
        a = {"version": 1, "images": {}}
    # add new image files, remove deleted ones
    present = sorted(n for n in os.listdir(_cs_folder(folder)) if IMAGE_NAME.match(n)) if os.path.isdir(_cs_folder(folder)) else []
    for n in present:
        a["images"].setdefault(n, {"use": True, "rating": {}, "source": "unknown"})
    for n in list(a["images"]):
        if n not in present:
            del a["images"][n]
    return a


def _evaluation_save(folder, a):
    p = os.path.join(_cs_folder(folder), "evaluation.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(a, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)


def _analyse(folder, name, d=None, info=None):
    """info: entry from evaluation.json (gives the hand-drawn boxes "manual" and tray points "tray_points")."""
    d = d or projects.load(workspace(), folder)
    info = info or {}
    img = cs.load_image(os.path.join(_cs_folder(folder), name))
    e = settings.load()
    return cs.analyse(img, tray_mm=(d.get("tray") or {}).get("diameter_mm"),
                      fov_diag_deg=float(e.get("fov_diag_deg", 66.0)),
                      param=d.get("surface"), manual=info.get("manual"),
                      die_mm=(d.get("die") or {}).get("edge_mm"),
                      tray_points=info.get("tray_points"))


def _analyse_all(folder, only_missing=True):
    d = projects.load(workspace(), folder)
    a = _evaluation_load(folder)
    for n, info in a["images"].items():
        if not only_missing or "analysis" not in info:
            try:
                info["analysis"] = _analyse(folder, n, d, info)
            except ValueError as ex:
                info["analysis"] = {"ok": False, "error": str(ex), "dice": []}
    _evaluation_save(folder, a)
    return a


def _new_image(folder, jpeg, source, extra=None):
    csf = _cs_folder(folder)
    name = projects.next_file_name(csf, ".jpg")
    projects.write_new(os.path.join(csf, name), jpeg)
    a = _evaluation_load(folder)
    a["images"][name] = {"use": True, "rating": {}, "source": source, "captured": projects.now(), **(extra or {})}
    a["images"][name]["analysis"] = _analyse(folder, name)
    _evaluation_save(folder, a)
    d = projects.load(workspace(), folder)
    if d["steps"]["1_camera_setup"]["status"] == "open":
        projects.set_step(d, "1_camera_setup", "in_progress")
        projects.save(workspace(), folder, d)
    return name


@app.route("/p/<folder>/camera-setup")
def camera_setup_page(folder):
    d = project_or_404(folder)
    a = _analyse_all(folder, only_missing=True)
    e = settings.load()
    try:
        stream = device.stream_url(e["device_address"]) if e["device_address"] else None
    except device.DeviceError:
        stream = None
    return render_template("camera_setup.html", p=d, folder=folder, active="1_camera_setup", step_id="1_camera_setup",
                           images=sorted(a["images"].items()), stream=stream,
                           folder_path=_cs_folder(folder), camera=projects.camera(d))


def _exposure_from_json(b):
    """{"level": -2..2} or {"aec": n, "gain": 0..30} checked; None = nothing given. Raises ValueError."""
    if not b:
        return None
    if "aec" in b:
        aec, gain = int(b["aec"]), int(b.get("gain", 0))
        if not (0 <= aec <= 65535 and 0 <= gain <= 30):
            raise ValueError("Fixed exposure: aec 0…65535, gain 0…30.")
        return {"aec": aec, "gain": gain}
    lv = int(b.get("level", 0))
    if not -2 <= lv <= 2:
        raise ValueError("Exposure correction: −2…+2.")
    return {"level": lv}


def _device_photo(folder, wanted, exposure, extra=None):
    """Fetch a photo from the camera board and save it. Returns (name, actual resolution, warning)."""
    jpeg, info = device.capture(settings.load()["device_address"], wanted, exposure)
    img = cs.image_from_bytes(jpeg)
    h, w = img.shape[:2]
    actual = f"{w}x{h}"
    data = {"resolution": actual, "resolution_wanted": wanted, "exposure": info["exposure"],
            "sensor": info.get("sensor"), **(extra or {})}
    name = _new_image(folder, jpeg, "device", data)
    warning = None
    if actual != wanted:
        warning = f"Wanted {wanted}, delivered {actual}. Is the Pokerdice firmware on the board up to date?"
    elif exposure and info["exposure"] is None:
        warning = "The camera board reports no exposure - is the Pokerdice firmware on the board up to date?"
    return name, actual, warning


@app.route("/p/<folder>/camera-setup/capture", methods=["POST"])
def cs_capture(folder):
    """Body (optional): {"resolution": "1024x768", "exposure": {"level": -1} or {"aec": 300, "gain": 0}}.
    Without anything: 640x480, exposure unchanged."""
    project_or_404(folder)
    j = request.get_json(silent=True) or {}
    try:
        name, actual, warning = _device_photo(folder, j.get("resolution") or "640x480", _exposure_from_json(j.get("exposure")))
    except (device.DeviceError, ValueError, TypeError) as e:
        return jsonify(ok=False, error=str(e))
    return jsonify(ok=True, name=name, resolution=actual, warning=warning)


@app.route("/p/<folder>/camera-setup/series", methods=["POST"])
def cs_series(folder):
    """Exposure series: 5 photos in a row with correction −2, −1, 0, +1, +2 (do not move the dice!).
    Body: {"resolution": "...", "after": {exposure for the live view afterwards}}."""
    project_or_404(folder)
    j = request.get_json(silent=True) or {}
    wanted = j.get("resolution") or "640x480"
    series = projects.now()
    names, warning = [], None
    try:
        after = _exposure_from_json(j.get("after")) or {"level": 0}
        for lv in (-2, -1, 0, 1, 2):
            name, actual, w = _device_photo(folder, wanted, {"level": lv}, {"series": series})
            names.append(name)
            if w:
                warning = w
                break
        # put the live view back to the chosen exposure (the photo for it is not saved)
        device.capture(settings.load()["device_address"], wanted, after)
    except (device.DeviceError, ValueError, TypeError) as e:
        return jsonify(ok=bool(names), names=names, error=str(e), name=names[0] if names else None)
    return jsonify(ok=True, names=names, name=names[0], warning=warning)


@app.route("/p/<folder>/camera-setup/upload", methods=["POST"])
def cs_upload(folder):
    project_or_404(folder)
    names = []
    for f in request.files.getlist("images"):
        data = f.read()
        if not data:
            continue
        try:
            img = cs.image_from_bytes(data)
        except ValueError:
            continue
        jpeg = data if data.startswith(b"\xff\xd8") else cs.jpeg_bytes(img, 95)
        names.append(_new_image(folder, jpeg, "uploaded"))
    return redirect(url_for("camera_setup_page", folder=folder) + ("#" + names[0] if names else ""))


@app.route("/p/<folder>/camera-setup/image/<name>")
def cs_image(folder, name):
    project_or_404(folder)
    if not IMAGE_NAME.match(name):
        abort(404)
    p = os.path.join(_cs_folder(folder), name)
    if not os.path.isfile(p):
        abort(404)
    kind = request.args.get("kind", "original")
    if kind == "original":
        with open(p, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    img = cs.load_image(p)
    a = _evaluation_load(folder)
    info = a["images"].get(name, {})
    res = info.get("analysis") or _analyse(folder, name, info=info)
    # marked/mask as JPEG (at 1600x1200 a PNG would be several MB), mosaic as PNG (small, pixels should stay exact)
    if kind == "marked":
        return Response(cs.jpeg_bytes(cs.marked(img, res), 90), mimetype="image/jpeg")
    if kind == "mask":
        out = cs.mask_image(img, projects.load(workspace(), folder).get("surface"), info.get("tray_points"))
        return Response(cs.jpeg_bytes(out, 90), mimetype="image/jpeg")
    if kind == "mosaic":
        cls = request.args.get("cls", type=int)
        cls = min(max(cls, 16), 160) if cls else None
        return Response(cs.png_bytes(cs.mosaic(img, res, tile=int(request.args.get("tile", 150)), cls=cls)), mimetype="image/png")
    abort(404)


@app.route("/p/<folder>/camera-setup/rating", methods=["POST"])
def cs_rating(folder):
    project_or_404(folder)
    j = request.get_json(force=True)
    name = j.get("image", "")
    if not IMAGE_NAME.match(name):
        abort(400)
    a = _evaluation_load(folder)
    info = a["images"].get(name)
    if info is None:
        abort(404)
    if "use" in j:
        info["use"] = bool(j["use"])
    if "all" in j:
        for f in (info.get("analysis") or {}).get("dice", []):
            info["rating"][str(f["nr"])] = j["all"]
    if "die" in j:
        value = j.get("value")
        if value in cs.RATINGS + ("wrong",):
            info["rating"][str(j["die"])] = value
        else:
            info["rating"].pop(str(j["die"]), None)
    _evaluation_save(folder, a)
    return jsonify(ok=True, rating=info["rating"], use=info["use"])


@app.route("/p/<folder>/camera-setup/ai/<name>", methods=["POST"])
def cs_ai(folder, name):
    project_or_404(folder)
    if not IMAGE_NAME.match(name):
        abort(400)
    a = _evaluation_load(folder)
    info = a["images"].get(name)
    if info is None:
        abort(404)
    p = os.path.join(_cs_folder(folder), name)
    img = cs.load_image(p)
    res = info.get("analysis") or _analyse(folder, name, info=info)
    with open(p, "rb") as f:
        original = f.read()
    mos = cs.jpeg_bytes(cs.mosaic(img, res, tile=160, pixelated=True), 92)
    try:
        info["ai"] = ai.assess_camera_setup(settings.load()["model"], original, mos, len(res.get("dice", [])),
                                            features=steps_dice.features_text(projects.load(workspace(), folder)))
    except ai.AIError as e:
        return jsonify(ok=False, error=str(e))
    _evaluation_save(folder, a)
    return jsonify(ok=True, ai=info["ai"])


@app.route("/p/<folder>/camera-setup/tray", methods=["POST"])
def cs_tray(folder):
    """Save the tray diameter and the die width (edge length, the same field as later in step 2)."""
    d = project_or_404(folder)

    def number(name, lo, hi):
        txt = request.form.get(name, "").replace(",", ".").strip()
        if not txt:
            return None
        try:
            w = float(txt)
        except ValueError:
            return "error"
        return w if lo <= w <= hi else "error"

    t = number("diameter_mm", 50, 1000)
    wm = number("die_mm", 5, 50)
    if t != "error":
        d["tray"]["diameter_mm"] = t
    if wm != "error":
        d.setdefault("die", {})["edge_mm"] = wm
    projects.save(workspace(), folder, d)
    _analyse_all(folder, only_missing=False)   # recompute distances
    return redirect(url_for("camera_setup_page", folder=folder))


@app.route("/p/<folder>/camera-setup/hints", methods=["POST"])
def cs_hints(folder):
    d = project_or_404(folder)
    d["symbol_hints"] = request.form.get("symbol_hints", "").strip()[:2000]
    projects.save(workspace(), folder, d)
    return redirect(url_for("camera_setup_page", folder=folder))


@app.route("/p/<folder>/camera-setup/reanalyse", methods=["POST"])
def cs_reanalyse(folder):
    project_or_404(folder)
    _analyse_all(folder, only_missing=False)
    return redirect(url_for("camera_setup_page", folder=folder))


# what a photo keeps from the capture itself; everything else is evaluation and is dropped on "start over"
CAPTURE_FIELDS = ("source", "captured", "resolution", "resolution_wanted", "exposure", "series", "sensor")


@app.route("/p/<folder>/camera-setup/start_over", methods=["POST"])
def cs_start_over(folder):
    """Evaluate all photos as if they had just been taken. The old evaluation.json moves to backup_<time>/."""
    project_or_404(folder)
    csf = _cs_folder(folder)
    p = os.path.join(csf, "evaluation.json")
    if os.path.isfile(p):
        old = os.path.join(csf, "backup_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(old, exist_ok=True)
        shutil.copy2(p, os.path.join(old, "evaluation.json"))
    a = _evaluation_load(folder)
    for n in list(a["images"]):
        old_info = a["images"][n]
        a["images"][n] = {"use": True, "rating": {}, **{k: old_info[k] for k in CAPTURE_FIELDS if k in old_info}}
        a["images"][n].setdefault("source", "unknown")
    _evaluation_save(folder, a)
    _analyse_all(folder, only_missing=True)
    return jsonify(ok=True, images=len(a["images"]))


def _entries(folder):
    a = _evaluation_load(folder)
    return [{"name": n, **i} for n, i in sorted(a["images"].items()) if "analysis" in i]


@app.route("/p/<folder>/camera-setup/summary")
def cs_summary(folder):
    d = project_or_404(folder)
    z = cs.summary(_entries(folder))
    z["camera_project"] = projects.camera(d) if d.get("camera") else None
    return jsonify(z)


def _measure_fixed_exposure(resolution, level):
    """Takes a photo with auto exposure (level), turns what the sensor reports into fixed values and checks them
    with a second photo at these fixed values. Returns (fixed or None, measurement for display)."""
    address = settings.load()["device_address"]
    _, info = device.capture(address, resolution, {"level": level})
    fixed = device.fixed_from_actual(info["exposure"])
    if not fixed:
        return None, {"error": "The camera board reported no exposure values."}
    _, info2 = device.capture(address, resolution, fixed)
    check = info2["exposure"] or {}
    m = {"auto": info["exposure"], "fixed": fixed, "check": check}
    a1, a2 = float(info["exposure"].get("actual_aec", 0)), float(check.get("actual_aec", -1))
    # the sensor should now report about the same exposure time as in auto mode
    m["ok"] = a2 >= 0 and abs(a2 - a1) <= max(8.0, 0.1 * a1)
    # back to auto for the live view
    device.capture(address, resolution, {"level": level})
    return (fixed if m["ok"] else None), m


@app.route("/p/<folder>/camera-setup/apply", methods=["POST"])
def cs_apply(folder):
    d = project_or_404(folder)
    entries = _entries(folder)
    z = cs.summary(entries)
    if z.get("status") in (None, "empty"):
        return jsonify(ok=False, error="Nothing to evaluate yet.")
    z["images"] = [e["name"] for e in entries if e.get("use", True)]
    z["applied"] = projects.now()
    z["tray_mm"] = d["tray"].get("diameter_mm")
    d["camera_setup"] = z
    # camera setting of the project (valid for test images, renders and firmware)
    j = request.get_json(silent=True) or {}
    res = j.get("resolution") or (z.get("resolution_recommended") or "").replace("×", "x") or projects.camera(d)["resolution"]
    if res not in projects.CAMERA_RESOLUTIONS:
        return jsonify(ok=False, error=f"Unknown resolution {res}.")
    try:
        level = int(j["exposure"]) if "exposure" in j else projects.camera(d)["exposure_level"]
    except (TypeError, ValueError):
        return jsonify(ok=False, error="Exposure invalid.")
    if not -2 <= level <= 2:
        return jsonify(ok=False, error="Exposure: −2 … +2.")
    fixed, measurement, warning = None, None, None
    if j.get("fixed"):
        try:
            fixed, measurement = _measure_fixed_exposure(res, level)
        except device.DeviceError as e:
            return jsonify(ok=False, error=str(e))
        if not fixed:
            warning = ("The fixed exposure could not be confirmed on the camera board (see the measurement) - "
                       "the project keeps auto exposure with the chosen correction.")
    d["camera"] = {"resolution": res, "exposure_level": level, "fixed_exposure": fixed, "set_at": projects.now()}
    if measurement:
        d["camera"]["measurement"] = measurement
    done = z["status"] in ("good", "borderline")
    projects.set_step(d, "1_camera_setup", "done" if done else "in_progress")
    projects.save(workspace(), folder, d)
    return jsonify(ok=True, done=done, status=z["status"], camera=d["camera"], warning=warning)


@app.route("/p/<folder>/camera-setup/surface", methods=["POST"])
def cs_surface(folder):
    """Measure the surface colour by click (image, x, y in pixels of the image) or reset it."""
    d = project_or_404(folder)
    j = request.get_json(force=True)
    if j.get("reset"):
        d["surface"] = None
    else:
        name = j.get("image", "")
        if not IMAGE_NAME.match(name):
            abort(400)
        img = cs.load_image(os.path.join(_cs_folder(folder), name))
        u = cs.surface_from_click(img, float(j["x"]), float(j["y"]))
        u["image"] = name
        u["measured"] = projects.now()
        d["surface"] = u
    projects.save(workspace(), folder, d)
    _analyse_all(folder, only_missing=False)
    return jsonify(ok=True, surface=d["surface"])


@app.route("/p/<folder>/camera-setup/manual", methods=["POST"])
def cs_manual(folder):
    """Die box by hand: {image, action:'new', x,y,w,h} or {image, action:'delete', nr}."""
    project_or_404(folder)
    j = request.get_json(force=True)
    name = j.get("image", "")
    if not IMAGE_NAME.match(name):
        abort(400)
    a = _evaluation_load(folder)
    info = a["images"].get(name)
    if info is None:
        abort(404)
    boxes = info.setdefault("manual", [])
    if j.get("action") == "new":
        x, y, w, h = [int(round(float(j[k]))) for k in ("x", "y", "w", "h")]
        if w < 5 or h < 5:
            return jsonify(ok=False, error="Box too small.")
        boxes.append({"x": max(0, x), "y": max(0, y), "w": w, "h": h})
    elif j.get("action") == "delete":
        idx = int(j.get("nr", 0)) - 101
        if 0 <= idx < len(boxes):
            boxes.pop(idx)
            # ratings of the following hand boxes move up by one number
            rat = info.get("rating", {})
            new = {k: v for k, v in rat.items() if int(k) < 101 + idx}
            for k, v in rat.items():
                if int(k) > 101 + idx:
                    new[str(int(k) - 1)] = v
            info["rating"] = new
    info["analysis"] = _analyse(folder, name, info=info)
    _evaluation_save(folder, a)
    return jsonify(ok=True)


@app.route("/p/<folder>/camera-setup/tray_points", methods=["POST"])
def cs_tray_points(folder):
    """Tray by hand: {image, points:[[x,y],...]} (at least 5 points on the tray edge, image pixels) or {image, points:[]} = discard."""
    project_or_404(folder)
    j = request.get_json(force=True)
    name = j.get("image", "")
    if not IMAGE_NAME.match(name):
        abort(400)
    a = _evaluation_load(folder)
    info = a["images"].get(name)
    if info is None:
        abort(404)
    try:
        points = [[round(float(x), 1), round(float(y), 1)] for x, y in (j.get("points") or [])][:200]
    except (TypeError, ValueError):
        return jsonify(ok=False, error="Points not readable.")
    if points and len(points) < 5:
        return jsonify(ok=False, error="Please set at least 5 points on the tray edge (spread all around).")
    if points:
        info["tray_points"] = points
    else:
        info.pop("tray_points", None)
    info["analysis"] = _analyse(folder, name, info=info)
    _evaluation_save(folder, a)
    t = info["analysis"].get("tray") or {}
    return jsonify(ok=True, source=t.get("source"), elevation_deg=info["analysis"].get("elevation_deg"),
                   distance_cm=info["analysis"].get("distance_cm"))


@app.route("/p/<folder>/camera-setup/delete/<name>", methods=["POST"])
def cs_photo_delete(folder, name):
    """Delete a photo for good (the page asks for confirmation first)."""
    project_or_404(folder)
    if not IMAGE_NAME.match(name):
        abort(400)
    p = os.path.join(_cs_folder(folder), name)
    if os.path.isfile(p):
        os.remove(p)
    a = _evaluation_load(folder)   # removes the entry of the deleted photo
    _evaluation_save(folder, a)
    return jsonify(ok=True)


def _rmtree_error(func, path, info):
    # Windows: make read-only files writable first, then try again
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


@app.route("/p/<folder>/delete", methods=["POST"])
def project_delete(folder):
    """Delete the whole project for good. Safety: the project name must be typed in."""
    d = project_or_404(folder)
    typed = request.form.get("confirmation", "").strip()
    if typed not in (d.get("name"), folder):
        return redirect(url_for("project", folder=folder, error="Name does not match - nothing deleted."))
    target = os.path.realpath(projects.path(workspace(), folder))
    root = os.path.realpath(projects.projects_folder(workspace()))
    if os.path.dirname(target) != root:   # only direct subfolders of projects/ may be deleted
        abort(400)
    shutil.rmtree(target, onerror=_rmtree_error)
    return redirect(url_for("index", message=f"Project “{d.get('name')}” was deleted."))


steps_dice.register(app, workspace, project_or_404)
steps_throws.register(app, workspace, project_or_404)
deploy.register(app, workspace, project_or_404)


# ====================================================================== error pages
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", text=getattr(e, "description", "Not found")), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", text=f"Internal error: {e}"), 500


# ====================================================================== start
def main():
    ap = argparse.ArgumentParser(
        description="Pokerdice Trainer (local web program).",
        epilog="Example: .venv\\Scripts\\python.exe trainer\\assistant\\app.py --port 5050",
    )
    ap.add_argument("--workspace", help="workspace folder with projects/ (default: workspace in the repository)")
    ap.add_argument("--port", type=int, default=5000, help="port (default 5000)")
    ap.add_argument("--local-only", action="store_true",
                    help="only reachable from this computer (then the phone upload via QR code does not work)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    ap.add_argument("--debug", action="store_true", help="Flask debug mode (reloads on code changes)")
    args = ap.parse_args()

    if args.workspace:
        app.config["WORKSPACE"] = os.path.abspath(args.workspace)
    os.makedirs(projects.projects_folder(workspace()), exist_ok=True)

    host = "127.0.0.1" if args.local_only else "0.0.0.0"
    app.config["HOST"], app.config["PORT"] = host, args.port
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Pokerdice Trainer {VERSION}")
    print(f"  Workspace (projects): {workspace()}")
    print(f"  Settings/key: {settings.config_folder()}")
    print(f"  In the browser: {url}   (quit: Ctrl+C)")
    if host == "0.0.0.0":
        print(f"  For the phone in the same Wi-Fi: http://{photos.lan_ip()}:{args.port}/  (Windows firewall the first time: “Allow”)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=args.port, debug=args.debug, use_reloader=args.debug, threaded=True)


if __name__ == "__main__":
    main()
