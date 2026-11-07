"""Phase D of the trainer: step 8 (capture real throws with the camera board) and step 9 (label).

Files in the project folder:
  raw/0001.jpg …                    unlabelled captures from the camera board (stay unchanged)
  raw/labels.json                   suggestions (boxes + AI) and confirmations per capture
  test_images/9_10_Q_K_A_0001.jpg   confirmed capture (copy), symbols in class order + number of the raw capture
  test_images/9_10_Q_K_A_0001.json  label: boxes (x,y,w,h in pixels) + symbol per die
  suggestions.csv                   overview of the suggestions (to look at in a spreadsheet)
"""
import csv
import json
import os
import re
import shutil

from flask import abort, jsonify, render_template, request, Response

import ai
import camera_setup as cs
import device
import local_model
import projects
import settings
import steps_dice

RAW_NAME = re.compile(r"^\d{4,}\.jpg$")
TARGET_MIN = 10      # at least this many confirmed throws for step 9 "done"
TARGET_MAX = 100


def raw_folder(workspace, folder):
    return projects.path(workspace, folder, "raw")


def test_folder(workspace, folder):
    return projects.path(workspace, folder, "test_images")


def raw_list(workspace, folder):
    o = raw_folder(workspace, folder)
    return sorted(n for n in os.listdir(o) if RAW_NAME.match(n)) if os.path.isdir(o) else []


def load_labels(workspace, folder):
    p = os.path.join(raw_folder(workspace, folder), "labels.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {"version": 1, "images": {}}
    present = set(raw_list(workspace, folder))
    for n in list(d["images"]):
        if n not in present:
            del d["images"][n]
    for n in present:
        d["images"].setdefault(n, {})
    return d


def save_labels(workspace, folder, d):
    o = raw_folder(workspace, folder)
    os.makedirs(o, exist_ok=True)
    p = os.path.join(o, "labels.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(p + ".tmp", p)


def write_csv(workspace, folder, d):
    p = projects.path(workspace, folder, "suggestions.csv")
    with open(p + ".tmp", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "nr", "x", "y", "w", "h", "several", "ai_symbol", "ai_sure", "confirmed_symbol"])
        for n, info in sorted(d["images"].items()):
            v = info.get("suggestion") or {}
            conf = {(b["x"], b["y"], b["w"], b["h"]): b["symbol"] for b in (info.get("confirmed") or {}).get("dice", [])}
            for r in v.get("boxes", []):
                w.writerow([n, r["nr"], r["x"], r["y"], r["w"], r["h"], int(r.get("several", False)), r.get("ai_symbol", ""),
                            "" if r.get("ai_sure") is None else int(r["ai_sure"]), conf.get((r["x"], r["y"], r["w"], r["h"]), "")])
    os.replace(p + ".tmp", p)


def suggest_boxes(workspace, folder, name, d_project):
    """Suggest boxes (and symbols):
    - with your own models from Colab if <project>/export/ is there (detector + classifier, like the firmware),
    - otherwise via the felt detection (like step 1, with the surface colour chosen there)."""
    img = cs.load_image(os.path.join(raw_folder(workspace, folder), name))
    notes = []
    ppath = projects.path(workspace, folder)
    ok, _ = local_model.available(ppath)
    if ok:
        try:
            kl = steps_dice.symbols(d_project)["classes"] or None
            m = local_model.suggest(ppath, img, kl)
            return {"boxes": m["boxes"], "notes": m["notes"], "error": None, "source": "model",
                    "model_version": m["version"], "created": projects.now()}
        except local_model.ModelError as e:
            notes.append(f"Own models not usable ({e}) - suggestion by colour.")
    res = cs.analyse(img, param=d_project.get("surface"), die_mm=(d_project.get("die") or {}).get("edge_mm"))
    boxes = [{"nr": f["nr"], "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"], "several": bool(f.get("several"))}
             for f in res.get("dice", [])]
    return {"boxes": boxes, "notes": notes + res.get("hints", []), "error": res.get("error"),
            "source": "color", "created": projects.now()}


def statistics(d):
    images = d["images"]
    total = len(images)
    conf = [i for i in images.values() if i.get("confirmed")]
    with_ai = sum(1 for i in images.values() if (i.get("suggestion") or {}).get("ai"))
    right = compared = m_right = m_compared = 0
    for i in conf:
        for w in i["confirmed"]["dice"]:
            if w.get("ai_symbol"):
                compared += 1
                right += int(w["ai_symbol"] == w["symbol"])
            if w.get("model_symbol"):
                m_compared += 1
                m_right += int(w["model_symbol"] == w["symbol"])
    syms = {}
    for i in conf:
        for w in i["confirmed"]["dice"]:
            syms[w["symbol"]] = syms.get(w["symbol"], 0) + 1
    return {"total": total, "confirmed": len(conf), "with_ai": with_ai,
            "ai_right": right, "ai_compared": compared,
            "ai_rate": round(right / float(compared), 3) if compared else None, "symbols": syms,
            "model_right": m_right, "model_compared": m_compared,
            "model_rate": round(m_right / float(m_compared), 3) if m_compared else None}


def test_file_name(syms, classes, raw_name):
    order = {k: i for i, k in enumerate(classes)}
    parts = sorted(syms, key=lambda s: order.get(s, 99))
    safe = [re.sub(r"[^\w\-]", "", s.replace("?", "X")) or "X" for s in parts]
    return ("_".join(safe) + "_" if safe else "empty_") + raw_name[:-4]


def confirm(workspace, folder, name, dice, classes):
    """Saves a copy + label in test_images/. An earlier confirmation of the same raw capture is replaced."""
    d = load_labels(workspace, folder)
    info = d["images"].get(name)
    if info is None:
        raise ValueError("Capture not found")
    to = test_folder(workspace, folder)
    os.makedirs(to, exist_ok=True)
    old = (info.get("confirmed") or {}).get("file")
    if old:
        for ext in (".jpg", ".json"):
            p = os.path.join(to, old + ext)
            if os.path.isfile(p):
                os.remove(p)
    clean = []
    for w in dice:
        sym = w.get("symbol")
        if sym not in classes:
            continue   # "no die" or invalid
        clean.append({"x": int(w["x"]), "y": int(w["y"]), "w": int(w["w"]), "h": int(w["h"]), "symbol": sym,
                      "class": classes.index(sym), "ai_symbol": w.get("ai_symbol") or None,
                      "model_symbol": w.get("model_symbol") or None, "source": w.get("source") or "auto"})
    base = test_file_name([w["symbol"] for w in clean], classes, name)
    shutil.copyfile(os.path.join(raw_folder(workspace, folder), name), os.path.join(to, base + ".jpg"))
    label = {"image": base + ".jpg", "raw": name, "size": jpeg_size(os.path.join(to, base + ".jpg")), "classes": classes,
             "dice": [{k: w[k] for k in ("x", "y", "w", "h", "symbol", "class")} for w in clean],
             "confirmed": projects.now()}
    with open(os.path.join(to, base + ".json"), "w", encoding="utf-8") as f:
        json.dump(label, f, indent=1, ensure_ascii=False)
    info["confirmed"] = {"file": base, "dice": clean, "at": projects.now()}
    save_labels(workspace, folder, d)
    write_csv(workspace, folder, d)
    return base


def jpeg_size(path):
    """[width, height] from the JPEG header (without decoding); [640, 480] if not readable."""
    try:
        with open(path, "rb") as f:
            data = f.read(65536)
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            if data[i + 1] in (0xC0, 0xC1, 0xC2):
                return [int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")]
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    except OSError:
        pass
    return [640, 480]


# ====================================================================== Flask
def register(app, workspace, project_or_404):

    def status_8_9(folder, d_project, d=None):
        d = d or load_labels(workspace(), folder)
        st = statistics(d)
        new8 = "done" if st["total"] >= TARGET_MIN else ("in_progress" if st["total"] else "open")
        new9 = "done" if st["confirmed"] >= TARGET_MIN and st["confirmed"] == st["total"] else \
            ("in_progress" if st["confirmed"] else "open")
        for sid, new in (("8_throws", new8), ("9_label", new9)):
            if d_project["steps"][sid]["status"] != new:
                projects.set_step(d_project, sid, new)
        projects.save(workspace(), folder, d_project)
        return st

    def classes_or_none(d_project):
        s = steps_dice.symbols(d_project)
        return s["classes"] if s["classes"] and s["confirmed"] else None

    # ------------------------------------------------------------ step 8
    @app.route("/p/<folder>/throws")
    def throws_page(folder):
        d = project_or_404(folder)
        e = settings.load()
        try:
            stream = device.stream_url(e["device_address"]) if e["device_address"] else None
        except device.DeviceError:
            stream = None
        st = status_8_9(folder, d)
        last = raw_list(workspace(), folder)[-12:][::-1]
        return render_template("step8_throws.html", p=d, folder=folder, active="8_throws", step_id="8_throws",
                               stream=stream, st=st, last=last, target_min=TARGET_MIN, target_max=TARGET_MAX,
                               raw_path=raw_folder(workspace(), folder), cs=d.get("camera_setup"), camera=projects.camera(d))

    @app.route("/p/<folder>/throws/capture", methods=["POST"])
    def throws_capture(folder):
        d = project_or_404(folder)
        try:
            k = projects.camera(d)   # test images like the firmware: resolution and exposure from step 1
            jpeg, _info = device.capture(settings.load()["device_address"], k["resolution"], projects.capture_exposure(d))
        except device.DeviceError as e:
            return jsonify(ok=False, error=str(e))
        o = raw_folder(workspace(), folder)
        name = projects.next_file_name(o, ".jpg")
        projects.write_new(os.path.join(o, name), jpeg)
        st = status_8_9(folder, d)
        return jsonify(ok=True, name=name, count=st["total"])

    @app.route("/p/<folder>/throws/upload", methods=["POST"])
    def throws_upload(folder):
        d = project_or_404(folder)
        o = raw_folder(workspace(), folder)
        names = []
        for f in request.files.getlist("images"):
            b = f.read()
            if not b.startswith(b"\xff\xd8"):
                continue
            name = projects.next_file_name(o, ".jpg")
            projects.write_new(os.path.join(o, name), b)
            names.append(name)
        status_8_9(folder, d)
        return jsonify(ok=True, names=names)

    @app.route("/p/<folder>/raw/<name>")
    def raw_image(folder, name):
        project_or_404(folder)
        if not RAW_NAME.match(name):
            abort(404)
        p = os.path.join(raw_folder(workspace(), folder), name)
        if not os.path.isfile(p):
            abort(404)
        g = request.args.get("g")
        if g:
            import cv2
            img = cs.load_image(p)
            f = int(g) / float(max(img.shape[:2]))
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            return Response(cs.jpeg_bytes(img, 80), mimetype="image/jpeg")
        with open(p, "rb") as fh:
            return Response(fh.read(), mimetype="image/jpeg")

    @app.route("/p/<folder>/raw/<name>/delete", methods=["POST"])
    def raw_delete(folder, name):
        d = project_or_404(folder)
        if not RAW_NAME.match(name):
            abort(400)
        dd = load_labels(workspace(), folder)
        old = (dd["images"].get(name, {}).get("confirmed") or {}).get("file")
        if old:
            for ext in (".jpg", ".json"):
                p = os.path.join(test_folder(workspace(), folder), old + ext)
                if os.path.isfile(p):
                    os.remove(p)
        p = os.path.join(raw_folder(workspace(), folder), name)
        if os.path.isfile(p):
            os.remove(p)
        dd = load_labels(workspace(), folder)
        save_labels(workspace(), folder, dd)
        status_8_9(folder, d, dd)
        return jsonify(ok=True)

    # ------------------------------------------------------------ step 9: overview
    @app.route("/p/<folder>/label")
    def label_page(folder):
        d = project_or_404(folder)
        dd = load_labels(workspace(), folder)
        st = status_8_9(folder, d, dd)
        images = sorted(dd["images"].items())
        model_ok, model_info = local_model.available(projects.path(workspace(), folder))
        return render_template("step9_label.html", model_ok=model_ok, model_info=model_info,
                               p=d, folder=folder, active="9_label", step_id="9_label",
                               st=st, images=images, classes=classes_or_none(d), target_min=TARGET_MIN, target_max=TARGET_MAX,
                               without_ai=[n for n, i in images if not (i.get("suggestion") or {}).get("ai") and not i.get("confirmed")],
                               test_path=test_folder(workspace(), folder))

    # ------------------------------------------------------------ step 9: suggestions
    def get_suggestion(folder, name, d_project, dd, with_ai):
        info = dd["images"][name]
        v = info.get("suggestion")
        # new or first available own models -> new suggestion (only for captures not confirmed yet;
        # an old Claude suggestion then no longer matches the box numbers and is dropped)
        if v and not info.get("confirmed"):
            ok, ver = local_model.available(projects.path(workspace(), folder))
            if ok and v.get("model_version") != ver:
                v = None
        if not v:
            v = suggest_boxes(workspace(), folder, name, d_project)
            info["suggestion"] = v
        if with_ai and not v.get("ai") and v["boxes"]:
            kl = classes_or_none(d_project)
            p = os.path.join(raw_folder(workspace(), folder), name)
            img = cs.load_image(p)
            res = {"dice": [dict(f, smallest_side=min(f["w"], f["h"]), largest_side=max(f["w"], f["h"]))
                            for f in v["boxes"]]}
            with open(p, "rb") as fh:
                original = fh.read()
            mos = cs.jpeg_bytes(cs.mosaic(img, res, tile=160, pixelated=False), 90)
            k = ai.label_throw(settings.load()["model"], original, mos, len(v["boxes"]), kl,
                               steps_dice.features_text(d_project))
            v["ai"] = k
            by_nr = {int(w.get("nr", 0)): w for w in k.get("dice", []) if isinstance(w.get("nr"), int)}
            for r in v["boxes"]:
                w = by_nr.get(r["nr"])
                if w:
                    r["ai_symbol"] = w.get("symbol") if w.get("is_die", True) and not w.get("several") else None
                    r["ai_sure"] = w.get("sure")
                    r["ai_several"] = bool(w.get("several"))
                    r["ai_no_die"] = not w.get("is_die", True)
        return v

    @app.route("/p/<folder>/label/ai/<name>", methods=["POST"])
    def label_ai(folder, name):
        d = project_or_404(folder)
        if not RAW_NAME.match(name):
            abort(400)
        if not classes_or_none(d):
            return jsonify(ok=False, error="First confirm the symbols in step 3 (class list).")
        dd = load_labels(workspace(), folder)
        if name not in dd["images"]:
            abort(404)
        try:
            get_suggestion(folder, name, d, dd, True)
        except ai.AIError as e:
            return jsonify(ok=False, error=str(e))
        save_labels(workspace(), folder, dd)
        write_csv(workspace(), folder, dd)
        return jsonify(ok=True)

    # ------------------------------------------------------------ step 9: check one capture
    @app.route("/p/<folder>/label/<name>")
    def label_image(folder, name):
        d = project_or_404(folder)
        if not RAW_NAME.match(name):
            abort(404)
        kl = classes_or_none(d)
        dd = load_labels(workspace(), folder)
        if name not in dd["images"]:
            abort(404)
        v = get_suggestion(folder, name, d, dd, False)
        save_labels(workspace(), folder, dd)
        names = sorted(dd["images"])
        i = names.index(name)
        open_ones = [n for n in names if not dd["images"][n].get("confirmed")]
        next_open = next((n for n in names[i + 1:] if n in open_ones), open_ones[0] if open_ones and open_ones[0] != name else None)
        info = dd["images"][name]
        if info.get("confirmed"):
            dice = [dict(w) for w in info["confirmed"]["dice"]]
        else:
            dice = []
            for r in v["boxes"]:
                if r.get("ai_no_die") or r.get("several") or r.get("ai_several"):
                    sym = "none"   # no die or several -> draw them one by one
                else:
                    # own model first (knows your dice), otherwise Claude
                    sym = r.get("model_symbol") or r.get("ai_symbol") or ""
                dice.append({"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "symbol": sym,
                             "ai_symbol": r.get("ai_symbol"), "several": r.get("several") or r.get("ai_several"),
                             "ai_sure": r.get("ai_sure"), "model_symbol": r.get("model_symbol"), "model_p": r.get("model_p"),
                             "model_unsure": r.get("model_unsure"), "source": "model" if r.get("model_symbol") else "auto"})
        return render_template("label_image.html", p=d, folder=folder, active="9_label", step_id="9_label",
                               name=name, classes=kl, dice=dice, info=info, suggestion=v,
                               prev=names[i - 1] if i > 0 else None, next=names[i + 1] if i + 1 < len(names) else None,
                               next_open=next_open, nr=i + 1, total=len(names),
                               st=statistics(dd), features=steps_dice.features_text(d))

    @app.route("/p/<folder>/label/<name>/confirm", methods=["POST"])
    def label_confirm(folder, name):
        d = project_or_404(folder)
        if not RAW_NAME.match(name):
            abort(400)
        kl = classes_or_none(d)
        if not kl:
            return jsonify(ok=False, error="No class list (step 3).")
        j = request.get_json(force=True)
        try:
            file = confirm(workspace(), folder, name, j.get("dice", []), kl)
        except ValueError as e:
            return jsonify(ok=False, error=str(e))
        st = status_8_9(folder, d)
        return jsonify(ok=True, file=file, confirmed=st["confirmed"], total=st["total"])
