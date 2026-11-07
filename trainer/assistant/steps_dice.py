"""Phase A of the trainer: steps 2-7, phone upload via QR code, "Ready for Colab?", Colab package."""
import os
import re

from flask import abort, jsonify, redirect, render_template, request, Response, url_for

import ai
import colab_package
import die_net
import photos
import projects
import settings

DEFAULT_ORDER = ["9", "10", "J", "Q", "K", "A"]
KINDS = ["pips", "figure", "sign", "letter", "number", "other"]


# ====================================================================== helpers (without Flask)
def symbols(d):
    s = d.get("symbols") or {}
    s.setdefault("faces", [])
    s.setdefault("classes", [])
    s.setdefault("frozen", False)
    s.setdefault("face_crops", {})
    s.setdefault("confirmed", False)
    d["symbols"] = s
    return s


def classes_without_unclear(d):
    return [k for k in symbols(d)["classes"] if k != "?"]


def features_text(d):
    """Features of the symbols for the AI: from step 3 (if confirmed), otherwise the free text from step 1."""
    s = symbols(d)
    if s["confirmed"] and s["faces"]:
        lines = []
        for e in s["faces"]:
            parts = [t for t in ((e.get("color") or "") + " " + (e.get("kind") or "")).split() if t]
            line = f"{e['name']}: {' '.join(parts)}"
            if e.get("position"):
                line += f", {e['position']}"
            lines.append(line)
        return "\n".join(lines)
    return d.get("symbol_hints") or ""


def checklist(workspace, folder, d):
    """All required points for "Ready for Colab?". Returns a list of dicts {id, nr, title, ok, text}."""
    w = d.get("die") or {}
    t = d.get("tray") or {}
    s = symbols(d)
    cnt = {k: len(photos.list_photos(workspace, folder, k)) for k in photos.PHOTO_STEPS}
    a = die_net.normalise(d.get("layout"))
    kl = classes_without_unclear(d)
    problems = die_net.check(a, kl) if kl else ["Confirm step 3 (symbols) first."]
    result = []

    def point(sid, ok, text):
        st = next(x for x in projects.STEPS if x["id"] == sid)
        result.append({"id": sid, "nr": st["nr"], "title": st["title"], "ok": bool(ok), "text": text})

    cs = d.get("camera_setup") or {}
    point("1_camera_setup", d["steps"]["1_camera_setup"]["status"] == "done",
          f"Result applied, status “{cs.get('status')}”" if cs else "Result not applied yet")
    miss2 = [n for n, v in (("edge length", w.get("edge_mm")), ("corner radius", w.get("corner_radius_mm")),
                            ("body color", w.get("body_color"))) if not v]
    point("2_measure", not miss2, "complete" if not miss2 else "missing: " + ", ".join(miss2))
    miss3 = []
    if cnt["3_faces"] < 6:
        miss3.append(f"{cnt['3_faces']} of 6 photos")
    if cnt["3_faces"] and len([f for f in photos.list_photos(workspace, folder, "3_faces") if f in s["face_crops"]]) < cnt["3_faces"]:
        miss3.append("die not marked on all photos")
    if not s["confirmed"]:
        miss3.append("symbols not confirmed")
    point("3_faces", not miss3, "6 photos, symbols confirmed: " + ", ".join(kl) if not miss3 else "; ".join(miss3))
    miss4 = []
    if cnt["4_corners"] < 4:
        miss4.append(f"{cnt['4_corners']} of 4 photos")
    if problems:
        miss4.extend(problems[:2])
    elif not die_net.cross_check_complete(a, kl):
        miss4.append("cross-check “what lies opposite” incomplete")
    if not a["confirmed"]:
        miss4.append("layout not confirmed")
    point("4_corners", not miss4, "net confirmed, cross-check matches" if not miss4 else "; ".join(miss4))
    miss5 = [n for n, v in (("diameter", t.get("diameter_mm")), ("wall height", t.get("wall_height_mm"))) if not v]
    if cnt["5_tray"] < 2:
        miss5.append(f"{cnt['5_tray']} of 2 photos")
    point("5_tray", not miss5, "complete" if not miss5 else "missing: " + ", ".join(miss5))
    point("6_felt", cnt["6_felt"] >= 1, f"{cnt['6_felt']} photo(s)" if cnt["6_felt"] else "no photo")
    point("7_surroundings", cnt["7_surroundings"] >= 10, f"{cnt['7_surroundings']} of at least 10 photos")
    return result


def update_status(workspace, folder, d):
    """Steps 2-7: done as soon as all required points are met; in progress as soon as something is there."""
    cnt = {k: len(photos.list_photos(workspace, folder, k)) for k in photos.PHOTO_STEPS}
    w, t, s = d.get("die") or {}, d.get("tray") or {}, symbols(d)
    something = {
        "2_measure": any([w.get("edge_mm"), w.get("corner_radius_mm"), w.get("body_color")]),
        "3_faces": cnt["3_faces"] > 0 or s["confirmed"],
        "4_corners": cnt["4_corners"] > 0 or bool(d.get("layout")),
        "5_tray": cnt["5_tray"] > 0 or bool(t.get("diameter_mm")) or bool(t.get("wall_height_mm")),
        "6_felt": cnt["6_felt"] > 0,
        "7_surroundings": cnt["7_surroundings"] > 0,
    }
    for p in checklist(workspace, folder, d):
        if p["id"] == "1_camera_setup":
            continue
        new = "done" if p["ok"] else ("in_progress" if something.get(p["id"]) else "open")
        if d["steps"][p["id"]]["status"] != new:
            projects.set_step(d, p["id"], new)


def add_face_crops(workspace, folder, d):
    """For face photos without a die mark, look for a suggestion automatically."""
    s = symbols(d)
    changed = False
    edge = (d.get("die") or {}).get("edge_mm")
    present = photos.list_photos(workspace, folder, "3_faces")
    for f in present:
        if f not in s["face_crops"]:
            z = photos.find_die(os.path.join(photos.folder(workspace, folder, "3_faces"), f), edge)
            if z:
                s["face_crops"][f] = z
                changed = True
    for f in list(s["face_crops"]):
        if f not in present:
            del s["face_crops"][f]
            changed = True
    return changed


# ====================================================================== Flask routes
def register(app, workspace, project_or_404):

    def save_with_status(folder, d):
        update_status(workspace(), folder, d)
        projects.save(workspace(), folder, d)

    def step_or_404(step):
        if step not in photos.PHOTO_STEPS:
            abort(404)

    def phone_url(folder, step):
        host = app.config.get("HOST", "127.0.0.1")
        ip = photos.lan_ip() if host == "0.0.0.0" else host
        return f"http://{ip}:{app.config.get('PORT', 5000)}" + url_for("phone_page", folder=folder, step=step)

    def photo_list_data(folder, step):
        o = photos.folder(workspace(), folder, step)
        result = []
        for n in photos.list_photos(workspace(), folder, step):
            try:
                p = photos.check(os.path.join(o, n))
            except ValueError as e:
                p = {"ok": False, "hints": [str(e)]}
            result.append({"name": n, "title": photos.NAMES_TEXT.get(n[:-4], n), "check": p})
        return result

    # ------------------------------------------------------------ step pages
    @app.route("/p/<folder>/a/<int:nr>")
    def step_a(folder, nr):
        d = project_or_404(folder)
        st = next((s for s in projects.STEPS if s["nr"] == nr), None)
        if not st or nr < 2 or nr > 7:
            abort(404)
        sid = st["id"]
        if sid == "3_faces" and add_face_crops(workspace(), folder, d):
            save_with_status(folder, d)
        data = {"p": d, "folder": folder, "active": sid, "s": st, "step_id": sid}
        if sid in photos.PHOTO_STEPS:
            data.update(photo_list=photo_list_data(folder, sid), cfg=photos.PHOTO_STEPS[sid],
                        phone_url=phone_url(folder, sid), local=app.config.get("HOST") not in ("0.0.0.0",),
                        folder_path=photos.folder(workspace(), folder, sid))
        if sid == "3_faces":
            data.update(sym=symbols(d), kinds=KINDS, default=DEFAULT_ORDER)
        if sid == "4_corners":
            kl = classes_without_unclear(d)
            a = die_net.normalise(d.get("layout"))
            data.update(classes=kl, layout=a, fields=die_net.FIELDS, offset=die_net.CSS_OFFSET,
                        problems=die_net.check(a, kl) if kl else [],
                        symbol_photo={e["name"]: e["photo"] for e in symbols(d)["faces"]})
        template = {2: "step2_measure.html", 3: "step3_faces.html", 4: "step4_net.html"}.get(nr, "photo_step.html")
        return render_template(template, **data)

    # ------------------------------------------------------------ photos: upload, show, delete
    @app.route("/p/<folder>/photos/<step>/upload", methods=["POST"])
    def photo_upload(folder, step):
        d = project_or_404(folder)
        step_or_404(step)
        names, errors = [], []
        wish = request.form.get("wish") or None
        for f in request.files.getlist("images"):
            data = f.read()
            if not data:
                continue
            name, err = photos.save(workspace(), folder, step, data, wish)
            if err:
                errors.append(f"{f.filename}: {err}")
            else:
                names.append(name)
                wish = None
        if step == "3_faces" and names:
            add_face_crops(workspace(), folder, d)
        save_with_status(folder, d)
        return jsonify(ok=not errors, names=names, errors=errors)

    @app.route("/p/<folder>/photos/<step>/list")
    def photo_list(folder, step):
        project_or_404(folder)
        step_or_404(step)
        cfg = photos.PHOTO_STEPS[step]
        return jsonify(photos=photo_list_data(folder, step), min=cfg["min"], max=cfg["max"], names=cfg["names"])

    @app.route("/p/<folder>/photos/<step>/<name>")
    def photo_image(folder, step, name):
        d = project_or_404(folder)
        step_or_404(step)
        if not photos.FILE_NAME.match(name):
            abort(404)
        p = os.path.join(photos.folder(workspace(), folder, step), name)
        if not os.path.isfile(p):
            abort(404)
        kind = request.args.get("kind", "preview")
        if kind == "original":
            with open(p, "rb") as f:
                return Response(f.read(), mimetype="image/jpeg")
        if kind == "face_crop":
            z = symbols(d)["face_crops"].get(name)
            if not z:
                abort(404)
            return Response(photos.jpeg(photos.face_crop(p, z, int(request.args.get("g", 256)))), mimetype="image/jpeg")
        return Response(photos.preview(p, int(request.args.get("g", 900))), mimetype="image/jpeg")

    @app.route("/p/<folder>/photos/<step>/<name>/delete", methods=["POST"])
    def photo_delete(folder, step, name):
        d = project_or_404(folder)
        step_or_404(step)
        try:
            photos.delete(workspace(), folder, step, name)
        except ValueError:
            abort(400)
        if step == "3_faces":
            s = symbols(d)
            s["face_crops"].pop(name, None)
            if any(e.get("photo") == name for e in s["faces"]):
                s["confirmed"] = False   # the confirmed assignment no longer fits
        save_with_status(folder, d)
        return jsonify(ok=True)

    @app.route("/p/<folder>/qr/<step>.svg")
    def qr(folder, step):
        project_or_404(folder)
        step_or_404(step)
        svg = photos.qr_svg(phone_url(folder, step))
        if not svg:
            abort(404, "Package qrcode is missing (pip install qrcode)")
        return Response(svg, mimetype="image/svg+xml")

    @app.route("/phone/<folder>/<step>")
    def phone_page(folder, step):
        d = project_or_404(folder)
        step_or_404(step)
        st = next(s for s in projects.STEPS if s["id"] == step)
        return render_template("phone.html", p=d, folder=folder, step=step, s=st, cfg=photos.PHOTO_STEPS[step],
                               names_text=photos.NAMES_TEXT)

    # ------------------------------------------------------------ step 2
    @app.route("/p/<folder>/step2", methods=["POST"])
    def step2_save(folder):
        d = project_or_404(folder)
        w = d.setdefault("die", {})

        def number(name, lo, hi):
            txt = request.form.get(name, "").replace(",", ".").strip()
            try:
                v = float(txt)
                return v if lo <= v <= hi else None
            except ValueError:
                return None
        w["edge_mm"] = number("edge_mm", 5, 50)
        w["corner_radius_mm"] = number("corner_radius_mm", 0, 10)
        color = request.form.get("body_color", "").strip()
        w["body_color"] = color if re.match(r"^#[0-9a-fA-F]{6}$", color) else None
        w["body_color_text"] = request.form.get("body_color_text", "").strip()[:60]
        save_with_status(folder, d)
        return redirect(url_for("step_a", folder=folder, nr=2) + "?saved=1")

    # ------------------------------------------------------------ step 3
    @app.route("/p/<folder>/faces/face_crop", methods=["POST"])
    def faces_face_crop(folder):
        d = project_or_404(folder)
        j = request.get_json(force=True)
        name = j.get("photo", "")
        if not photos.FILE_NAME.match(name):
            abort(400)
        s = int(round(float(j["s"])))
        if s < 20:
            return jsonify(ok=False, error="Crop too small")
        z = {"x": int(round(float(j["x"]))), "y": int(round(float(j["y"]))), "s": s, "auto": False}
        edge = (d.get("die") or {}).get("edge_mm")
        if edge:
            z["px_per_mm"] = round(s / float(edge), 1)
        symbols(d)["face_crops"][name] = z
        save_with_status(folder, d)
        return jsonify(ok=True)

    @app.route("/p/<folder>/faces/detect_again", methods=["POST"])
    def faces_detect_again(folder):
        """Look for the die mark again. photo="" = all; also_hand=False leaves boxes set by hand alone."""
        d = project_or_404(folder)
        j = request.get_json(force=True) or {}
        name = j.get("photo") or ""
        also_hand = bool(j.get("also_hand"))
        if name and not photos.FILE_NAME.match(name):
            abort(400)
        o = photos.folder(workspace(), folder, "3_faces")
        present = photos.list_photos(workspace(), folder, "3_faces")
        targets = [name] if name else present
        s = symbols(d)
        edge = (d.get("die") or {}).get("edge_mm")
        found, not_found, skipped = [], [], []
        for f in targets:
            if f not in present:
                continue
            old = s["face_crops"].get(f)
            if old and not old.get("auto") and not also_hand:
                skipped.append(f)
                continue
            z = photos.find_die(os.path.join(o, f), edge)
            if z:
                s["face_crops"][f] = z
                found.append(f)
            else:
                not_found.append(f)
                if old and old.get("auto"):
                    del s["face_crops"][f]   # the old automatic box was obviously wrong
        save_with_status(folder, d)
        return jsonify(ok=True, found=found, not_found=not_found, skipped=skipped)

    @app.route("/p/<folder>/faces/ai", methods=["POST"])
    def faces_ai(folder):
        d = project_or_404(folder)
        o = photos.folder(workspace(), folder, "3_faces")
        s = symbols(d)
        images = []
        for n in photos.list_photos(workspace(), folder, "3_faces"):
            p = os.path.join(o, n)
            z = s["face_crops"].get(n)
            b = photos.jpeg(photos.face_crop(p, z, 512)) if z else photos.preview(p, 1024)
            images.append((n, b))
        if not images:
            return jsonify(ok=False, error="No face photos yet.")
        try:
            res = ai.recognise_faces(settings.load()["model"], images, d.get("symbol_hints") or "")
        except ai.AIError as e:
            return jsonify(ok=False, error=str(e))
        s["suggestion"] = res
        projects.save(workspace(), folder, d)
        return jsonify(ok=True, suggestion=res)

    @app.route("/p/<folder>/faces/save", methods=["POST"])
    def faces_save(folder):
        d = project_or_404(folder)
        s = symbols(d)
        if s["frozen"]:
            return jsonify(ok=False, error="The class list is frozen (rendering has started) and can no longer be changed.")
        j = request.get_json(force=True)
        faces = []
        for e in j.get("faces", []):
            name = re.sub(r"[^\w\-]", "", (e.get("name") or "").strip())[:12]
            faces.append({"photo": e.get("photo"), "name": name, "kind": e.get("kind") if e.get("kind") in KINDS else "other",
                          "color": (e.get("color") or "").strip()[:30], "position": (e.get("position") or "").strip()[:60],
                          "description": (e.get("description") or "").strip()[:200]})
        names = [e["name"] for e in faces]
        if len(faces) != 6 or any(not n for n in names):
            return jsonify(ok=False, error="Please give all 6 photos a name.")
        if len(set(names)) != 6:
            return jsonify(ok=False, error="Each name may occur only once.")
        if "?" in names:
            return jsonify(ok=False, error="“?” is reserved for unclear dice.")
        order = [k for k in DEFAULT_ORDER if k in names] if set(names) == set(DEFAULT_ORDER) else names
        s["faces"] = faces
        s["classes"] = order + ["?"]
        s["confirmed"] = True
        s["confirmed_at"] = projects.now()
        save_with_status(folder, d)
        return jsonify(ok=True, classes=s["classes"])

    # ------------------------------------------------------------ step 4
    @app.route("/p/<folder>/layout/ai", methods=["POST"])
    def layout_ai(folder):
        d = project_or_404(folder)
        s = symbols(d)
        if not s["confirmed"]:
            return jsonify(ok=False, error="First confirm the symbols in step 3.")
        o3 = photos.folder(workspace(), folder, "3_faces")
        o4 = photos.folder(workspace(), folder, "4_corners")
        faces = []
        for e in s["faces"]:
            z = s["face_crops"].get(e["photo"])
            p = os.path.join(o3, e["photo"])
            if os.path.isfile(p):
                faces.append((e["name"], photos.jpeg(photos.face_crop(p, z, 384)) if z else photos.preview(p, 600)))
        corners = [(n, photos.preview(os.path.join(o4, n), 1024)) for n in photos.list_photos(workspace(), folder, "4_corners")]
        if not corners:
            return jsonify(ok=False, error="No corner photos yet.")
        try:
            res = ai.suggest_layout(settings.load()["model"], faces, corners, classes_without_unclear(d))
        except ai.AIError as e:
            return jsonify(ok=False, error=str(e))
        a = die_net.normalise({"net": res.get("net"), "source": "ai"})
        old = die_net.normalise(d.get("layout"))
        a["cross_check"] = old["cross_check"]       # your cross-check stays
        a["ai_remark"] = (res.get("remark") or "") + (f" (confidence: {res['confidence']})" if res.get("confidence") else "")
        d["layout"] = a
        save_with_status(folder, d)
        return jsonify(ok=True)

    @app.route("/p/<folder>/layout/save", methods=["POST"])
    def layout_save(folder):
        d = project_or_404(folder)
        j = request.get_json(force=True)
        old = die_net.normalise(d.get("layout"))
        a = die_net.normalise({"net": j.get("net") or old["net"], "cross_check": j.get("cross_check", old["cross_check"]),
                               "source": old["source"] or "hand"})
        kl = classes_without_unclear(d)
        problems = die_net.check(a, kl)
        if j.get("confirm"):
            if problems:
                return jsonify(ok=False, error="Not consistent yet: " + " ".join(problems))
            if not die_net.cross_check_complete(a, kl):
                return jsonify(ok=False, error="Please first fill in the cross-check “what lies opposite” for all symbols.")
            a["confirmed"] = True
            a["confirmed_at"] = projects.now()
        else:
            a["confirmed"] = False   # every change cancels the confirmation
        if old.get("ai_remark"):
            a["ai_remark"] = old["ai_remark"]
        d["layout"] = a
        save_with_status(folder, d)
        return jsonify(ok=True, problems=problems, confirmed=a["confirmed"])

    # ------------------------------------------------------------ step 5
    @app.route("/p/<folder>/tray_size", methods=["POST"])
    def tray_size(folder):
        d = project_or_404(folder)
        t = d.setdefault("tray", {})
        for name, lo, hi in (("diameter_mm", 50, 1000), ("wall_height_mm", 0, 200)):
            txt = request.form.get(name, "").replace(",", ".").strip()
            try:
                v = float(txt)
                t[name] = v if lo <= v <= hi else t.get(name)
            except ValueError:
                if not txt:
                    t[name] = None
        save_with_status(folder, d)
        return redirect(url_for("step_a", folder=folder, nr=5))

    # ------------------------------------------------------------ ready for Colab
    def drive_folder():
        return re.sub(r"[^\w\-]", "", settings.load().get("drive_folder") or "Pokerdice") or "Pokerdice"

    @app.route("/p/<folder>/colab")
    def colab_page(folder):
        d = project_or_404(folder)
        update_status(workspace(), folder, d)
        projects.save(workspace(), folder, d)
        items = checklist(workspace(), folder, d)
        pf = colab_package.packages_folder(workspace())
        packages = sorted([n for n in os.listdir(pf) if n.startswith(f"Colab_{folder}_") or n.startswith(f"1_setup_{folder}_")],
                          reverse=True) if os.path.isdir(pf) else []
        to = projects.path(workspace(), folder, "test_images")
        test_count = len([n for n in os.listdir(to) if n.endswith(".json")]) if os.path.isdir(to) else 0
        return render_template("colab.html", p=d, folder=folder, active="colab", items=items, test_count=test_count,
                               ready=all(x["ok"] for x in items), packages=packages, packages_folder=pf,
                               drive=drive_folder(), message=request.args.get("message"), error=request.args.get("error"),
                               missing_notebooks=colab_package.missing_notebooks(app.config["TRAINER"]))

    @app.route("/p/<folder>/colab/package", methods=["POST"])
    def colab_package_create(folder):
        project_or_404(folder)
        try:
            path, nb = colab_package.build(workspace(), folder, drive_folder(), app.config["TRAINER"])
        except (ValueError, OSError) as e:
            return redirect(url_for("colab_page", folder=folder, error=str(e)))
        return redirect(url_for("colab_page", folder=folder,
                                message=f"Package created: {path}  ·  notebook (only for the first opening): {nb}"))
