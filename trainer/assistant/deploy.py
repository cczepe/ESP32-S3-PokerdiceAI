"""Page "Training & device" (/p/<project>/deploy).

Shows which files are there already (export from Colab, converted models, firmware models), the results
of the reports and the exact commands with real paths. The trainer itself runs none of them.
"""
import datetime
import filecmp
import json
import os

from flask import render_template

import projects
import settings

EXPORT_FILES = ["dice_det.onnx", "det_calib.npz", "det_test.npz", "det_info.json",
                "dice_cls.onnx", "cls_calib.npz", "cls_test.npz", "cls_info.json"]
MODEL_FILES = ["dice_det.espdl", "det_report.json", "dice_cls.espdl", "cls_report.json"]
FIRMWARE_FILES = ["dice_det.espdl", "dice_cls.espdl", "model_config.h"]

TQT_NOTE = ("The default is with <b>TQT</b> (fine-tuning of the int8 values, 50 steps). It takes clearly longer than without "
            "(minutes instead of under one minute on a laptop) and usually finds a few more dice. "
            "For a quick try append <code>--no-tqt</code>, for a little more accuracy "
            "<code>--tqt-steps 200</code> (takes correspondingly longer). Older results are moved to "
            "<code>models/backup_…</code>.")


def file_info(folder, names):
    result = []
    for n in names:
        p = os.path.join(folder, n)
        if os.path.isfile(p):
            result.append({"name": n, "present": True, "kb": round(os.path.getsize(p) / 1024, 1),
                           "time": datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")})
        else:
            result.append({"name": n, "present": False})
    return result


def json_or_none(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def register(app, workspace, project_or_404):

    @app.route("/p/<folder>/deploy")
    def deploy_page(folder):
        d = project_or_404(folder)
        repo = app.config["REPO"]
        rel = lambda p: os.path.relpath(p, repo)   # paths relative to the repository folder (the shell stands there)
        p_folder = projects.path(workspace(), folder)
        export = os.path.join(p_folder, "export")
        models = os.path.join(p_folder, "models")
        firmware = os.path.join(repo, "firmware")
        e = settings.load()
        ex = file_info(export, EXPORT_FILES)
        mo = file_info(models, MODEL_FILES)
        fw_models = os.path.join(firmware, "models")
        fw = file_info(fw_models, FIRMWARE_FILES)
        # the firmware does not know the project name - "built from this project" = same files as here
        fw_same = all(x["present"] for x in fw) and all(
            os.path.isfile(os.path.join(models, n)) and filecmp.cmp(os.path.join(models, n), os.path.join(fw_models, n), shallow=False)
            for n in FIRMWARE_FILES)
        return render_template(
            "deploy.html", p=d, folder=folder, active="deploy",
            export=ex, models=mo, fw=fw,
            export_ok=all(x["present"] for x in ex), models_ok=all(x["present"] for x in mo[0::2]),
            fw_ok=fw_same, fw_other=all(x["present"] for x in fw) and not fw_same,
            firmware_present=os.path.isfile(os.path.join(firmware, "CMakeLists.txt")),
            det_report=json_or_none(os.path.join(models, "det_report.json")),
            cls_report=json_or_none(os.path.join(models, "cls_report.json")),
            det_info=json_or_none(os.path.join(export, "det_info.json")),
            cls_info=json_or_none(os.path.join(export, "cls_info.json")),
            r_export=rel(export), r_models=rel(models), r_project=rel(p_folder),
            r_firmware=rel(firmware), drive=e.get("drive_folder") or "Pokerdice",
            REPO_PATH=repo, TQT_NOTE=TQT_NOTE, camera=projects.camera(d))
