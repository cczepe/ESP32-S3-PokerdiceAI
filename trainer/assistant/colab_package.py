"""Colab package: one ZIP with the Drive folder structure (project data, scripts, notebooks).

Unpacked into "My Drive" it gives:
  <drive_folder>/projects/<project>/   project.json, photos/, test_images/, camera_setup/
  <drive_folder>/scripts/              trainer/colab/scripts/*.py
  <drive_folder>/notebooks/            the 5 notebooks + version.json
Next to the ZIP the trainer writes 1_setup_<project>_<time>.ipynb (only for opening Colab the very first time).
"""
import datetime
import json
import os
import zipfile

import projects

NOTEBOOKS = ["1_setup", "2_render", "3_detector", "4_classifier", "5_all_in_one"]
PACKAGE_FOLDERS = ("photos", "test_images", "camera_setup")   # from the project folder into the package
PACKAGE_FILES = ("project.json",)


def packages_folder(workspace):
    return os.path.join(workspace, "colab_packages")


def notebook_versions(trainer):
    """Versions of the notebooks (written by build_notebooks.py); the notebooks compare them with Drive."""
    p = os.path.join(trainer, "colab", "notebooks", "version.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def notebook_text(trainer, name, drive_folder, project):
    p = os.path.join(trainer, "colab", "notebooks", name + ".ipynb")
    with open(p, encoding="utf-8") as f:
        return f.read().replace("__DRIVE__", drive_folder).replace("__PROJECT__", project)


def missing_notebooks(trainer):
    return [n for n in NOTEBOOKS if not os.path.isfile(os.path.join(trainer, "colab", "notebooks", n + ".ipynb"))]


def build(workspace, folder, drive_folder, trainer):
    """Builds <workspace>/colab_packages/Colab_<project>_<time>.zip and next to it
    1_setup_<project>_<time>.ipynb. Returns (zip, notebook)."""
    missing = missing_notebooks(trainer)
    if missing:
        raise ValueError("Notebooks missing in trainer/colab/notebooks: " + ", ".join(missing)
                         + " (python trainer/colab/build_notebooks.py creates them).")
    target_folder = packages_folder(workspace)
    os.makedirs(target_folder, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(target_folder, f"Colab_{folder}_{stamp}.zip")
    source = projects.path(workspace, folder)
    scripts = os.path.join(trainer, "colab", "scripts")
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(source):
            rel = os.path.relpath(root, source)
            parts = [] if rel == "." else rel.split(os.sep)
            # only what Colab needs (a fixed list - backups such as export_backup_..., evaluation_...
            # or models/backup_... would otherwise blow up the package many times over)
            if parts and parts[0] not in PACKAGE_FOLDERS:
                continue
            for fn in files:
                if not parts and fn not in PACKAGE_FILES:
                    continue
                if fn.endswith(".tmp"):
                    continue
                arc = "/".join([drive_folder, "projects", folder] + parts + [fn])
                z.write(os.path.join(root, fn), arc)
        for fn in sorted(os.listdir(scripts)) if os.path.isdir(scripts) else []:
            if fn.endswith(".py"):
                z.write(os.path.join(scripts, fn), f"{drive_folder}/scripts/{fn}")
        for name in NOTEBOOKS:
            z.writestr(f"{drive_folder}/notebooks/{name}.ipynb", notebook_text(trainer, name, drive_folder, folder))
        z.writestr(f"{drive_folder}/notebooks/version.json", json.dumps(notebook_versions(trainer)))
    nb = os.path.join(target_folder, f"1_setup_{folder}_{stamp}.ipynb")
    with open(nb, "x", encoding="utf-8") as f:
        f.write(notebook_text(trainer, "1_setup", drive_folder, folder))
    return target, nb
