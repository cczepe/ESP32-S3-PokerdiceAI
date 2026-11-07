"""Builds the Colab notebook templates in trainer/colab/notebooks/.

The notebooks contain the placeholders __DRIVE__ and __PROJECT__; the trainer fills them in when it
builds a Colab package (assistant/colab_package.py). version.json holds the version of each notebook,
so that a notebook can tell when the package contains a newer copy of itself.

  1_setup        Drive + package, Blender, textures, corner check (confirm), preview with timing
  2_render       full render (one mixed run: normal, close, keep, leaning dice)
  3_detector     detector training (ESPDet-Pico), test, export
  4_classifier   classifier training, comparison, export
  5_all_in_one   2 + 3 + 4 in one session with a log, then end the session

Examples (from the repository folder):
  .venv\\Scripts\\python.exe trainer\\colab\\build_notebooks.py --help
  .venv\\Scripts\\python.exe trainer\\colab\\build_notebooks.py            writes the notebooks (refuses to overwrite)
  .venv\\Scripts\\python.exe trainer\\colab\\build_notebooks.py --force    replaces existing notebooks
  (Linux/macOS: .venv/bin/python trainer/colab/build_notebooks.py --force)
"""
import argparse
import json
import os

VERSIONS = {"1_setup": 1, "2_render": 1, "3_detector": 2, "4_classifier": 1, "5_all_in_one": 2}

ESP_DETECTION_COMMIT = "28de5902a357d99dd9a5db0f52905255641c5a78"


class Notebook:
    def __init__(self, name):
        self.name = name
        self.cells = []

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})

    def code(self, text):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": text.strip("\n").splitlines(True)})

    def json(self):
        return {"cells": self.cells,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 0}


# ====================================================================== shared cells
PATHS = """
DRIVE = f"/content/drive/MyDrive/{DRIVE_FOLDER}"
P = f"{DRIVE}/projects/{PROJECT}"
SCRIPTS = f"{DRIVE}/scripts"
SCENE = f"{P}/scene"
RENDERS = f"{P}/renders"
PREVIEW = f"{P}/renders_preview"
EXPORT = f"{P}/export"
DET = f"{P}/detector"
CLS = f"{P}/classifier"
ESP_DETECTION = "/content/esp-detection"
print("Project folder:", P)
"""

GPU_MD = """
## 2 · Check the GPU
Expected: **NVIDIA L4** (faster) or **Tesla T4**. If it says "no GPU": *Runtime → Change runtime type → L4 GPU*
(or T4), then start again from the top.
"""
GPU_CODE = """
import subprocess
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = ""
print("✔ GPU:", gpu) if gpu else print("✘ no GPU! Runtime → Change runtime type → L4 or T4 GPU, then run everything again.")
"""

DRIVE_MD = """
## 3 · Connect Google Drive and unpack the newest package
Colab asks whether the notebook may access your Drive → **Allow**.

Then the **newest Colab package** in `packages/` is unpacked (the ZIP file from the trainer - upload it to
*My Drive → __DRIVE__ → packages*, do **not** unpack it). Scripts, notebooks, `project.json`, photos, test images and the
camera setup are updated. **Renders and training results are not touched.**
A package is unpacked only once; a newer package replaces photos, test images and camera setup completely.
"""
DRIVE_CODE = """
import glob, json, os, shutil, zipfile
from google.colab import drive
drive.mount("/content/drive")
PACKAGES = f"{DRIVE}/packages"
os.makedirs(PACKAGES, exist_ok=True)
packages = sorted(glob.glob(f"{PACKAGES}/Colab_{PROJECT}_*.zip"))
state_file = f"{DRIVE}/package_state.json"
state = json.load(open(state_file)) if os.path.exists(state_file) else {}
if not packages:
    print(f"⚠ No package “Colab_{PROJECT}_….zip” in {PACKAGES} - create one in the trainer and upload the ZIP file there.")
elif state.get(PROJECT) == os.path.basename(packages[-1]):
    print("✔ newest package is unpacked already:", os.path.basename(packages[-1]))
else:
    new = packages[-1]
    print("unpacking:", os.path.basename(new))
    with zipfile.ZipFile(new) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        prefix = names[0].split("/")[0]
        # mirror of the laptop: these folders are replaced completely (deleted photos disappear here too)
        for part in ("scripts", "notebooks", f"projects/{PROJECT}/photos", f"projects/{PROJECT}/test_images",
                     f"projects/{PROJECT}/camera_setup"):
            if any(n.startswith(f"{prefix}/{part}/") for n in names) and os.path.isdir(f"{DRIVE}/{part}"):
                shutil.rmtree(f"{DRIVE}/{part}")
        for n in names:
            target = os.path.join(DRIVE, n.split("/", 1)[1])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(n) as src, open(target, "wb") as f:
                shutil.copyfileobj(src, f)
    state[PROJECT] = os.path.basename(new)
    json.dump(state, open(state_file, "w"), indent=1)
    print(f"✔ unpacked ({len(names)} files)")
    try:
        v = json.load(open(f"{DRIVE}/notebooks/version.json")).get(NB_NAME, 0)
        if v > NB_VERSION:
            print(f"⚠ The package contains a NEWER version of this notebook: please open notebooks/{NB_NAME}.ipynb in Drive again.")
    except (OSError, ValueError):
        pass
missing = [p for p in (f"{P}/project.json", f"{SCRIPTS}/render.py") if not os.path.exists(p)]
if missing:
    print("✘ not found:", *missing, sep="\\n   ")
else:
    print("✔ project and scripts found")
"""

BLENDER_MD = """
## 4 · Install Blender (automatic, about 1-3 minutes)
Downloads Blender 4.5 LTS from **download.blender.org** and unpacks it on the Colab disk. The first time the download
is also stored in your Drive (`blender/`) - after that it is faster and it always stays **the same version**.
Colab forgets everything installed in every new session, so this cell runs every time.
At the end it shows whether Blender sees the GPU (expected: **OPTIX** or **CUDA**).
"""
BLENDER_CODE = """
import json, os, re, shutil, subprocess, urllib.request
CACHE = f"{DRIVE}/blender"
os.makedirs(CACHE, exist_ok=True)
vfile = f"{CACHE}/version.json"
if os.path.exists(vfile):
    version = json.load(open(vfile))["version"]
else:
    html = urllib.request.urlopen(f"https://download.blender.org/release/Blender{BLENDER_SERIES}/").read().decode()
    found = set(re.findall(rf"blender-({re.escape(BLENDER_SERIES)}\\.\\d+)-linux-x64\\.tar\\.xz", html))
    version = sorted(found, key=lambda v: int(v.split(".")[-1]))[-1]
    json.dump({"version": version}, open(vfile, "w"))
archive = f"blender-{version}-linux-x64.tar.xz"
if not os.path.exists(f"{CACHE}/{archive}"):
    print("downloading", archive, "…")
    urllib.request.urlretrieve(f"https://download.blender.org/release/Blender{BLENDER_SERIES}/{archive}", f"/content/{archive}")
    shutil.copy(f"/content/{archive}", f"{CACHE}/{archive}")
BLENDER_DIR = f"/content/blender-{version}-linux-x64"
if not os.path.isdir(BLENDER_DIR):
    source = f"/content/{archive}" if os.path.exists(f"/content/{archive}") else f"{CACHE}/{archive}"
    subprocess.run(["tar", "-xf", source, "-C", "/content"], check=True)
subprocess.run("apt-get -qq update > /dev/null; apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 "
               "libxxf86vm1 libxfixes3 libgl1 libegl1 > /dev/null", shell=True)
BLENDER = f"{BLENDER_DIR}/blender"
print(subprocess.run([BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
test = ("import bpy; p=bpy.context.preferences.addons['cycles'].preferences\\n"
        "for t in ('OPTIX','CUDA'):\\n"
        "    try:\\n        p.compute_device_type=t; p.get_devices()\\n"
        "        g=[d.name for d in p.devices if d.type==t]\\n"
        "        if g: print('GPU RENDERING:', t, g); break\\n"
        "    except TypeError: pass\\n"
        "else: print('GPU RENDERING: none - rendering on the CPU (very slow)')")
out = subprocess.run([BLENDER, "-b", "--python-expr", test], capture_output=True, text=True).stdout
print([z for z in out.splitlines() if z.startswith("GPU RENDERING")] or out[-500:])
"""

CHECK_FUNCS = """
import hashlib, json, os

def atlas_hash():
    return hashlib.sha1(open(f"{SCENE}/atlas.png", "rb").read() + open(f"{SCENE}/scene.json", "rb").read()).hexdigest()

def check_confirmed():
    f = f"{SCENE}/check_ok.json"
    return (os.path.exists(f) and os.path.exists(f"{SCENE}/atlas.png")
            and json.load(open(f)).get("hash") == atlas_hash())
"""

INSTALL_MD = """
## 4 · Install the training programs (about 1-2 minutes, every session)
Fixed versions so that the result stays reproducible: **ultralytics 8.4.189**, **albumentations 2.0.8**,
**onnx 1.23.2**, **onnxsim 0.8.1** and Espressif's **esp-detection** (fixed commit), which describes the detector
*ESPDet-Pico*. PyTorch comes with Colab.
"""
INSTALL_CODE = """
import os, subprocess, sys, json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.189", "albumentations==2.0.8",
                "onnx==1.23.2", "onnxsim==0.8.1"], check=True)
if not os.path.isdir(ESP_DETECTION):
    subprocess.run(["git", "clone", "-q", "https://github.com/espressif/esp-detection.git", ESP_DETECTION], check=True)
subprocess.run(["git", "-C", ESP_DETECTION, "checkout", "-q", "__COMMIT__"], check=True)
sys.path.insert(0, SCRIPTS)
import torch, ultralytics, ml_tools
d = ml_tools.load_project(P)
IMGSZ, CROP = ml_tools.sizes(d)
print(f"✔ torch {torch.__version__} (CUDA: {torch.cuda.is_available()}), ultralytics {ultralytics.__version__}")
print(f"✔ classes: {', '.join(ml_tools.classes(d))} · detector input {IMGSZ} px · crop {CROP} px")
""".replace("__COMMIT__", ESP_DETECTION_COMMIT)


def run_code(cmd_expr, filter_expr="True"):
    """Cell code that starts a script and shows its output line by line (progress bars: only the last state)."""
    return f'''
import re, subprocess, time
t0 = time.time()
proc = subprocess.Popen({cmd_expr}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    z = re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", "", line.rstrip("\\n").split("\\r")[-1])   # no colour codes
    if z.strip() and ({filter_expr}):
        print(z, flush=True)
proc.wait()
print(f"⏱ {{(time.time() - t0) / 60:.1f}} min · return code {{proc.returncode}}")
'''


def indent(text, n=4):
    return "\n".join((" " * n + z) if z.strip() else z for z in text.strip("\n").splitlines())


def settings_cell(nb, lines, note="", paths=""):
    nb.md("""
## 1 · Settings
The trainer has filled in Drive folder and project. """ + (note or "Normally nothing needs to be changed here."))
    nb.code(f'''
DRIVE_FOLDER = "__DRIVE__"        # folder in "My Drive"
PROJECT = "__PROJECT__"           # project folder name (under projects/)
NB_NAME, NB_VERSION = "{nb.name}", {VERSIONS[nb.name]}   # version of this notebook (to detect updates)
{lines.strip()}
''' + PATHS + paths)


def end_session(nb):
    nb.md("""
## End the session (optional)
A connected Colab session uses **compute units even when it does nothing**. If you leave while something is still
running: set `END_SESSION = True` below and start this cell with ▶ - it queues up and ends the session when all cells
before it are done. All results are on Drive, nothing is lost.
**Note:** if a cell before it stops with an error, Colab discards the waiting cells - this one too. The session then
keeps running until Colab disconnects it (or you choose *Runtime → Disconnect and delete runtime*).
With *Run all* nothing happens here as long as `END_SESSION = False`.
""")
    nb.code("""
END_SESSION = False   # <- set to True and press ▶ if the session should end after the running cells
if END_SESSION:
    from google.colab import runtime
    print("Ending the session - results are on Drive.")
    runtime.unassign()
else:
    print("Session keeps running (END_SESSION = False).")
""")


RENDER_SETTINGS = """
BLENDER_SERIES = "4.5"            # Blender LTS series; the exact version is recorded the first time
IMAGES = 28000                    # renders in total
IMAGES_PER_CHUNK = 250            # rendered in one go, then saved to Drive
SAMPLES = 32                      # quality (more = nicer, but slower)
SEED = 1
SHARE_CLOSE = 0.35                # throws with dice lying close together
SHARE_LEAN_WALL = 0.2             # images with a die leaning against the wall (“?”)
SHARE_KEEP = 0.2                  # “keep and re-roll”: 2-4 dice pushed together at the wall, the rest thrown
SHARE_LEAN_DIE = 0.05             # images with a die leaning against another die (“?”)
"""

RENDER_ARGS = ('"--samples", str(SAMPLES), "--share-close", str(SHARE_CLOSE), "--share-lean-wall", str(SHARE_LEAN_WALL), '
               '"--share-keep", str(SHARE_KEEP), "--share-lean-die", str(SHARE_LEAN_DIE)')

RENDER_FILTER = ('"✔" in z or "⚠" in z or "skipped" in z or "ERROR" in z or "Error" in z or "render device" in z '
                 'or "image 1/" in z')


# ====================================================================== 1_setup
def nb_setup():
    nb = Notebook("1_setup")
    nb.md("""
# 1 · Setup - die model, corner check and preview

This notebook rebuilds your die in **Blender**, checks it against your corner photos and renders a small **preview**
with timing. The full render is notebook **2_render**. Everything is stored in your Google Drive in the project folder.

**Once before:** in the menu **Runtime → Change runtime type → L4 GPU** (or T4) and save.
The Colab package from the trainer (ZIP file, not unpacked) is in *My Drive → __DRIVE__ → packages*.

**How:** start the cells in order (▶ left of each cell, or *Runtime → Run all*).
At cell 7 you have to compare and confirm yourself - nothing is rendered before that.
""")
    settings_cell(nb, RENDER_SETTINGS, note="""Normally nothing needs to be changed here.
- `IMAGES`: how many renders notebook 2 makes in total - here only used for the time estimate after the preview (cell 8).
- `SAMPLES`: quality (more = nicer, but slower). 32 is a good start.
- `SHARE_…`: shares of the special throw types. The preview uses the same values as the full render.
""")
    nb.md(GPU_MD)
    nb.code(GPU_CODE)
    nb.md(DRIVE_MD)
    nb.code(DRIVE_CODE)
    nb.md(BLENDER_MD)
    nb.code(BLENDER_CODE)
    nb.md("""
## 5 · Prepare the photos for Blender
Crops the 6 face photos exactly, straightens them, matches the colours to the white paper and combines them according to
your die net into a **texture atlas**. Also: felt, tray wall, surroundings photos and all sizes/ranges
(`scene/scene.json`). Takes less than a minute.

**Good if:** below you see 6 clean die faces - no paper edge, no shadow, not skewed.
If a face looks bad: in the trainer, draw the yellow frame of this photo again (or take a new photo), build a new Colab
package, upload it and repeat from here.
""")
    nb.code("""
import os, subprocess
from IPython.display import Image, display
r = subprocess.run(["python", f"{SCRIPTS}/prepare_textures.py", "--project", P, "--output", SCENE],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
if os.path.exists(f"{SCENE}/atlas.png"):
    display(Image(f"{SCENE}/atlas.png", width=600))
""")
    nb.md("""
## 6 · Render the 3D die from 8 corners
Renders your 3D die from the same 8 corner views as the 3D preview in the trainer and puts your corner photos below.
Takes about 1 minute.
""")
    nb.code("""
r = subprocess.run([BLENDER, "-b", "-P", f"{SCRIPTS}/check_corners.py", "--", "--scene", SCENE],
                   capture_output=True, text=True)
print("\\n".join(z for z in r.stdout.splitlines() if "rendered" in z or "✔" in z or "⚠" in z or "Error" in z))
if r.returncode != 0 or "✔ check done" not in r.stdout:
    print("✖ Rendering failed:\\n", r.stderr[-2000:])
else:
    r = subprocess.run(["python", f"{SCRIPTS}/contact_sheet.py", "check", "--scene", SCENE, "--project", P,
                        "--output", f"{SCENE}/check.jpg"], capture_output=True, text=True)
    print(r.stdout, r.stderr[-1500:])
    if os.path.exists(f"{SCENE}/check.jpg"):
        display(Image(f"{SCENE}/check.jpg", width=1000))
    else:
        print("✖ The check image was not created (see the output above).")
""")
    nb.md("""
## 7 · Compare and confirm (required!)
Find the matching rendered view for each of your corner photos (bottom row). Check:
1. **The same symbols** on the three visible faces?
2. **The same rotation** of every symbol (where does the head of a figure point, how do the rows of pips run)?

**If everything matches:** set `CHECK_OK = True` below and run the cell.
**If something is wrong:** correct the die net in the trainer, build a new Colab package, upload it and repeat from cell 5.
Otherwise the model learns a die that does not exist.

The confirmation is stored on Drive and is valid only for exactly this texture atlas - if the die changes, confirm again.
""")
    nb.code(CHECK_FUNCS + """
import time
CHECK_OK = False   # <- set to True after comparing

if CHECK_OK:
    json.dump({"hash": atlas_hash(), "confirmed": time.strftime("%Y-%m-%d %H:%M:%S")}, open(f"{SCENE}/check_ok.json", "w"))
print("✔ Check confirmed - rendering is allowed." if check_confirmed()
      else "✘ Not confirmed yet. Compare first, then set CHECK_OK = True and run this cell.")
""")
    nb.md("""
## 8 · Preview with timing (about 2-5 minutes)
Renders **30 images** into a separate folder (`renders_preview`) with the same throw types as the full render and
measures the time per image. From that it calculates how long `IMAGES` take. Measured, not estimated.
Measure again (e.g. with other SAMPLES)? Set `PREVIEW_AGAIN = True`.
""")
    nb.code("""
import shutil
PREVIEW_AGAIN = False
assert check_confirmed(), "First cell 7: confirm the check."
if PREVIEW_AGAIN and os.path.isdir(PREVIEW):
    shutil.rmtree(PREVIEW)
proc = subprocess.Popen([BLENDER, "-b", "-P", f"{SCRIPTS}/render.py", "--", "--scene", SCENE, "--output", PREVIEW,
                         "--chunk-from", "0", "--chunk-to", "0", "--images", "30", "--seed", "999", __ARGS__],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for z in proc.stdout:
    if "Chunk" in z or "✔" in z or "ERROR" in z or "Error" in z or "render device" in z:
        print(z, end="")
proc.wait()
info = json.load(open(f"{PREVIEW}/chunk_000/done.json"))
s = info["s_per_image"]
print(f"\\n⏱ measured: {s:.2f} s per image ({info['device']}, {info['samples']} samples)")
h = IMAGES * s / 3600
print(f"   estimate for {IMAGES} images: {h:.1f} hours ({h * 60:.0f} minutes)")
""".replace("__ARGS__", RENDER_ARGS))
    nb.md("""
## 9 · Contact sheet of the preview
Green = die with symbol, **orange = “?”** (tilted/covered, the actual top symbol in brackets).
**Good if:** the images look like real throws, the labels match the dice, and distances/angles roughly match your
camera position. If something looks unreal (too bright, wrong colours …), fix it before the full render runs.
""")
    nb.code("""
r = subprocess.run(["python", f"{SCRIPTS}/contact_sheet.py", "renders", "--folder", PREVIEW, "--output",
                    f"{PREVIEW}/sheet.jpg", "--count", "16"], capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
display(Image(f"{PREVIEW}/sheet.jpg", width=1100))
""")
    nb.md("""
## What next?
1. `notebooks/2_render.ipynb` renders the full set (hours - see the estimate in cell 8),
2. then `notebooks/3_detector.ipynb` (finds the dice in the image)
3. and `notebooks/4_classifier.ipynb` (recognises the top symbol).

**Or all in one go** (e.g. overnight): `notebooks/5_all_in_one.ipynb` renders, trains detector and classifier,
exports both and then ends the session.

All of them are already in your Drive (they came with the package). Real throws labelled in the trainer are used
there as test images - the more, the more honest the measurement.
""")
    end_session(nb)
    return nb


# ====================================================================== 2_render
def nb_render():
    nb = Notebook("2_render")
    nb.md("""
# 2 · Render - synthetic training images

Renders `IMAGES` images with labels in chunks of `IMAGES_PER_CHUNK` into `renders/`. One mixed run: normal throws,
dice lying close together, **keep and re-roll** (2-4 dice pushed together at the wall) and dice leaning against the
wall or another die (“?”). The shares are set in cell 1.

**Before:** notebook 1 is done (check confirmed), *Runtime → Change runtime type → L4 GPU*.

**If Colab loses the connection:** just run everything again. Finished chunks are skipped and it continues where it stopped.
""")
    settings_cell(nb, RENDER_SETTINGS, note="""Use the same values as in notebook 1 (the preview measured with them).
Changing `SEED` or a `SHARE_…` value later only affects chunks that are not finished yet.
""")
    nb.md(GPU_MD)
    nb.code(GPU_CODE)
    nb.md(DRIVE_MD)
    nb.code(DRIVE_CODE)
    nb.md(BLENDER_MD)
    nb.code(BLENDER_CODE)
    nb.md("""
## 5 · Render the full set (takes hours - see the estimate in notebook 1)
Every finished chunk is saved to Drive right away. **If Colab stops:** open the notebook again, *Runtime → Run all* -
finished chunks are skipped. Tip: keep the browser tab open and do not let the laptop sleep.
""")
    nb.code(CHECK_FUNCS + """
import math, subprocess
assert check_confirmed(), "First notebook 1, cell 7: confirm the check."
chunks = math.ceil(IMAGES / IMAGES_PER_CHUNK)
proc = subprocess.Popen([BLENDER, "-b", "-P", f"{SCRIPTS}/render.py", "--", "--scene", SCENE, "--output", RENDERS,
                         "--chunk-from", "0", "--chunk-to", str(chunks - 1), "--images", str(IMAGES_PER_CHUNK),
                         "--seed", str(SEED), __ARGS__],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for z in proc.stdout:
    if __FILTER__:
        print(z, end="", flush=True)
proc.wait()
print("return code", proc.returncode)
""".replace("__ARGS__", RENDER_ARGS).replace("__FILTER__", RENDER_FILTER))
    nb.md("""
## 6 · Status
Counts the finished chunks and images, shows a contact sheet of the full set and the render statistics
(throw types, dice at the wall, “?” shares). You can run this cell any time (also in a new session after cells 1-3).
""")
    nb.code("""
import glob, json, os, subprocess
from IPython.display import Image, display
done = sorted(glob.glob(f"{RENDERS}/chunk_*/done.json"))
n = sum(json.load(open(f))["images"] for f in done)
print(f"{len(done)} chunks done: {n} images (target {IMAGES})")
if done:
    r = subprocess.run(["python", f"{SCRIPTS}/contact_sheet.py", "renders", "--folder", RENDERS, "--output",
                        f"{P}/sheet_renders.jpg", "--count", "16", "--seed", str(len(done))], capture_output=True, text=True)
    print(r.stdout, r.stderr[-1000:])
    display(Image(f"{P}/sheet_renders.jpg", width=1100))
    r = subprocess.run(["python", f"{SCRIPTS}/render_stats.py", "--renders", RENDERS, "--scene", SCENE],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-1000:])
""")
    nb.md("""
## What next?
When the full set is done: `notebooks/3_detector.ipynb`, then `notebooks/4_classifier.ipynb`.
""")
    end_session(nb)
    return nb


# ====================================================================== 3_detector
DET_SETTINGS = """
EPOCHS = 300                # from scratch: passes over all images (upper limit)
PATIENCE = 100              # from scratch: stop when nothing improves for this many epochs
BATCH = 64
RUN = "run1"                # name of the training run (a new name = a new run, the old one stays)
START_FROM = ""             # continue from the best.pt of this run; "" (or missing) = from scratch
EPOCHS_CONTINUE = 80        # when continuing: upper limit …
PATIENCE_CONTINUE = 25      # … and patience
DET_DATA = "/content/det_data"   # local (fast), rebuilt in every session
"""


def nb_detector():
    nb = Notebook("3_detector")
    nb.md("""
# 3 · Train the detector - finds the dice in the image

Trains **ESPDet-Pico** (a very small detector by Espressif for the ESP32-S3) on your renders - the first time
**from scratch**; later (e.g. after more renders) it can **continue from an earlier run** (`START_FROM`), which is much
shorter and does not forget anything because the old images train along. It only learns “there is a die here” - which
symbol is on top is the job of the classifier (notebook 4). At the end the detector is prepared for the ESP32
(ONNX + image sets) and stored in `export/`.

**Before:** notebook 2 is done, *Runtime → Change runtime type → L4 GPU*.

**If Colab stops:** run everything again. Training continues from the last saved state
(the checkpoints are on Drive under `projects/…/detector/`).
""")
    settings_cell(nb, DET_SETTINGS)
    nb.md(GPU_MD)
    nb.code(GPU_CODE)
    nb.md(DRIVE_MD)
    nb.code(DRIVE_CODE)
    nb.md(INSTALL_MD)
    nb.code(INSTALL_CODE)
    nb.md("""
## 5 · Build the dataset (about 10-15 minutes, in every new session)
Copies the renders (32 at a time, with progress) to the fast Colab disk and splits them into training (90 %) and
validation (10 %). Your real labelled throws from the trainer are added as **real test images** - they are never used
for training, only for measuring.
""")
    nb.code(run_code('["python", f"{SCRIPTS}/det_dataset.py", "--project", P, "--renders", RENDERS, "--output", DET_DATA]'))
    nb.md("""
## 6 · Timing: 1 epoch
Trains exactly one epoch and extrapolates how long `EPOCHS` would take (usually `PATIENCE` stops the training earlier).
Skipping this cell is fine. **Note:** the measurement includes the one-off start (filling the cache, building the
model) and therefore overestimates a lot. The real time per epoch is shown in cell 7.
""")
    nb.code("""
import shutil, time
START = f"{DET}/runs/{START_FROM}/weights/best.pt" if START_FROM else ""
CONTINUE = bool(START) and os.path.exists(START)
print(f"Run {RUN}: " + (f"continue from {START_FROM} (at most {EPOCHS_CONTINUE} epochs)" if CONTINUE
                        else f"from scratch (at most {EPOCHS} epochs)" + (f" - {START} is missing" if START else "")))
timing = f"{DET}/runs/timing"
if os.path.isdir(timing):
    shutil.rmtree(timing)
t0 = time.time()
r = subprocess.run(["python", f"{SCRIPTS}/det_train.py", "--data", f"{DET_DATA}/data.yaml", "--output", f"{DET}/runs",
                    "--name", "timing", "--imgsz", str(IMGSZ), "--epochs", "1", "--batch", str(BATCH),
                    "--esp-detection", ESP_DETECTION], capture_output=True, text=True)
s = time.time() - t0
print(r.stdout[-600:] if r.returncode == 0 else r.stdout[-2000:] + r.stderr[-2000:])
E = EPOCHS_CONTINUE if CONTINUE else EPOCHS
print(f"⏱ 1 epoch incl. start: {s:.0f} s  →  {E} epochs at most about {E * s / 3600:.1f} h "
      "(the start - filling the cache, building the model - happens only once, so the real time is lower)")
""")
    nb.md("""
## 7 · Training (takes a while - see the timing)
After every epoch one line: the validation (`mAP50`, recall, precision - should go up, 1.0 would be perfect), the
duration of the epoch, the best epoch so far and how many epochs are left at most (`PATIENCE` stops earlier). Checkpoints are saved to Drive continuously.
**Stopped?** Just run all cells again - this cell resumes automatically.
""")
    nb.code("""
run = f"{DET}/runs/{RUN}"
if os.path.exists(f"{run}/result.json"):
    print(f"✔ Run “{RUN}” is finished already - for a new attempt change RUN above.")
else:
    START = f"{DET}/runs/{START_FROM}/weights/best.pt" if START_FROM else ""
    CONTINUE = bool(START) and os.path.exists(START)
    cmd = ["python", f"{SCRIPTS}/det_train.py", "--data", f"{DET_DATA}/data.yaml", "--output", f"{DET}/runs",
           "--name", RUN, "--imgsz", str(IMGSZ), "--batch", str(BATCH), "--esp-detection", ESP_DETECTION]
    cmd += (["--start-from", START, "--epochs", str(EPOCHS_CONTINUE), "--patience", str(PATIENCE_CONTINUE)] if CONTINUE
            else ["--epochs", str(EPOCHS), "--patience", str(PATIENCE)])
    if os.path.exists(f"{run}/weights/last.pt"):
        cmd.append("--resume")
        print("resuming from", f"{run}/weights/last.pt")
""" + indent(run_code("cmd", 'z.startswith("  epoch ") or "final validation" in z or "✔" in z or "⏱" in z or "rror" in z or "continuing" in z or "Resuming" in z')) + "\n")
    nb.md("""
## 8 · Measure the result (test only, no new training)
Measures the finished detector **every time anew** on the current test images - also when the training was skipped.
So you get the values on **real** images as soon as you have added them (trainer, new package).
**mAP50** (0-1): how well boxes and dice match. **Recall**: share of dice found. **Precision**: share of boxes that
really are dice. The values on **real** images matter most; a recall clearly below 0.95 there needs a closer look.
When continuing, `START_FROM` is measured on **the same** test images (before/after comparison).
""")
    nb.code("""
run = f"{DET}/runs/{RUN}"
r = subprocess.run(["python", f"{SCRIPTS}/det_train.py", "--data", f"{DET_DATA}/data.yaml", "--output", f"{DET}/runs",
                    "--name", RUN, "--imgsz", str(IMGSZ), "--esp-detection", ESP_DETECTION, "--test-only"],
                   capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-1500:] + r.stderr[-1500:])
res = json.load(open(f"{run}/result.json"))
before = None
if START_FROM and START_FROM != RUN and os.path.exists(f"{DET}/runs/{START_FROM}/weights/best.pt"):
    subprocess.run(["python", f"{SCRIPTS}/det_train.py", "--data", f"{DET_DATA}/data.yaml", "--output", f"{DET}/runs",
                    "--name", START_FROM, "--imgsz", str(IMGSZ), "--esp-detection", ESP_DETECTION, "--test-only",
                    "--result", "result_compare.json"], capture_output=True, text=True)
    before = json.load(open(f"{DET}/runs/{START_FROM}/result_compare.json"))
for k in ("synthetic", "real"):
    if k in res:
        e = res[k]
        print(f"{k:10s} {RUN:8s}: mAP50 {e['mAP50']:.3f} · recall {e['recall']:.3f} · precision {e['precision']:.3f}")
        if before and k in before:
            a = before[k]
            print(f"{'':10s} {START_FROM:8s}: mAP50 {a['mAP50']:.3f} · recall {a['recall']:.3f} · precision {a['precision']:.3f}  (before)")
if "real" not in res:
    print("⚠ no real test images - result measured on synthetic images only")
""")
    nb.md("""
## 9 · Prepare for the ESP32 (export)
Saves the detector as `export/dice_det.onnx` with calibration and test images (`det_calib.npz`, `det_test.npz`,
`det_info.json`). The actual conversion into the ESP32 format happens afterwards **on the laptop**.
""")
    nb.code(run_code('["python", f"{SCRIPTS}/det_export.py", "--weights", f"{DET}/runs/{RUN}/weights/best.pt", "--project", P, '
                     '"--renders", RENDERS, "--esp-detection", ESP_DETECTION, "--imgsz", str(IMGSZ), "--output", EXPORT]')
            + """for f in sorted(os.listdir(EXPORT)):
    print(f"  {f:24s} {os.path.getsize(os.path.join(EXPORT, f)) / 1024:8.1f} KB")
""")
    nb.md("""
## What next?
Open **`notebooks/4_classifier.ipynb`**. When both are done, download the folder `export` from Drive to the laptop -
how is described in the trainer on the page **“Training & device”**.
""")
    end_session(nb)
    return nb


# ====================================================================== 4_classifier
CLS_SETTINGS = """
# name: (width, fast[, blur, color_cast]). "fast" = first stage with stride 2: half the compute.
# blur / color_cast (default 1.0) strengthen the degradation during training - for real camera images that are
# blurrier and have a stronger colour cast than the renders.
VARIANTS = {"s025": (0.25, True), "b025": (0.25, False), "b050": (0.5, False), "b100": (1.0, False),
            "s025f": (0.25, True, 1.0, 2.0),     # like s025, double colour cast
            "s025fu": (0.25, True, 1.5, 2.0)}    # like s025, double colour cast + 1.5× blur
EPOCHS = 60
TRAIN = ["s025f"]          # which variants are trained (compare all: list(VARIANTS))
CHOSEN = "s025f"           # exported variant; after the comparison (cell 8) change it if needed, then run cells 1 + 9
MODELS_FOLDER = "models"   # under classifier/; a new name keeps the older models for comparison
COMPARE_ALSO = []          # older models to measure along in cell 8, e.g. ["models_old/s025f"] (under classifier/)
CROPS_FOLDER = "crops"     # under classifier/; crops on Drive (built once, reused in every session)
"""


def nb_classifier():
    nb = Notebook("4_classifier")
    nb.md("""
# 4 · Train the classifier - recognises the top symbol

Trains a small convolutional network (CNN) **from scratch** that gets a crop around one die and says which symbol is on
top (or **“?”** if the die is tilted or not clear). The variants in `TRAIN` are trained (default: only `s025f`, the
fast design - the first stage halves the image right away, about half the compute on the ESP32). To try more designs:
`TRAIN = list(VARIANTS)`. Bigger is usually more accurate, but slower on the ESP32 (column “Compute”). At the end
`CHOSEN` is exported.

**Before:** notebook 2 is done, *Runtime → Change runtime type → L4 GPU*.
**If Colab stops:** run everything again - finished variants are skipped, started ones resume.
""")
    settings_cell(nb, CLS_SETTINGS, paths="""CLS_DATA = f"{CLS}/{CROPS_FOLDER}"
MODELS = f"{CLS}/{MODELS_FOLDER}"
""")
    nb.md(GPU_MD)
    nb.code(GPU_CODE)
    nb.md(DRIVE_MD)
    nb.code(DRIVE_CODE)
    nb.md(INSTALL_MD)
    nb.code(INSTALL_CODE)
    nb.md("""
## 5 · Build the crops (once; depending on Drive 15 minutes to 3 hours)
Cuts every die out of every render (a square around the die with a margin, exactly as later on the ESP32) and stores
them bundled on Drive. Real labelled throws are added as a real test set.
If the crops exist already, this cell takes seconds. New renders are cropped on their own; files whose labels have
changed are detected and rebuilt.

**Duration:** Drive sometimes delivers very slowly (e.g. with two sessions at the same time). The line
“… images/s” shows the speed. This cell needs **no GPU**: if it takes long, stop the cell as soon as a file
`crops_….npz` is finished (it is kept), *Change runtime type → CPU*, run cells 1-5, then L4 again and *Run all*.
""")
    nb.code(run_code('["python", f"{SCRIPTS}/cls_crops.py", "--project", P, "--renders", RENDERS, "--output", CLS_DATA, '
                     '"--size", str(CROP)]'))
    nb.md("""
## 6 · Timing: 1 epoch of the smallest variant
""")
    nb.code("""
import time
t0 = time.time()
r = subprocess.run(["python", f"{SCRIPTS}/cls_train.py", "--data", CLS_DATA, "--output", f"{CLS}/timing",
                    "--name", "timing", "--width", "0.25", "--epochs", "1"], capture_output=True, text=True)
s = time.time() - t0
print(r.stdout[-800:] if r.returncode == 0 else r.stdout[-2000:] + r.stderr[-2000:])
print(f"⏱ 1 epoch (width 0.25) incl. start: {s:.0f} s  →  {EPOCHS} epochs about {EPOCHS * s / 60:.0f} min; "
      "width 0.5 / 1.0 take about 1.5× / 3×")
""")
    nb.md("""
## 7 · Train the variants
Trains every variant in `TRAIN` one after the other. One line per epoch: loss (should go down) and accuracy on the
validation crops (should go up). Finished variants are skipped, started ones resume.
""")
    nb.code("""
for name in TRAIN:
    width, fast, *extra = VARIANTS[name]
    blur, color_cast = (extra + [1.0, 1.0])[:2]
    if os.path.exists(f"{MODELS}/{name}.done.json"):
        print(f"✔ {name} is done already"); continue
    cmd = ["python", f"{SCRIPTS}/cls_train.py", "--data", CLS_DATA, "--output", MODELS, "--name", name,
           "--width", str(width), "--epochs", str(EPOCHS)] + (["--fast"] if fast else []) + \\
          (["--blur", str(blur)] if blur != 1.0 else []) + (["--color-cast", str(color_cast)] if color_cast != 1.0 else [])
    if width >= 1.0:
        cmd += ["--lr", "0.001"]   # large variants need a smaller learning rate
    if os.path.exists(f"{MODELS}/{name}.ckpt"):
        cmd.append("--resume")
    print(f"=== {name} (width {width}{', fast' if fast else ''}"
          f"{f', blur {blur}' if blur != 1.0 else ''}{f', colour cast {color_cast}' if color_cast != 1.0 else ''}) ===")
""" + indent(run_code("cmd")) + "\n")
    nb.md("""
## 8 · Comparison
Table with size (parameters, estimated size on the ESP32), accuracy **per die** and **per throw** (all 5 dice right)
and the most frequent confusions. The values on **real** images matter most. It is measured **every time anew**
(test only, no new training), so real images you added later (new package, cell 5) are included right away.
Rule of thumb: take the variant with the smallest compute that is hardly worse per throw than the best one.
""")
    nb.code("""
from IPython.display import Markdown, display
done = [n for n in VARIANTS if os.path.exists(f"{MODELS}/{n}.done.json")]
also = [f"{CLS}/{a}.pth" for a in COMPARE_ALSO if os.path.exists(f"{CLS}/{a}.pth")]
r = subprocess.run(["python", f"{SCRIPTS}/cls_train.py", "--data", CLS_DATA, "--output", MODELS,
                    "--compare", *done, *also], capture_output=True, text=True)
print(r.stderr[-1500:] if r.returncode else "")
display(Markdown(open(f"{MODELS}/compare.md", encoding="utf-8").read()))
""")
    nb.md("""
## 9 · Prepare the chosen variant for the ESP32 (export)
Saves `export/dice_cls.onnx` with calibration and test crops (`cls_calib.npz`, `cls_test.npz`, `cls_info.json`).
Afterwards it continues **on the laptop**.
""")
    nb.code(run_code('["python", f"{SCRIPTS}/cls_export.py", "--model", f"{MODELS}/{CHOSEN}.pth", "--data", CLS_DATA, '
                     '"--project", P, "--output", EXPORT]')
            + """for f in sorted(os.listdir(EXPORT)):
    print(f"  {f:24s} {os.path.getsize(os.path.join(EXPORT, f)) / 1024:8.1f} KB")
""")
    nb.md("""
## Continue on the laptop
Drive now holds `projects/__PROJECT__/export/` with both models. Download the folder and put it into the project folder
of the trainer (`export/`). The commands for converting, checking and flashing are in the trainer on the page
**“Training & device”**.
""")
    end_session(nb)
    return nb


# ====================================================================== 5_all_in_one
def nb_all_in_one():
    nb = Notebook("5_all_in_one")
    nb.md("""
# 5 · All in one - render, train detector and classifier, export

For a run **overnight** in **one** session:
1. render the full set (finished chunks are skipped),
2. train the detector (from scratch, or from `START_FROM`), measure, export,
3. build the classifier crops, train, compare, export,
4. then **end** the session (saves compute units).

**Before:** notebook 1 has run once (check confirmed). Then: *Runtime → Change runtime type → L4 GPU*,
*Runtime → Run all*. Keep the browser tab open and do not let the laptop sleep.

**Stopped?** Just *Run all* again: finished chunks, crops and training runs are skipped, started ones resume.
The log is shown below and also stored on Drive (`projects/…/all_in_one.log`).
The previous export files are backed up to `export_backup_<time>/` before they are overwritten.
""")
    settings_cell(nb, RENDER_SETTINGS + "# detector (as notebook 3)" + DET_SETTINGS + "# classifier (as notebook 4)"
                  + CLS_SETTINGS.replace("EPOCHS = 60", "EPOCHS_CLS = 60") + """
END_SESSION = True          # at the end (also after an error) end the session - the results are on Drive
""", note="Same meaning as in notebooks 1-4.")
    nb.md(GPU_MD)
    nb.code(GPU_CODE)
    nb.md(DRIVE_MD)
    nb.code(DRIVE_CODE)
    nb.md(BLENDER_MD.replace("## 4 ·", "## 4 ·"))
    nb.code(BLENDER_CODE)
    nb.md(INSTALL_MD.replace("## 4 ·", "## 5 ·"))
    nb.code(INSTALL_CODE)
    nb.md("""
## 6 · All steps
All steps one after the other. For every step a heading with the time, shortened output and the duration. If a step
fails, the following steps stop (so that nothing runs on half the data) - the message is below and in the log.
Afterwards the session is ended (`END_SESSION`).
""")
    nb.code(CHECK_FUNCS + r'''
import datetime, math, re, shutil, subprocess, time
LOG = f"{P}/all_in_one.log"
CLS_DATA = f"{CLS}/{CROPS_FOLDER}"
MODELS = f"{CLS}/{MODELS_FOLDER}"
start_total = time.time()

def log(text):
    line = f"{datetime.datetime.now():%H:%M:%S} {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def step(title, cmd, filt=lambda z: True):
    log(f"=== {title}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        z = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line.rstrip("\n").split("\r")[-1])
        tail = (tail + [z])[-40:]
        if z.strip() and filt(z):
            print(z, flush=True)
    proc.wait()
    log(f"    {title}: {(time.time() - t0) / 60:.1f} min · return code {proc.returncode}")
    if proc.returncode:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(tail) + "\n")
        raise RuntimeError(f"step “{title}” failed (last lines above and in {LOG})")

def backup_export():
    if os.path.isdir(EXPORT) and os.listdir(EXPORT):
        target = f"{P}/export_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        shutil.copytree(EXPORT, target)
        log(f"    previous export backed up in {target}")

try:
    log(f"All in one {PROJECT} started · GPU {gpu or '-'}")
    # ---------------------------------------------------------------- 1. render
    if not (os.path.exists(f"{SCENE}/scene.json") and check_confirmed()):
        raise RuntimeError("Texture atlas not confirmed - first notebook 1, cells 5-7.")
    chunks = math.ceil(IMAGES / IMAGES_PER_CHUNK)
    step(f"Render ({IMAGES} images, chunks 0-{chunks - 1})",
         [BLENDER, "-b", "-P", f"{SCRIPTS}/render.py", "--", "--scene", SCENE, "--output", RENDERS,
          "--chunk-from", "0", "--chunk-to", str(chunks - 1), "--images", str(IMAGES_PER_CHUNK),
          "--seed", str(SEED), __ARGS__],
         lambda z: __FILTER__)
    step("Render statistics", ["python", f"{SCRIPTS}/render_stats.py", "--renders", RENDERS, "--scene", SCENE])
    # ---------------------------------------------------------------- 2. detector
    step("Detector: build the dataset", ["python", f"{SCRIPTS}/det_dataset.py", "--project", P, "--renders", RENDERS,
                                         "--output", DET_DATA], lambda z: "✔" in z or "⚠" in z)
    start = f"{DET}/runs/{START_FROM}/weights/best.pt" if START_FROM else ""
    run = f"{DET}/runs/{RUN}"
    base = ["python", f"{SCRIPTS}/det_train.py", "--data", f"{DET_DATA}/data.yaml", "--output", f"{DET}/runs",
            "--name", RUN, "--imgsz", str(IMGSZ), "--esp-detection", ESP_DETECTION]
    if os.path.exists(f"{run}/result.json"):
        log(f"    detector run {RUN} is finished already - training skipped")
    else:
        cmd = base + ["--batch", str(BATCH)]
        if os.path.exists(f"{run}/weights/last.pt"):
            cmd.append("--resume")
        elif start and os.path.exists(start):
            cmd += ["--start-from", start, "--epochs", str(EPOCHS_CONTINUE), "--patience", str(PATIENCE_CONTINUE)]
        else:
            if start:
                log(f"    {start} is missing - the detector is trained from scratch")
            cmd += ["--epochs", str(EPOCHS), "--patience", str(PATIENCE)]
        step(f"Detector: train ({RUN})", cmd, lambda z: z.startswith("  epoch ") or "final validation" in z or "✔" in z or "⏱" in z or "rror" in z
             or "continuing" in z or "Resuming" in z)
    step(f"Detector: measure ({RUN})", base + ["--test-only"], lambda z: "✔" in z)
    if start and START_FROM != RUN and os.path.exists(start):
        step(f"Detector: measure ({START_FROM}, before)",
             ["python", f"{SCRIPTS}/det_train.py", "--data", f"{DET_DATA}/data.yaml", "--output", f"{DET}/runs",
              "--name", START_FROM, "--imgsz", str(IMGSZ), "--esp-detection", ESP_DETECTION, "--test-only",
              "--result", "result_compare.json"], lambda z: "✔" in z)
    backup_export()
    step("Detector: export", ["python", f"{SCRIPTS}/det_export.py", "--weights", f"{run}/weights/best.pt",
                              "--project", P, "--renders", RENDERS, "--esp-detection", ESP_DETECTION,
                              "--imgsz", str(IMGSZ), "--output", EXPORT], lambda z: "✔" in z or "rror" in z)
    # ---------------------------------------------------------------- 3. classifier
    step("Classifier: crops (only new ones)", ["python", f"{SCRIPTS}/cls_crops.py", "--project", P, "--renders", RENDERS,
                                                "--output", CLS_DATA, "--size", str(CROP)],
         lambda z: "✔" in z or "⚠" in z or "crops from" in z or "rebuilt" in z)
    for name in TRAIN:
        width, fast, *extra = VARIANTS[name]
        blur, color_cast = (extra + [1.0, 1.0])[:2]
        if os.path.exists(f"{MODELS}/{name}.done.json"):
            log(f"    {name} is done already - skipped")
            continue
        cmd = ["python", f"{SCRIPTS}/cls_train.py", "--data", CLS_DATA, "--output", MODELS, "--name", name,
               "--width", str(width), "--epochs", str(EPOCHS_CLS)] + (["--fast"] if fast else []) + \
              (["--blur", str(blur)] if blur != 1.0 else []) + (["--color-cast", str(color_cast)] if color_cast != 1.0 else [])
        if width >= 1.0:
            cmd += ["--lr", "0.001"]
        if os.path.exists(f"{MODELS}/{name}.ckpt"):
            cmd.append("--resume")
        step(f"Classifier: train ({name})", cmd, lambda z: "poch" in z or "✔" in z or "rror" in z)
    done = [n for n in VARIANTS if os.path.exists(f"{MODELS}/{n}.done.json")]
    also = [f"{CLS}/{a}.pth" for a in COMPARE_ALSO if os.path.exists(f"{CLS}/{a}.pth")]
    step("Classifier: compare", ["python", f"{SCRIPTS}/cls_train.py", "--data", CLS_DATA, "--output", MODELS,
                                 "--compare", *done, *also], lambda z: False)
    step(f"Classifier: export ({CHOSEN})", ["python", f"{SCRIPTS}/cls_export.py", "--model", f"{MODELS}/{CHOSEN}.pth",
                                            "--data", CLS_DATA, "--project", P, "--output", EXPORT],
         lambda z: "✔" in z or "rror" in z)
    # ---------------------------------------------------------------- summary
    log("=== Result")
    for name, fname in ((RUN, "result.json"), (START_FROM, "result_compare.json")):
        f = f"{DET}/runs/{name}/{fname}"
        if name and os.path.exists(f) and (name == RUN or START_FROM != RUN):
            e = json.load(open(f))
            for k in ("synthetic", "real"):
                if k in e:
                    log(f"    detector {name:8s} {k:10s}: mAP50 {e[k]['mAP50']:.3f} · recall {e[k]['recall']:.3f} · "
                        f"precision {e[k]['precision']:.3f}")
    comparison = open(f"{MODELS}/compare.md", encoding="utf-8").read()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(comparison + "\n")
    try:
        from IPython.display import Markdown, display
        display(Markdown(comparison))
    except ImportError:
        print(comparison)
    for f in sorted(os.listdir(EXPORT)):
        log(f"    export/{f:24s} {os.path.getsize(os.path.join(EXPORT, f)) / 1024:8.1f} KB")
    log(f"✔ All in one done after {(time.time() - start_total) / 3600:.1f} h - continue on the laptop (trainer, “Training & device”)")
except Exception as e:
    log(f"✘ stopped: {e}")
finally:
    if END_SESSION:
        log("Ending the session (END_SESSION = True).")
        from google.colab import runtime
        runtime.unassign()
'''.replace("__ARGS__", RENDER_ARGS).replace("__FILTER__", RENDER_FILTER))
    return nb


BUILDERS = [nb_setup, nb_render, nb_detector, nb_classifier, nb_all_in_one]


def main():
    ap = argparse.ArgumentParser(description="Builds the Colab notebooks (1_setup … 5_all_in_one) and version.json.",
                                 epilog="Examples (from the repository folder):\n"
                                        "  .venv\\Scripts\\python.exe trainer\\colab\\build_notebooks.py\n"
                                        "  .venv\\Scripts\\python.exe trainer\\colab\\build_notebooks.py --force\n"
                                        "  (Linux/macOS: .venv/bin/python trainer/colab/build_notebooks.py --force)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks"),
                    help="target folder (default: notebooks/ next to this script)")
    ap.add_argument("--force", action="store_true", help="replace existing notebooks")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    files = {b.__name__: b() for b in BUILDERS}
    targets = [os.path.join(args.output, nb.name + ".ipynb") for nb in files.values()] + \
              [os.path.join(args.output, "version.json")]
    existing = [t for t in targets if os.path.exists(t)]
    if existing and not args.force:
        raise SystemExit("exists already (use --force to replace): " + ", ".join(os.path.basename(t) for t in existing))
    for nb in files.values():
        with open(os.path.join(args.output, nb.name + ".ipynb"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb.json(), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("written:", nb.name + ".ipynb")
    with open(os.path.join(args.output, "version.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(VERSIONS, f, indent=1)
        f.write("\n")
    print("written: version.json")


if __name__ == "__main__":
    main()
