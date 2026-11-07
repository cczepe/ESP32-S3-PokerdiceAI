"""Project management: one project = one folder under <workspace>/projects/ with project.json.

Rules:
- Projects never overwrite each other (same folder name -> error).
- Photos are never overwritten: new files get the next free number.
- project.json is written atomically (first .tmp, then rename).
"""
import datetime
import json
import os
import re
import shutil
import unicodedata

SCHEMA_VERSION = 1

SUBFOLDERS = [
    "camera_setup",
    os.path.join("photos", "faces"),
    os.path.join("photos", "corners"),
    os.path.join("photos", "tray"),
    os.path.join("photos", "felt"),
    os.path.join("photos", "surroundings"),
    "raw",
    "test_images",
]

# Steps of the trainer. Step 0 (setup) is global, not per project.
STEPS = [
    {"id": "1_camera_setup", "nr": 1, "title": "Camera setup (feasibility check)", "phase": "0"},
    {"id": "2_measure", "nr": 2, "title": "Measure the die", "phase": "A"},
    {"id": "3_faces", "nr": 3, "title": "Photograph the 6 faces", "phase": "A"},
    {"id": "4_corners", "nr": 4, "title": "4 corner photos / die layout", "phase": "A"},
    {"id": "5_tray", "nr": 5, "title": "Dice tray", "phase": "A"},
    {"id": "6_felt", "nr": 6, "title": "Felt", "phase": "A"},
    {"id": "7_surroundings", "nr": 7, "title": "Surroundings", "phase": "A"},
    {"id": "8_throws", "nr": 8, "title": "Capture real throws", "phase": "D"},
    {"id": "9_label", "nr": 9, "title": "Label", "phase": "D"},
]

# Parts that can be copied from an existing project when creating a new one
PARTS = {
    "tray": {"title": "Dice tray (size + photos)", "folders": [os.path.join("photos", "tray")], "json": ["tray"], "steps": ["5_tray"]},
    "felt": {"title": "Felt photo", "folders": [os.path.join("photos", "felt")], "json": [], "steps": ["6_felt"]},
    "surroundings": {"title": "Surroundings photos", "folders": [os.path.join("photos", "surroundings")], "json": [], "steps": ["7_surroundings"]},
    "camera_setup": {"title": "Camera setup (only useful for dice of the same size)", "folders": ["camera_setup"],
                     "json": ["camera_setup", "camera"], "steps": ["1_camera_setup"]},
}


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def projects_folder(workspace):
    return os.path.join(workspace, "projects")


def folder_name(name):
    """File-system friendly folder name: 'Blue Dice' -> 'Blue_Dice', 'Café' -> 'Cafe'."""
    s = name.strip()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    s = re.sub(r"_+", "_", s).strip("_-")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if s.upper() in reserved:
        s = s + "_project"
    return s[:60]


def empty_project(name, folder):
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "folder": folder,
        "created": now(),
        "modified": now(),
        "copied_from": None,
        "die": {"edge_mm": None, "corner_radius_mm": None, "body_color": None},
        "symbols": {"faces": [], "classes": [], "frozen": False},
        "layout": None,
        "tray": {"diameter_mm": None, "wall_height_mm": None},
        "surface": None,          # colour of the felt (measured by click), None = default "green"
        "symbol_hints": "",       # free text for the AI ("king: red figure ...") until step 3 records the faces
        "camera_setup": None,
        "steps": {s["id"]: {"status": "open", "modified": None} for s in STEPS},
    }


CAMERA_DEFAULT = {"resolution": "640x480", "exposure_level": 0, "fixed_exposure": None}
CAMERA_RESOLUTIONS = ["640x480", "800x600", "1024x768", "1280x1024", "1600x1200"]


def camera(data):
    """Camera setting of the project (from step 1):
    {"resolution": "1024x768", "width", "height", "exposure_level": -2..2, "fixed_exposure": {"aec", "gain"} or None}.
    Valid for test images (step 8), renders and firmware."""
    k = dict(CAMERA_DEFAULT)
    k.update({x: v for x, v in (data.get("camera") or {}).items() if x in CAMERA_DEFAULT})
    if k["resolution"] not in CAMERA_RESOLUTIONS:
        k["resolution"] = CAMERA_DEFAULT["resolution"]
    k["exposure_level"] = max(-2, min(2, int(k.get("exposure_level") or 0)))
    fx = k.get("fixed_exposure")
    k["fixed_exposure"] = {"aec": int(fx["aec"]), "gain": int(fx["gain"])} if isinstance(fx, dict) and "aec" in fx else None
    k["width"], k["height"] = [int(v) for v in k["resolution"].split("x")]
    return k


def capture_exposure(data):
    """Exposure for captures of this project: fixed values if saved, otherwise auto with correction."""
    k = camera(data)
    return dict(k["fixed_exposure"]) if k["fixed_exposure"] else {"level": k["exposure_level"]}


def path(workspace, folder, *parts):
    return os.path.join(projects_folder(workspace), folder, *parts)


def list_projects(workspace):
    po = projects_folder(workspace)
    if not os.path.isdir(po):
        return []
    result = []
    for name in sorted(os.listdir(po), key=str.lower):
        pj = os.path.join(po, name, "project.json")
        if os.path.isfile(pj):
            try:
                d = load(workspace, name)
            except (OSError, ValueError) as e:
                result.append({"folder": name, "name": name, "error": str(e)})
                continue
            done = sum(1 for s in d["steps"].values() if s.get("status") == "done")
            result.append({"folder": name, "name": d.get("name", name), "done": done,
                           "total": len(STEPS), "modified": d.get("modified")})
    return result


def load(workspace, folder):
    with open(path(workspace, folder, "project.json"), encoding="utf-8") as f:
        d = json.load(f)
    # add missing fields with defaults
    template = empty_project(d.get("name", folder), folder)
    for k, v in template.items():
        d.setdefault(k, v)
    for s in STEPS:
        d["steps"].setdefault(s["id"], {"status": "open", "modified": None})
    return d


def save(workspace, folder, data):
    data["modified"] = now()
    target = path(workspace, folder, "project.json")
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, target)


def set_step(data, step_id, status):
    data["steps"][step_id] = {"status": status, "modified": now()}


def create(workspace, name, source=None, parts=()):
    """Creates a new project. Raises ValueError with an understandable message."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Please enter a project name.")
    folder = folder_name(name)
    if not folder:
        raise ValueError("The name contains no usable characters (letters/digits).")
    target = path(workspace, folder)
    if os.path.exists(target):
        raise ValueError(f"There is already a folder “{folder}”. Please choose another name.")

    source_data = None
    if source:
        if not os.path.isfile(path(workspace, source, "project.json")):
            raise ValueError(f"Source project “{source}” not found.")
        source_data = load(workspace, source)

    os.makedirs(target)
    for u in SUBFOLDERS:
        os.makedirs(os.path.join(target, u), exist_ok=True)
    data = empty_project(name, folder)

    if source_data and parts:
        copied = []
        for t in parts:
            if t not in PARTS:
                continue
            info = PARTS[t]
            for u in info["folders"]:
                q = path(workspace, source, u)
                if os.path.isdir(q):
                    for fn in os.listdir(q):
                        qd = os.path.join(q, fn)
                        if os.path.isfile(qd):
                            shutil.copy2(qd, os.path.join(target, u, fn))  # target is new and empty
            for k in info["json"]:
                data[k] = source_data.get(k)
            for s in info["steps"]:
                data["steps"][s] = dict(source_data["steps"].get(s, {"status": "open"}))
            copied.append(t)
        data["copied_from"] = {"project": source, "parts": copied, "at": now()}

    save(workspace, folder, data)
    return folder


def next_file_name(folder_path, ext=".jpg"):
    """Next free number: 0001.jpg, 0002.jpg ... (never overwrite)."""
    os.makedirs(folder_path, exist_ok=True)
    highest = 0
    for d in os.listdir(folder_path):
        m = re.match(r"^(\d{4,})\.", d)
        if m:
            highest = max(highest, int(m.group(1)))
    n = highest + 1
    while True:
        name = f"{n:04d}{ext}"
        if not os.path.exists(os.path.join(folder_path, name)):
            return name
        n += 1


def write_new(target_path, data_bytes):
    """Writes only if the file does not exist yet (exclusive)."""
    with open(target_path, "xb") as f:
        f.write(data_bytes)
