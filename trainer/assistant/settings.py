"""Local settings of the trainer (valid for all projects).

Stored in the user folder under  ~/.pokerdice/  (Windows: %USERPROFILE%\\.pokerdice\\)
  settings.json  - camera board address, Claude model, Drive folder name ...
  api_key.txt    - Anthropic API key (only here, never in project folders, Drive or the repository)

An environment variable ANTHROPIC_API_KEY takes precedence over the file.
"""
import json
import os

DEFAULTS = {
    "model": "claude-sonnet-5-5",
    "device_address": "pokerdice.local",
    "drive_folder": "Pokerdice",
    "fov_diag_deg": 66.0,   # typical lens of OV2640/OV3660 modules; only used to estimate the distance in step 1
}


def config_folder():
    return os.path.join(os.path.expanduser("~"), ".pokerdice")


def _path(name):
    return os.path.join(config_folder(), name)


def _write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load():
    data = dict(DEFAULTS)
    try:
        with open(_path("settings.json"), encoding="utf-8") as f:
            data.update(json.load(f))
    except (OSError, ValueError):
        pass
    return data


def save(data):
    allowed = {k: data[k] for k in DEFAULTS if k in data}
    old = load()
    old.update(allowed)
    _write_atomic(_path("settings.json"), json.dumps(old, indent=2, ensure_ascii=False))
    return old


def api_key():
    """Returns (key, source) - source is 'environment', 'file' or None."""
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k, "environment"
    try:
        with open(_path("api_key.txt"), encoding="utf-8") as f:
            k = f.read().strip()
        if k:
            return k, "file"
    except OSError:
        pass
    return None, None


def api_key_save(key):
    _write_atomic(_path("api_key.txt"), key.strip() + "\n")


def api_key_delete():
    try:
        os.remove(_path("api_key.txt"))
    except OSError:
        pass


def masked(k):
    if not k:
        return ""
    return k[:7] + "…" + k[-4:] if len(k) > 14 else "…"
