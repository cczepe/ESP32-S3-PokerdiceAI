"""Access to the Pokerdice camera board (firmware in firmware/, home Wi-Fi).

API used (see docs/firmware.md):
  GET /api/info                                   firmware, sensor, resolutions, ...
  GET /api/capture?res=1024x768&exposure=-1       JPEG, auto exposure with correction -2..2
  GET /api/capture?res=1024x768&aec=300&gain=0    JPEG, fixed exposure
      answer headers: X-Resolution, X-Exposure ("auto level=0 actual_aec=512 actual_gain=1.50"), X-Sensor
  live view: http://<board>:81/stream
"""
import json
import urllib.error
import urllib.request

RESOLUTIONS = ["640x480", "800x600", "1024x768", "1280x1024", "1600x1200"]


class DeviceError(Exception):
    pass


def _get(url, timeout):
    """Returns (bytes, headers)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read(), r.headers
    except urllib.error.HTTPError as e:
        raise DeviceError(f"Camera board answers with error {e.code}: {e.read()[:200].decode('utf-8', 'replace')}")
    except (urllib.error.URLError, OSError) as e:
        raise DeviceError(f"Camera board not reachable at {url} ({getattr(e, 'reason', e)}). "
                          "Is it switched on and in the same Wi-Fi? Try its IP address instead of pokerdice.local.")


def _split(address):
    """'192.168.1.57' -> ('192.168.1.57', 80). 'host:8080' is for the PC preview (stream then on port+1)."""
    address = (address or "").strip()
    if not address:
        raise DeviceError("No camera board address entered (step 0 “Setup”).")
    if "//" in address:
        address = address.split("//", 1)[1]
    address = address.split("/")[0]
    if ":" in address:
        host, port = address.rsplit(":", 1)
        try:
            return host, int(port)
        except ValueError:
            raise DeviceError(f"Invalid address: {address}")
    return address, 80


def base_url(address):
    host, port = _split(address)
    return f"http://{host}" if port == 80 else f"http://{host}:{port}"


def stream_url(address):
    host, port = _split(address)
    return f"http://{host}:{81 if port == 80 else port + 1}/stream"


def info(address, timeout=5):
    data, _ = _get(base_url(address) + "/api/info", timeout)
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError:
        raise DeviceError("Unexpected answer - is the Pokerdice firmware running on the board?")


def parse_exposure(text):
    """'auto level=-1 actual_aec=312 actual_gain=1.50' -> {"mode": "auto", "level": -1, "actual_aec": 312, "actual_gain": 1.5}"""
    parts = (text or "").split()
    if not parts:
        return None
    result = {"mode": parts[0]}
    for t in parts[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            try:
                result[k] = float(v) if "." in v else int(v)
            except ValueError:
                result[k] = v
    return result


def capture(address, resolution=None, exposure=None, timeout=20):
    """Fresh single image. resolution e.g. "1024x768" (None = keep the current one).
    exposure: None (keep), {"level": -2..2} (auto with correction) or {"aec": n, "gain": 0..30} (fixed).
    Returns (jpeg_bytes, {"resolution": "1024x768", "exposure": {...}, "sensor": "OV3660"})."""
    q = []
    if resolution:
        if resolution not in RESOLUTIONS:
            raise DeviceError(f"Unknown resolution {resolution}. Possible: {', '.join(RESOLUTIONS)}")
        q.append("res=" + resolution)
    if exposure:
        if "aec" in exposure:
            q.append(f"aec={int(exposure['aec'])}&gain={int(exposure.get('gain', 0))}")
        else:
            q.append(f"exposure={int(exposure.get('level', 0))}")
    url = base_url(address) + "/api/capture" + ("?" + "&".join(q) if q else "")
    data, headers = _get(url, timeout)
    if not data.startswith(b"\xff\xd8"):
        raise DeviceError("The camera board did not deliver a JPEG.")
    return data, {"resolution": headers.get("X-Resolution"), "exposure": parse_exposure(headers.get("X-Exposure")),
                  "sensor": headers.get("X-Sensor")}


def fixed_from_actual(exposure):
    """Fixed exposure values from what the sensor reports in auto mode.
    actual_gain is a factor (1.0 = no gain); the firmware expects a gain step 0..30 (about factor - 1)."""
    if not exposure or "actual_aec" not in exposure:
        return None
    gain = int(round(float(exposure.get("actual_gain", 1.0)) - 1.0))
    return {"aec": max(1, int(exposure["actual_aec"])), "gain": max(0, min(30, gain))}
