"""Web page of the Pokerdice firmware on the PC, without a board (for working on main/web/index.html).

Serves main/web/index.html on port 8080 and answers the API with made-up values; the live view (port 8081)
and single images are generated test pictures. With --play the board "has models" and simulates rolls
(a new result every few seconds). Needs Pillow (pip install pillow).

Examples:
  python firmware/tools/web_preview.py
  python firmware/tools/web_preview.py --play
  python firmware/tools/web_preview.py --port 8080 --mode hotspot
  python firmware/tools/web_preview.py --help
"""
import argparse
import io
import json
import os
import random
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "main", "web", "index.html")
RESOLUTIONS = ["640x480", "800x600", "1024x768", "1280x1024", "1600x1200"]

state = {"resolution": "640x480", "captures": 0, "frames": 0,
         "detection": {"threshold": 35, "stable_n": 3, "stable_m": 5, "led": 1, "sort": 0, "det_cores": 0, "status_log": 1,
                       "model_threshold": 35},
         "camera": {
    "quality": 10, "vflip": 0, "hmirror": 0, "brightness": 0, "contrast": 0, "saturation": 0}}


def picture(res, text):
    w, h = (int(v) for v in res.split("x"))
    im = Image.new("RGB", (w, h), (24, 70, 44))
    d = ImageDraw.Draw(im)
    d.ellipse([w * 0.1, h * 0.1, w * 0.9, h * 0.9], outline=(10, 10, 10), width=max(4, w // 80))
    for i, (x, y) in enumerate([(0.35, 0.4), (0.55, 0.45), (0.45, 0.62)]):
        s = w * 0.06
        d.rectangle([w * x, h * y, w * x + s, h * y + s], fill=(235, 235, 228))
    d.text((10, 10), text, fill=(255, 255, 255))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return b.getvalue()


CLASSES = ["9", "10", "J", "Q", "K", "A", "?"]
RANK = [0, 1, 2, 3, 4, 5, -1]
HANDS = ["", "Nothing", "One Pair", "Two Pairs", "Three of a Kind", "Small Straight", "Large Straight",
         "Full House", "Four of a Kind", "Five of a Kind"]
PLAY_W, PLAY_H = 1024, 768
T0 = time.time()


def score_hand(ranks):
    if len(ranks) != 5 or min(ranks) < 0:
        return 0
    counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True) + [0]
    a, b = counts[0], counts[1]
    if a == 5: return 9
    if a == 4: return 8
    if a == 3 and b == 2: return 7
    if a == 3: return 4
    if a == 2 and b == 2: return 3
    if a == 2: return 2
    if max(ranks) - min(ranks) == 4: return 6 if max(ranks) == 5 else 5
    return 1


class Game:
    """Simulated detection: a new roll every 6 s (2 s rolling, then stable)."""
    def __init__(self):
        self.history, self.roll_no, self.frame, self.paused, self.cur = [], 0, 0, False, None

    def tick(self):
        k = int((time.time() - T0) // 6)
        phase = (time.time() - T0) % 6
        rnd = random.Random(k)
        cls = [rnd.randrange(6) for _ in range(5)]
        if rnd.random() < 0.15:
            cls[rnd.randrange(5)] = 6   # a tilted die
        pos = [(120 + i * 165 + rnd.randint(-30, 30), 300 + rnd.randint(-120, 160)) for i in range(5)]
        dice = [[x, y, x + 95, y + 95, round(0.8 + rnd.random() * 0.19, 3), c, round(0.75 + rnd.random() * 0.24, 3)]
                for (x, y), c in zip(pos, cls)]
        rolling = phase < 2
        if rolling:
            j = random.Random(int(time.time() * 3))
            dice = [[d[0] + j.randint(-20, 20), d[1] + j.randint(-20, 20), d[2], d[3], d[4], j.randrange(7), round(j.random(), 3)] for d in dice]
            for d in dice:
                d[2], d[3] = d[0] + 95, d[1] + 95
        status = "rolling" if rolling else ("searching" if 6 in cls else "stable")
        if status == "stable" and (not self.history or self.history[0]["k"] != k):
            ranks = sorted(cls, key=lambda c: -RANK[c])
            self.roll_no += 1
            self.history.insert(0, {"k": k, "cls": ranks, "hand": score_hand([RANK[c] for c in ranks]), "t": time.time()})
            del self.history[24:]
        self.cur = (status, dice)
        return status, dice

    def roll(self, r):
        return {"cls": r["cls"], "hand": r["hand"], "hand_name": HANDS[r["hand"]], "age_s": int(time.time() - r["t"])}


game = Game()


def play_picture(dice, boxes_text):
    im = Image.new("RGB", (PLAY_W, PLAY_H), (24, 70, 44))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 20, PLAY_W - 60, PLAY_H - 20], outline=(10, 10, 10), width=14)
    for x0, y0, x1, y1, _, c, _ in dice:
        d.rounded_rectangle([x0 + 6, y0 + 6, x1 - 6, y1 - 6], radius=12, fill=(235, 235, 228))
        d.text(((x0 + x1) / 2 - 6, (y0 + y1) / 2 - 6), CLASSES[c], fill=(20, 20, 20))
    d.text((10, 10), boxes_text, fill=(255, 255, 255))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return b.getvalue()


def make_handler(mode, play):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj):
            self.send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            if u.path == "/":
                with open(PAGE, "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif u.path == "/api/info":
                self.json({"firmware": "1.0.0", "board": "ESP32-S3-CAM", "sensor": "OV3660",
                           "camera_ok": True, "mode": "detection" if play else "camera", "models": play,
                           "resolution": state["resolution"],
                           "resolutions": RESOLUTIONS,
                           "wifi": {"mode": "station" if mode == "station" else "hotspot", "ssid": "HomeNet" if mode == "station" else "",
                                    "ip": "192.168.1.57" if mode == "station" else "", "rssi": -58, "hostname": "pokerdice.local",
                                    "hotspot": "Pokerdice", "hotspot_clients": 0 if mode == "station" else 1, "saved_mode": "home"},
                           "captures": state["captures"], "stream_frames": state["frames"], "stream_active": False,
                           "last_jpeg_bytes": 61234, "heap_free": 142000, "psram_free": 7300000, "uptime_s": int(time.time()) % 5000,
                           "model": {"config": play, "version": "07.11.26 10:00" if play else "-", "width": PLAY_W, "height": PLAY_H,
                                     "det_imgsz": 224, "cls_size": 64, "dice": 5, "unsure": 6, "det_kb": 674 if play else 0,
                                     "cls_kb": 99 if play else 0, "det_test": 1 if play else -1, "cls_test": 1 if play else -1,
                                     "strips_internal": True, "exposure": -1, "aec": -1, "gain": 0, "classes": CLASSES, "rank": RANK}})
            elif u.path == "/api/state":
                if not play:
                    return self.json({"frame": 0, "status": "no_models", "pause": False, "camera_busy": False,
                                      "message": "firmware/models/model_config.h missing", "candidates": 0, "dice": [], "roll_no": 0,
                                      "stable": None, "history": [], "mon": {}, "sys": {}})
                status, dice = game.tick()
                busy = time.time() < state.get("busy_until", 0)
                mon = {"fps": 1.6, "cam": 2.1, "jpg": 61.3, "box": 3.2, "det": 280.4, "dec": 0.42, "crop": 48.0, "cls": 205.5,
                       "total": 610.2, "strips": 38, "strips_total": 48, "jpeg_kb": 88.4}
                if q.get("timing") == "1":
                    mon["timing"] = [600 + 40 * ((i * 7) % 5) for i in range(60)]
                self.json({"frame": game.frame, "status": status, "pause": game.paused, "camera_busy": busy, "message": "",
                           "candidates": 9, "dice": dice, "roll_no": game.roll_no,
                           "stable": game.roll(game.history[0]) if game.history else None,
                           "history": [game.roll(r) for r in game.history],
                           "mon": mon, "sys": {"psram_free": 7020, "psram_min": 6890, "ram_free": 96, "ram_min": 71,
                                               "uptime_s": int(time.time() - T0)}})
            elif u.path == "/api/frame":
                if not play:
                    return self.send(204, b"", "image/jpeg")
                time.sleep(0.4)
                status, dice = game.tick()
                game.frame += 1
                det = {"frame": game.frame, "models": "own", "built": "07.11.26 10:00", "firmware": "1.0.0", "threshold": 0.35,
                       "width": PLAY_W, "height": PLAY_H, "status": "paused" if game.paused else status,
                       "hand": HANDS[game.history[0]["hand"]] if status == "stable" and game.history else "",
                       "dice": [] if game.paused else [{"x0": d[0], "y0": d[1], "x1": d[2], "y1": d[3], "score": d[4],
                                                        "class": CLASSES[d[5]], "cls": d[5], "p": d[6]} for d in dice]}
                self.send(200, play_picture(dice, f"frame {game.frame}"), "image/jpeg",
                          {"X-Frame": str(game.frame), "X-Detection": json.dumps(det)})
            elif u.path == "/api/settings":
                self.json(state["detection"])
            elif u.path == "/api/camera":
                c = dict(state["camera"])
                c["exposure"] = {"manual": False, "level": 0, "aec": 0, "gain": 0, "actual_aec": 512, "actual_gain": 1.5}
                self.json(c)
            elif u.path == "/api/capture":
                res = q.get("res", state["resolution"])
                if res not in RESOLUTIONS:
                    return self.send(400, b"unknown resolution (see /api/info)", "text/plain")
                state["resolution"] = res
                state["captures"] += 1
                state["busy_until"] = time.time() + 5
                # exposure like the firmware reports it: fixed values are used as given, auto picks something plausible
                if "aec" in q:
                    exp = f"manual aec={q['aec']} gain={q.get('gain', 0)} actual_aec={q['aec']} actual_gain={int(q.get('gain', 0)) + 1:.2f}"
                    state["exposure"] = exp
                elif "exposure" in q:
                    lv = int(q["exposure"])
                    exp = f"auto level={lv} actual_aec={512 + 60 * lv} actual_gain=1.50"
                    state["exposure"] = exp
                else:
                    exp = state.get("exposure", "auto level=0 actual_aec=512 actual_gain=1.50")
                self.send(200, picture(res, f"capture {res} exposure {q.get('exposure', q.get('aec', '-'))}"), "image/jpeg",
                          {"X-Resolution": res, "X-Exposure": exp, "X-Sensor": "OV3660"})
            elif u.path == "/api/wifi/scan":
                time.sleep(0.5)
                self.json([{"ssid": "HomeNet", "rssi": -52}, {"ssid": "Neighbour \"5G\"", "rssi": -80}])
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            form = dict(urllib.parse.parse_qsl(self.rfile.read(n).decode()))
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/camera":
                for k in state["camera"]:
                    if k in form:
                        state["camera"][k] = int(form[k])
                self.json(state["camera"])
            elif u.path == "/api/wifi":
                if form.get("mode", "home") == "home" and not form.get("ssid"):
                    return self.send(400, b"network name missing", "text/plain")
                self.json({"ok": True, "restart": True})
            elif u.path == "/api/settings":
                if form.get("defaults") == "1":
                    state["detection"].update({"threshold": 35, "stable_n": 3, "stable_m": 5, "led": 1, "sort": 0, "det_cores": 0,
                                               "status_log": 1})
                for k in state["detection"]:
                    if k in form:
                        state["detection"][k] = int(form[k])
                self.json(state["detection"])
            elif u.path == "/api/cmd":
                if form.get("do") == "pause":
                    game.paused = True
                elif form.get("do") == "continue":
                    game.paused = False
                elif form.get("do") == "clear":
                    game.history, game.roll_no = [], 0
                else:
                    return self.send(400, b"unknown command", "text/plain")
                self.json({"ok": True, "pause": game.paused})
            elif u.path in ("/api/wifi/forget", "/api/restart"):
                self.json({"ok": True, "restart": True})
            else:
                self.send(404, b"not found", "text/plain")
    return Handler


class Stream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=pokerdiceframe")
        self.end_headers()
        try:
            while True:
                state["frames"] += 1
                state["busy_until"] = time.time() + 5
                jpg = picture(state["resolution"], f"live {state['frames']}")
                self.wfile.write(b"\r\n--pokerdiceframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg)
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser(description="Pokerdice firmware web page on the PC with a simulated board.")
    ap.add_argument("--port", type=int, default=8080, help="web page port (the live view uses port+1)")
    ap.add_argument("--mode", choices=["station", "hotspot"], default="station", help="simulated Wi-Fi state")
    ap.add_argument("--play", action="store_true", help="simulate a board with models (detection, rolls)")
    args = ap.parse_args()
    threading.Thread(target=lambda: ThreadingHTTPServer(("127.0.0.1", args.port + 1), Stream).serve_forever(), daemon=True).start()
    print(f"open http://127.0.0.1:{args.port}/  (live view on port {args.port + 1})")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.mode, args.play)).serve_forever()


if __name__ == "__main__":
    main()
