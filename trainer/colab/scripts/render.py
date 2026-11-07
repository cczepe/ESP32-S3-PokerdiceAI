"""Render synthetic training images (runs INSIDE Blender).

Per image: 5 dice are thrown into the tray by physics; camera (elevation, distance, roll), light and
background are chosen at random within the ranges from scene.json. CLEAN images are saved
(size from scene.json "camera.resolution", e.g. 640x480 or 1024x768; JPEG quality 92); the degradation to
camera board quality happens only during training.

Output per chunk (folder <output>/chunk_000 …):
  images/000000.jpg …           images
  labels/000000.txt …           boxes in Ultralytics format: "0 cx cy w h" (one class "die", values 0-1)
  meta.jsonl                    one line per image: camera, light, per die box, symbol, class, tilt, visibility
  done.json                     written only at the end -> finished chunks are skipped when continuing
                                (only if all images and labels are really on Drive, otherwise rendered again)
Same seed = same throws (chunk n uses seed seed*1000+n).

Throw types (one run, mixed):
  normal        all dice are thrown
  keep          --share-keep: 2-4 "kept" dice lie upright at ONE place at the tray wall - pushed there quickly,
                so not neatly aligned (each turned ±25°, small gaps, 0-5 mm from the wall), mostly as a row
                along the wall, sometimes as a small block; the others are thrown in the middle.
                This is how the dice lie in a real game from the second throw on.
  lean_wall     --share-lean-wall: one die leans against the tray wall (class "?")
  lean_die      --share-lean-die: one die leans against ANOTHER die (class "?"), so that "?" does not
                mean "lies at the wall"

Call (in Colab):
  blender -b -P render.py -- --scene .../scene --output .../renders --chunk-from 0 --chunk-to 3 --images 200
  blender -b -P render.py -- --scene .../scene --output .../renders_preview --chunk-from 0 --chunk-to 0 --images 30
  blender -b -P render.py -- --scene .../scene --output .../renders --chunk-from 0 --chunk-to 111 --images 250 --share-keep 0.3
  blender -b -P render.py -- --help
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector, Quaternion  # noqa: E402
import blender_tools as bt  # noqa: E402

VERSION = 1


def parse_args():
    ap = argparse.ArgumentParser(description="Render synthetic dice images.")
    ap.add_argument("--scene", required=True, help="folder with scene.json (from prepare_textures.py)")
    ap.add_argument("--output", required=True, help="target folder for the chunks (e.g. .../renders)")
    ap.add_argument("--chunk-from", type=int, default=0, help="first chunk (default 0)")
    ap.add_argument("--chunk-to", type=int, default=0, help="last chunk, inclusive (default 0)")
    ap.add_argument("--images", type=int, default=200, help="images per chunk (default 200)")
    ap.add_argument("--seed", type=int, default=1, help="base seed (default 1)")
    ap.add_argument("--samples", type=int, default=32, help="Cycles samples (default 32; more = nicer, slower)")
    ap.add_argument("--share-close", type=float, default=0.35, help="share of throws with dice lying close together (default 0.35)")
    ap.add_argument("--share-lean-wall", type=float, default=0.2,
                    help="share of images in which a die leans against the tray wall -> class “?” (default 0.2)")
    ap.add_argument("--share-keep", type=float, default=0.2,
                    help="share of throws “keep and re-roll”: 2-4 dice at one place at the wall (default 0.2)")
    ap.add_argument("--share-lean-die", type=float, default=0.05,
                    help="share of images in which a die leans against another die -> “?” (default 0.05)")
    ap.add_argument("--dice", type=int, default=5, help="dice per image (default 5)")
    ap.add_argument("--cpu", action="store_true", help="render without GPU (only for testing)")
    ap.add_argument("--work-dir", help="local working folder (default: temporary)")
    return ap.parse_args(bt.script_args())


def random_rotation(rng):
    """Uniformly distributed random rotation (Shoemake). Three random angles would NOT be uniform -
    then some faces would land on top more often."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    return Quaternion((b * math.cos(2 * math.pi * u3), a * math.sin(2 * math.pi * u2),
                       a * math.cos(2 * math.pi * u2), b * math.sin(2 * math.pi * u3)))


class Scene:
    """Builds the scene once and then creates any number of throws/images."""

    def __init__(self, sc_data, folder, args):
        self.sd, self.args = sc_data, args
        self.sc = bt.empty_scene()
        self.device = bt.setup_render(self.sc, *sc_data["camera"]["resolution"], samples=args.samples, gpu=not args.cpu)
        self.k = sc_data["die"]["edge_mm"] / 1000.0
        self.bottom, self.wall, self.R = bt.create_tray(sc_data, folder)
        self.table, self.back, self.sur_mats = bt.table_and_background(sc_data, folder)
        template = bt.create_die(sc_data, folder, "Die0")
        self.dice = [template] + [bt.copy_die(template, f"Die{i}") for i in range(1, args.dice)]
        self.cam = bt.create_camera(sc_data)
        self.light = bt.create_light()
        self.world = bt.setup_world(self.sc)
        bt.setup_physics(self.sc)
        t = sc_data["tray"]
        for ob in bt.tray_collision(self.R, t.get("wall_width_mm", 12.0) / 1000.0, t["wall_height_mm"] / 1000.0):
            bt.rigid_body(ob, "PASSIVE", "BOX", friction=0.6, bounce=0.3)
        for w in self.dice:
            bt.rigid_body(w, "ACTIVE", "BOX", mass=0.005, friction=0.5, bounce=0.35)

    # -------------------------------------------------------------- throw
    def throw(self, rng):
        k, R = self.k, self.R
        self.leaning = None
        self.throw_type = "normal"
        for w in self.dice:
            if w.rigid_body.kinematic:
                w.rigid_body.kinematic = False
        kept = []
        # draw a random number only if the throw type is switched on: so a share of 0 gives exactly the same
        # throws as without this throw type (same seed = same throws)
        if self.args.share_keep > 0 and rng.random() < self.args.share_keep and len(self.dice) >= 3:
            kept = self.place_kept(rng, rng.randint(2, min(4, len(self.dice) - 1)))
            if kept:
                self.throw_type = "keep"
        close = rng.random() < self.args.share_close
        centre = Vector((rng.uniform(-0.3, 0.3) * R, rng.uniform(-0.3, 0.3) * R, 0))
        positions = [Vector((self.dice[i].location.x, self.dice[i].location.y, 0)) for i in kept]
        for i, w in enumerate(self.dice):
            if i in kept:
                continue
            for _ in range(50):
                if close:
                    p = centre + Vector((rng.gauss(0, 1.6 * k), rng.gauss(0, 1.6 * k), 0))
                else:
                    a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random()) * 0.65 * R
                    p = Vector((math.cos(a) * r, math.sin(a) * r, 0))
                if p.length < R - 0.9 * k and all((p - q).length > 1.5 * k for q in positions):
                    break
            positions.append(p)
            w.location = (p.x, p.y, k * (1.0 + 1.6 * i) + rng.uniform(0.005, 0.03))
            w.rotation_mode = "QUATERNION"
            w.rotation_quaternion = random_rotation(rng)
        bt.simulate(self.sc)
        # all dice still in the tray?
        for w in self.dice:
            m = w.matrix_world.translation
            if Vector((m.x, m.y)).length > self.R or m.z > 0.06 or m.z < -0.01:
                return False
        # keep the physics result (simulation off), so that a die can be moved afterwards
        self.end_poses = [w.matrix_world.copy() for w in self.dice]
        if rng.random() < self.args.share_lean_wall:
            self.lean_on_wall(rng)
        elif self.args.share_lean_die > 0 and rng.random() < self.args.share_lean_die:
            self.lean_on_die(rng)
        return True

    # -------------------------------------------------------------- keep: dice at the wall
    @staticmethod
    def _extent(angle):
        """Extent of a square (edge 1) turned by 'angle' in one direction: |cos| + |sin|."""
        return abs(math.cos(angle)) + abs(math.sin(angle))

    def place_kept(self, rng, count):
        """Lay count dice upright at ONE place at the wall (pushed there quickly) and hold them (kinematic, the
        thrown ones bounce off them). Returns the indices of the placed dice (empty if it does not fit)."""
        k, R = self.k, self.R
        idx = list(range(len(self.dice)))
        rng.shuffle(idx)
        idx = idx[:count]
        phi = rng.uniform(0, 2 * math.pi)
        block = count >= 3 and rng.random() < 0.25          # otherwise a row along the wall
        # symbols: mostly equal ones are kept (pair, three of a kind …), sometimes not
        fields = [rng.choice(bt.FIELDS)] * count if rng.random() < 0.6 else [rng.choice(bt.FIELDS) for _ in idx]
        if rng.random() < 0.5 and count >= 3:
            fields[-1] = rng.choice(bt.FIELDS)
        yaws = [rng.uniform(-25, 25) for _ in idx]          # turn against the wall direction (degrees)
        # position per die: (angle at the wall, distance to the wall) - row: side by side along the wall
        poses = []
        row = idx if not block else idx[:2]
        inner = [] if not block else idx[2:]
        angle = phi
        for n, i in enumerate(row):
            e = self._extent(math.radians(yaws[idx.index(i)]))
            gap_wall = rng.choice([rng.uniform(0, 0.0015), rng.uniform(0, 0.0015), rng.uniform(0.0015, 0.005)])
            r = R - k / 2 * e - gap_wall - 0.0005
            if n:
                e_prev = self._extent(math.radians(yaws[idx.index(row[n - 1])]))
                chord = k / 2 * (e + e_prev) + rng.uniform(0.0003, 0.003)    # gap 0.3-3 mm
                angle += chord / r
            poses.append((i, angle, r))
        for n, i in enumerate(inner):                         # second row inside the first one (block)
            e = self._extent(math.radians(yaws[idx.index(i)]))
            i0, w0, r0 = poses[min(n, len(poses) - 1)]
            e0 = self._extent(math.radians(yaws[idx.index(i0)]))
            r = r0 - k / 2 * (e + e0) - rng.uniform(0.0003, 0.003)
            poses.append((i, w0 + rng.uniform(-0.15, 0.15) * k / r0, r))
        # distribute evenly around phi (the row should not only grow in one direction)
        mid = (min(w for _, w, _ in poses) + max(w for _, w, _ in poses)) / 2
        for i, w, r in poses:
            w = w - mid + phi
            field = fields[idx.index(i)]
            to_top = Vector(bt.NORMAL[field]).rotation_difference(Vector((0, 0, 1)))
            yaw = Quaternion((0, 0, 1), w + math.radians(yaws[idx.index(i)]) + rng.choice([0, 0.5, 1, 1.5]) * math.pi)
            ob = self.dice[i]
            ob.rigid_body.kinematic = True
            mat = (yaw @ to_top).to_matrix().to_4x4()
            mat.translation = Vector((math.cos(w) * r, math.sin(w) * r, k / 2 + 0.0002))
            ob.matrix_world = mat
        bpy.context.view_layer.update()
        return idx

    # -------------------------------------------------------------- leaning against another die
    def lean_on_die(self, rng):
        """A die leans (22-45°) against the side face of another die lying flat -> "?"."""
        k, R = self.k, self.R
        n = len(self.dice)
        for _ in range(20):
            i, j = rng.sample(range(n), 2)
            mj = self.end_poses[j]
            rot = mj.to_3x3().normalized()
            if bt.top_face(self.dice[j])[1] > 5:
                continue                                       # the support must lie flat
            sides = [rot @ Vector(bt.NORMAL[f]) for f in bt.FIELDS]
            sides = [Vector((v.x, v.y, 0)).normalized() for v in sides if abs(v.z) < 0.2]
            if not sides:
                continue
            u = rng.choice(sides)                              # points away from j, towards the leaning die
            tilt = math.radians(rng.uniform(22, 45))
            cj = Vector((mj.translation.x, mj.translation.y, 0))
            edge = cj + u * (k / 2 + k * math.cos(tilt) + 0.0003)
            o = -u                                             # direction to the "wall" (die j)
            f = o * math.cos(tilt) + Vector((0, 0, math.sin(tilt)))
            nrm = -o * math.sin(tilt) + Vector((0, 0, math.cos(tilt)))
            centre = edge + f * (k / 2) + nrm * (k / 2)
            if Vector((centre.x, centre.y)).length > R - 0.9 * k:
                continue
            free = all((Vector((m.translation.x, m.translation.y, 0)) - Vector((centre.x, centre.y, 0))).length > 1.5 * k
                       for jj, m in enumerate(self.end_poses) if jj not in (i, j))
            if not free:
                continue
            t = Vector((-o.y, o.x, 0))
            field = rng.choice(bt.FIELDS)
            to_top = Vector(bt.NORMAL[field]).rotation_difference(Vector((0, 0, 1)))
            yaw = Quaternion((0, 0, 1), math.atan2(o.y, o.x) + rng.choice([0, 0.5, 1, 1.5]) * math.pi)
            tip = Quaternion(t, -tilt)
            w = self.dice[i]
            w.rigid_body.kinematic = True
            mat = (tip @ yaw @ to_top).to_matrix().to_4x4()
            mat.translation = centre
            w.matrix_world = mat
            self.leaning = i
            self.throw_type = self.throw_type + "+lean_die"
            return True
        return False

    def lean_on_wall(self, rng):
        """A die leans against the tray wall (tilt 20-50°): lower edge on the felt, upper edge at the wall."""
        k, R = self.k, self.R
        for _ in range(20):
            i = rng.randrange(len(self.dice))
            phi = rng.uniform(0, 2 * math.pi)
            tilt = math.radians(rng.uniform(22, 50))
            u = Vector((math.cos(phi), math.sin(phi), 0))       # outwards
            t = Vector((-math.sin(phi), math.cos(phi), 0))      # along the wall
            edge = u * (R - k * math.cos(tilt) - 0.0005)
            f = u * math.cos(tilt) + Vector((0, 0, math.sin(tilt)))
            n = -u * math.sin(tilt) + Vector((0, 0, math.cos(tilt)))
            centre = edge + f * (k / 2) + n * (k / 2)
            free = all((Vector((m.translation.x, m.translation.y, 0)) - Vector((centre.x, centre.y, 0))).length > 1.6 * k
                       for j, m in enumerate(self.end_poses) if j != i)
            if not free:
                continue
            field = rng.choice(bt.FIELDS)
            to_top = Vector(bt.NORMAL[field]).rotation_difference(Vector((0, 0, 1)))
            yaw = Quaternion((0, 0, 1), rng.uniform(0, 2 * math.pi))
            tip = Quaternion(t, -tilt)
            w = self.dice[i]
            w.rigid_body.kinematic = True          # no longer moved by the physics
            mat = (tip @ yaw @ to_top).to_matrix().to_4x4()
            mat.translation = centre
            w.matrix_world = mat
            self.leaning = i
            self.throw_type = self.throw_type + "+lean_wall"
            return True
        return False

    # -------------------------------------------------------------- camera, light, background
    def view(self, rng):
        cam = self.sd["camera"]
        e = rng.uniform(*cam["elevation_deg"])
        d = rng.uniform(*cam["distance_cm"]) / 100.0
        a = rng.uniform(0, 360)
        roll = rng.uniform(-cam["roll_deg"], cam["roll_deg"])
        # target: mostly the group of dice, sometimes the tray centre (then dice also at the image border)
        centre = sum((w.matrix_world.translation for w in self.dice), Vector()) / len(self.dice)
        target = centre if rng.random() < 0.7 else Vector((rng.uniform(-0.3, 0.3) * self.R, rng.uniform(-0.3, 0.3) * self.R, 0))
        target = Vector((target.x + rng.gauss(0, 0.01), target.y + rng.gauss(0, 0.01), 0))
        bt.aim_camera(self.cam, target, e, a, d, roll)
        # light
        le, la = rng.uniform(30, 85), rng.uniform(0, 360)
        bt.aim_camera(self.light, (0, 0, 0), le, la, rng.uniform(0.6, 1.8), 0)
        self.light.data.size = rng.uniform(0.1, 1.2)
        self.light.data.energy = rng.uniform(4, 35)
        kelvin = rng.uniform(2700, 6500)
        self.light.data.color = bt.color_temperature_rgb(kelvin)
        self.world.inputs["Color"].default_value = (*bt.color_temperature_rgb(rng.uniform(3500, 7500)), 1)
        self.world.inputs["Strength"].default_value = rng.uniform(0.04, 0.35)
        self.sc.view_settings.exposure = rng.uniform(-0.5, 0.3)
        # background: table + wall behind the tray (seen from the camera)
        bt.set_material(self.table, rng.choice(self.sur_mats))
        wm = rng.choice(self.sur_mats)
        bt.set_material(self.back, wm)
        ar = math.radians(a + 180)
        self.back.location = (math.cos(ar) * 0.9, math.sin(ar) * 0.9, 0.5)
        self.back.rotation_euler = (math.radians(90), 0, ar - math.radians(90))
        self.table.rotation_euler = (0, 0, rng.uniform(0, 2 * math.pi))
        return {"elevation_deg": round(e, 2), "distance_cm": round(d * 100, 1), "azimuth_deg": round(a, 1),
                "roll_deg": round(roll, 2), "light_k": round(kelvin), "light_elevation_deg": round(le, 1),
                "exposure": round(self.sc.view_settings.exposure, 2)}

    # -------------------------------------------------------------- labels
    def labels(self):
        sd, sc = self.sd, self.sc
        classes = sd["classes"]
        unclear = classes.index("?") if "?" in classes else len(classes) - 1
        dg = bpy.context.evaluated_depsgraph_get()
        result = []
        for w in self.dice:
            box, vis = bt.box_in_image(sc, self.cam, w, self.k)
            if box is None or vis < sd["labels"]["min_visible"]:
                continue
            field, tilt = bt.top_face(w)
            symbol = sd["faces"][field]["symbol"]
            top_vis = bt.top_visible(sc, self.cam, w, self.k, field, dg)
            tilted = tilt > sd["labels"]["tilt_deg"]
            cls = unclear if (tilted or top_vis < 0.5) else classes.index(symbol)
            x0, y0, x1, y1 = box
            result.append({"box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                           "symbol": symbol, "class": cls, "class_name": classes[cls],
                           "tilt_deg": round(tilt, 1), "tilted": tilted,
                           "visible": round(vis, 2), "top_visible": round(top_vis, 2),
                           "height_mm": round(w.matrix_world.translation.z * 1000, 1),
                           "wall_distance_mm": round((self.R - Vector((w.matrix_world.translation.x,
                                                                       w.matrix_world.translation.y)).length) * 1000, 1)})
        return result


def missing_files(folder, source=None):
    """Relative paths (images/…, labels/…) missing in 'folder'. Without 'source': target list from meta.jsonl;
    with 'source': all files of the local working copy."""
    if source:
        wanted = [os.path.relpath(os.path.join(w, d), source) for w, _, ds in os.walk(source) for d in ds]
    else:
        wanted = ["meta.jsonl", "done.json"]
        try:
            with open(os.path.join(folder, "meta.jsonl"), encoding="utf-8") as m:
                for line in m:
                    if line.strip():
                        b = json.loads(line)["image"]
                        wanted += [f"images/{b}", f"labels/{b[:-4]}.txt"]
        except (OSError, ValueError):
            return ["meta.jsonl"]
    present = {}
    result = []
    for rel in wanted:
        part, name = os.path.split(rel)
        if part not in present:
            try:
                present[part] = set(os.listdir(os.path.join(folder, part)))
            except OSError:
                present[part] = set()
        if name not in present[part]:
            result.append(rel)
    return result


def render_chunk(scene, nr, args, out_root, sd):
    target = os.path.join(out_root, f"chunk_{nr:03d}")
    if os.path.isfile(os.path.join(target, "done.json")):
        missing = missing_files(target)
        if not missing:
            print(f"Chunk {nr:03d}: done already - skipped")
            return 0, 0.0
        print(f"⚠ Chunk {nr:03d}: {len(missing)} file(s) missing on Drive ({', '.join(missing[:5])}) - rendering it again",
              flush=True)
    work = tempfile.mkdtemp(prefix=f"chunk_{nr:03d}_", dir=args.work_dir)
    os.makedirs(os.path.join(work, "images"))
    os.makedirs(os.path.join(work, "labels"))
    seed = args.seed * 1000 + nr
    rng = random.Random(seed)
    sc = scene.sc
    t0 = time.time()
    with open(os.path.join(work, "meta.jsonl"), "w", encoding="utf-8") as meta:
        for i in range(args.images):
            ti = time.time()
            for attempt in range(5):
                if scene.throw(rng):
                    break
            camera = scene.view(rng)
            name = f"{i:06d}"
            sc.render.filepath = os.path.join(work, "images", name + ".jpg")
            bt.render(sc)
            lab = scene.labels()
            w, h = sc.render.resolution_x, sc.render.resolution_y
            with open(os.path.join(work, "labels", name + ".txt"), "w") as f:
                for d in lab:
                    x0, y0, x1, y1 = d["box"]
                    f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
            meta.write(json.dumps({"image": name + ".jpg", "size": [w, h], "camera": camera, "dice": lab,
                                   "throw_type": scene.throw_type, "attempts": attempt + 1,
                                   "seconds": round(time.time() - ti, 2)}, ensure_ascii=False) + "\n")
            if i == 0 or (i + 1) % 10 == 0:
                print(f"  chunk {nr:03d}: image {i + 1}/{args.images}, {time.time() - ti:.1f} s/image", flush=True)
    duration = time.time() - t0
    with open(os.path.join(work, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"chunk": nr, "images": args.images, "seed": seed, "seconds": round(duration, 1),
                   "s_per_image": round(duration / max(1, args.images), 3), "device": scene.device,
                   "samples": args.samples, "classes": sd["classes"], "version": VERSION,
                   "shares": {"close": args.share_close, "lean_wall": args.share_lean_wall, "keep": args.share_keep,
                              "lean_die": args.share_lean_die}}, f, indent=1)
    # copy to Drive completely first, then rename (atomic: half chunks never exist)
    os.makedirs(out_root, exist_ok=True)
    tmp_target = target + ".tmp"
    if os.path.exists(tmp_target):
        shutil.rmtree(tmp_target)
    shutil.copytree(work, tmp_target)
    # Drive occasionally loses single files when copying -> check and copy missing ones again one by one
    for attempt in range(3):
        missing = missing_files(tmp_target, work)
        if not missing:
            break
        print(f"⚠ Chunk {nr:03d}: {len(missing)} file(s) did not arrive on Drive - copying again ({attempt + 1}/3)",
              flush=True)
        time.sleep(5)
        for rel in missing:
            shutil.copyfile(os.path.join(work, rel), os.path.join(tmp_target, rel))
    else:
        if missing_files(tmp_target, work):
            raise RuntimeError(f"Chunk {nr:03d}: files do not arrive on Drive")
    if os.path.exists(target):
        shutil.rmtree(target)   # unfinished rest of an aborted run
    os.rename(tmp_target, target)
    shutil.rmtree(work, ignore_errors=True)
    print(f"✔ Chunk {nr:03d} done: {args.images} images in {duration / 60:.1f} min ({duration / max(1, args.images):.2f} s/image)", flush=True)
    return args.images, duration


def main():
    args = parse_args()
    sd = bt.load_scene(args.scene)
    # freeze the class list: recorded at the first rendering, afterwards it must stay the same
    os.makedirs(args.output, exist_ok=True)
    cf = os.path.join(args.output, "classes.json")
    if os.path.isfile(cf):
        with open(cf, encoding="utf-8") as f:
            old = json.load(f)["classes"]
        if old != sd["classes"]:
            sys.exit(f"ERROR: the class list has changed ({old} -> {sd['classes']}). It is frozen since the first rendering.")
    else:
        with open(cf, "w", encoding="utf-8") as f:
            json.dump({"classes": sd["classes"], "frozen": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    scene = Scene(sd, args.scene, args)
    print(f"Scene built - render device: {scene.device}, {args.samples} samples", flush=True)
    total_i, total_s = 0, 0.0
    for nr in range(args.chunk_from, args.chunk_to + 1):
        i, s = render_chunk(scene, nr, args, args.output, sd)
        total_i += i
        total_s += s
    if total_i:
        print(f"✔ done: {total_i} images, {total_s / total_i:.2f} s/image on average")


if __name__ == "__main__":
    main()
