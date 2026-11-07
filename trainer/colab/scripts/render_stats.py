"""Statistics of the renders: throw types, dice at the wall, rows of touching dice, camera angles,
"?" for upright dice at the wall and overlapping boxes.

Reads only the labels (meta.jsonl of the finished chunks) - no images, no GPU, takes 1-2 minutes.
  die at the wall  = centre at most --wall-mm from the wall (default 1.2 edge lengths)
  wall row         = at least 2 dice at the wall whose boxes touch or overlap in the image
  upright          = tilt at most tilt_deg from scene.json (above it "?" is right)

Questions behind it:
  1. Do UPRIGHT dice at the wall often get "?" in the renders? Then the classifier would learn "at the wall = ?"
     although the real camera sees the top face.
  2. How often do the boxes of two dice overlap so much that the firmware (NMS 0.5) can keep only one?
  3. Which camera angles occur at all (to compare with the real camera)?

Examples (in Colab, after the cell "Connect Drive" of a notebook):
  python render_stats.py --help
  python render_stats.py --renders /content/drive/MyDrive/Pokerdice/projects/My_Dice/renders --scene .../scene
  python render_stats.py --renders .../renders --edge-mm 16 --json render_stats.json
"""
import argparse
import collections
import glob
import json
import os


def boxes_touch(a, b, tol=2.0):
    """Do two boxes (x0, y0, x1, y1) touch/overlap in the image? tol in pixels."""
    return not (a[2] + tol < b[0] or b[2] + tol < a[0] or a[3] + tol < b[1] or b[3] + tol < a[1])


def groups(boxes):
    """Connected groups of touching boxes (indices)."""
    open_, result = set(range(len(boxes))), []
    while open_:
        stack, g = [open_.pop()], []
        while stack:
            i = stack.pop()
            g.append(i)
            for j in list(open_):
                if boxes_touch(boxes[i], boxes[j]):
                    open_.discard(j)
                    stack.append(j)
        result.append(g)
    return result


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def share(z, n):
    return f"{100.0 * z / n:5.1f} %" if n else "      –"


def main():
    ap = argparse.ArgumentParser(description="Render statistics: throw types, dice at the wall, camera angles, overlapping boxes.")
    ap.add_argument("--renders", required=True, help="folder with chunk_*/ (e.g. .../projects/<project>/renders)")
    ap.add_argument("--scene", help="folder scene with scene.json (camera range, tray, tilt limit, edge length)")
    ap.add_argument("--edge-mm", type=float, help="edge length of the dice in mm (default: from scene.json, otherwise 16)")
    ap.add_argument("--wall-mm", type=float, help="“at the wall” up to this distance centre-wall (default 1.2 × edge)")
    ap.add_argument("--nms", type=float, default=0.5, help="NMS threshold of the firmware (default 0.5)")
    ap.add_argument("--json", help="also save the result as a JSON file")
    args = ap.parse_args()

    tilt_limit, edge = 20.0, args.edge_mm or 16.0
    if args.scene and os.path.isfile(os.path.join(args.scene, "scene.json")):
        with open(os.path.join(args.scene, "scene.json"), encoding="utf-8") as f:
            sd = json.load(f)
        tilt_limit = float(sd.get("labels", {}).get("tilt_deg", tilt_limit))
        edge = args.edge_mm or float(sd.get("die", {}).get("edge_mm", edge))
        k, t = sd.get("camera", {}), sd.get("tray", {})
        print(f"Scene: camera elevation {k.get('elevation_deg')}°, distance {k.get('distance_cm')} cm · "
              f"tray Ø {t.get('diameter_mm')} mm, wall {t.get('wall_height_mm')} mm high · tilted from {tilt_limit}°")
    wall_mm = args.wall_mm or 1.2 * edge

    def eclass(e):
        return min(int(e // 10) * 10, 80)
    z = collections.Counter()
    cams = collections.Counter()
    wall_e = collections.defaultdict(lambda: [0, 0])     # upright dice at the wall per elevation class: [n, "?"]
    inner_e = collections.defaultdict(lambda: [0, 0])
    wall_pos = collections.defaultdict(lambda: [0, 0])   # bottom/middle/top in the image
    top_vis = []
    pairs = collections.Counter()
    over_nms = collections.Counter()
    types = collections.Counter()
    rows = collections.Counter()

    for f in sorted(glob.glob(os.path.join(args.renders, "chunk_*", "done.json"))):
        z["chunks"] += 1
        with open(os.path.join(os.path.dirname(f), "meta.jsonl"), encoding="utf-8") as m:
            for line in m:
                if not line.strip():
                    continue
                d = json.loads(line)
                z["images"] += 1
                tt = d.get("throw_type", "normal")
                types[tt] += 1
                base = tt.split("+")[0]
                e = float(d.get("camera", {}).get("elevation_deg", -1))
                cams[eclass(e)] += 1
                h = (d.get("size") or [1024, 768])[1]
                ws = d.get("dice", [])
                z["dice"] += len(ws)
                z["dice ?"] += sum(1 for x in ws if x["class_name"] == "?")
                wall = [x for x in ws if x.get("wall_distance_mm", 1e9) <= wall_mm]
                z["dice at the wall"] += len(wall)
                rws = [g for g in groups([x["box"] for x in wall]) if len(g) >= 2]
                if rws:
                    z["images with a wall row"] += 1
                    for g in rws:
                        rows[len(g)] += 1
                for w in ws:
                    if w.get("tilt_deg", 0) > tilt_limit:
                        continue                      # tilted: "?" is right there
                    q = w.get("class_name") == "?"
                    if w.get("wall_distance_mm", 1e9) <= wall_mm:
                        wall_e[eclass(e)][0] += 1
                        wall_e[eclass(e)][1] += q
                        cy = (w["box"][1] + w["box"][3]) / 2 / h
                        pos = "bottom (near wall)" if cy > 2 / 3 else ("top (far wall)" if cy < 1 / 3 else "middle (side)")
                        wall_pos[pos][0] += 1
                        wall_pos[pos][1] += q
                        top_vis.append(w.get("top_visible", 1.0))
                    else:
                        inner_e[eclass(e)][0] += 1
                        inner_e[eclass(e)][1] += q
                r = [w["box"] for w in ws]
                over = False
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        u = iou(r[i], r[j])
                        pairs["≥ 0.7" if u >= 0.7 else "0.5-0.7" if u >= 0.5 else "0.3-0.5" if u >= 0.3 else "0-0.3" if u > 0 else "none"] += 1
                        over = over or u >= args.nms
                if over:
                    over_nms[base] += 1

    if not z["images"]:
        raise SystemExit(f"No finished chunks in {args.renders}.")
    n = z["images"]
    print(f"\n{z['chunks']} chunks, {n} images, {z['dice']} dice (of which “?” {z['dice ?']}) · "
          f"“at the wall” = centre ≤ {wall_mm:.0f} mm from the wall")
    print("\n1 · Throw types")
    for k, v in sorted(types.items()):
        print(f"    {k:24s} {v:7d}  {share(v, n)}")
    print(f"    dice at the wall: {z['dice at the wall']} · images with a wall row (≥ 2 touching): "
          f"{z['images with a wall row']} ({share(z['images with a wall row'], n).strip()})")
    for k in sorted(rows):
        print(f"      wall row of {k} dice: {rows[k]}")
    print("\n2 · Camera elevation of the renders (90° = straight from above)")
    for w in range(0, 90, 10):
        if cams[w]:
            print(f"    {w:2d}-{w + 10:2d}°  {cams[w]:6d} images  {share(cams[w], n)}")
    print("\n3 · UPRIGHT dice: share with label “?” (top face less than half visible)")
    print(f"    {'elevation':12s} {'at the wall':>22s} {'further inside':>22s}")
    for w in range(0, 90, 10):
        a, b = wall_e[w], inner_e[w]
        if a[0] or b[0]:
            print(f"    {w:2d}-{w + 10:2d}°      {a[1]:6d}/{a[0]:<6d} {share(a[1], a[0])}   {b[1]:6d}/{b[0]:<6d} {share(b[1], b[0])}")
    sa = [sum(x[0] for x in wall_e.values()), sum(x[1] for x in wall_e.values())]
    print(f"    at the wall in total: {sa[1]} of {sa[0]} upright dice have “?” ({share(sa[1], sa[0]).strip()})")
    for pos in ("bottom (near wall)", "middle (side)", "top (far wall)"):
        x = wall_pos[pos]
        if x[0]:
            print(f"      {pos:20s} {x[1]:6d}/{x[0]:<6d} {share(x[1], x[0])} “?”")
    if top_vis:
        s = sorted(top_vis)
        print(f"    visible share of the top face at the wall: median {s[len(s) // 2]:.2f}, below 0.5 for {sum(1 for x in s if x < 0.5)} dice")
    print(f"\n4 · Overlap of the boxes of two dice in the same image (IoU; the firmware keeps only one at ≥ {args.nms})")
    for k in ("0-0.3", "0.3-0.5", "0.5-0.7", "≥ 0.7"):
        print(f"    {k:8s} {pairs[k]:7d} pairs")
    total_over = sum(over_nms.values())
    print(f"    images with at least one pair ≥ {args.nms}: {total_over} ({share(total_over, n).strip()})")
    if args.json:
        res = {"counts": dict(z), "throw_types": dict(types), "wall_rows": dict(rows), "camera_elevation": dict(cams),
               "upright_wall": {str(k): v for k, v in wall_e.items()}, "upright_inner": {str(k): v for k, v in inner_e.items()},
               "wall_position": dict(wall_pos), "pairs_iou": dict(pairs), "images_over_nms": dict(over_nms)}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"\n✔ {args.json}")


if __name__ == "__main__":
    main()
