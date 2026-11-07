"""Required check: render the 3D die from 8 corner views (like the 3D preview in the trainer).

Then contact_sheet.py builds an image "renders next to your corner photos" from them. Only when they match
(same symbols at the same places, rotated the same way) is rendering started.

Call (in Colab):
  blender -b -P check_corners.py -- --scene /content/.../projects/My_Dice/scene
  blender -b -P check_corners.py -- --scene ... --size 400 --samples 32
  blender -b -P check_corners.py -- --help
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_tools as bt  # noqa: E402

VIEWS = [  # (name, elevation, azimuth) - azimuth 0 = +X (right), -90 = -Y (front)
    ("top_front-right", 35.26, -45), ("top_back-right", 35.26, 45),
    ("top_back-left", 35.26, 135), ("top_front-left", 35.26, -135),
    ("bottom_front-right", -35.26, -45), ("bottom_back-right", -35.26, 45),
    ("bottom_back-left", -35.26, 135), ("bottom_front-left", -35.26, -135),
]


def main():
    ap = argparse.ArgumentParser(description="Render the 3D die from 8 corner views (check).")
    ap.add_argument("--scene", required=True, help="folder with scene.json (output of prepare_textures.py)")
    ap.add_argument("--output", help="output folder (default: <scene>/check)")
    ap.add_argument("--size", type=int, default=400, help="image size in pixels (default 400)")
    ap.add_argument("--samples", type=int, default=32, help="Cycles samples (default 32)")
    ap.add_argument("--cpu", action="store_true", help="render without GPU")
    args = ap.parse_args(bt.script_args())

    sd = bt.load_scene(args.scene)
    out = args.output or os.path.join(args.scene, "check")
    os.makedirs(out, exist_ok=True)
    sc = bt.empty_scene()
    device = bt.setup_render(sc, args.size, args.size, args.samples, gpu=not args.cpu)
    sc.render.image_settings.file_format = "PNG"
    bt.create_die(sd, args.scene)
    k = sd["die"]["edge_mm"] / 1000.0
    bg = bt.setup_world(sc)
    bg.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1)
    bg.inputs["Strength"].default_value = 0.45
    light = bt.create_light()
    light.data.size = 0.2
    light.data.energy = 0.8
    cam = bt.create_camera({"camera": {"resolution": [args.size, args.size], "fov_diag_deg": 30}})
    for name, e, a in VIEWS:
        bt.aim_camera(cam, (0, 0, 0), e, a, k * 6.0, 0)
        # the light comes at an angle from the camera side, so that all 3 visible faces are bright
        bt.aim_camera(light, (0, 0, 0), e + 20 if e > 0 else e - 20, a + 30, 0.3, 0)
        sc.render.filepath = os.path.join(out, f"{name}.png")
        bt.render(sc)
        print(f"  rendered: {name}")
    print(f"✔ check done ({device}): {out}")


if __name__ == "__main__":
    main()
