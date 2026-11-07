"""Removes the metadata of photos (EXIF with GPS position, camera, time; XMP; comments) before you publish them,
e.g. the pictures for the README.

Only the pixels are kept. The EXIF rotation is applied first, so the picture keeps the right orientation.
The original stays untouched: the result is written as <name>_clean.jpg (or .png) next to it, or into --output.
An existing file is never overwritten.

Examples (from the repository folder, with the virtual environment):
  .venv\\Scripts\\python.exe trainer\\tools\\clean_photo.py --help
  .venv\\Scripts\\python.exe trainer\\tools\\clean_photo.py C:\\Users\\Me\\Pictures\\setup.jpg
  .venv\\Scripts\\python.exe trainer\\tools\\clean_photo.py setup.jpg screenshot.png --output docs\\images --max-size 1600
  (Linux/macOS: .venv/bin/python trainer/tools/clean_photo.py setup.jpg)
"""
import argparse
import os
import sys

from PIL import Image, ImageOps


def clean(path, output, max_size, quality):
    with Image.open(path) as im:
        had = sorted(k for k in im.info if k not in ("dpi", "jfif", "jfif_version", "jfif_unit", "jfif_density"))
        exif = im.getexif()
        gps = bool(exif.get_ifd(0x8825)) if exif else False
        im = ImageOps.exif_transpose(im)
        fmt = "PNG" if path.lower().endswith(".png") else "JPEG"
        if fmt == "JPEG" and im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if max_size and max(im.size) > max_size:
            im.thumbnail((max_size, max_size), Image.LANCZOS)
        if im.mode == "P":
            im = im.convert("RGBA")
        clean_im = Image.new(im.mode, im.size)       # a fresh image: no info, no EXIF, no ICC, no XMP
        clean_im.paste(im)
    base, ext = os.path.splitext(os.path.basename(path))
    folder = output or os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"{base}_clean{'.png' if fmt == 'PNG' else '.jpg'}")
    if os.path.exists(target):
        print(f"✘ {target} exists already - not overwritten")
        return False
    if fmt == "PNG":
        clean_im.save(target, "PNG", optimize=True)
    else:
        clean_im.save(target, "JPEG", quality=quality, optimize=True)
    with Image.open(target) as check:
        left = [k for k in check.info if k not in ("dpi", "jfif", "jfif_version", "jfif_unit", "jfif_density")]
        if check.getexif() or left:
            print(f"✘ {target}: metadata still present ({left}) - please check")
            return False
    print(f"✔ {target}  {clean_im.size[0]}×{clean_im.size[1]}  removed: {', '.join(had + (['exif'] if exif and 'exif' not in had else [])) or 'nothing'}"
          f"{'  (contained a GPS position!)' if gps else ''}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Removes the metadata (EXIF/GPS, XMP, comments) of photos before publishing.",
                                 epilog="Example: .venv\\Scripts\\python.exe trainer\\tools\\clean_photo.py setup.jpg --output docs\\images",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="JPEG or PNG files")
    ap.add_argument("--output", help="target folder (default: next to the original)")
    ap.add_argument("--max-size", type=int, default=0, help="scale down so that the longer side is at most this many pixels")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    args = ap.parse_args()
    ok = True
    for p in args.photos:
        if not os.path.isfile(p):
            print(f"✘ {p} not found")
            ok = False
            continue
        ok = clean(p, args.output, args.max_size, args.quality) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
