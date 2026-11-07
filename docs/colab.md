# Rendering and training in Google Colab

The trainer builds a **Colab package** (page *Ready for Colab?*). Google Colab then renders synthetic training images
with Blender and trains two small networks on a GPU:

- **detector** (ESPDet-Pico from Espressif's esp-detection): finds the dice in the camera image,
- **classifier** (a small CNN): reads the symbol on top of each die, or `?`.

## Getting the package to Colab

1. Google Drive → *My Drive* → a folder with the name from step 0 (e.g. `Pokerdice`), inside it a folder `packages`.
2. Upload the newest `Colab_<project>_<time>.zip` from `workspace/colab_packages/` into `packages` - **do not unpack it**.
3. colab.research.google.com → *File → Upload notebook* → `1_setup_<project>_<time>.ipynb` (next to the ZIP file).
4. *Runtime → Change runtime type → L4 GPU* (or T4), then run the cells in order.

The first notebook unpacks the package; after that all notebooks are in Drive under `notebooks/` (double-click →
*Open with Google Colaboratory*). A newer package is unpacked automatically by the next notebook you start;
photos, test images and camera setup are replaced, renders and training results are kept.

## The notebooks

| Notebook | What it does |
|---|---|
| `1_setup` | Blender, the textures of your die, the corner check (you compare and confirm with `CHECK_OK = True`), a preview of 30 images with timing and an estimate for the full render |
| `2_render` | the full set (`IMAGES`, default 28 000) in chunks of 250 - normal throws, dice close together, "keep and re-roll" (dice pushed together at the wall) and dice leaning against the wall or another die (`?`) |
| `3_detector` | dataset, training (from scratch or continuing from an earlier run with `START_FROM`), test on synthetic and real images, export |
| `4_classifier` | crops of every die, training of one or more variants, comparison table, export of `CHOSEN` |
| `5_all_in_one` | 2 + 3 + 4 in one go (e.g. overnight), with a log in `projects/<project>/all_in_one.log`, then ends the session |

Each notebook has a settings cell at the top; the defaults are a good start.
The result is the folder `projects/<project>/export/` in your Drive.

## Practical notes

- **Resuming**: every finished chunk, crop file and checkpoint is on Drive. If Colab stops, open the notebook again
  and *Run all* - finished parts are skipped, started ones continue.
- **Session length**: Colab does not guarantee how long a session runs. If the preview estimate is many hours,
  run `2_render` and the training in separate sessions.
- **Compute units**: a connected session uses compute units even when idle. Every notebook ends with an
  `END_SESSION` cell; `5_all_in_one` ends the session by itself (`END_SESSION = True`).
- **Do not upload a new package while a notebook is running** - scripts and notebooks on Drive would be replaced.
- **Download `export` only when both models are exported** (detector and classifier).
- **More test images** (trainer steps 8 and 9) make the measurements more honest: build a new package, the notebooks
  measure again without training again.

## Folders on Google Drive

```
<Drive folder>/
  packages/        the ZIP files from the trainer
  scripts/         the Python scripts (from the package)
  notebooks/       the notebooks (from the package)
  blender/         the Blender download (cached, so the version stays the same)
  projects/<project>/
    scene/         textures, scene.json, corner check
    renders/       chunk_000 … with images and meta.jsonl
    detector/      training runs (runs/run1 …)
    classifier/    crops/, models/
    export/        dice_det.onnx, dice_cls.onnx, calibration and test sets, info files  ◄─ this goes to the laptop
```
