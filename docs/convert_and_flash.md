# From the Colab export to the flashed board

The page **Training & device** in the trainer shows these commands with the paths of your project, ready to copy.
Below they are written for a project folder `My_Dice` (Windows; on Linux/macOS use `.venv/bin/python` and `/`).

## 1. Get the export

Download the folder `projects/My_Dice/export/` from Google Drive and put it into `workspace\projects\My_Dice\export\`.
An older `export` is renamed first (the page shows the command), nothing is overwritten.

## 2. Convert for the ESP32 (esp-ppq, int8)

```
.venv\Scripts\python.exe trainer\convert\quantize_detector.py --export workspace\projects\My_Dice\export --dest workspace\projects\My_Dice\models
.venv\Scripts\python.exe trainer\convert\quantize_classifier.py --export workspace\projects\My_Dice\export --dest workspace\projects\My_Dice\models
```

Both write the `.espdl` model plus a report: how the converted (int8) model compares with the original (float) on the
synthetic and on your real test images. The calibration uses the images from the export (about 256 or more).

## 3. Check on the test images (optional)

```
.venv\Scripts\python.exe trainer\tools\evaluate_test_images.py --project workspace\projects\My_Dice
```

Runs detector and classifier on all labelled real throws and writes `evaluation_<time>/` with one image per throw
(green = your label, orange = what the models found, red = missed) and `report.json`.

## 4. Into the firmware

```
.venv\Scripts\python.exe trainer\convert\select_project.py --project workspace\projects\My_Dice
```

Writes `model_config.h` (class names, ranks, sizes, thresholds, the camera settings of step 1) and copies
`dice_det.espdl`, `dice_cls.espdl` and `model_config.h` into `firmware/models/`. What was there before moves to
`workspace/firmware_models_backup/<time>/`. The project name does not appear in the firmware.

## 5. Build and flash

```
cd firmware
idf.py set-target esp32s3        (only the first time)
idf.py build flash monitor
```

The models are written into their own flash partitions (`det`, `cls`); settings and Wi-Fi on the board stay.
Then open http://pokerdice.local: tab **Play** shows the hand, tab **Monitoring** the self-test of both models and the
time per image.
