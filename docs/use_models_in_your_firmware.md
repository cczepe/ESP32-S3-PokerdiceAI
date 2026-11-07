# Using the models in your own firmware

The firmware in this repository is one way to use the models. They work in any ESP-IDF project with
[ESP-DL](https://github.com/espressif/esp-dl) (~3.3) on an ESP32-S3. Three files make up one set:

| File | Content |
|---|---|
| `dice_det.espdl` | detector (ESPDet-Pico), int8 |
| `dice_cls.espdl` | classifier (small CNN), int8 |
| `model_config.h` | everything the code needs to know about the two models |

`select_project.py` copies them into `firmware/models/`; `--firmware <folder>` (several times possible) copies the same
set into other firmware folders as well (into `<folder>/models/`):

```
.venv\Scripts\python.exe trainer\convert\select_project.py --project workspace\projects\My_Dice --firmware firmware --firmware D:\my\firmware
```

## model_config.h

```c
#define MC_VERSION          "07.11.26 10:00"   // when the set was made
#define MC_DET_IMGSZ        224                 // detector input: square letterbox, RGB
#define MC_DET_THRESHOLD    0.350f              // box score threshold
#define MC_DET_NMS_IOU      0.500f              // non-maximum suppression
#define MC_CLS_SIZE         64                  // classifier input: square crop, RGB
#define MC_CLS_MARGIN       0.250f              // margin around the box for the crop
#define MC_CLASSES          7
#define MC_CLASS_UNSURE     6                   // index of "?" (tilted / not readable)
#define MC_DICE             5                   // dice in the game
#define MC_CLASS_NAMES_EN   { "9", "10", "J", "Q", "K", "A", "?" }
#define MC_CLASS_RANK       { 0, 1, 2, 3, 4, 5, -1 }   // poker rank per class, -1 = no rank
#define MC_CAMERA_W         1024                // camera image the models were trained for
#define MC_CAMERA_H         768
#define MC_CAMERA_EXPOSURE  -1                  // auto exposure correction -2..2 (when MC_CAMERA_AEC = -1)
#define MC_CAMERA_AEC       -1                  // >= 0: fixed exposure time ...
#define MC_CAMERA_GAIN      0                   // ... and gain 0..30
```

Below the values the file carries the conversion reports as comments. Use the same camera image (size, exposure,
mounting) as during training: the models only know what they have seen.

## Flashing the models

The firmware here uses two data partitions and lets ESP-IDF write the files when flashing
(`firmware/partitions.csv`, `firmware/main/CMakeLists.txt`):

```
det,      data, 0x40,    0x400000, 0x200000
cls,      data, 0x41,    0x600000, 0x100000
```

```cmake
esptool_py_flash_to_partition(flash "det" "${models}/dice_det.espdl")
esptool_py_flash_to_partition(flash "cls" "${models}/dice_cls.espdl")
```

Loading (C++):

```cpp
dl::Model *det = new dl::Model("det", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
dl::Model *cls = new dl::Model("cls", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
```

Alternatively ESP-DL can embed the files into the app (`MODEL_LOCATION_IN_FLASH_RODATA`) - see the ESP-DL documentation.

## Pre- and post-processing

Both must match the training exactly. The reference is `firmware/main/preprocess.hpp` (plain C++, no ESP dependencies,
tested against the Python code with `firmware/tools/test_pc/compare.py`); in Python the same steps are in
`trainer/colab/scripts/ml_tools.py` and `trainer/convert/decode.py`.

1. **Detector input**: scale the camera image (aspect ratio kept) into an `MC_DET_IMGSZ` square, fill the rest with gray
   114 (letterbox), RGB, values `v / 255` quantised with the input exponent of the model (`pd::input_table`).
2. **Detector output**: three levels (strides 8, 16, 32), each with `box<i>` (distribution of the four box sides,
   16 bins) and `score<i>` (one class: "die"); `pd::decode` turns them into boxes in camera pixels, with threshold and NMS.
3. **Classifier input**: per box a square crop around the centre, side = larger box side × (1 + 2 × `MC_CLS_MARGIN`),
   scaled to `MC_CLS_SIZE`, gray 114 outside the image (`pd::crop`), same quantisation as above.
4. **Classifier output**: `MC_CLASSES` values; the largest is the symbol. Low probability or `MC_CLASS_UNSURE` = "?".
5. **Result**: sort the dice by `MC_CLASS_RANK`, wait until a few frames agree (`stabilizer.hpp`), then
   `pd::score_hand` gives the poker hand.

`firmware/main/detection.cpp` shows the whole chain including the memory-saving variant that decodes the JPEG in strips
and cuts the crops on the way.
