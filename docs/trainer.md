# The trainer

The trainer is a web program on your computer (`start.bat` / `start.sh`). Every page explains what the step is for,
what to watch out for and what typical mistakes look like. This page is the overview.

```
0 Setup ─► 1 camera setup ─► 2-7 the die and its surroundings ─► Ready for Colab? ─► Colab (render, train)
                         └─► 8-9 real throws (test images) ───────────────────────┘
                                                          Training & device ◄─ export from Colab
                                                          convert ─► check ─► into the firmware ─► flash
```

## 0 · Setup (once)

Board address, Google Drive folder name, Anthropic API key (optional) and model. Applies to all projects.

## 1 · Camera setup (feasibility check)

Mount the board above the tray as it will be used later and take images with it. The trainer measures how large the
dice appear, how sharp the images are and how many pixels a symbol gets; the AI (optional) rates whether the symbols
are readable. You choose the resolution and the exposure here:

- **auto exposure** with a correction (-2 darker … +2 brighter), or
- **fixed exposure**: the trainer measures what the auto exposure chooses, sets these values fixed and checks them with
  a second image. Fixed values keep the brightness constant from throw to throw.

Resolution and exposure end up in `model_config.h`: the board later plays with exactly these settings.

## 2 · Measure the die

Edge length and corner radius with a ruler or caliper. Blender builds the 3D die in real size.

## 3 · Photograph the 6 faces

With a phone or camera, straight from above on white paper. You draw a frame around each face; the trainer crops,
straightens and colour-matches them. The AI (optional) reads the symbols.

## 4 · 4 corner photos / die layout

Poker dice differ in which symbol lies next to which and how it is rotated. From four photos of the corners the trainer
(with AI help, optional) builds the die net; the 3D preview shows the die from the same corners for comparison.

## 5 · Dice tray, 6 · Felt, 7 · Surroundings

Size and photos of the tray, a photo of the felt (texture and colour), and 10-30 photos of the room around it
(they become reflections and background). Steps 5-7 can be copied from another project.

## 8 · Capture real throws

Real throws taken with the board in its final position. These are the **test images**: they are never used for training,
only to measure how good the models really are. Mix deliberately: dice touching, at the wall, tilted, different light.

## 9 · Label

A box and the symbol per die, `?` for a die that is tilted or not readable. Suggestions come from the AI (optional) or,
as soon as there are models, from your own models.

## Ready for Colab?

Shows whether everything the Colab notebooks need is there and builds the **Colab package** (ZIP) with the photos,
the test images, the camera setup, the scripts and the notebooks. How it gets to Google Drive: see [colab.md](colab.md).

## Training & device

After Colab: download the folder `export`, convert the models for the ESP32, check them on the test images and copy
them into the firmware - see [convert_and_flash.md](convert_and_flash.md). The page shows all commands with the paths of
your project.

## Guides in the trainer

- **Python environment** - creating `.venv` again, common errors.
- **Camera board firmware** - building, flashing, Wi-Fi, LED, the board's own web page.
