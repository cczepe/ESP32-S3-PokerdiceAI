// Model values: come from firmware/models/model_config.h (written by trainer/convert/select_project.py together
// with the two models dice_det.espdl and dice_cls.espdl). Without that file these defaults apply - the firmware
// still builds, works as a camera for the trainer and the web page shows "No models".
#pragma once

#if defined(__has_include)
#if __has_include("model_config.h")
#include "model_config.h"
#define MC_FROM_FILE 1
#endif
#endif

#ifndef MC_FROM_FILE
#define MC_FROM_FILE        0
#define MC_VERSION          "-"
#define MC_DET_IMGSZ        224
#define MC_DET_THRESHOLD    0.35f
#define MC_DET_NMS_IOU      0.5f
#define MC_CLS_SIZE         64
#define MC_CLS_MARGIN       0.25f
#define MC_CLASSES          7
#define MC_CLASS_UNSURE     6
#define MC_DICE             5
#define MC_CLASS_NAMES_EN   { "9", "10", "J", "Q", "K", "A", "?" }
#define MC_CLASS_RANK       { 0, 1, 2, 3, 4, 5, -1 }
#endif

// camera for playing (the models were trained for exactly this image)
#ifndef MC_CAMERA_W
#define MC_CAMERA_W         640
#define MC_CAMERA_H         480
#endif
#ifndef MC_CAMERA_EXPOSURE
#define MC_CAMERA_EXPOSURE  0     // auto exposure with correction -2..+2
#endif
#ifndef MC_CAMERA_AEC
#define MC_CAMERA_AEC       -1    // -1 = auto exposure; otherwise fixed exposure time (sensor lines) ...
#define MC_CAMERA_GAIN      0     // ... and gain 0..30
#endif

#define MC_STRIDES          { 8, 16, 32 }
#define IMG_W               MC_CAMERA_W   // camera image while playing
#define IMG_H               MC_CAMERA_H
