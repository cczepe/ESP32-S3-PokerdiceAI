// Target board: ESP32-S3-CAM (GOOUUU, ESP32-S3-WROOM-1 N16R8, 24-pin camera connector) with an OV3660 sensor.
// The camera pins are the same as on the Espressif ESP32-S3-EYE, which many ESP32-S3 camera boards copy
// (e.g. Freenove ESP32-S3-WROOM CAM) - for those this file may fit as well.
#pragma once

#define BOARD_NAME          "ESP32-S3-CAM"

#define PIN_CAM_PWDN        -1
#define PIN_CAM_RESET       -1
#define PIN_CAM_XCLK        15
#define PIN_CAM_SDA         4
#define PIN_CAM_SCL         5
#define PIN_CAM_D7          16      // Y9
#define PIN_CAM_D6          17      // Y8
#define PIN_CAM_D5          18      // Y7
#define PIN_CAM_D4          12      // Y6
#define PIN_CAM_D3          10      // Y5
#define PIN_CAM_D2          8       // Y4
#define PIN_CAM_D1          9       // Y3
#define PIN_CAM_D0          11      // Y2
#define PIN_CAM_VSYNC       6
#define PIN_CAM_HREF        7
#define PIN_CAM_PCLK        13
#define CAM_XCLK_HZ         20000000

// image orientation per sensor as long as nothing is saved on the web page (the OV3660 delivers the image upside down without vflip)
#define CAM_DEFAULT_VFLIP(pid)    ((pid) == OV3660_PID ? 1 : 0)
#define CAM_DEFAULT_HMIRROR(pid)  0

// colour order of the RGB LED: 1 = red-green-blue (target board), 0 = green-red-blue (WS2812 standard)
#define LED_ORDER_RGB       1

#define PIN_RGB_LED         48      // WS2812 RGB LED, -1 if the board has none
#define PIN_BUTTON          0       // BOOT button (LOW when pressed)
