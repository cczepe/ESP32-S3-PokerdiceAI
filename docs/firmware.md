# Firmware

ESP-IDF project in `firmware/` (C and C++). It does two jobs:

- **Playing**: detection of the dice, the poker hand, a web page with live image, history and monitoring.
- **Camera for the trainer**: live view, single images in several resolutions and exposures, camera settings.

## Build and flash

You need [ESP-IDF 6.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/) or VS Code with
the ESP-IDF extension. In the ESP-IDF terminal:

```
cd firmware
idf.py set-target esp32s3        (only the first time)
idf.py build flash monitor
```

Add `-p <port>` if several serial ports exist (e.g. `-p COMx` on Windows, `-p /dev/ttyACM0` on Linux).
Quit the monitor with `Ctrl+]`. The first build downloads the components (esp32-camera, mdns, esp-dl, esp_new_jpeg).

**Models**: `firmware/models/` holds `dice_det.espdl`, `dice_cls.espdl` and `model_config.h`. When flashing, the two
models are written into the partitions `det` and `cls`; `model_config.h` is compiled in. Without these files the
firmware still builds and works as a camera only (the page shows "No models").

## Wi-Fi

- Without a saved home network the board opens its own hotspot **Pokerdice** (password in
  `firmware/main/wifi_config.h` - please change it). Phones usually open the page by themselves, otherwise
  http://192.168.4.1.
- Tab **Wi-Fi**: choose the home network, enter the password, save. The board restarts and is reachable at
  http://pokerdice.local (or its IP address, shown in the serial log and in the router).
- If the home network is not reachable for 30 s, the hotspot comes on as well and the board keeps trying.
- **Forget the home network**: hold the BOOT button for 5 s, or tab Wi-Fi → *Forget home Wi-Fi*.
- Wi-Fi passwords are stored only on the board (NVS). Only 2.4 GHz networks work.

## LED

| LED | Meaning |
|---|---|
| yellow blinking | connecting to the home Wi-Fi |
| green | connected to the home Wi-Fi |
| blue | own hotspot |
| blue, blinking green | home Wi-Fi not reachable, hotspot on, still trying |
| short white flash | new result (twice for a straight or better) - can be switched off in tab Settings |
| white blinking | BOOT button held: the home Wi-Fi is forgotten |
| red blinking | camera not found |

If green and red look swapped, change `LED_ORDER_RGB` in `firmware/main/boards/esp32s3_cam.h`.

## The web page

| Tab | Content |
|---|---|
| Play | the hand and the five symbols, the live image with boxes (orange = `?`), Pause, **Save snapshot** (camera image, detection as JSON, image with boxes) |
| History | the last results |
| Monitoring | time per step (camera, JPEG, detector, crops, classifier), frames per second, memory, self-test of the models |
| Settings | detector threshold, stability (n of m frames), LED flash, order of the dice, detector cores, status line in the serial log |
| Camera | live view, single images, camera settings (JPEG quality, flip, mirror, brightness, contrast, saturation) |
| Wi-Fi, Info | network, firmware, board, models |

While the trainer or the tab Camera uses the camera, the detection waits; it continues 5 s after the last image.

## API

| Call | Meaning |
|---|---|
| `GET /api/info` | firmware, board, sensor, resolutions, Wi-Fi, models (class names, ranks, sizes, self-test) |
| `GET /api/state` | detection: status (`searching`, `rolling`, `stable` …), dice `[x0,y0,x1,y1,score,class,p]`, last result, history, times; `?timing=1` adds the times of the last frames |
| `GET /api/frame?after=N` | newest camera image of the detection (JPEG); header `X-Detection` = JSON with boxes and symbols of exactly this image, `X-Frame` = its number; `204` = nothing newer than N |
| `GET/POST /api/settings` | detection settings: `threshold`, `stable_n`, `stable_m`, `led`, `sort`, `det_cores`, `status_log`; `defaults=1` |
| `POST /api/cmd` | `do=pause`, `do=continue`, `do=clear` (history) |
| `GET/POST /api/camera` | camera settings |
| `GET /api/capture` | single image: `res=1024x768` (640x480, 800x600, 1024x768, 1280x1024, 1600x1200), `exposure=-2…2` (auto with correction) or `aec=…&gain=0…30` (fixed); answer headers `X-Exposure`, `X-Resolution` |
| port 81 `/stream` | MJPEG live view (one viewer at a time) |
| `GET /api/wifi/scan`, `POST /api/wifi`, `/api/wifi/forget`, `/api/restart` | Wi-Fi setup |

## Other boards

The camera pins and the LED are in `firmware/main/boards/esp32s3_cam.h` (the same camera pins as the Espressif
ESP32-S3-EYE, which many ESP32-S3 camera boards copy). For another board copy the file, adapt the pins and include it in
`firmware/main/board.h`. Needed: ESP32-S3 with at least 8 MB flash and PSRAM (the partition table in
`firmware/partitions.csv` uses 7 MB: app 4 MB, detector 2 MB, classifier 1 MB). The sensors OV2640, OV3660 and OV5640
are set up in `firmware/main/camera.c`; for others check the flip default in the board file.

## Tools

- `firmware/tools/web_preview.py` - the web page on the PC without a board, with made-up values (`--play` simulates rolls).
- `firmware/tools/test_pc/compare.py` - compares the pre- and post-processing of the firmware (`main/preprocess.hpp`)
  with the Python code of the trainer; needs g++.

## Source overview

| File | Content |
|---|---|
| `main.c` | start order, status line |
| `detection.cpp`, `preprocess.hpp`, `stabilizer.hpp` | JPEG → detector → crops → classifier → stable result → poker hand |
| `camera.c` | camera, resolutions, exposure, shared use by detection and trainer |
| `web.c`, `web/index.html` | web server, API, the page |
| `wifi.c`, `dns_server.c` | home network, hotspot, captive portal, mDNS |
| `settings.c` | settings in NVS |
| `led.c`, `button.c` | status LED, BOOT button |
| `config.h` | defaults when `firmware/models/model_config.h` is missing |
