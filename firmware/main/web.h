// Web server: page and JSON interface (port 80), MJPEG live stream for the trainer (port 81).
//   GET  /                   the page (web/index.html, embedded into the firmware)
//   GET  /api/info           firmware, board, camera, Wi-Fi, models (class names, ranks, self-test)
//   GET  /api/state          detection: status, dice, result, history, monitoring; ?timing=1 adds the chart values
//   GET  /api/frame?after=N  newest camera JPEG of the detection + header X-Detection (JSON of exactly this frame),
//                            204 = nothing newer yet
//   GET  /api/settings       detection settings          POST /api/settings  change them (saved at once)
//   POST /api/cmd            do=pause | continue | clear
//   GET  /api/camera         camera settings              POST /api/camera    change them
//   GET  /api/capture        single image for the trainer (res, exposure or aec+gain)
//   GET  /api/wifi/scan      POST /api/wifi, /api/wifi/forget, /api/restart
//   port 81: /stream         MJPEG live view for the trainer (the detection waits meanwhile)
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t web_start(void);

// called by the detection task:
bool web_frame_wanted(void);                                  // a page asked for images in the last 3 s
bool web_frame_store(const uint8_t *jpg, size_t len);         // copy the JPEG (false = not wanted / no memory)
void web_frame_publish(uint32_t frame_no, const char *json);  // detection of the stored JPEG -> visible

#ifdef __cplusplus
}
#endif
