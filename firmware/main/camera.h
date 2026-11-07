// Camera: start, sensor detection, resolution switching, single images and frames for the live stream.
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"
#include "esp_camera.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CAMERA_RESOLUTIONS 5   // 640x480, 800x600, 1024x768, 1280x1024, 1600x1200

typedef struct {
    bool manual;     // false = auto exposure with correction ae_level, true = fixed values
    int  ae_level;   // -2..2
    int  aec;        // exposure time in sensor lines (manual)
    int  gain;       // 0..30 (manual)
} camera_exposure_t;

typedef esp_err_t (*camera_send_t)(const uint8_t *jpg, size_t len, void *ctx);

esp_err_t camera_start(void);
void camera_apply_settings(void);
int  camera_resolution(void);                          // current index
int  camera_vflip(void);                               // effective values (FLIP_AUTO resolved)
int  camera_hmirror(void);
void camera_resolution_size(int idx, int *w, int *h);
int  camera_resolution_find(const char *text);         // "1024x768" -> index, -1 = unknown
void camera_exposure(camera_exposure_t *e);            // requested exposure
void camera_exposure_actual(int *aec, float *gain);    // read back from the sensor with the last image

// Takes one image (resolution -1 / exposure NULL = keep the current one) and hands it to send().
// fresh = drop an already waiting frame so the image is taken now.
esp_err_t camera_capture(int resolution, const camera_exposure_t *exposure, bool fresh, camera_send_t send, void *ctx);

// ---- for the detection (playing)
// Resolution and exposure for playing (from model_config.h). The camera switches to them with the next frame_take.
void camera_play_set(int resolution, const camera_exposure_t *exposure);
// true while the trainer uses the camera (single images, live stream) or did so in the last seconds:
// the detection then leaves the camera alone, so the two do not switch resolution and exposure back and forth.
bool camera_in_use_by_trainer(void);
// One frame with the play settings. The camera stays reserved until camera_frame_release (the detection
// still reads the JPEG for the crops). NULL = no frame within timeout_ms.
camera_fb_t *camera_frame_take(int timeout_ms);
void camera_frame_release(camera_fb_t *fb);

#ifdef __cplusplus
}
#endif
