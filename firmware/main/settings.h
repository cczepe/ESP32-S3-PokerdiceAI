// Settings in NVS (namespace "pokerdice"). The struct carries a version number; on a new version the defaults apply.
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SETTINGS_VERSION 2   // 2: detection settings added at the end (version 1 is taken over)

typedef struct {
    uint16_t version;
    uint8_t  jpeg_quality;   // 4..40, lower = better quality and bigger file
    uint8_t  vflip;          // 1 = flip vertically, FLIP_AUTO = default for the detected sensor (board.h)
    uint8_t  hmirror;        // 1 = mirror horizontally (both on = camera mounted upside down), FLIP_AUTO as above
    int8_t   brightness;     // -2..2
    int8_t   contrast;       // -2..2
    int8_t   saturation;     // -2..2
    uint8_t  wifi_mode;      // WIFI_MODE_HOME / WIFI_MODE_HOTSPOT
    // --- since version 2: detection (new fields only at the end)
    uint8_t  threshold_pct;  // detector threshold in % (10..90)
    uint8_t  stable_n;       // this many equal frames ...
    uint8_t  stable_m;       // ... out of the last m give a stable result
    uint8_t  led_result;     // 1 = the LED flashes white on a new result (twice for a straight or better)
    uint8_t  sort_by_pos;    // web page: 0 = dice by rank, 1 = by position (left -> right)
    uint8_t  det_cores;      // detector: 0 = 1 core, 1 = automatic, 2 = 2 cores
    uint8_t  status_log;     // 1 = a status line every 30 s on the serial output
} settings_t;

enum { WIFI_MODE_HOME = 0, WIFI_MODE_HOTSPOT = 1 };
#define FLIP_AUTO 255

extern settings_t g_settings;

void settings_defaults(settings_t *s);
void settings_detection_defaults(settings_t *s);   // only the detection fields ("Defaults" button)
void settings_validate(settings_t *s);
esp_err_t settings_load(void);
esp_err_t settings_save(void);

// Home Wi-Fi credentials (own NVS namespace "wifi"; never in the source code)
bool wifi_credentials_load(char *ssid, size_t ssid_len, char *pass, size_t pass_len);
esp_err_t wifi_credentials_save(const char *ssid, const char *pass);
esp_err_t wifi_credentials_forget(void);

#ifdef __cplusplus
}
#endif
