// Wi-Fi: home network (station) for the trainer, own hotspot with captive portal for playing and for the first setup.
#pragma once
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t wifi_start(void);
esp_err_t wifi_scan_json(char *buf, size_t len);   // nearby networks as JSON list

#ifdef __cplusplus
}
#endif
