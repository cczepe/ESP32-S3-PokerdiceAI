// Pokerdice firmware
// - detection: recognizes the poker dice on the tray and shows hand, live view with boxes, history and
//   monitoring on the web page (models from firmware/models, flashed into the partitions det and cls)
// - camera for the Pokerdice Trainer: live view, single images in several resolutions, camera settings
//   (while the trainer uses the camera, the detection waits)
// - Wi-Fi: home network (http://pokerdice.local) or own hotspot "Pokerdice" with captive portal
// - RGB LED shows the Wi-Fi state and flashes on a new result, the BOOT button (hold 5 s) forgets the home Wi-Fi
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "board.h"
#include "button.h"
#include "camera.h"
#include "detection.h"
#include "led.h"
#include "settings.h"
#include "state.h"
#include "version.h"
#include "web.h"
#include "wifi.h"

static const char *TAG = "pokerdice";

void app_main(void)
{
    ESP_LOGI(TAG, "Pokerdice firmware %s on %s", POKERDICE_VERSION, BOARD_NAME);
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    settings_load();
    state_init();
    led_start();
    button_start();

    if (camera_start() != ESP_OK) {
        g_state.camera_ok = false;
        g_state.camera_failed = true;
    }

    wifi_start();
    web_start();
    detection_start();   // loads the models (self-test) and starts the detection on core 1 - or camera mode only
    ESP_LOGI(TAG, "ready - free RAM %u KB internal, %u KB PSRAM", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(30000));
        if (!g_settings.status_log) continue;
        static const char *const NET_NAMES[] = {"starting", "connecting", "home ", "hotspot", "hotspot (home Wi-Fi not reached)"};
        ESP_LOGI(TAG, "status: wifi %s%s, rssi %d, captures %lu, stream frames %lu, free RAM %u KB internal, %u KB PSRAM",
                 NET_NAMES[g_state.net <= NET_HOTSPOT_FALLBACK ? g_state.net : 0],
                 g_state.net == NET_STATION ? g_state.ip : "", g_state.rssi, (unsigned long)g_state.captures,
                 (unsigned long)g_state.stream_frames, (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        if (g_state.status >= ST_SEARCHING)
            ESP_LOGI(TAG, "detection: %s, %.1f frames/s (%.0f ms: camera %.0f, jpeg %.0f, detector %.0f, crops %.0f, "
                     "classifier %.0f), %lu results",
                     g_state.pause ? "paused" : g_state.camera_busy ? "waiting (trainer uses the camera)" : "running",
                     g_state.fps, g_state.ms_total, g_state.ms_camera, g_state.ms_jpeg, g_state.ms_det, g_state.ms_crop,
                     g_state.ms_cls, (unsigned long)g_state.roll_no);
    }
}
