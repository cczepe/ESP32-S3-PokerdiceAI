#include <string.h>
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"
#include "settings.h"
#include "config.h"

static const char *TAG = "settings";
settings_t g_settings;

void settings_defaults(settings_t *s)
{
    memset(s, 0, sizeof(*s));
    s->version = SETTINGS_VERSION;
    s->jpeg_quality = 10;
    s->vflip = FLIP_AUTO;
    s->hmirror = FLIP_AUTO;
    s->wifi_mode = WIFI_MODE_HOME;
    settings_detection_defaults(s);
}

void settings_detection_defaults(settings_t *s)
{
    s->threshold_pct = (uint8_t)(MC_DET_THRESHOLD * 100.0f + 0.5f);
    s->stable_n = 3;
    s->stable_m = 5;
    s->led_result = 1;
    s->sort_by_pos = 0;
    s->det_cores = 0;   // 1 core: the safe default
    s->status_log = 1;
}

// size of the version-1 struct (camera + Wi-Fi): version + 7 bytes, padded to 2
#define SETTINGS_V1_SIZE 10

static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void settings_validate(settings_t *s)
{
    s->jpeg_quality = (uint8_t)clamp(s->jpeg_quality, 4, 40);
    if (s->vflip != FLIP_AUTO) s->vflip = s->vflip ? 1 : 0;
    if (s->hmirror != FLIP_AUTO) s->hmirror = s->hmirror ? 1 : 0;
    s->brightness = (int8_t)clamp(s->brightness, -2, 2);
    s->contrast = (int8_t)clamp(s->contrast, -2, 2);
    s->saturation = (int8_t)clamp(s->saturation, -2, 2);
    if (s->wifi_mode > WIFI_MODE_HOTSPOT) s->wifi_mode = WIFI_MODE_HOME;
    s->threshold_pct = (uint8_t)clamp(s->threshold_pct, 10, 90);
    s->stable_m = (uint8_t)clamp(s->stable_m, 1, 8);
    s->stable_n = (uint8_t)clamp(s->stable_n, 1, s->stable_m);
    s->led_result = s->led_result ? 1 : 0;
    s->sort_by_pos = s->sort_by_pos ? 1 : 0;
    s->det_cores = (uint8_t)clamp(s->det_cores, 0, 2);
    s->status_log = s->status_log ? 1 : 0;
}

esp_err_t settings_load(void)
{
    settings_defaults(&g_settings);
    nvs_handle_t h;
    if (nvs_open("pokerdice", NVS_READONLY, &h) != ESP_OK) return ESP_OK;   // first start
    settings_t s;
    size_t len = sizeof(s);
    esp_err_t err = nvs_get_blob(h, "settings", &s, &len);
    nvs_close(h);
    if (err == ESP_OK && len == sizeof(s) && s.version == SETTINGS_VERSION) {
        settings_validate(&s);
        g_settings = s;
    } else if (err == ESP_OK && s.version == 1 && len == SETTINGS_V1_SIZE) {
        // version 1 (camera + Wi-Fi only): take it over, detection settings get their defaults
        settings_detection_defaults(&s);
        s.version = SETTINGS_VERSION;
        settings_validate(&s);
        g_settings = s;
        ESP_LOGI(TAG, "settings of version 1 taken over");
    } else if (err == ESP_OK) {
        ESP_LOGW(TAG, "stored settings have another version - defaults apply");
    }
    return ESP_OK;
}

esp_err_t settings_save(void)
{
    settings_validate(&g_settings);
    nvs_handle_t h;
    esp_err_t err = nvs_open("pokerdice", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_blob(h, "settings", &g_settings, sizeof(g_settings));
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    ESP_LOGI(TAG, "saved: quality %d, vflip %d, hmirror %d (255 = sensor default), brightness %d, contrast %d, saturation %d, "
             "wifi %s, threshold %d %%, stable %d of %d, led %d, det cores %d",
             g_settings.jpeg_quality, g_settings.vflip, g_settings.hmirror, g_settings.brightness, g_settings.contrast,
             g_settings.saturation, g_settings.wifi_mode == WIFI_MODE_HOTSPOT ? "hotspot" : "home", g_settings.threshold_pct,
             g_settings.stable_n, g_settings.stable_m, g_settings.led_result, g_settings.det_cores);
    return err;
}

bool wifi_credentials_load(char *ssid, size_t ssid_len, char *pass, size_t pass_len)
{
    nvs_handle_t h;
    if (nvs_open("wifi", NVS_READONLY, &h) != ESP_OK) return false;
    size_t a = ssid_len, b = pass_len;
    bool ok = nvs_get_str(h, "ssid", ssid, &a) == ESP_OK && ssid[0];
    if (ok && nvs_get_str(h, "pass", pass, &b) != ESP_OK) pass[0] = 0;
    nvs_close(h);
    return ok;
}

esp_err_t wifi_credentials_save(const char *ssid, const char *pass)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("wifi", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_str(h, "ssid", ssid);
    if (err == ESP_OK) err = nvs_set_str(h, "pass", pass ? pass : "");
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    ESP_LOGI(TAG, "home Wi-Fi '%s' saved", ssid);
    return err;
}

esp_err_t wifi_credentials_forget(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("wifi", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    nvs_erase_all(h);
    err = nvs_commit(h);
    nvs_close(h);
    ESP_LOGW(TAG, "home Wi-Fi forgotten");
    return err;
}
