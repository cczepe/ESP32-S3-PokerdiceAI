#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "board.h"
#include "camera.h"
#include "settings.h"
#include "state.h"

static const char *TAG = "camera";
static SemaphoreHandle_t s_mutex;
static int s_res = 0;
static camera_exposure_t s_exp = { .manual = false, .ae_level = 0 };
static int s_actual_aec = -1;
static float s_actual_gain = -1;
static int s_pid = 0;
static int s_play_res = -1;                 // -1 = no play settings (no models)
static camera_exposure_t s_play_exp = { .manual = false, .ae_level = 0 };
static volatile int64_t s_trainer_until = 0;
#define TRAINER_HOLD_US 5000000             // detection waits this long after the last image for the trainer

static const struct { framesize_t fs; int w, h; } RES[CAMERA_RESOLUTIONS] = {
    {FRAMESIZE_VGA, 640, 480}, {FRAMESIZE_SVGA, 800, 600}, {FRAMESIZE_XGA, 1024, 768},
    {FRAMESIZE_SXGA, 1280, 1024}, {FRAMESIZE_UXGA, 1600, 1200},
};

int camera_resolution(void) { return s_res; }

int camera_vflip(void) { return g_settings.vflip == FLIP_AUTO ? CAM_DEFAULT_VFLIP(s_pid) : g_settings.vflip; }
int camera_hmirror(void) { return g_settings.hmirror == FLIP_AUTO ? CAM_DEFAULT_HMIRROR(s_pid) : g_settings.hmirror; }

void camera_resolution_size(int idx, int *w, int *h)
{
    if (idx < 0 || idx >= CAMERA_RESOLUTIONS) idx = 0;
    *w = RES[idx].w;
    *h = RES[idx].h;
}

int camera_resolution_find(const char *text)
{
    int w = 0, h = 0;
    if (sscanf(text, "%dx%d", &w, &h) != 2) return -1;
    for (int i = 0; i < CAMERA_RESOLUTIONS; i++)
        if (RES[i].w == w && RES[i].h == h) return i;
    return -1;
}

void camera_exposure(camera_exposure_t *e) { *e = s_exp; }

void camera_exposure_actual(int *aec, float *gain)
{
    *aec = s_actual_aec;
    *gain = s_actual_gain;
}

// big images are compressed a bit more so the JPEG always fits into the frame buffer
static int quality_for(int idx)
{
    int q = g_settings.jpeg_quality;
    if (idx >= 3 && q < 12) q = 12;
    return q;
}

void camera_apply_settings(void)
{
    sensor_t *s = esp_camera_sensor_get();
    if (!s) return;
    s->set_quality(s, quality_for(s_res));
    s->set_vflip(s, camera_vflip());
    s->set_hmirror(s, camera_hmirror());
    s->set_brightness(s, g_settings.brightness);
    s->set_contrast(s, g_settings.contrast);
    s->set_saturation(s, g_settings.saturation);
}

static void exposure_apply(void)
{
    sensor_t *s = esp_camera_sensor_get();
    if (!s) return;
    if (s_exp.manual) {
        s->set_exposure_ctrl(s, 0);
        s->set_gain_ctrl(s, 0);
        s->set_aec_value(s, s_exp.aec);
        s->set_agc_gain(s, s_exp.gain);
    } else {
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s, 1);
        s->set_ae_level(s, s_exp.ae_level);
    }
}

// Exposure time (lines) and gain as the sensor uses them right now
static void exposure_read(void)
{
    sensor_t *s = esp_camera_sensor_get();
    if (!s || !s->get_reg) return;
    if (s_pid == OV2640_PID) {
        // bank 1 (sensor) registers are addressed as 0x1xx
        int r04 = s->get_reg(s, 0x104, 0x03), r10 = s->get_reg(s, 0x110, 0xFF), r45 = s->get_reg(s, 0x145, 0x3F);
        int g = s->get_reg(s, 0x100, 0xFF);
        if (r04 < 0 || r10 < 0 || r45 < 0 || g < 0) return;
        s_actual_aec = r04 | (r10 << 2) | (r45 << 10);
        // gain register: bits 7..4 one doubling each, bits 3..0 = 1 + n/16
        s_actual_gain = (1.0f + (g & 0x0F) / 16.0f) * ((g & 0x10) ? 2 : 1) * ((g & 0x20) ? 2 : 1) *
                        ((g & 0x40) ? 2 : 1) * ((g & 0x80) ? 2 : 1);
    } else if (s_pid == OV3660_PID || s_pid == OV5640_PID) {
        // 0x3500..0x3502: exposure in 1/16 lines, 0x350A..0x350B: gain in 1/16
        int e0 = s->get_reg(s, 0x3500, 0x0F), e1 = s->get_reg(s, 0x3501, 0xFF), e2 = s->get_reg(s, 0x3502, 0xFF);
        int g0 = s->get_reg(s, 0x350A, 0x03), g1 = s->get_reg(s, 0x350B, 0xFF);
        if (e0 < 0 || e1 < 0 || e2 < 0 || g0 < 0 || g1 < 0) return;
        s_actual_aec = ((e0 << 16) | (e1 << 8) | e2) >> 4;
        s_actual_gain = ((g0 << 8) | g1) / 16.0f;
    }
}

static void drop_frames(int n)
{
    for (int i = 0; i < n; i++) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
    }
}

// mutex must be held
static void resolution_set(int idx)
{
    if (idx < 0 || idx >= CAMERA_RESOLUTIONS || idx == s_res) return;
    sensor_t *s = esp_camera_sensor_get();
    if (!s) return;
    s->set_framesize(s, RES[idx].fs);
    s_res = idx;
    camera_apply_settings();   // quality for the size, flips etc. again
    exposure_apply();          // the size tables of the sensor may overwrite exposure registers
    drop_frames(3);            // until the new size is surely there
    ESP_LOGI(TAG, "resolution now %dx%d", RES[idx].w, RES[idx].h);
}

static bool exposure_equal(const camera_exposure_t *a, const camera_exposure_t *b)
{
    if (a->manual != b->manual) return false;
    return a->manual ? (a->aec == b->aec && a->gain == b->gain) : (a->ae_level == b->ae_level);
}

// mutex must be held
static void exposure_set(const camera_exposure_t *exposure)
{
    if (!exposure || exposure_equal(exposure, &s_exp)) return;
    s_exp = *exposure;
    exposure_apply();
    drop_frames(s_exp.manual ? 3 : 8);   // auto exposure needs some frames to settle
    if (s_exp.manual) ESP_LOGI(TAG, "exposure: fixed, aec %d, gain %d", s_exp.aec, s_exp.gain);
    else ESP_LOGI(TAG, "exposure: auto, correction %+d", s_exp.ae_level);
}

esp_err_t camera_capture(int resolution, const camera_exposure_t *exposure, bool fresh, camera_send_t send, void *ctx)
{
    if (!g_state.camera_ok) return ESP_ERR_INVALID_STATE;
    s_trainer_until = esp_timer_get_time() + TRAINER_HOLD_US;   // before waiting: the detection pauses from now on
    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(8000)) != pdTRUE) return ESP_ERR_TIMEOUT;
    if (resolution >= 0) resolution_set(resolution);
    exposure_set(exposure);
    esp_err_t err = ESP_OK;
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb && fresh) {
        esp_camera_fb_return(fb);
        fb = esp_camera_fb_get();
    }
    if (!fb) {
        err = ESP_FAIL;
    } else {
        g_state.last_jpeg_bytes = fb->len;
        exposure_read();
        err = send(fb->buf, fb->len, ctx);
        esp_camera_fb_return(fb);
    }
    xSemaphoreGive(s_mutex);
    s_trainer_until = esp_timer_get_time() + TRAINER_HOLD_US;
    return err;
}

void camera_play_set(int resolution, const camera_exposure_t *exposure)
{
    s_play_exp = *exposure;
    s_play_res = resolution;
}

bool camera_in_use_by_trainer(void)
{
    return g_state.stream_active || esp_timer_get_time() < s_trainer_until;
}

camera_fb_t *camera_frame_take(int timeout_ms)
{
    if (!g_state.camera_ok || s_play_res < 0) return NULL;
    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) return NULL;
    resolution_set(s_play_res);
    exposure_set(&s_play_exp);
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        xSemaphoreGive(s_mutex);
        return NULL;
    }
    g_state.last_jpeg_bytes = fb->len;
    return fb;
}

void camera_frame_release(camera_fb_t *fb)
{
    if (fb) esp_camera_fb_return(fb);
    xSemaphoreGive(s_mutex);
}

esp_err_t camera_start(void)
{
    s_mutex = xSemaphoreCreateMutex();
    camera_config_t cfg = {
        .pin_pwdn = PIN_CAM_PWDN,
        .pin_reset = PIN_CAM_RESET,
        .pin_xclk = PIN_CAM_XCLK,
        .pin_sccb_sda = PIN_CAM_SDA,
        .pin_sccb_scl = PIN_CAM_SCL,
        .pin_d7 = PIN_CAM_D7, .pin_d6 = PIN_CAM_D6, .pin_d5 = PIN_CAM_D5, .pin_d4 = PIN_CAM_D4,
        .pin_d3 = PIN_CAM_D3, .pin_d2 = PIN_CAM_D2, .pin_d1 = PIN_CAM_D1, .pin_d0 = PIN_CAM_D0,
        .pin_vsync = PIN_CAM_VSYNC,
        .pin_href = PIN_CAM_HREF,
        .pin_pclk = PIN_CAM_PCLK,
        .xclk_freq_hz = CAM_XCLK_HZ,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_UXGA,   // buffers for the biggest size ...
        .jpeg_quality = 12,
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_LATEST,
    };
    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera start failed: %s (check the camera cable and board.h)", esp_err_to_name(err));
        snprintf(g_state.sensor, sizeof(g_state.sensor), "none");
        return err;
    }
    sensor_t *s = esp_camera_sensor_get();
    s_pid = s ? s->id.PID : 0;
    const char *name = s_pid == OV2640_PID ? "OV2640" : s_pid == OV3660_PID ? "OV3660" : s_pid == OV5640_PID ? "OV5640" : "unknown";
    snprintf(g_state.sensor, sizeof(g_state.sensor), "%s", name);
    if (s) s->set_framesize(s, FRAMESIZE_VGA);   // ... and start with 640x480
    s_res = 0;
    camera_apply_settings();
    exposure_apply();
    drop_frames(5);   // the first images are often too bright or too dark
    g_state.camera_ok = true;
    ESP_LOGI(TAG, "camera ready: sensor %s (PID 0x%04x), 640x480 JPEG, switchable up to 1600x1200, quality %d, "
             "vflip %d, hmirror %d%s", name, s_pid, g_settings.jpeg_quality, camera_vflip(), camera_hmirror(),
             g_settings.vflip == FLIP_AUTO ? " (default for this sensor)" : "");
    return ESP_OK;
}
