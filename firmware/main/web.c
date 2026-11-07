#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_log.h"
#include "board.h"
#include "camera.h"
#include "config.h"
#include "detection.h"
#include "settings.h"
#include "state.h"
#include "version.h"
#include "wifi.h"
#include "wifi_config.h"
#include "web.h"

static const char *TAG = "web";

extern const char index_html_start[] asm("_binary_index_html_start");
extern const char index_html_end[]   asm("_binary_index_html_end");

// ---------------------------------------------------------------- helpers
static void url_decode(char *s)
{
    char *o = s;
    for (; *s; s++) {
        if (*s == '+') *o++ = ' ';
        else if (*s == '%' && isxdigit((unsigned char)s[1]) && isxdigit((unsigned char)s[2])) {
            char h[3] = { s[1], s[2], 0 };
            *o++ = (char)strtol(h, NULL, 16);
            s += 2;
        } else *o++ = *s;
    }
    *o = 0;
}

static int read_body(httpd_req_t *req, char *buf, size_t len)
{
    int total = 0;
    while (total < (int)len - 1 && total < req->content_len) {
        int r = httpd_req_recv(req, buf + total, len - 1 - total);
        if (r <= 0) return -1;
        total += r;
    }
    buf[total] = 0;
    return total;
}

static bool field(const char *form, const char *name, char *value, size_t len)
{
    if (httpd_query_key_value(form, name, value, len) != ESP_OK) return false;
    url_decode(value);
    return true;
}

static bool int_field(const char *form, const char *name, int lo, int hi, int *out, bool *bad)
{
    char v[12], *end;
    if (httpd_query_key_value(form, name, v, sizeof(v)) != ESP_OK) return false;
    long x = strtol(v, &end, 10);
    if (end == v || *end || x < lo || x > hi) { *bad = true; return false; }
    *out = (int)x;
    return true;
}

static void common_headers(httpd_req_t *req)
{
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
}

static esp_err_t send_json(httpd_req_t *req, const char *json)
{
    common_headers(req);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

static esp_err_t bad_request(httpd_req_t *req, const char *text)
{
    httpd_resp_set_status(req, "400 Bad Request");
    common_headers(req);
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_sendstr(req, text);
}

static void json_escape(const char *in, char *out, size_t len)
{
    size_t k = 0;
    for (; *in && k + 2 < len; in++) {
        if (*in == '"' || *in == '\\') out[k++] = '\\';
        out[k++] = *in;
    }
    out[k] = 0;
}

static const char *net_name(net_mode_t m)
{
    switch (m) {
        case NET_CONNECTING:       return "connecting";
        case NET_STATION:          return "station";
        case NET_HOTSPOT:          return "hotspot";
        case NET_HOTSPOT_FALLBACK: return "hotspot_fallback";
        default:                   return "starting";
    }
}

static void restart_later(void *arg)
{
    vTaskDelay(pdMS_TO_TICKS(1500));   // let the answer reach the browser first
    esp_restart();
}

// ---------------------------------------------------------------- page
static esp_err_t h_index(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, index_html_start, index_html_end - index_html_start);
}

// ---------------------------------------------------------------- detection: camera images for the page
// Two JPEG buffers in PSRAM: the detection writes into the one that is neither the newest nor being sent.
static SemaphoreHandle_t s_flock;
static uint8_t *s_buf[2];
static size_t s_cap[2], s_len[2];
static char *s_json[2];
static uint32_t s_frame[2];
static int s_latest = -1, s_pending = -1, s_sending = -1;
static volatile int64_t s_want_until = 0;
#define JSON_MAX 1600

bool web_frame_wanted(void) { return esp_timer_get_time() < s_want_until; }

bool web_frame_store(const uint8_t *jpg, size_t len)
{
    if (!web_frame_wanted() || !s_flock) return false;
    bool ok = false;
    xSemaphoreTake(s_flock, portMAX_DELAY);
    int i = -1;
    for (int k = 0; k < 2; k++)
        if (k != s_sending && k != s_latest) i = k;
    if (i < 0) for (int k = 0; k < 2; k++) if (k != s_sending) i = k;
    if (i >= 0) {
        if (s_cap[i] < len) {
            size_t cap = len + 16 * 1024;   // a little reserve, JPEG sizes vary
            uint8_t *b = (uint8_t *)heap_caps_realloc(s_buf[i], cap, MALLOC_CAP_SPIRAM);
            if (b) { s_buf[i] = b; s_cap[i] = cap; }
        }
        if (s_buf[i] && s_cap[i] >= len) {
            memcpy(s_buf[i], jpg, len);
            s_len[i] = len;
            if (s_latest == i) s_latest = -1;
            s_pending = i;
            ok = true;
        }
    }
    xSemaphoreGive(s_flock);
    if (!ok) ESP_LOGW(TAG, "no memory for a camera image of %u bytes", (unsigned)len);
    return ok;
}

void web_frame_publish(uint32_t frame_no, const char *json)
{
    if (!s_flock) return;
    xSemaphoreTake(s_flock, portMAX_DELAY);
    if (s_pending >= 0) {
        snprintf(s_json[s_pending], JSON_MAX, "%s", json);
        s_frame[s_pending] = frame_no;
        s_latest = s_pending;
        s_pending = -1;
    }
    xSemaphoreGive(s_flock);
}

// GET /api/frame?after=N
static esp_err_t h_frame(httpd_req_t *req)
{
    s_want_until = esp_timer_get_time() + 3000000;
    uint32_t after = 0;
    char q[48], v[16];
    if (httpd_req_get_url_query_str(req, q, sizeof(q)) == ESP_OK && httpd_query_key_value(q, "after", v, sizeof(v)) == ESP_OK)
        after = (uint32_t)strtoul(v, NULL, 10);
    xSemaphoreTake(s_flock, portMAX_DELAY);
    int i = s_latest;
    if (i >= 0 && s_frame[i] == after) i = -1;   // nothing new (any other number = new, also after a restart)
    if (i >= 0) s_sending = i;
    xSemaphoreGive(s_flock);
    common_headers(req);
    if (i < 0) {
        httpd_resp_set_status(req, "204 No Content");
        return httpd_resp_send(req, NULL, 0);
    }
    char no[12];
    snprintf(no, sizeof(no), "%lu", (unsigned long)s_frame[i]);
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "X-Frame", no);
    httpd_resp_set_hdr(req, "X-Detection", s_json[i]);
    httpd_resp_set_hdr(req, "Access-Control-Expose-Headers", "X-Frame, X-Detection");
    esp_err_t err = httpd_resp_send(req, (const char *)s_buf[i], (ssize_t)s_len[i]);
    xSemaphoreTake(s_flock, portMAX_DELAY);
    s_sending = -1;
    xSemaphoreGive(s_flock);
    return err;
}

// ---------------------------------------------------------------- JSON output into a buffer
typedef struct { char *b; size_t size; size_t len; } out_t;

static void put(out_t *o, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void put(out_t *o, const char *fmt, ...)
{
    if (o->len >= o->size) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(o->b + o->len, o->size - o->len, fmt, ap);
    va_end(ap);
    if (n > 0) o->len = o->len + (size_t)n < o->size ? o->len + (size_t)n : o->size - 1;
}

static void put_num(out_t *o, const char *key, float v, int decimals)
{
    if (isnan(v) || isinf(v)) put(o, "\"%s\":null", key);
    else put(o, "\"%s\":%.*f", key, decimals, v);
}

static char *s_out;          // response buffer (PSRAM); the httpd task handles one request at a time
#define OUT_SIZE (12 * 1024)
static state_t *s_copy;      // copy of the state (PSRAM)

static esp_err_t send_out(httpd_req_t *req, out_t *o)
{
    common_headers(req);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, o->b, (ssize_t)o->len);
}

static const char *status_name(int s)
{
    static const char *n[] = {"start", "no_models", "error", "searching", "rolling", "stable"};
    return s >= 0 && s <= ST_STABLE ? n[s] : "start";
}

static void put_roll(out_t *o, const roll_t *r, int64_t now)
{
    put(o, "{\"cls\":[");
    for (int i = 0; i < r->count && i < MAX_DICE; i++) put(o, "%s%d", i ? "," : "", r->cls[i]);
    put(o, "],\"hand\":%d,\"hand_name\":\"%s\",\"age_s\":%lld}", r->hand, detection_hand_name(r->hand),
        (long long)((now - r->time_us) / 1000000));
}

// GET /api/state
static esp_err_t h_state(httpd_req_t *req)
{
    char q[32], v[4];
    bool timing = httpd_req_get_url_query_str(req, q, sizeof(q)) == ESP_OK &&
                  httpd_query_key_value(q, "timing", v, sizeof(v)) == ESP_OK && v[0] == '1';
    state_lock();
    memcpy(s_copy, &g_state, sizeof(state_t));
    state_unlock();
    const state_t *z = s_copy;
    int64_t now = esp_timer_get_time();
    out_t o = {s_out, OUT_SIZE, 0};
    char msg[140];
    json_escape(z->message, msg, sizeof(msg));
    put(&o, "{\"frame\":%lu,\"status\":\"%s\",\"pause\":%s,\"camera_busy\":%s,\"message\":\"%s\",\"candidates\":%d,\"dice\":[",
        (unsigned long)z->frame_no, status_name(z->status), z->pause ? "true" : "false", z->camera_busy ? "true" : "false",
        msg, z->candidates);
    for (int i = 0; i < z->n && i < MAX_DICE; i++) {
        const die_t *d = &z->dice[i];
        put(&o, "%s[%.1f,%.1f,%.1f,%.1f,%.3f,%d,%.3f]", i ? "," : "", d->x0, d->y0, d->x1, d->y1, d->score, d->cls, d->p);
    }
    put(&o, "],\"roll_no\":%lu,\"stable\":", (unsigned long)z->roll_no);
    if (z->roll_no) put_roll(&o, &z->stable, now);
    else put(&o, "null");
    put(&o, ",\"history\":[");
    for (int i = 0; i < z->history_n && i < HISTORY_MAX; i++) {
        if (i) put(&o, ",");
        put_roll(&o, &z->history[i], now);
    }
    put(&o, "],\"mon\":{");
    put_num(&o, "fps", z->fps, 2);
    put(&o, ",");
    put_num(&o, "cam", z->ms_camera, 1);
    put(&o, ",");
    put_num(&o, "jpg", z->ms_jpeg, 1);
    put(&o, ",");
    put_num(&o, "box", z->ms_letterbox, 1);
    put(&o, ",");
    put_num(&o, "det", z->ms_det, 1);
    put(&o, ",");
    put_num(&o, "dec", z->ms_decode, 2);
    put(&o, ",");
    put_num(&o, "crop", z->ms_crop, 1);
    put(&o, ",");
    put_num(&o, "cls", z->ms_cls, 1);
    put(&o, ",");
    put_num(&o, "total", z->ms_total, 1);
    put(&o, ",\"strips\":%d,\"strips_total\":%d,\"jpeg_kb\":%.1f", z->strips, z->strips_total, z->jpeg_bytes / 1024.0f);
    if (timing) {
        put(&o, ",\"timing\":[");
        for (int i = 0; i < z->timing_n; i++) {   // oldest first
            int k = (z->timing_pos - z->timing_n + i + TIMING_MAX) % TIMING_MAX;
            put(&o, "%s%.0f", i ? "," : "", z->timing[k]);
        }
        put(&o, "]");
    }
    put(&o, "},\"sys\":{\"psram_free\":%u,\"psram_min\":%u,\"ram_free\":%u,\"ram_min\":%u,\"uptime_s\":%lu}}",
        (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
        (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM) / 1024),
        (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
        (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) / 1024), (unsigned long)(now / 1000000));
    return send_out(req, &o);
}

// GET/POST /api/settings  (detection)
static esp_err_t settings_send(httpd_req_t *req)
{
    const settings_t *s = &g_settings;
    out_t o = {s_out, OUT_SIZE, 0};
    put(&o, "{\"threshold\":%d,\"stable_n\":%d,\"stable_m\":%d,\"led\":%d,\"sort\":%d,\"det_cores\":%d,\"status_log\":%d,"
            "\"model_threshold\":%d}",
        s->threshold_pct, s->stable_n, s->stable_m, s->led_result, s->sort_by_pos, s->det_cores, s->status_log,
        (int)(MC_DET_THRESHOLD * 100.0f + 0.5f));
    return send_out(req, &o);
}

static esp_err_t h_settings_get(httpd_req_t *req) { return settings_send(req); }

// POST /api/settings  threshold=35&stable_n=3&stable_m=5&led=1&sort=0&det_cores=0&status_log=1  or  defaults=1
static esp_err_t h_settings_post(httpd_req_t *req)
{
    char body[256], v[8];
    if (read_body(req, body, sizeof(body)) < 0) return bad_request(req, "no data");
    settings_t s = g_settings;
    bool bad = false;
    int x;
    if (httpd_query_key_value(body, "defaults", v, sizeof(v)) == ESP_OK && v[0] == '1') {
        settings_detection_defaults(&s);
    } else {
        if (int_field(body, "threshold", 10, 90, &x, &bad)) s.threshold_pct = (uint8_t)x;
        if (int_field(body, "stable_n", 1, 8, &x, &bad)) s.stable_n = (uint8_t)x;
        if (int_field(body, "stable_m", 1, 8, &x, &bad)) s.stable_m = (uint8_t)x;
        if (int_field(body, "led", 0, 1, &x, &bad)) s.led_result = (uint8_t)x;
        if (int_field(body, "sort", 0, 1, &x, &bad)) s.sort_by_pos = (uint8_t)x;
        if (int_field(body, "det_cores", 0, 2, &x, &bad)) s.det_cores = (uint8_t)x;
        if (int_field(body, "status_log", 0, 1, &x, &bad)) s.status_log = (uint8_t)x;
        if (bad) return bad_request(req, "value out of range");
    }
    settings_validate(&s);
    g_settings = s;   // the detection reads single bytes - a torn read for one frame is harmless
    settings_save();
    return settings_send(req);
}

// POST /api/cmd  do=pause | continue | clear
static esp_err_t h_cmd(httpd_req_t *req)
{
    char body[64], v[16];
    if (read_body(req, body, sizeof(body)) < 0 || httpd_query_key_value(body, "do", v, sizeof(v)) != ESP_OK)
        return bad_request(req, "do=?");
    if (strcmp(v, "pause") == 0) g_state.pause = true;
    else if (strcmp(v, "continue") == 0) g_state.pause = false;
    else if (strcmp(v, "clear") == 0) detection_clear_history();
    else return bad_request(req, "unknown command");
    ESP_LOGI(TAG, "web: %s", v);
    out_t o = {s_out, OUT_SIZE, 0};
    put(&o, "{\"ok\":true,\"pause\":%s}", g_state.pause ? "true" : "false");
    return send_out(req, &o);
}

// ---------------------------------------------------------------- API
// GET /api/info: everything the trainer and the page need to know about the board
static const char *const NAMES_EN[MC_CLASSES] = MC_CLASS_NAMES_EN;
static const int RANK[MC_CLASSES] = MC_CLASS_RANK;

static esp_err_t h_info(httpd_req_t *req)
{
    char ssid[70], name[24];
    json_escape(g_state.ssid, ssid, sizeof(ssid));
    int w, h;
    camera_resolution_size(camera_resolution(), &w, &h);
    const bool playing = g_state.status >= ST_SEARCHING;
    out_t o = {s_out, OUT_SIZE, 0};
    put(&o, "{\"firmware\":\"%s\",\"board\":\"%s\",\"sensor\":\"%s\",\"camera_ok\":%s,"
            "\"mode\":\"%s\",\"models\":%s,\"resolution\":\"%dx%d\",\"resolutions\":[",
        POKERDICE_VERSION, BOARD_NAME, g_state.sensor, g_state.camera_ok ? "true" : "false",
        playing ? "detection" : "camera", g_state.models_present ? "true" : "false", w, h);
    for (int i = 0; i < CAMERA_RESOLUTIONS; i++) {
        camera_resolution_size(i, &w, &h);
        put(&o, "%s\"%dx%d\"", i ? "," : "", w, h);
    }
    put(&o, "],\"wifi\":{\"mode\":\"%s\",\"ssid\":\"%s\",\"ip\":\"%s\",\"rssi\":%d,\"hostname\":\"%s.local\","
            "\"hotspot\":\"%s\",\"hotspot_clients\":%d,\"saved_mode\":\"%s\"},"
            "\"captures\":%lu,\"stream_frames\":%lu,\"stream_active\":%s,\"last_jpeg_bytes\":%lu,"
            "\"heap_free\":%u,\"psram_free\":%u,\"uptime_s\":%lld,",
        net_name(g_state.net), ssid, g_state.ip, g_state.rssi, MDNS_HOSTNAME, AP_SSID, g_state.hotspot_clients,
        g_settings.wifi_mode == WIFI_MODE_HOTSPOT ? "hotspot" : "home",
        (unsigned long)g_state.captures, (unsigned long)g_state.stream_frames, g_state.stream_active ? "true" : "false",
        (unsigned long)g_state.last_jpeg_bytes, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), esp_timer_get_time() / 1000000);
    // models (values of model_config.h compiled into the firmware)
    put(&o, "\"model\":{\"config\":%s,\"version\":\"%s\",\"width\":%d,\"height\":%d,\"det_imgsz\":%d,\"cls_size\":%d,"
            "\"dice\":%d,\"unsure\":%d,\"det_kb\":%lu,\"cls_kb\":%lu,\"det_test\":%d,\"cls_test\":%d,"
            "\"strips_internal\":%s,\"exposure\":%d,\"aec\":%d,\"gain\":%d,\"classes\":[",
        MC_FROM_FILE ? "true" : "false", MC_VERSION, IMG_W, IMG_H, MC_DET_IMGSZ, MC_CLS_SIZE, MC_DICE, MC_CLASS_UNSURE,
        (unsigned long)g_state.det_kb, (unsigned long)g_state.cls_kb, g_state.det_test, g_state.cls_test,
        g_state.strips_internal ? "true" : "false", MC_CAMERA_EXPOSURE, MC_CAMERA_AEC, MC_CAMERA_GAIN);
    for (int i = 0; i < MC_CLASSES; i++) {
        json_escape(NAMES_EN[i], name, sizeof(name));
        put(&o, "%s\"%s\"", i ? "," : "", name);
    }
    put(&o, "],\"rank\":[");
    for (int i = 0; i < MC_CLASSES; i++) put(&o, "%s%d", i ? "," : "", RANK[i]);
    put(&o, "]}}");
    return send_out(req, &o);
}

static esp_err_t camera_json(httpd_req_t *req)
{
    char buf[400];
    camera_exposure_t e;
    camera_exposure(&e);
    int aec;
    float gain;
    camera_exposure_actual(&aec, &gain);
    snprintf(buf, sizeof(buf),
        "{\"quality\":%d,\"vflip\":%d,\"hmirror\":%d,\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
        "\"exposure\":{\"manual\":%s,\"level\":%d,\"aec\":%d,\"gain\":%d,\"actual_aec\":%d,\"actual_gain\":%.2f}}",
        g_settings.jpeg_quality, camera_vflip(), camera_hmirror(), g_settings.brightness, g_settings.contrast,
        g_settings.saturation, e.manual ? "true" : "false", e.ae_level, e.aec, e.gain, aec, gain);
    return send_json(req, buf);
}

static esp_err_t h_camera_get(httpd_req_t *req) { return camera_json(req); }

// POST /api/camera  quality=10&vflip=0&hmirror=0&brightness=0&contrast=0&saturation=0   (all optional)
static esp_err_t h_camera_post(httpd_req_t *req)
{
    char body[256];
    if (read_body(req, body, sizeof(body)) < 0) return bad_request(req, "no data");
    settings_t s = g_settings;
    bool bad = false;
    int v;
    if (int_field(body, "quality", 4, 40, &v, &bad)) s.jpeg_quality = (uint8_t)v;
    if (int_field(body, "vflip", 0, 1, &v, &bad)) s.vflip = (uint8_t)v;
    if (int_field(body, "hmirror", 0, 1, &v, &bad)) s.hmirror = (uint8_t)v;
    if (int_field(body, "brightness", -2, 2, &v, &bad)) s.brightness = (int8_t)v;
    if (int_field(body, "contrast", -2, 2, &v, &bad)) s.contrast = (int8_t)v;
    if (int_field(body, "saturation", -2, 2, &v, &bad)) s.saturation = (int8_t)v;
    if (bad) return bad_request(req, "value out of range");
    g_settings = s;
    settings_save();
    camera_apply_settings();
    return camera_json(req);
}

typedef struct { httpd_req_t *req; char res[16]; char exposure[96]; } capture_ctx_t;

static esp_err_t capture_send(const uint8_t *jpg, size_t len, void *arg)
{
    capture_ctx_t *c = (capture_ctx_t *)arg;
    int w, h;
    camera_resolution_size(camera_resolution(), &w, &h);
    snprintf(c->res, sizeof(c->res), "%dx%d", w, h);
    camera_exposure_t e;
    camera_exposure(&e);
    int aec;
    float gain;
    camera_exposure_actual(&aec, &gain);
    if (e.manual) snprintf(c->exposure, sizeof(c->exposure), "manual aec=%d gain=%d actual_aec=%d actual_gain=%.2f", e.aec, e.gain, aec, gain);
    else snprintf(c->exposure, sizeof(c->exposure), "auto level=%d actual_aec=%d actual_gain=%.2f", e.ae_level, aec, gain);
    common_headers(c->req);
    httpd_resp_set_type(c->req, "image/jpeg");
    httpd_resp_set_hdr(c->req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(c->req, "X-Resolution", c->res);
    httpd_resp_set_hdr(c->req, "X-Exposure", c->exposure);
    httpd_resp_set_hdr(c->req, "X-Sensor", g_state.sensor);
    httpd_resp_set_hdr(c->req, "Access-Control-Expose-Headers", "X-Resolution, X-Exposure, X-Sensor");
    g_state.captures++;
    return httpd_resp_send(c->req, (const char *)jpg, len);
}

// GET /api/capture?res=1024x768&exposure=-1        auto exposure with correction -2..2
// GET /api/capture?res=1024x768&aec=300&gain=0     fixed exposure time and gain
// Without a value the last resolution / exposure stays.
static esp_err_t h_capture(httpd_req_t *req)
{
    char q[128] = {0}, v[16];
    httpd_req_get_url_query_str(req, q, sizeof(q));
    int res = -1;
    if (httpd_query_key_value(q, "res", v, sizeof(v)) == ESP_OK) {
        res = camera_resolution_find(v);
        if (res < 0) return bad_request(req, "unknown resolution (see /api/info)");
    }
    camera_exposure_t e;
    camera_exposure(&e);
    bool bad = false, set = false;
    int x, aec = -1, gain = -1;
    if (int_field(q, "exposure", -2, 2, &x, &bad)) { e.manual = false; e.ae_level = x; set = true; }
    if (int_field(q, "aec", 0, 65535, &aec, &bad) | int_field(q, "gain", 0, 30, &gain, &bad)) {
        if (aec < 0 || gain < 0) return bad_request(req, "aec and gain belong together");
        e.manual = true; e.aec = aec; e.gain = gain; set = true;
    }
    if (bad) return bad_request(req, "value out of range");
    capture_ctx_t *c = calloc(1, sizeof(*c));
    if (!c) return ESP_ERR_NO_MEM;
    c->req = req;
    esp_err_t err = camera_capture(res, set ? &e : NULL, true, capture_send, c);
    free(c);
    if (err == ESP_ERR_INVALID_STATE) return bad_request(req, "camera not available");
    if (err != ESP_OK) {
        httpd_resp_set_status(req, "503 Service Unavailable");
        return httpd_resp_sendstr(req, "no image");
    }
    return ESP_OK;
}

static esp_err_t h_wifi_scan(httpd_req_t *req)
{
    char *buf = malloc(2200);
    if (!buf) return ESP_ERR_NO_MEM;
    wifi_scan_json(buf, 2200);
    esp_err_t err = send_json(req, buf);
    free(buf);
    return err;
}

// POST /api/wifi  mode=home&ssid=...&password=...   or   mode=hotspot   -> saved, the board restarts
static esp_err_t h_wifi_post(httpd_req_t *req)
{
    char body[300], mode[12] = "home", ssid[33] = {0}, pass[65] = {0};
    if (read_body(req, body, sizeof(body)) < 0) return bad_request(req, "no data");
    field(body, "mode", mode, sizeof(mode));
    if (strcmp(mode, "hotspot") == 0) {
        g_settings.wifi_mode = WIFI_MODE_HOTSPOT;
    } else {
        if (!field(body, "ssid", ssid, sizeof(ssid)) || !ssid[0]) return bad_request(req, "network name missing");
        field(body, "password", pass, sizeof(pass));
        if (pass[0] && strlen(pass) < 8) return bad_request(req, "password too short (at least 8 characters)");
        wifi_credentials_save(ssid, pass);
        memset(pass, 0, sizeof(pass));
        g_settings.wifi_mode = WIFI_MODE_HOME;
    }
    settings_save();
    send_json(req, "{\"ok\":true,\"restart\":true}");
    xTaskCreate(restart_later, "restart", 2048, NULL, 5, NULL);
    return ESP_OK;
}

static esp_err_t h_wifi_forget(httpd_req_t *req)
{
    wifi_credentials_forget();
    g_settings.wifi_mode = WIFI_MODE_HOME;
    settings_save();
    send_json(req, "{\"ok\":true,\"restart\":true}");
    xTaskCreate(restart_later, "restart", 2048, NULL, 5, NULL);
    return ESP_OK;
}

static esp_err_t h_restart(httpd_req_t *req)
{
    send_json(req, "{\"ok\":true,\"restart\":true}");
    xTaskCreate(restart_later, "restart", 2048, NULL, 5, NULL);
    return ESP_OK;
}

// Captive portal: in hotspot mode every unknown address leads to the page
static esp_err_t h_not_found(httpd_req_t *req, httpd_err_code_t err)
{
    if (g_state.net == NET_HOTSPOT || g_state.net == NET_HOTSPOT_FALLBACK) {
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/");
        return httpd_resp_send(req, NULL, 0);
    }
    httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "not found");
    return ESP_OK;
}

// ---------------------------------------------------------------- live stream (port 81)
#define BOUNDARY "pokerdiceframe"

typedef struct { httpd_req_t *req; } stream_ctx_t;

static esp_err_t stream_part(const uint8_t *jpg, size_t len, void *arg)
{
    stream_ctx_t *c = (stream_ctx_t *)arg;
    char head[96];
    int n = snprintf(head, sizeof(head), "\r\n--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", (unsigned)len);
    if (httpd_resp_send_chunk(c->req, head, n) != ESP_OK) return ESP_FAIL;
    if (httpd_resp_send_chunk(c->req, (const char *)jpg, len) != ESP_OK) return ESP_FAIL;
    g_state.stream_frames++;
    return ESP_OK;
}

static esp_err_t h_stream(httpd_req_t *req)
{
    if (g_state.stream_active) {
        httpd_resp_set_status(req, "503 Service Unavailable");
        return httpd_resp_sendstr(req, "the live view already has a viewer");
    }
    g_state.stream_active = true;
    httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" BOUNDARY);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    stream_ctx_t c = { req };
    while (camera_capture(-1, NULL, false, stream_part, &c) == ESP_OK) {
        vTaskDelay(pdMS_TO_TICKS(20));   // leave room for single images
    }
    g_state.stream_active = false;
    return ESP_OK;
}

static esp_err_t h_to_stream(httpd_req_t *req)
{
    httpd_resp_set_status(req, "302 Found");
    httpd_resp_set_hdr(req, "Location", "/stream");
    return httpd_resp_send(req, NULL, 0);
}

esp_err_t web_start(void)
{
    s_flock = xSemaphoreCreateMutex();
    s_out = (char *)heap_caps_malloc(OUT_SIZE, MALLOC_CAP_SPIRAM);
    s_copy = (state_t *)heap_caps_malloc(sizeof(state_t), MALLOC_CAP_SPIRAM);
    for (int i = 0; i < 2; i++) s_json[i] = (char *)heap_caps_calloc(1, JSON_MAX, MALLOC_CAP_SPIRAM);
    if (!s_flock || !s_out || !s_copy || !s_json[0] || !s_json[1]) {
        ESP_LOGE(TAG, "no memory for the web server");
        return ESP_ERR_NO_MEM;
    }
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.max_uri_handlers = 20;
    cfg.lru_purge_enable = true;
    cfg.stack_size = 8192;
    cfg.core_id = 0;
    httpd_handle_t srv = NULL;
    esp_err_t err = httpd_start(&srv, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "web server failed: %s", esp_err_to_name(err));
        return err;
    }
    const httpd_uri_t uris[] = {
        { .uri = "/",                .method = HTTP_GET,  .handler = h_index },
        { .uri = "/api/info",        .method = HTTP_GET,  .handler = h_info },
        { .uri = "/api/camera",      .method = HTTP_GET,  .handler = h_camera_get },
        { .uri = "/api/camera",      .method = HTTP_POST, .handler = h_camera_post },
        { .uri = "/api/capture",     .method = HTTP_GET,  .handler = h_capture },
        { .uri = "/api/wifi/scan",   .method = HTTP_GET,  .handler = h_wifi_scan },
        { .uri = "/api/wifi",        .method = HTTP_POST, .handler = h_wifi_post },
        { .uri = "/api/wifi/forget", .method = HTTP_POST, .handler = h_wifi_forget },
        { .uri = "/api/restart",     .method = HTTP_POST, .handler = h_restart },
        { .uri = "/api/state",       .method = HTTP_GET,  .handler = h_state },
        { .uri = "/api/frame",       .method = HTTP_GET,  .handler = h_frame },
        { .uri = "/api/settings",    .method = HTTP_GET,  .handler = h_settings_get },
        { .uri = "/api/settings",    .method = HTTP_POST, .handler = h_settings_post },
        { .uri = "/api/cmd",         .method = HTTP_POST, .handler = h_cmd },
    };
    for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); i++) httpd_register_uri_handler(srv, &uris[i]);
    httpd_register_err_handler(srv, HTTPD_404_NOT_FOUND, h_not_found);

    httpd_config_t sc = HTTPD_DEFAULT_CONFIG();
    sc.server_port = 81;
    sc.ctrl_port = 32769;
    sc.max_uri_handlers = 2;
    sc.stack_size = 6144;
    sc.core_id = 0;
    httpd_handle_t ss = NULL;
    if (httpd_start(&ss, &sc) == ESP_OK) {
        const httpd_uri_t us = { .uri = "/stream", .method = HTTP_GET, .handler = h_stream };
        const httpd_uri_t ur = { .uri = "/", .method = HTTP_GET, .handler = h_to_stream };
        httpd_register_uri_handler(ss, &us);
        httpd_register_uri_handler(ss, &ur);
    }
    esp_log_level_set("httpd_txrx", ESP_LOG_ERROR);   // "error in send: 104" = browser closed the live view
    esp_log_level_set("httpd_uri", ESP_LOG_ERROR);    // unknown addresses are answered by the captive portal
    ESP_LOGI(TAG, "web page on port 80, live stream on port 81");
    return ESP_OK;
}
