// Detection task (core 1). Per frame (camera JPEG at the model's resolution, e.g. 1024x768):
//   1. decode the JPEG scaled down, straight to letterbox size (e.g. 288x216) -> borders + int8 -> detector
//   2. decode boxes + NMS (in camera pixels)
//   3. decode the JPEG a second time at full resolution, in block mode (strips of 8/16 rows):
//      meanwhile the die crops (int8, classifier size) are computed row by row;
//      as soon as all are done, decoding stops. A full image is never held in memory.
//   4. classifier per die -> stabilizer
// A full RGB image at 1024x768 would need 2.3 MB (PSRAM: 2 MB) - hence this approach.
// While a web page shows the live view, the JPEG of each frame is handed to the web server together with
// the detection of exactly this frame (web_frame_store / web_frame_publish).
// While the trainer uses the camera (single images, live stream), the detection waits.
// Pre- and post-processing: preprocess.hpp (identical to the Python scripts of the trainer).
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#include "esp_partition.h"
#include "camera.h"
#include "esp_jpeg_dec.h"
#include "dl_model_base.hpp"
#include "sdkconfig.h"
#include "config.h"
#include "state.h"
#include "settings.h"
#include "detection.h"
#include "preprocess.hpp"
#include "stabilizer.hpp"
#include "led.h"
#include "version.h"
#include "web.h"

static const char *TAG = "detection";

// The block-mode strips go into internal RAM (faster) only if at least this much internal RAM stays free
// afterwards - Wi-Fi, TCP/IP and the web server need it. Otherwise they go into PSRAM.
#define INTERNAL_RESERVE_KB 48

static const int RANK[MC_CLASSES] = MC_CLASS_RANK;
static const char *const NAMES_EN[MC_CLASSES] = MC_CLASS_NAMES_EN;   // serial log + frame JSON
static const char *hand_name_en(int h) { return pd::hand_name(h); }
const char *detection_hand_name(int hand) { return hand_name_en(hand); }

static dl::Model *s_det = nullptr, *s_cls = nullptr;
static uint8_t *s_rgb = nullptr;            // scaled-down image (detector input)
static int s_rgb_w = 0, s_rgb_h = 0;
static pd::Letterbox s_lb;                   // letterbox of the camera image
static uint8_t *s_strip[2] = {nullptr, nullptr};   // block mode: current and previous strip
static int s_strip_len = 0;
static int8_t *s_crop[MAX_DICE];             // finished classifier inputs per die
static int8_t s_tab_det[256], s_tab_cls[256];
static pd::Axis s_ax[1024], s_ay[1024];
static pd::Box s_cand[256], s_boxes[MAX_DICE];
static pd::Stabilizer s_stab;
static int s_n_ranks = 0;
static volatile bool s_clear_history = false;

// ------------------------------------------------------------------ models
static bool partition_has_model(const char *label, uint32_t *kb)
{
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
    if (!p) { ESP_LOGE(TAG, "partition '%s' missing (partitions.csv?)", label); return false; }
    uint32_t head[4];
    if (esp_partition_read(p, 0, head, sizeof(head)) != ESP_OK) return false;
    if (memcmp(head, "EDL2", 4) != 0 && memcmp(head, "EDL1", 4) != 0) {
        ESP_LOGW(TAG, "partition '%s' contains no model", label);
        return false;
    }
    uint32_t size = head[2];
    if (size == 0 || size + 16 > p->size) {
        ESP_LOGE(TAG, "model in '%s' does not fit (%u bytes, partition %u)", label, (unsigned)size, (unsigned)p->size);
        return false;
    }
    *kb = (size + 1023) / 1024;
    return true;
}

static dl::Model *model_load(const char *label, bool copy_params, int *test)
{
    dl::Model *m = new dl::Model(label, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY,
                                 nullptr, copy_params);
    if (!m || m->get_inputs().empty()) {
        ESP_LOGE(TAG, "model '%s' cannot be loaded", label);
        delete m;
        return nullptr;
    }
    int64_t t = esp_timer_get_time();
    *test = m->test() == ESP_OK ? 1 : 0;   // compares with the test values from esp-ppq
    ESP_LOGI(TAG, "self-test '%s': %s (%lld ms)", label, *test ? "ok" : "MISMATCH",
             (long long)((esp_timer_get_time() - t) / 1000));
    m->minimize();
    return m;
}

static bool outputs_check(void)
{
    const char *names[6] = {"box0", "score0", "box1", "score1", "box2", "score2"};
    for (int i = 0; i < 6; i++) {
        dl::TensorBase *t = s_det->get_output(names[i]);
        if (!t || t->shape.size() != 4 || t->dtype != dl::DATA_TYPE_INT8) {
            ESP_LOGE(TAG, "detector output %s missing or wrong format", names[i]);
            return false;
        }
    }
    dl::TensorBase *in = s_det->get_input();
    if (in->shape.size() != 4 || in->shape[1] != MC_DET_IMGSZ || in->shape[2] != MC_DET_IMGSZ || in->shape[3] != 3) {
        ESP_LOGE(TAG, "detector input does not match model_config.h (imgsz %d)", MC_DET_IMGSZ);
        return false;
    }
    dl::TensorBase *ci = s_cls->get_input();
    if (ci->shape.size() != 4 || ci->shape[1] != MC_CLS_SIZE || ci->shape[3] != 3) {
        ESP_LOGE(TAG, "classifier input does not match model_config.h (%d px)", MC_CLS_SIZE);
        return false;
    }
    dl::TensorBase *co = s_cls->get_output();
    if (co->get_size() < MC_CLASSES) {
        ESP_LOGE(TAG, "classifier has too few outputs");
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ get and decode the image
// Read the header; the decoder reports the output size (when scaling down, already the small size)
static bool jpeg_header(jpeg_dec_handle_t h, jpeg_dec_io_t *io, const uint8_t *jpg, size_t len, int w, int hgt)
{
    io->inbuf = (uint8_t *)jpg;
    io->inbuf_len = (int)len;
    jpeg_dec_header_info_t info;
    if (jpeg_dec_parse_header(h, io, &info) != JPEG_ERR_OK) return false;
    if (info.width != w || info.height != hgt) {
        ESP_LOGW(TAG, "JPEG delivers %dx%d, expected %dx%d - does the image match model_config.h (%dx%d)?", info.width,
                 info.height, w, hgt, IMG_W, IMG_H);
        return false;
    }
    return true;
}

// 1. decode scaled down (straight to letterbox size)
static bool jpeg_scaled(const uint8_t *jpg, size_t len)
{
    jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
    cfg.output_type = JPEG_PIXEL_FORMAT_RGB888;
    cfg.scale.width = (uint16_t)s_rgb_w;
    cfg.scale.height = (uint16_t)s_rgb_h;
    jpeg_dec_handle_t h = nullptr;
    if (jpeg_dec_open(&cfg, &h) != JPEG_ERR_OK) return false;
    jpeg_dec_io_t io = {};
    bool ok = jpeg_header(h, &io, jpg, len, s_rgb_w, s_rgb_h);
    if (ok) {
        io.outbuf = s_rgb;
        ok = jpeg_dec_process(h, &io) == JPEG_ERR_OK;
    }
    jpeg_dec_close(h);
    return ok;
}

// Detector input: scaled-down image + gray borders, uint8 -> int8 via table.
// If the size matches exactly (e.g. 288x216 at 4:3), only copy; otherwise (e.g. 5:4) letterbox once more.
static void detector_input(int8_t *in)
{
    const int D = MC_DET_IMGSZ;
    if (s_rgb_w != s_lb.nw || s_rgb_h != s_lb.nh) {
        pd::letterbox(s_rgb, s_rgb_w, s_rgb_h, D, in, [](uint8_t v) { return s_tab_det[v]; }, s_ax, s_ay);
        return;
    }
    const int8_t gray = s_tab_det[pd::GRAY];
    memset(in, gray, (size_t)D * s_lb.dy * 3);                                                   // top
    memset(in + (size_t)(s_lb.dy + s_lb.nh) * D * 3, gray, (size_t)(D - s_lb.dy - s_lb.nh) * D * 3);   // bottom
    for (int y = 0; y < s_lb.nh; y++) {
        int8_t *r = in + (size_t)(y + s_lb.dy) * D * 3;
        const uint8_t *q = s_rgb + (size_t)y * s_lb.nw * 3;
        if (s_lb.dx) memset(r, gray, (size_t)s_lb.dx * 3);
        int8_t *o = r + s_lb.dx * 3;
        for (int i = 0; i < s_lb.nw * 3; i++) o[i] = s_tab_det[q[i]];
        int right = D - s_lb.dx - s_lb.nw;
        if (right > 0) memset(o + s_lb.nw * 3, gray, (size_t)right * 3);
    }
}

// 3. full resolution in block mode; compute the crops row by row, stop as soon as all are done.
// Returns the number of decoded strips (0 = error); *total = strips in the whole image.
static int crops_block(const uint8_t *jpg, size_t len, const pd::Box *b, int n, int *total)
{
    static pd::CropStream<int8_t, MAX_DICE> stream;
    stream.start(n, IMG_W, IMG_H);
    for (int i = 0; i < stream.n; i++)
        stream.set(i, pd::crop_plan(1.0f, b[i].x0, b[i].y0, b[i].x1, b[i].y1, MC_CLS_SIZE, MC_CLS_MARGIN), s_crop[i]);
    jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
    cfg.output_type = JPEG_PIXEL_FORMAT_RGB888;
    cfg.block_enable = true;
    jpeg_dec_handle_t h = nullptr;
    if (jpeg_dec_open(&cfg, &h) != JPEG_ERR_OK) return 0;
    jpeg_dec_io_t io = {};
    int count = 0, blen = 0, done_strips = 0;
    bool ok = jpeg_header(h, &io, jpg, len, IMG_W, IMG_H) && jpeg_dec_get_outbuf_len(h, &blen) == JPEG_ERR_OK &&
              jpeg_dec_get_process_count(h, &count) == JPEG_ERR_OK && count > 0 && blen <= s_strip_len;
    if (!ok) ESP_LOGW(TAG, "block mode: strip %d bytes (buffer %d), %d strips", blen, s_strip_len, count);
    const int sh = ok ? IMG_H / count : 0;
    if (total) *total = count;
    for (int s = 0; ok && s < count; s++) {
        uint8_t *cur = s_strip[s & 1];
        io.outbuf = cur;
        if (jpeg_dec_process(h, &io) != JPEG_ERR_OK) { ok = false; break; }
        done_strips = s + 1;
        bool all = stream.strip(cur, s ? s_strip[(s + 1) & 1] : nullptr, s * sh, sh, s == count - 1,
                                [](uint8_t v) { return s_tab_cls[v]; });
        if (all) break;   // all crops done -> do not decode the rest of the image
    }
    jpeg_dec_close(h);
    return ok ? done_strips : 0;
}

// JSON with the detection of one frame (also the content of a snapshot): "cls" = class index
static void frame_json(char *json, size_t size, const die_t *d, int n, int status, int hand)
{
    int p = snprintf(json, size,
                     "{\"frame\":%lu,\"uptime_ms\":%lld,\"models\":\"%s\",\"built\":\"%s\","
                     "\"firmware\":\"%s\",\"threshold\":%.2f,\"width\":%d,\"height\":%d,\"status\":\"%s\","
                     "\"hand\":\"%s\",\"dice\":[",
                     (unsigned long)g_state.frame_no, (long long)(esp_timer_get_time() / 1000),
                     MC_FROM_FILE ? "own" : "none", MC_VERSION, POKERDICE_VERSION, g_settings.threshold_pct / 100.0f,
                     IMG_W, IMG_H, status == pd::S_STABLE ? "stable" : (status == pd::S_ROLLING ? "rolling" : "searching"),
                     hand_name_en(hand));
    for (int i = 0; i < n && p < (int)size - 160; i++) {
        const char *k = (d[i].cls >= 0 && d[i].cls < MC_CLASSES) ? NAMES_EN[d[i].cls] : "-";
        p += snprintf(json + p, size - p,
                      "%s{\"x0\":%.1f,\"y0\":%.1f,\"x1\":%.1f,\"y1\":%.1f,\"score\":%.3f,\"class\":\"%s\",\"cls\":%d,"
                      "\"p\":%.3f}",
                      i ? "," : "", d[i].x0, d[i].y0, d[i].x1, d[i].y1, d[i].score, k, d[i].cls, d[i].p);
    }
    snprintf(json + p, size - p, "]}");
}

static inline void average(float *value, float now) { *value = *value == 0 ? now : *value * 0.8f + now * 0.2f; }

// ------------------------------------------------------------------ task
static void detection_task(void *arg)
{
    const bool models = s_det && s_cls;
    dl::TensorBase *det_in = models ? s_det->get_input() : nullptr;
    dl::TensorBase *cls_in = models ? s_cls->get_input() : nullptr;
    dl::TensorBase *cls_out = models ? s_cls->get_output() : nullptr;
    pd::Level lv[3];
    if (models) {
        const char *names[6] = {"box0", "score0", "box1", "score1", "box2", "score2"};
        const int strides[3] = MC_STRIDES;
        for (int e = 0; e < 3; e++) {
            dl::TensorBase *b = s_det->get_output(names[2 * e]), *s = s_det->get_output(names[2 * e + 1]);
            lv[e].box = (const int8_t *)b->data;
            lv[e].box_exp = (int)b->exponent;
            lv[e].score = (const int8_t *)s->data;
            lv[e].score_exp = (int)s->exponent;
            lv[e].h = b->shape[1];
            lv[e].w = b->shape[2];
            lv[e].channels = b->shape[3];
            lv[e].stride = strides[e];
        }
    }
    int64_t last = esp_timer_get_time();

    while (1) {
        const bool trainer = camera_in_use_by_trainer();
        g_state.camera_busy = trainer;
        if (g_state.pause || trainer) {
            // paused by the user: the web live view keeps getting images (without detection);
            // camera in use by the trainer: leave the camera alone
            if (!trainer && web_frame_wanted()) {
                camera_fb_t *fb = camera_frame_take(500);
                if (fb) {
                    bool stored = web_frame_store(fb->buf, fb->len);
                    camera_frame_release(fb);
                    if (stored) {
                        static char json[200];
                        state_lock();
                        g_state.frame_no++;
                        state_unlock();
                        snprintf(json, sizeof(json),
                                 "{\"frame\":%lu,\"models\":\"%s\",\"built\":\"%s\",\"firmware\":\"%s\","
                                 "\"width\":%d,\"height\":%d,\"status\":\"paused\",\"hand\":\"\",\"dice\":[]}",
                                 (unsigned long)g_state.frame_no, MC_FROM_FILE ? "own" : "none", MC_VERSION,
                                 POKERDICE_VERSION, IMG_W, IMG_H);
                        web_frame_publish(g_state.frame_no, json);
                    }
                }
                vTaskDelay(pdMS_TO_TICKS(250));
            } else {
                vTaskDelay(pdMS_TO_TICKS(100));
            }
            last = esp_timer_get_time();
            continue;
        }
        if (s_clear_history) {
            s_stab.reset();
            s_clear_history = false;
        }
        int64_t t0 = esp_timer_get_time();
        bool ok;
        size_t jpeg_len = 0;
        // the JPEG stays in use until after the crops (second decoding in block mode)
        camera_fb_t *fb = camera_frame_take(1000);
        int64_t t1 = esp_timer_get_time();
        if (!fb) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
        const uint8_t *jpg = fb->buf;
        jpeg_len = fb->len;
        ok = jpeg_scaled(jpg, jpeg_len);
        // live view / snapshot on a web page: keep a copy of this JPEG (published after the detection)
        bool web_frame = ok && web_frame_store(jpg, jpeg_len);
        int64_t t2 = esp_timer_get_time();
        if (!ok) {
            ESP_LOGW(TAG, "JPEG error");
            camera_frame_release(fb);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        int n = 0, cand = 0;
        int64_t us_crop = 0;   // time for the crops (block mode; rest of t5..t6 = classifier network)
        int64_t t3 = t2, t4 = t2, t5 = t2, t6 = t2;
        int classes[MAX_DICE] = {0};
        float probs[MAX_DICE] = {0};
        if (models) {
            detector_input((int8_t *)det_in->data);
            t3 = esp_timer_get_time();
            static const dl::runtime_mode_t MODES[3] = {dl::RUNTIME_MODE_SINGLE_CORE, dl::RUNTIME_MODE_AUTO,
                                                        dl::RUNTIME_MODE_MULTI_CORE};
            s_det->run(MODES[g_settings.det_cores < 3 ? g_settings.det_cores : 0]);
            t4 = esp_timer_get_time();
            n = pd::decode(lv, 3, MC_DET_IMGSZ, IMG_W, IMG_H, g_settings.threshold_pct / 100.0f, MC_DET_NMS_IOU,
                           s_cand, 256, s_boxes, MAX_DICE, &cand);
            t5 = esp_timer_get_time();
            if (n > MAX_DICE) n = MAX_DICE;
            if (n > 0) {
                int64_t ta = esp_timer_get_time();
                int total = 0, strips = crops_block(jpg, jpeg_len, s_boxes, n, &total);
                us_crop = esp_timer_get_time() - ta;
                g_state.strips = strips;
                g_state.strips_total = total;
                if (!strips) { ESP_LOGW(TAG, "crops: JPEG error in block mode"); n = 0; }
            }
            const size_t cls_bytes = (size_t)MC_CLS_SIZE * MC_CLS_SIZE * 3;
            for (int i = 0; i < n; i++) {
                memcpy(cls_in->data, s_crop[i], cls_bytes);
                s_cls->run();
                classes[i] = pd::choose_class((const int8_t *)cls_out->data, (int)cls_out->exponent, MC_CLASSES,
                                              &probs[i]);
            }
            t6 = esp_timer_get_time();
        }

        camera_frame_release(fb);
        // stabilizer (only the expected number of best boxes counts as a roll)
        bool is_new = false;
        pd::Signature sig;
        pd::StabStatus st = s_stab.step(classes, probs, n, RANK, MC_DICE, MC_CLASS_UNSURE,
                                        g_settings.stable_n, g_settings.stable_m, 0.5f, &is_new, &sig);
        int64_t t7 = esp_timer_get_time();

        // sort dice by x (for the live view and "by position")
        die_t d[MAX_DICE];
        for (int i = 0; i < n; i++) {
            pd::Box &b = s_boxes[i];
            d[i] = {std::max(0.f, b.x0), std::max(0.f, b.y0), std::min((float)IMG_W, b.x1),
                    std::min((float)IMG_H, b.y1), b.score, classes[i], probs[i]};
        }
        for (int i = 1; i < n && i < MAX_DICE; i++) {   // insertion sort (n <= 10)
            die_t t = d[i];
            int j = i - 1;
            while (j >= 0 && d[j].x0 + d[j].x1 > t.x0 + t.x1) { d[j + 1] = d[j]; j--; }
            d[j + 1] = t;
        }

        int hand = pd::H_NONE;
        if (st == pd::S_STABLE) {
            int r[MAX_DICE];
            for (int i = 0; i < sig.n; i++) r[i] = RANK[sig.k[i]];
            hand = pd::score_hand(r, sig.n, s_n_ranks);
        }

        state_lock();
        g_state.frame_no++;
        g_state.n = n;
        memcpy(g_state.dice, d, sizeof(die_t) * n);
        g_state.candidates = cand;
        if (!models) {
            // status stays ST_NO_MODELS / ST_ERROR
        } else if (st == pd::S_STABLE) {
            g_state.status = ST_STABLE;
        } else if (st == pd::S_ROLLING) {
            g_state.status = ST_ROLLING;
        } else {
            g_state.status = ST_SEARCHING;
        }
        if (is_new) {
            roll_t r = {};
            r.count = sig.n;
            memcpy(r.cls, sig.k, sizeof(int) * sig.n);
            r.hand = hand;
            r.frame_no = g_state.frame_no;
            r.time_us = t7;
            g_state.stable = r;
            g_state.roll_no++;
            memmove(&g_state.history[1], &g_state.history[0], sizeof(roll_t) * (HISTORY_MAX - 1));
            g_state.history[0] = r;
            if (g_state.history_n < HISTORY_MAX) g_state.history_n++;
        }
        float ms = (t7 - t0) / 1000.0f;
        average(&g_state.ms_camera, (t1 - t0) / 1000.0f);
        average(&g_state.ms_jpeg, (t2 - t1) / 1000.0f);
        average(&g_state.ms_letterbox, (t3 - t2) / 1000.0f);
        average(&g_state.ms_det, (t4 - t3) / 1000.0f);
        average(&g_state.ms_decode, (t5 - t4) / 1000.0f);
        average(&g_state.ms_crop, us_crop / 1000.0f);
        average(&g_state.ms_cls, (t6 - t5 - us_crop) / 1000.0f);
        average(&g_state.ms_total, ms);
        int64_t now = esp_timer_get_time();
        average(&g_state.fps, 1e6f / (float)std::max<int64_t>(1, now - last));
        last = now;
        g_state.jpeg_bytes = (uint32_t)jpeg_len;
        g_state.timing[g_state.timing_pos] = ms;
        g_state.timing_pos = (g_state.timing_pos + 1) % TIMING_MAX;
        if (g_state.timing_n < TIMING_MAX) g_state.timing_n++;
        state_unlock();

        if (web_frame) {
            static char json[1600];
            frame_json(json, sizeof(json), d, n, st, hand);
            web_frame_publish(g_state.frame_no, json);
        }

        if (is_new) {
            if (g_settings.led_result) led_signal(hand >= pd::H_SMALL_STRAIGHT ? 2 : 1);
            char text[64];
            int p = 0;
            for (int i = 0; i < sig.n; i++) p += snprintf(text + p, sizeof(text) - p, "%s ", NAMES_EN[sig.k[i]]);
            ESP_LOGI(TAG, "roll #%lu: %s-> %s", (unsigned long)g_state.roll_no, text, hand_name_en(hand));
        }
        vTaskDelay(1);   // release core 1 briefly (otherwise the watchdog reports IDLE1)
    }
}

void detection_clear_history(void)
{
    state_lock();
    g_state.history_n = 0;
    g_state.roll_no = 0;
    memset(&g_state.stable, 0, sizeof(g_state.stable));
    state_unlock();
    s_clear_history = true;
}

esp_err_t detection_start(void)
{
    for (int i = 0; i < MC_CLASSES; i++) if (RANK[i] + 1 > s_n_ranks) s_n_ranks = RANK[i] + 1;
    g_state.status = ST_START;

    // 1. models in flash? (without them the board is a camera for the trainer only)
    g_state.det_test = g_state.cls_test = -1;
    g_state.det_present = partition_has_model("det", &g_state.det_kb);
    g_state.cls_present = partition_has_model("cls", &g_state.cls_kb);
    g_state.models_present = g_state.det_present && g_state.cls_present;
    if (!MC_FROM_FILE || !g_state.models_present) {
        snprintf(g_state.message, sizeof(g_state.message), "%s", !MC_FROM_FILE ? "firmware/models/model_config.h missing"
                 : (!g_state.det_present ? "detector model missing" : "classifier model missing"));
        g_state.status = ST_NO_MODELS;
        ESP_LOGW(TAG, "%s - camera mode only", g_state.message);
        return ESP_ERR_NOT_FOUND;
    }
    // 2. camera for playing: resolution and exposure of model_config.h
    char res_text[16];
    snprintf(res_text, sizeof(res_text), "%dx%d", IMG_W, IMG_H);
    int res = camera_resolution_find(res_text);
    if (!g_state.camera_ok || res < 0) {
        snprintf(g_state.message, sizeof(g_state.message), !g_state.camera_ok ? "camera not available"
                 : "camera size %s of model_config.h not supported", res_text);
        g_state.status = ST_ERROR;
        ESP_LOGE(TAG, "%s", g_state.message);
        return ESP_ERR_INVALID_STATE;
    }
    camera_exposure_t e = {};
    if (MC_CAMERA_AEC >= 0) { e.manual = true; e.aec = MC_CAMERA_AEC; e.gain = MC_CAMERA_GAIN; }
    else { e.manual = false; e.ae_level = MC_CAMERA_EXPOSURE; }
    camera_play_set(res, &e);
    if (e.manual) ESP_LOGI(TAG, "playing with %s, fixed exposure aec %d gain %d", res_text, e.aec, e.gain);
    else ESP_LOGI(TAG, "playing with %s, auto exposure with correction %+d", res_text, e.ae_level);

    // 3. Buffers. Scaled-down image at letterbox size (multiples of 8 for the JPEG decoder),
    //    two strips for the block mode (internal RAM if enough stays free for Wi-Fi/web, otherwise PSRAM),
    //    crops per die.
    s_lb = pd::letterbox_params(IMG_W, IMG_H, MC_DET_IMGSZ);
    s_rgb_w = (s_lb.nw + 7) / 8 * 8;
    s_rgb_h = (s_lb.nh + 7) / 8 * 8;
    s_rgb = (uint8_t *)jpeg_calloc_align((size_t)s_rgb_w * s_rgb_h * 3, 16);
    s_strip_len = IMG_W * 16 * 3;
    const size_t reserve = INTERNAL_RESERVE_KB * 1024;
    for (int i = 0; i < 2; i++) {
        if (heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) >= (size_t)s_strip_len + 64 &&
            heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= (size_t)s_strip_len + reserve)
            s_strip[i] = (uint8_t *)heap_caps_aligned_calloc(16, 1, s_strip_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!s_strip[i]) s_strip[i] = (uint8_t *)heap_caps_aligned_calloc(16, 1, s_strip_len, MALLOC_CAP_SPIRAM);
    }
    g_state.strips_internal = s_strip[0] && esp_ptr_internal(s_strip[0]) && s_strip[1] && esp_ptr_internal(s_strip[1]);
    bool crops_ok = true;
    for (int i = 0; i < MAX_DICE; i++) {
        s_crop[i] = (int8_t *)heap_caps_malloc((size_t)MC_CLS_SIZE * MC_CLS_SIZE * 3, MALLOC_CAP_SPIRAM);
        if (!s_crop[i]) crops_ok = false;
    }
    if (!s_rgb || !s_strip[0] || !s_strip[1] || !crops_ok) {
        snprintf(g_state.message, sizeof(g_state.message), "out of memory");
        g_state.status = ST_ERROR;
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "camera image %dx%d -> detector %d (scaled %dx%d, letterbox %dx%d), strip buffers %s "
             "(internal RAM free now %u KB)", IMG_W, IMG_H, MC_DET_IMGSZ, s_rgb_w, s_rgb_h, s_lb.nw, s_lb.nh,
             g_state.strips_internal ? "internal" : "PSRAM",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    // 4. load the models (with self-test against the test values of the conversion)
    ESP_LOGI(TAG, "loading detector (%u KB)", (unsigned)g_state.det_kb);
    s_det = model_load("det", false, &g_state.det_test);     // parameters stay in flash (saves PSRAM)
    ESP_LOGI(TAG, "loading classifier (%u KB)", (unsigned)g_state.cls_kb);
    s_cls = model_load("cls", true, &g_state.cls_test);      // small -> into PSRAM (faster)
    if (!s_det || !s_cls || !outputs_check()) {
        snprintf(g_state.message, sizeof(g_state.message), "model does not match model_config.h");
        g_state.status = ST_ERROR;
        delete s_det;
        delete s_cls;
        s_det = s_cls = nullptr;
        return ESP_ERR_INVALID_STATE;
    }
    pd::input_table(s_tab_det, (int)s_det->get_input()->exponent);
    pd::input_table(s_tab_cls, (int)s_cls->get_input()->exponent);
    g_state.message[0] = 0;
    g_state.status = ST_SEARCHING;
    ESP_LOGI(TAG, "models ready (%s): det input exp %d, cls input exp %d, cls output exp %d", MC_VERSION,
             (int)s_det->get_input()->exponent, (int)s_cls->get_input()->exponent, (int)s_cls->get_output()->exponent);
    ESP_LOGI(TAG, "PSRAM free after model start: %u KB (lowest so far %u KB)",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
             (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM) / 1024));
    xTaskCreatePinnedToCore(detection_task, "detection", 12 * 1024, nullptr, 4, nullptr, 1);
    return ESP_OK;
}
