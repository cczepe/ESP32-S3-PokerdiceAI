// Shared state of the tasks: Wi-Fi, web server, LED (core 0) and detection (core 1).
// The detection fields are written only between state_lock() and state_unlock().
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DICE      10      // at most this many boxes are processed further
#define HISTORY_MAX   24
#define TIMING_MAX    96      // values for the chart (total time per frame)

typedef enum {
    NET_STARTING = 0,
    NET_CONNECTING,       // home Wi-Fi is being searched
    NET_STATION,          // connected to the home Wi-Fi, IP available
    NET_HOTSPOT,          // own hotspot only (no home Wi-Fi saved or chosen)
    NET_HOTSPOT_FALLBACK, // home Wi-Fi not reachable: hotspot on, still trying to connect
} net_mode_t;

typedef struct {
    float x0, y0, x1, y1;     // camera pixels
    float score;              // detector
    int   cls;                // index into the class names, -1 = not classified
    float p;                  // probability of the class
} die_t;

typedef enum {
    ST_START = 0,             // loading models / self-test
    ST_NO_MODELS,             // partition empty or model_config.h missing
    ST_ERROR,                 // camera or model broken
    ST_SEARCHING,             // no / too few dice
    ST_ROLLING,               // result still changing
    ST_STABLE,                // result settled (n of m frames equal)
} detection_status_t;

typedef struct {
    int      count;
    int      cls[MAX_DICE];         // sorted by rank
    int      hand;                  // pd::Hand
    uint32_t frame_no;
    int64_t  time_us;
} roll_t;

typedef struct {
    // --- network and camera
    volatile net_mode_t net;
    char ssid[33];               // home Wi-Fi name (if saved)
    char ip[16];                 // IP in the home Wi-Fi
    volatile int rssi;           // dBm
    volatile int hotspot_clients;
    volatile bool camera_ok;
    volatile bool camera_failed;      // start was tried and failed (LED red)
    char sensor[16];             // "OV3660", "OV2640", ...
    volatile uint32_t captures;          // single images delivered
    volatile uint32_t stream_frames;     // frames sent in the live stream
    volatile bool stream_active;
    volatile uint32_t last_jpeg_bytes;
    volatile bool models_present;        // detector and classifier found in flash

    // --- detection, per frame
    volatile detection_status_t status;
    int       n;                    // dice found (after NMS)
    die_t     dice[MAX_DICE];       // sorted by x (left -> right)
    int       candidates;           // before NMS
    uint32_t  frame_no;
    roll_t    stable;               // last stable result
    uint32_t  roll_no;              // counts new stable results
    // monitoring (ms per stage, moving average)
    float ms_camera, ms_jpeg, ms_letterbox, ms_det, ms_decode, ms_crop, ms_cls, ms_total, fps;   // ms_cls = network only
    uint32_t jpeg_bytes;
    int      strips, strips_total;  // block mode: strips decoded until all crops were done / strips in the whole image
    float    timing[TIMING_MAX];    // total time per frame, ring buffer
    int      timing_pos;
    int      timing_n;
    // start / self-test
    bool det_present, cls_present;  // model found in the partition
    int  det_test, cls_test;        // 1 = ok, 0 = mismatch, -1 = not tested
    uint32_t det_kb, cls_kb;
    bool strips_internal;           // block-mode buffers in internal RAM (faster) or PSRAM
    char message[64];               // start message / error text
    // history (newest first)
    roll_t history[HISTORY_MAX];
    int    history_n;
    // control
    volatile bool pause;            // paused by the user (web page)
    volatile bool camera_busy;      // the trainer uses the camera (live view or single images): detection waits
} state_t;

extern state_t g_state;
void state_init(void);
void state_lock(void);
void state_unlock(void);

#ifdef __cplusplus
}
#endif
