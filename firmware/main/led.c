#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/rmt_tx.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "board.h"
#include "led.h"
#include "state.h"

static const char *TAG = "led";
static rmt_channel_handle_t s_chan = NULL;
static rmt_encoder_handle_t s_enc = NULL;
static volatile int64_t s_flash_until = 0;
static volatile uint8_t s_flash[3];
static volatile int s_signal = 0;   // remaining phases of the white flashes (on/off)

#define LEVEL 12   // kept dim: the LED sits next to the camera

static void put(uint8_t r, uint8_t g, uint8_t b)
{
    if (!s_chan) return;
#if defined(LED_ORDER_RGB) && LED_ORDER_RGB
    uint8_t px[3] = { r, g, b };
#else
    uint8_t px[3] = { g, r, b };
#endif
    rmt_transmit_config_t tx = { .loop_count = 0 };
    rmt_transmit(s_chan, s_enc, px, sizeof(px), &tx);
    rmt_tx_wait_all_done(s_chan, 50);
}

void led_flash(int r, int g, int b, int ms)
{
    s_flash[0] = (uint8_t)r; s_flash[1] = (uint8_t)g; s_flash[2] = (uint8_t)b;
    s_flash_until = esp_timer_get_time() + (int64_t)ms * 1000;
}

void led_signal(int count)
{
    s_signal = count * 2;
}

static void led_task(void *arg)
{
    int t = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(100));
        t++;
        bool blink = (t / 4) % 2;
        if (esp_timer_get_time() < s_flash_until) { put(s_flash[0], s_flash[1], s_flash[2]); continue; }
        if (s_signal > 0) {   // phases of 200 ms: white, dark, white, ...
            if (t % 2 == 0) s_signal--;
            if (s_signal > 0) { put(s_signal % 2 ? 2 * LEVEL : 0, s_signal % 2 ? 2 * LEVEL : 0, s_signal % 2 ? 2 * LEVEL : 0); continue; }
        }
        if (g_state.camera_failed) { put(blink ? LEVEL : 0, 0, 0); continue; }
        switch (g_state.net) {
            case NET_STATION:          put(0, LEVEL, 0); break;
            case NET_HOTSPOT:          put(0, 0, LEVEL); break;
            case NET_HOTSPOT_FALLBACK: put(0, blink ? LEVEL : 0, LEVEL); break;   // blue, green blinking: still trying
            default:                   put(blink ? LEVEL : 0, blink ? LEVEL : 0, 0); break;
        }
    }
}

void led_start(void)
{
    if (PIN_RGB_LED < 0) return;
    rmt_tx_channel_config_t cc = {
        .gpio_num = PIN_RGB_LED,
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10000000,   // 0.1 us per tick
        .mem_block_symbols = 64,
        .trans_queue_depth = 4,
    };
    if (rmt_new_tx_channel(&cc, &s_chan) != ESP_OK) {
        ESP_LOGW(TAG, "no RMT channel for the LED");
        s_chan = NULL;
        return;
    }
    rmt_bytes_encoder_config_t ec = {
        .bit0 = { .level0 = 1, .duration0 = 3, .level1 = 0, .duration1 = 9 },   // 0.3 us high, 0.9 us low
        .bit1 = { .level0 = 1, .duration0 = 9, .level1 = 0, .duration1 = 3 },   // 0.9 us high, 0.3 us low
        .flags.msb_first = 1,
    };
    rmt_new_bytes_encoder(&ec, &s_enc);
    rmt_enable(s_chan);
    put(0, 0, 0);
    xTaskCreatePinnedToCore(led_task, "led", 3072, NULL, 2, NULL, 0);
}
