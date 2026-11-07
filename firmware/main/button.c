#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_system.h"
#include "esp_log.h"
#include "board.h"
#include "button.h"
#include "led.h"
#include "settings.h"

static const char *TAG = "button";
#define HOLD_RESET_MS 5000

static void button_task(void *arg)
{
    int held = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(20));
        if (gpio_get_level((gpio_num_t)PIN_BUTTON) == 0) {
            held += 20;
            if (held == 1000) ESP_LOGI(TAG, "keep holding for %d s to forget the home Wi-Fi", HOLD_RESET_MS / 1000);
            if (held >= 1000) led_flash(held / 250 % 2 ? 20 : 0, held / 250 % 2 ? 20 : 0, held / 250 % 2 ? 20 : 0, 100);
            if (held == HOLD_RESET_MS) {
                wifi_credentials_forget();
                g_settings.wifi_mode = WIFI_MODE_HOME;
                settings_save();
                ESP_LOGW(TAG, "home Wi-Fi forgotten - restarting into hotspot mode");
                vTaskDelay(pdMS_TO_TICKS(500));
                esp_restart();
            }
        } else {
            held = 0;
        }
    }
}

void button_start(void)
{
    if (PIN_BUTTON < 0) return;
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << PIN_BUTTON,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    gpio_config(&io);
    xTaskCreatePinnedToCore(button_task, "button", 3072, NULL, 3, NULL, 0);
}
