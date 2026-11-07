// RGB status LED (WS2812): blue = hotspot, green = home Wi-Fi, yellow blinking = connecting, red blinking = camera error.
// A new poker result: short white flash (twice for a straight or better), can be switched off in the settings.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void led_start(void);
void led_flash(int r, int g, int b, int ms);   // short signal, then back to the status colour
void led_signal(int count);                    // count white flashes (new result)
#ifdef __cplusplus
}
#endif
