#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "state.h"

state_t g_state;
static SemaphoreHandle_t s_lock;

void state_init(void)
{
    s_lock = xSemaphoreCreateMutex();
    g_state.det_test = g_state.cls_test = -1;
}

void state_lock(void)
{
    if (s_lock) xSemaphoreTake(s_lock, portMAX_DELAY);
}

void state_unlock(void)
{
    if (s_lock) xSemaphoreGive(s_lock);
}
