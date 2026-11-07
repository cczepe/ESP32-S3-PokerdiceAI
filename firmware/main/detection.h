// Detection: camera JPEG -> detector -> crops -> classifier -> stable result -> poker hand
#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loads the models (with self-test) and starts the detection task on core 1.
// Without models (or with models that do not fit model_config.h) no task is started; g_state.status tells why.
esp_err_t detection_start(void);
void detection_clear_history(void);
const char *detection_hand_name(int hand);   // English name of a poker hand ("" = no hand)

#ifdef __cplusplus
}
#endif
