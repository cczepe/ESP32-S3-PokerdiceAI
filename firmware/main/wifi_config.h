// Hotspot of the board (used for playing without a home network and for the first Wi-Fi setup).
// Please change the password before you use the board somewhere others can reach it (8..63 characters).
#pragma once
#define AP_SSID         "Pokerdice"
#define AP_PASSWORD     "pokerdice"
#define AP_CHANNEL      6
#define AP_MAX_CONN     4
#define MDNS_HOSTNAME   "pokerdice"     // reachable as http://pokerdice.local in the home network
#define STA_TIMEOUT_S   30              // home Wi-Fi not reached within this time -> hotspot on as well
