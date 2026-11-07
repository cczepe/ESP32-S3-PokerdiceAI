#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lwip/inet.h"
#include "mdns.h"
#include "dns_server.h"
#include "wifi_config.h"
#include "wifi.h"
#include "settings.h"
#include "state.h"

static const char *TAG = "wifi";

static bool s_have_credentials = false;
static bool s_ap_on = false;
static volatile bool s_connected = false;
static volatile int64_t s_next_try_us = 0;
static int64_t s_start_us = 0;
static esp_netif_t *s_ap_netif = NULL;
static dns_server_handle_t s_dns = NULL;

#define RETRY_MS     3000     // without hotspot
#define RETRY_AP_MS  15000    // with hotspot less often (scanning disturbs the hotspot)

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected = false;
        g_state.ip[0] = 0;
        if (s_have_credentials && !s_ap_on) g_state.net = NET_CONNECTING;
        s_next_try_us = esp_timer_get_time() + (int64_t)(s_ap_on ? RETRY_AP_MS : RETRY_MS) * 1000;
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        snprintf(g_state.ip, sizeof(g_state.ip), IPSTR, IP2STR(&e->ip_info.ip));
        s_connected = true;
        g_state.net = NET_STATION;
        ESP_LOGI(TAG, "connected to '%s', IP %s, http://%s.local", g_state.ssid, g_state.ip, MDNS_HOSTNAME);
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *e = (wifi_event_ap_staconnected_t *)data;
        g_state.hotspot_clients++;
        ESP_LOGI(TAG, "device " MACSTR " joined the hotspot (%d now)", MAC2STR(e->mac), g_state.hotspot_clients);
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STADISCONNECTED) {
        if (g_state.hotspot_clients > 0) g_state.hotspot_clients--;
    }
}

static void hotspot_on(void)
{
    if (s_ap_on) return;
    wifi_config_t ap = { 0 };
    strncpy((char *)ap.ap.ssid, AP_SSID, sizeof(ap.ap.ssid));
    ap.ap.ssid_len = (uint8_t)strlen(AP_SSID);
    strncpy((char *)ap.ap.password, AP_PASSWORD, sizeof(ap.ap.password));
    ap.ap.channel = AP_CHANNEL;
    ap.ap.max_connection = AP_MAX_CONN;
    ap.ap.authmode = strlen(AP_PASSWORD) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    if (strlen(AP_PASSWORD) > 0 && strlen(AP_PASSWORD) < 8)
        ESP_LOGE(TAG, "AP_PASSWORD in wifi_config.h is shorter than 8 characters - the hotspot will not start");
    esp_wifi_set_mode(s_have_credentials ? WIFI_MODE_APSTA : WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap);
    s_ap_on = true;
    g_state.net = s_have_credentials ? NET_HOTSPOT_FALLBACK : NET_HOTSPOT;

    // DHCP option 114 tells newer phones where the captive portal is
    esp_netif_ip_info_t ip;
    esp_netif_get_ip_info(s_ap_netif, &ip);
    static char uri[32];
    char addr[16];
    inet_ntoa_r(ip.ip.addr, addr, sizeof(addr));
    snprintf(uri, sizeof(uri), "http://%s", addr);
    esp_netif_dhcps_stop(s_ap_netif);
    esp_netif_dhcps_option(s_ap_netif, ESP_NETIF_OP_SET, ESP_NETIF_CAPTIVEPORTAL_URI, uri, strlen(uri));
    esp_netif_dhcps_start(s_ap_netif);
    // DNS: every name is answered with our own address (captive portal)
    if (!s_dns) {
        esp_log_level_set("example_dns_redirect_server", ESP_LOG_NONE);   // phones also ask for names it cannot answer (IPv6)
        dns_server_config_t dns = DNS_SERVER_CONFIG_SINGLE("*", "WIFI_AP_DEF");
        s_dns = start_dns_server(&dns);
    }
    ESP_LOGI(TAG, "hotspot \"%s\" on channel %d (%s), web page http://%s", AP_SSID, AP_CHANNEL,
             strlen(AP_PASSWORD) ? "password from wifi_config.h" : "open", addr);
}

static void hotspot_off(void)
{
    if (!s_ap_on) return;
    if (s_dns) {
        stop_dns_server(s_dns);
        s_dns = NULL;
    }
    esp_wifi_set_mode(WIFI_MODE_STA);
    s_ap_on = false;
    ESP_LOGI(TAG, "hotspot off (home Wi-Fi connected)");
}

// watches the connection, the hotspot and the signal strength
static void wifi_task(void *arg)
{
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(500));
        int64_t now = esp_timer_get_time();
        if (s_have_credentials && !s_connected && s_next_try_us && now >= s_next_try_us) {
            s_next_try_us = 0;
            esp_wifi_connect();
        }
        if (s_have_credentials && !s_connected && !s_ap_on && (now - s_start_us) > STA_TIMEOUT_S * 1000000LL) {
            ESP_LOGW(TAG, "home Wi-Fi '%s' not reachable -> hotspot on as well (still trying)", g_state.ssid);
            hotspot_on();
        }
        if (s_connected && s_ap_on && s_have_credentials) {
            vTaskDelay(pdMS_TO_TICKS(3000));   // let the "saved" page reach the phone first
            hotspot_off();
        }
        if (s_connected) {
            wifi_ap_record_t info;
            if (esp_wifi_sta_get_ap_info(&info) == ESP_OK) g_state.rssi = info.rssi;
        }
    }
}

esp_err_t wifi_start(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *sta = esp_netif_create_default_wifi_sta();
    s_ap_netif = esp_netif_create_default_wifi_ap();
    esp_netif_set_hostname(sta, MDNS_HOSTNAME);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_err_t err = esp_wifi_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi start failed: %s", esp_err_to_name(err));
        return err;
    }
    esp_wifi_set_storage(WIFI_STORAGE_RAM);   // the credentials are kept in our own NVS namespace
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL);

    char ssid[33] = {0}, pass[65] = {0};
    s_have_credentials = wifi_credentials_load(ssid, sizeof(ssid), pass, sizeof(pass)) &&
                         g_settings.wifi_mode == WIFI_MODE_HOME;
    if (ssid[0]) snprintf(g_state.ssid, sizeof(g_state.ssid), "%s", ssid);
    s_start_us = esp_timer_get_time();

    if (s_have_credentials) {
        wifi_config_t sc = { 0 };
        memcpy(sc.sta.ssid, ssid, strnlen(ssid, sizeof(sc.sta.ssid)));   // without terminating zero allowed
        memcpy(sc.sta.password, pass, strnlen(pass, sizeof(sc.sta.password)));
        sc.sta.threshold.authmode = pass[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
        sc.sta.pmf_cfg.capable = true;
        esp_wifi_set_mode(WIFI_MODE_STA);
        esp_wifi_set_config(WIFI_IF_STA, &sc);
        g_state.net = NET_CONNECTING;
        ESP_ERROR_CHECK(esp_wifi_start());
        esp_wifi_connect();
        ESP_LOGI(TAG, "connecting to '%s' ...", ssid);
    } else {
        ESP_LOGI(TAG, "%s", ssid[0] ? "hotspot mode chosen" : "no home Wi-Fi saved -> hotspot (set up the home Wi-Fi on the web page)");
        hotspot_on();
        ESP_ERROR_CHECK(esp_wifi_start());
    }
    esp_wifi_set_ps(WIFI_PS_NONE);   // no power saving: smoother live view, always full reaction time
    memset(pass, 0, sizeof(pass));

    if (mdns_init() == ESP_OK) {
        mdns_hostname_set(MDNS_HOSTNAME);
        mdns_instance_name_set("Pokerdice");
        mdns_service_add("Pokerdice", "_http", "_tcp", 80, NULL, 0);
    }
    xTaskCreatePinnedToCore(wifi_task, "wifi", 4096, NULL, 4, NULL, 0);
    return ESP_OK;
}

esp_err_t wifi_scan_json(char *buf, size_t len)
{
    // in pure hotspot mode the station part is off and cannot scan -> switch it on (stays on, no connection)
    wifi_mode_t mode;
    if (esp_wifi_get_mode(&mode) == ESP_OK && mode == WIFI_MODE_AP) esp_wifi_set_mode(WIFI_MODE_APSTA);
    wifi_scan_config_t sc = { 0 };
    esp_err_t err = esp_wifi_scan_start(&sc, true);
    ESP_LOGI(TAG, "scan: %s", esp_err_to_name(err));
    if (err != ESP_OK) {
        snprintf(buf, len, "[]");
        return err;
    }
    uint16_t n = 20;
    wifi_ap_record_t recs[20];
    esp_wifi_scan_get_ap_records(&n, recs);
    ESP_LOGI(TAG, "scan: %d networks found", n);
    size_t pos = snprintf(buf, len, "[");
    for (int i = 0; i < n && pos + 80 < len; i++) {
        char name[70];
        size_t k = 0;
        for (const char *c = (const char *)recs[i].ssid; *c && k < sizeof(name) - 2; c++) {
            if (*c == '"' || *c == '\\') name[k++] = '\\';   // escape for JSON
            name[k++] = *c;
        }
        name[k] = 0;
        if (!name[0]) continue;
        pos += snprintf(buf + pos, len - pos, "%s{\"ssid\":\"%s\",\"rssi\":%d}", pos > 1 ? "," : "", name, recs[i].rssi);
    }
    snprintf(buf + pos, len - pos, "]");
    return ESP_OK;
}
