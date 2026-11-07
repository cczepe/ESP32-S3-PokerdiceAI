# ---------------------------------------------------------------- models
# The models live in firmware/models/:  dice_det.espdl, dice_cls.espdl, model_config.h
# (written there by trainer/convert/select_project.py). Without them the firmware still builds and works as a
# camera for the trainer; the web page then shows "No models".
get_filename_component(models "${CMAKE_CURRENT_LIST_DIR}/../models" ABSOLUTE)
# folder as a dependency: when files are added or replaced, the next build reconfigures
if(EXISTS ${models})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${models})
endif()
if(EXISTS ${models}/model_config.h)
    message(STATUS "Pokerdice: models from ${models}")
else()
    message(WARNING "Pokerdice: ${models}/model_config.h missing - the firmware runs as a camera only. "
                    "Choose a project with trainer/convert/select_project.py first.")
    set(models "")
endif()

idf_component_register(
    SRCS main.c state.c settings.c camera.c wifi.c dns_server.c web.c led.c button.c detection.cpp
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_rmt esp_timer esp_psram esp_partition nvs_flash
             esp_wifi esp_netif esp_event esp_http_server lwip
    EMBED_FILES web/index.html
)

if(models)
    target_include_directories(${COMPONENT_LIB} PRIVATE ${models})
    # write the models into the partitions 'det' and 'cls' when flashing (idf.py flash)
    foreach(part det cls)
        if(EXISTS ${models}/dice_${part}.espdl)
            esptool_py_flash_to_partition(flash "${part}" "${models}/dice_${part}.espdl")
        else()
            message(WARNING "Pokerdice: ${models}/dice_${part}.espdl missing - not flashed.")
        endif()
    endforeach()
endif()

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-missing-field-initializers)
