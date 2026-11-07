"""ESPDet-Pico with ultralytics 8.4.189: adjustments (patches) and degradation to camera board quality.

esp-detection must be cloned and pinned to commit 28de5902a357d99dd9a5db0f52905255641c5a78
(the notebook does this). Path via --esp-detection or the environment variable ESP_DETECTION.
"""
import os
import sys

ESP_DETECTION_COMMIT = "28de5902a357d99dd9a5db0f52905255641c5a78"


def patches(esp_detection):
    """The two adjustments needed for ESPDet-Pico with this ultralytics version. Returns the model yaml."""
    if esp_detection not in sys.path:
        sys.path.insert(0, esp_detection)
    import ultralytics.nn.tasks as tasks
    from nn.esp_tasks import custom_parse_model
    tasks.parse_model = custom_parse_model
    from nn.modules import DSConv
    if not hasattr(DSConv, "conv"):
        DSConv.conv = property(lambda s: s.depthwise)
    return os.path.join(esp_detection, "cfg", "models", "espdet_pico.yaml")


def degradation(strength=1.0):
    """Random degradation to the quality of a small camera board (OV2640/OV3660 class) - new for every image in
    every epoch. Only pixel changes (no shift), colour only slightly (colour is a feature!)."""
    import albumentations as A
    s = strength
    return [
        A.OneOf([A.MotionBlur(blur_limit=(3, 7)), A.GaussianBlur(blur_limit=(3, 5)), A.Defocus(radius=(1, 2))], p=0.5 * s),
        A.Downscale(scale_range=(0.55, 0.95), p=0.3 * s),
        A.RandomBrightnessContrast(brightness_limit=0.25 * s, contrast_limit=0.25 * s, p=0.7),
        A.RandomGamma(gamma_limit=(80, 125), p=0.4),
        A.RGBShift(r_shift_limit=int(10 * s), g_shift_limit=int(8 * s), b_shift_limit=int(10 * s), p=0.4),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=int(20 * s), val_shift_limit=0, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.06 * s), p=0.6),
        A.ISONoise(p=0.25 * s),
        A.ImageCompression(quality_range=(30, 85), p=0.8),
    ]
