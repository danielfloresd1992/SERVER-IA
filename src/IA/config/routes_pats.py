import os


BASE_DIR = r"D:\VVR\IA-JARVIS\v4-Inference-main"
VIDEO_NAME = "traffic_signs.mp4"
LOG_FILE = os.path.join(BASE_DIR, "detection_log.txt")
VIDEO_PATH = os.path.join(BASE_DIR, VIDEO_NAME)
CAR_EXIT_DIR = os.path.join(BASE_DIR, "Car Exit")
CONFIG_FILE = os.path.join(BASE_DIR, "roi_config.json")