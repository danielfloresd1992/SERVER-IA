import os
from typing import Dict, Any

# Configuración global
APP_CONFIG: Dict[str, Any] = {
    "model_path": "models/yolo11m.pt",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.4,
    "device": "auto",
    "websocket_port": 9000,
    "host": "0.0.0.0"
}

# ROI ajustado: esquinas inferiores movidas más a la derecha
DEFAULT_ROI = [
    [500, 250],   # Esquina superior izquierda
    [900, 250],   # Esquina superior derecha
    [1040, 560],  # Esquina inferior derecha (más a la derecha)
    [600, 560]    # Esquina inferior izquierda (más a la derecha)
]



def get_config():
    return APP_CONFIG.copy()