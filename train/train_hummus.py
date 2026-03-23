from ultralytics import YOLO
import os
import dotenv
from .class_base_train import Train
from src.analityc.core.hardware_available import device_hardware
dotenv.load_dotenv()


api_key = os.getenv('api_keyroboflow')
workspace_name_roboflow = os.getenv('workspace_name_roboflow')
project_name_roboflow = 'hummus'
version = 6



# Intenta detectar la 1080 Ti, si falla usa la primera disponible
try:
    # Si sabes que la 1080 Ti es la segunda en la lista:
    selected_gpu = device_hardware.gpu_tuple[1]['gpu_use']
except (IndexError, KeyError):
    # Fallback a la primera GPU o directamente al ID 1 de Torch
    selected_gpu = 1 

train = Train(
    api_key = api_key,
    workspace_name_roboflow = workspace_name_roboflow,
    project_name_roboflow = project_name_roboflow,
    version = version,
    device_train = selected_gpu, # Aquí pasamos el ID 1
    model_path='models/base/yolov8x.pt'
)


if __name__ == '__main__':                                                               
    # Esto es OBLIGATORIO para Windows + multiprocesamiento
    # Importar freeze_support solo si estamos en un entorno congelado (ejecutable)
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except ImportError:
        pass

    train.run_train(epochs=150, patience=40, batch=8)
