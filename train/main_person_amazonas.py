from ultralytics import YOLO
import os
import dotenv
from .class_base_train import Train
from src.analityc.core.hardware_available import device_hardware
dotenv.load_dotenv()
import torch



api_key = os.getenv('api_keyroboflow')
workspace_name_roboflow = os.getenv('workspace_name_roboflow')
project_name_roboflow = 'amazonas365_completo'
version = 8


print(device_hardware.gpu_tuple)

train = Train(
    api_key = api_key,
    workspace_name_roboflow = workspace_name_roboflow,
    project_name_roboflow = project_name_roboflow,
    version = version ,
    device_train = device_hardware.gpu_tuple[1]['gpu_use'],
    model = YOLO('models/base/YOLO12m.pt')
)



if __name__ == '__main__':
    # Esto es OBLIGATORIO para Windows + multiprocesamiento
    # Importar freeze_support solo si estamos en un entorno congelado (ejecutable)
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except ImportError:
        pass

    train.run_train(epochs=150, patience=20, batch=16)




   