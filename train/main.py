from roboflow import Roboflow
from ultralytics import YOLO
import os
import dotenv
from src.analityc.core.hardware_available import device_hardware
dotenv.load_dotenv()


api_key = os.getenv('api_keyroboflow')
workspace_name_roboflow = os.getenv('workspace_name_roboflow')
project_name_roboflow = os.getenv('project_name_roboflow')

path_result = f'train/datasets/{project_name_roboflow}'
path_yamal = f'{path_result}/data.yaml'


if not  os.path.exists(path_result):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name_roboflow).project(project_name_roboflow)
    version = project.version(3)
    dataset = version.download('yolov12', location=f'train/datasets/{project_name_roboflow}')




if __name__ == '__main__':
    # Esto es OBLIGATORIO para Windows + multiprocesamiento
    # Importar freeze_support solo si estamos en un entorno congelado (ejecutable)
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except ImportError:
        pass
    
    # Ejecutar función principal
    model_for_training = YOLO('models/base/yolo12s.pt')

    model_for_training.train(data = path_yamal, epochs=100, patience=40, batch=20)
    model_for_training.save()