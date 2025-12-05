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
    version = project.version(1)
    dataset = version.download('yolov12', location=f'train/datasets/{project_name_roboflow}')


model_for_training = YOLO('models/base/yolo12x.pt')
print(path_yamal)
model_for_training.train(data = path_yamal, epochs=150, imgsz=640)
