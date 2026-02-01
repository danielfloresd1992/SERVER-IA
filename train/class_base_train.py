from roboflow import Roboflow
from ultralytics import YOLO
import os



class Train:

    def __init__(
            self,
            api_key: str = None,
            workspace_name_roboflow: str = None,
            project_name_roboflow: str = None,
            version: int = None,
            device_train = 'cpu_use',
            model = YOLO
    ):
        self.api_key = api_key
        self.workspace_name_roboflow = workspace_name_roboflow
        self.project_name_roboflow = project_name_roboflow
        self.version = version

        self.path_dataset = f'train/datasets/{self.project_name_roboflow}'
        self.path_yamal = f'{self.path_dataset}/data.yaml'
        self.path_result = f'models'

        self.device = device_train

        self.model = model
        # PREPARATE DATASET
        self._download_dataset()




    def _download_dataset(self):

        if not  os.path.exists(self.path_dataset):
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace_name_roboflow).project(self.project_name_roboflow)
            version = project.version(self.version)
            dataset = version.download('yolov12', location=self.path_dataset)



    def run_train(self, epochs=10, patience=5, batch=5):
        self.model.train(
            data = self.path_yamal, 
            epochs=epochs, 
            patience=patience, 
            batch=batch, 
            device=self.device,
            project=self.path_result,
            name=self.project_name_roboflow,
            exist_ok = True,
            workers = 8,
            imgsz=512,
            pretrained = True,
            amp = True,  # Mixed precision
            cache = False,  # Desactivar si problemas de memoria
            save = True,
            save_period = 10,
            verbose = True,
            # DESACTIVAR DDP EXPLÍCITAMENTE
            single_cls = False,
            rect = False,
            cos_lr = False,
            label_smoothing = 0.0,
            overlap_mask = True,
            mask_ratio = 4,
            dropout = 0.0,
        )
            
        self.model.save()

        