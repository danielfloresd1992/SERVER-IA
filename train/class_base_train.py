from roboflow import Roboflow
from ultralytics import YOLO
import os


class Train:
    """
    Clase optimizada para entrenar YOLOv12m con Roboflow.
    """

    def __init__(
        self,
        api_key: str = None,
        workspace_name_roboflow: str = None,
        project_name_roboflow: str = None,
        version: int = None,
        device_train: str = '0',
        model_path: str = 'yolov8x.pt'      # 🔥 ¡NUEVO! Ruta del modelo pre-entrenado
    ):
        self.api_key = api_key
        self.workspace_name_roboflow = workspace_name_roboflow
        self.project_name_roboflow = project_name_roboflow
        self.version = version

        self.path_dataset = f'train/datasets/{self.project_name_roboflow}'
        self.path_yamal = f'{self.path_dataset}/data.yaml'
        self.path_result = 'models'

        self.device = device_train
        self.model = YOLO(model_path)        # 🔥 AHORA instancia el modelo con pesos

        self._download_dataset()

    def _download_dataset(self):
        """Descarga el dataset desde Roboflow en formato YOLOv12"""
        if not os.path.exists(self.path_dataset):
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace_name_roboflow).project(self.project_name_roboflow)
            version = project.version(self.version)
            version.download('yolov12', location=self.path_dataset)
            print(f"✅ Dataset descargado en: {self.path_dataset}")
        else:
            print(f"📁 Dataset ya existe: {self.path_dataset}")

    def run_train(self, epochs=300, patience=30, batch=16, imgsz=640):
        """
        Entrena el modelo con hiperparámetros optimizados para YOLOv12m.
        """
        self.model.train(
            data=self.path_yamal,
            epochs=epochs,
            patience=patience,
            batch=batch,
            device=self.device,
            project=self.path_result,
            name=self.project_name_roboflow,
            exist_ok=True,
            workers=8,
            imgsz=imgsz,
            pretrained=True,      # Ya está pre-entrenado por la instancia
            amp=True,
            cache=False,
            save=True,
            save_period=50,
            verbose=True,

            # ============ HIPERPARÁMETROS OPTIMIZADOS ============
            lr0=0.0005,
            lrf=0.01,
            cos_lr=True,
            optimizer='AdamW',
            weight_decay=0.0005,
            dropout=0.15,
            label_smoothing=0.05,

            # ============ DATA AUGMENTATION ============
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0005,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,

            # ============ OTROS ============
            single_cls=False,
            rect=False,
            overlap_mask=True,
            mask_ratio=4,
        )

        # ❌ NO necesitas self.model.save(), el mejor modelo ya se guardó
        print(f"✅ Entrenamiento completado. Modelo guardado en: {self.path_result}/{self.project_name_roboflow}/weights/best.pt")