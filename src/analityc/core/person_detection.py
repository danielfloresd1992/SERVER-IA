from ultralytics import YOLO
from .hardware_available import device_hardware
import cv2
import numpy as np
from pathlib import Path

from ...libs.files_save import save_frame_png


project_root = Path(__file__).resolve().parents[3]  # sube hasta la raíz del proyecto
image_path = project_root / 'debugger' / 'Amazonas_Camera5_Amazonas_20251203152218_1576750.bmp'
frame = cv2.imread(str(image_path))



class PersonDetector:
    
    def __init__(self, device='cpu', model_path: str = 'models/best_model_trained.pt', dir_output: str = 'output/Person_Detection'):
        
        
        self.device = device_hardware.device_default['gpu_use'] if device_hardware.cuda_available else 'cpu'
        
        self.model = YOLO(model_path)
        #self.model.to(self.device)
        self.output_dir = dir_output
        
        
        
    def analyze_image(self, frame, roi_coordinates=None):
        """Analiza un frame y devuelve las detecciones de personas"""
        
   
        results = self.model.predict(
            source=frame,
            conf=0.3,
            iou=0.4,
            device=self.device,
            save=False,
            save_dir=self.output_dir,
            verbose=False,
            show=False,
            dnn=True,    
        )
        
        print(results)
        
      
        for result in results:
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                person_indices = np.where(classes == 0)[0]
                
                for idx in person_indices:
                    bbox = boxes[idx]  # [x1, y1, x2, y2]
                    conf = confidences[idx]  # 0.95
                    cls_id = classes[idx]  # 0
                    
                    print(f"Persona detectada:")
                    print(f"  Bounding Box: {bbox}")
                    print(f"  Confianza: {conf:.2%}")
                    print(f"  Coordenadas: ({bbox[0]:.0f}, {bbox[1]:.0f}) a ({bbox[2]:.0f}, {bbox[3]:.0f})")
                    print(f"  Ancho: {bbox[2]-bbox[0]:.0f}px, Alto: {bbox[3]-bbox[1]:.0f}px")
                    
                    # Dibujar en la imagen
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),  (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    
                    # Añadir etiqueta
                    label = f"Persona: {conf:.1%}"
                    cv2.putText(frame, label, 
                              (int(bbox[0]), int(bbox[1]-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        save_frame_png(frame=frame)
                 
        
        
        
        
personDetector = PersonDetector(
    device = device_hardware.device_default,
    dir_output ='output/Person_Detection'
)


personDetector.analyze_image(frame=frame)