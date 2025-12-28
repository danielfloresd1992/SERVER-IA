import os
from ultralytics import YOLO
from .hardware_available import device_hardware
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from ...libs.files_save import save_frame_png
from datetime import datetime


project_root = Path(__file__).resolve().parents[3]  # sube hasta la raíz del proyecto



class PersonDetector:
    
    def __init__(self, device='cpu', model_path: str = 'models/base/yolo12s.pt', dir_output: str = 'output/Person_Detection'):
    
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)
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
                    
        processed_image = frame
        return processed_image
        
   


path_files_test = project_root / 'debugger' / 'test' / 'data_test'
patf_files_result = project_root / 'debugger' / 'test' / 'result'

list_files_test = os.listdir(path_files_test)



personDetector = PersonDetector(
    device = device_hardware.device_default['gpu_use'],
    model_path ='runs/detect/train/weights/best.pt',
    dir_output = patf_files_result
)



for file in list_files_test:
    if file == 'Thumbs.db': continue
    path_file = os.path.join(path_files_test, file)
    image = cv2.imread(str(path_file))
   
    results = personDetector.model(image)
    
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1,y1, x2, y2 = map(int, box.xyxy[0])

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f'{personDetector.model.names[cls_id]}'
            coords = box.xyxy[0].tolist()

            cv2.rectangle(image, (x1,y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label,(x1 + 5, y1 - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),2 )

    cv2.imwrite(f'/debugger/ouput/test/model_person_amazonas/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg', image)

       
       

    #save_frame_png(result, patf_files_result)
    




