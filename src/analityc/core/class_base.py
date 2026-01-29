
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque

from typing import Tuple, Dict, Any, List

import datetime
import os




class ObjectDetector:
    """
    Clase base para detección de objetos usando YOLO.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', output_dir: str = 'output'):
        """
        Inicializa el detector de objetos.
        
        Args:
            model_path: Ruta al modelo entrenado (.pt file)
            device: Dispositivo a usar ('cpu' o 'cuda')
            output_dir: Directorio para guardar resultados
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener nombres de clases del modelo entrenado
        self.class_names = self.model.names
        print(f"Modelo cargado. Clases disponibles: {self.class_names}")
    
    def detect(self, image):
        """
        Detecta objetos en una imagen.
        
        Args:
            image: Imagen en formato numpy array o ruta a la imagen
        
        Returns:
            Lista de resultados con detecciones
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            
        results = self.model.predict(
            source=image,
            conf=0.3,
            iou=0.4,
            device=self.device,
            verbose=False
        )
        
        return results
    
    def draw_detections(self, image, results, save_path=None):
        """
        Dibuja las detecciones en la imagen.
        
        Args:
            image: Imagen original
            results: Resultados de la detección
            save_path: Ruta para guardar la imagen (opcional)
        
        Returns:
            Imagen con detecciones dibujadas
        """
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Usar el nombre de clase del modelo entrenado
                    label = f'{self.class_names[cls_id]}: {conf:.2f}'
                    
                    # Dibujar rectángulo
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Dibujar etiqueta
                    cv2.putText(image, label, (x1 + 5, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
        
        return image
    
    def process_image(self, image_path, output_path=None):
        """
        Procesa una imagen completa: detección y dibujo.
        
        Args:
            image_path: Ruta a la imagen de entrada
            output_path: Ruta para guardar la imagen procesada (opcional)
        
        Returns:
            Imagen procesada con detecciones
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: No se pudo cargar la imagen {image_path}")
            return None
        
        # Realizar detección
        results = self.detect(image)
        
        # Dibujar detecciones
        processed_image = self.draw_detections(image.copy(), results)
        
        # Guardar si se especifica ruta
        if output_path:
            cv2.imwrite(output_path, processed_image)
        
        return processed_image, results
    
    def process_directory(self, input_dir, output_dir=None):
        """
        Procesa todas las imágenes en un directorio.
        
        Args:
            input_dir: Directorio con imágenes de entrada
            output_dir: Directorio para guardar resultados (opcional)
        """
        input_dir = Path(input_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = self.output_dir / input_dir.name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Listar archivos de imagen
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Procesando {len(image_files)} imágenes de {input_dir}")
        
        for image_file in image_files:
            print(f"Procesando: {image_file.name}")
            
            # Crear ruta de salida
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output_path = output_dir / f"{image_file.stem}_{timestamp}.jpg"
            
            # Procesar imagen
            self.process_image(str(image_file), str(output_path))
        
        print(f"Procesamiento completado. Resultados en: {output_dir}")


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar rutas
    project_root = Path(__file__).resolve().parents[3]
    
    # Rutas de ejemplo - ajusta según tu estructura
    model_path = 'runs/detect/train/weights/best.pt'  # Tu modelo entrenado
    test_dir = project_root / 'debugger' / 'test' / 'data_test'
    output_dir = project_root / 'debugger' / 'test' / 'result'
    
    # Crear detector
    detector = ObjectDetector(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Auto-detect GPU
        output_dir=output_dir
    )
    
    # Procesar directorio completo
    detector.process_directory(test_dir)
    
    # O procesar una imagen individual
    # image_path = test_dir / 'mi_imagen.jpg'
    # processed_image, results = detector.process_image(str(image_path))
    
    # Mostrar resultados
    # cv2.imshow('Resultado', processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()