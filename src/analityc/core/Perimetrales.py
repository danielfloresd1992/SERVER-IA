import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from typing import Tuple, Dict, Any, List
import asyncio
import threading
import base64
import httpx
import datetime
import os
from ..config.config import DEFAULT_ROI

logger = logging.getLogger(__name__)




class MultiObjectProcessor:
    """Procesador de múltiples objetos con reconocimiento de colores EXACTO"""
    
    def __init__(self, 
                client_id: None = None,
                model_path: str = "yolo11x.pt",
                confidence_threshold: float = 0.6,
                iou_threshold: float = 0.4,
                device: str = 'cpu',
                log_file: str = "output/detection_log.txt",
                car_exit_dir: str = "output/Car_Exit",
                image_quality: int = 50,
                min_time_in_roi: int = 10,
                max_frames_out: int = 5,
                min_track_frames: int = 3,
                show_minimal_info: bool = True,
                exit_frames_threshold: int = 1,
                max_frames_without_detection: int = 5,
                max_image_size: tuple = (640, 480)):
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.log_file = log_file
        self.car_exit_dir = car_exit_dir
        self.image_quality = image_quality
        self.show_minimal_info = show_minimal_info
        self.exit_frames_threshold = exit_frames_threshold
        self.max_image_size = max_image_size
        
        # Configuración de tiempos
        self.min_time_in_roi = min_time_in_roi
        self.max_frames_out = max_frames_out
        self.min_track_frames = min_track_frames
        self.max_frames_without_detection = max_frames_without_detection
        
        # Estado interno
        self.frame_counter = 0
        self.vehiculos_en_area = 0
        self.personas_en_area = 0
        self.active_tracks = {}
        self.next_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
        # Contadores por tipo de vehículo
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        
        # Contadores de personas dentro
        self.person_count_inside = 0
        
        # ROI por defecto
        self.roi_polygon = np.array(DEFAULT_ROI if DEFAULT_ROI else [(0, 0), (640, 0), (640, 480), (0, 480)], np.int32)
        
        # Para evitar reconteo
        self.counted_tracks = set()
        self.recent_counted_vehicles = deque(maxlen=30)
        self.vehicle_cooldown = defaultdict(int)
        
        # Para personas
        self.counted_persons = set()
        self.recent_counted_persons = deque(maxlen=30)
        self.person_cooldown = defaultdict(int)
        
        # Estados de seguimiento
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.debug_mode = True
        
        # Historial de posiciones para validar movimiento
        self.movement_history = defaultdict(lambda: deque(maxlen=10))
        
        # Clases a detectar - SOLO PERSONAS Y VEHÍCULOS, SIN ANIMALES
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.person_class = [0]  # person
        self.all_classes = self.vehicle_classes + self.person_class
        

        # Diccionario de nombres en español
        self.class_names_es = {
            'person': 'Persona',
            'car': 'Carro',
            'truck': 'Camion',
            'motorcycle': 'Motocicleta',
            'bus': 'Camioneta'
        }
        

        # Diccionario de colores en español (COMPLETO)
        self.color_names_es = {
            'red': 'rojo',
            'rojo': 'rojo',
            'crimson': 'carmesí',
            'scarlet': 'escarlata',
            'ruby': 'rubí',
            'cherry': 'cereza',
            'burgundy': 'burdeos',
            'maroon': 'granate',
            'blue': 'azul',
            'azul': 'azul',
            'navy': 'azul marino',
            'royal_blue': 'azul real',
            'sky_blue': 'azul cielo',
            'turquoise': 'turquesa',
            'teal': 'verde azulado',
            'cyan': 'cian',
            'green': 'verde',
            'verde': 'verde',
            'lime': 'lima',
            'olive': 'oliva',
            'emerald': 'esmeralda',
            'forest_green': 'verde bosque',
            'yellow': 'amarillo',
            'amarillo': 'amarillo',
            'gold': 'dorado',
            'amber': 'ámbar',
            'mustard': 'mostaza',
            'orange': 'naranja',
            'naranja': 'naranja',
            'coral': 'coral',
            'peach': 'durazno',
            'tangerine': 'mandarina',
            'purple': 'morado',
            'morado': 'morado',
            'violet': 'violeta',
            'lavender': 'lavanda',
            'lilac': 'lila',
            'plum': 'ciruela',
            'pink': 'rosa',
            'rosa': 'rosa',
            'magenta': 'magenta',
            'hot_pink': 'rosa fuerte',
            'salmon': 'salmón',
            'brown': 'café',
            'cafe': 'café',
            'chocolate': 'chocolate',
            'caramel': 'caramelo',
            'tan': 'marrón claro',
            'beige': 'beige',
            'black': 'negro',
            'negro': 'negro',
            'charcoal': 'carbón',
            'onyx': 'ónix',
            'white': 'blanco',
            'blanco': 'blanco',
            'ivory': 'marfil',
            'cream': 'crema',
            'gray': 'gris',
            'gris': 'gris',
            'silver': 'plateado',
            'ash_gray': 'gris ceniza',
            'slate_gray': 'gris pizarra',
            'silver_metallic': 'plateado metálico',
            'gold_metallic': 'dorado metálico',
            'bronze_metallic': 'bronce metálico',
            'unknown': 'desconocido',
            'desconocido': 'desconocido'
        }
        


        # PALETA DE COLORES EXACTA CON VALORES HSV ESPECÍFICOS
        self.color_ranges_hsv = {
            # ROJOS
            'red': [(0, 150, 50), (10, 255, 255), (170, 150, 50), (180, 255, 255)],
            'crimson': [(0, 180, 60), (5, 255, 200), (175, 180, 60), (180, 255, 200)],
            'scarlet': [(0, 200, 150), (8, 255, 255)],
            'ruby': [(0, 180, 100), (10, 255, 200)],
            'cherry': [(170, 150, 80), (180, 255, 200)],
            'burgundy': [(150, 100, 40), (180, 255, 120)],
            'maroon': [(0, 100, 30), (10, 200, 100), (170, 100, 30), (180, 200, 100)],
            
            # AZULES
            'blue': [(90, 100, 50), (130, 255, 255)],
            'navy': [(100, 150, 30), (140, 255, 120)],
            'royal_blue': [(100, 150, 100), (130, 255, 200)],
            'sky_blue': [(90, 50, 150), (120, 150, 255)],
            'turquoise': [(75, 100, 100), (95, 255, 255)],
            'teal': [(85, 100, 50), (105, 255, 200)],
            'cyan': [(85, 100, 150), (95, 255, 255)],
            
            # VERDES
            'green': [(40, 100, 50), (80, 255, 255)],
            'lime': [(50, 150, 150), (70, 255, 255)],
            'olive': [(30, 100, 50), (50, 200, 150)],
            'emerald': [(70, 150, 100), (85, 255, 200)],
            'forest_green': [(50, 100, 40), (70, 255, 150)],
            
            # AMARILLOS
            'yellow': [(20, 100, 150), (30, 255, 255)],
            'gold': [(25, 100, 150), (35, 255, 220)],
            'amber': [(25, 150, 150), (35, 255, 255)],
            'mustard': [(35, 100, 100), (45, 200, 200)],
            
            # NARANJAS
            'orange': [(10, 150, 150), (20, 255, 255)],
            'coral': [(5, 100, 150), (15, 255, 255)],
            'peach': [(15, 50, 180), (25, 150, 255)],
            'tangerine': [(10, 180, 180), (20, 255, 255)],
            
            # MORADOS
            'purple': [(130, 100, 80), (150, 255, 200)],
            'violet': [(120, 100, 100), (140, 255, 200)],
            'lavender': [(130, 50, 150), (150, 150, 255)],
            'lilac': [(140, 50, 150), (160, 150, 255)],
            'plum': [(145, 100, 80), (160, 255, 180)],
            
            # ROSAS
            'pink': [(150, 50, 150), (170, 200, 255)],
            'magenta': [(145, 150, 150), (155, 255, 255)],
            'hot_pink': [(160, 150, 150), (170, 255, 255)],
            'salmon': [(5, 50, 180), (15, 150, 255)],
            
            # CAFÉS
            'brown': [(10, 100, 30), (20, 200, 150)],
            'chocolate': [(10, 100, 30), (20, 200, 120)],
            'caramel': [(20, 100, 100), (30, 200, 200)],
            'tan': [(20, 50, 150), (30, 150, 220)],
            'beige': [(20, 30, 180), (30, 80, 240)],
            
            # NEGROS
            'black': [(0, 0, 0), (180, 255, 40)],
            'charcoal': [(0, 0, 30), (180, 50, 80)],
            'onyx': [(0, 0, 10), (180, 50, 30)],
            
            # BLANCOS
            'white': [(0, 0, 220), (180, 30, 255)],
            'ivory': [(20, 10, 220), (40, 50, 255)],
            'cream': [(25, 20, 220), (35, 80, 255)],
            
            # GRISES
            'gray': [(0, 0, 80), (180, 30, 180)],
            'ash_gray': [(0, 0, 100), (180, 30, 150)],
            'slate_gray': [(0, 0, 60), (180, 30, 120)],
            
            # METÁLICOS
            'silver': [(0, 0, 150), (180, 30, 220)],
            'silver_metallic': [(0, 0, 180), (180, 50, 220)],
            'gold_metallic': [(25, 50, 150), (35, 150, 220)],
            'bronze_metallic': [(15, 50, 100), (25, 150, 180)]
        }
        
        # Mapeo de colores principales para agrupación
        self.color_groups = {
            'rojo': ['red', 'crimson', 'scarlet', 'ruby', 'cherry', 'burgundy', 'maroon'],
            'azul': ['blue', 'navy', 'royal_blue', 'sky_blue', 'turquoise', 'teal', 'cyan'],
            'verde': ['green', 'lime', 'olive', 'emerald', 'forest_green'],
            'amarillo': ['yellow', 'gold', 'amber', 'mustard'],
            'naranja': ['orange', 'coral', 'peach', 'tangerine'],
            'morado': ['purple', 'violet', 'lavender', 'lilac', 'plum'],
            'rosa': ['pink', 'magenta', 'hot_pink', 'salmon'],
            'cafe': ['brown', 'chocolate', 'caramel', 'tan', 'beige'],
            'negro': ['black', 'charcoal', 'onyx'],
            'blanco': ['white', 'ivory', 'cream'],
            'gris': ['gray', 'ash_gray', 'slate_gray'],
            'plateado': ['silver', 'silver_metallic'],
            'dorado': ['gold_metallic'],
            'bronce': ['bronze_metallic']
        }
        
        # Para controlar que no se envíen múltiples fotos del mismo evento
        self.sent_entry_photos = defaultdict(lambda: deque(maxlen=2))
        self.sent_exit_photos = defaultdict(lambda: deque(maxlen=2))
        
        # Control de frecuencia de envío para evitar sobrecarga
        self.last_sent_time = defaultdict(float)
        self.send_cooldown = 1.0
        
        # Para controlar alertas periódicas
        self.alert_minutes_sent = defaultdict(list)  # {track_id: [minutos_enviados]}
        
        self.model = None
        self.device = device
        self._initialize_model()
        
        os.makedirs(self.car_exit_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        self._log_buffer = []
        self.setup_log_file()
        


        print(f'Modelo inicializado para {client_id}')
        print(f'Analisis procesado desde: {self.device}')
        print(f'🔔 Modo: Alertas a los 1, 3, 6, 9... minutos')
        print(f'🚫 Animales ignorados')
        print(f'🎯 Confianza aumentada: {confidence_threshold}')
        print(f'⏱️  Alertas: 1 minuto, luego cada 3 minutos')
        print(f'🌈 Reconocimiento de colores EXACTO activado')
        print(f'🎨 Paleta de colores: {len(self.color_ranges_hsv)} colores específicos')




    def _initialize_model(self):
        try:
            print(f"🚀 Inicializando modelo YOLO en {self.device}...")
            self.model = YOLO(self.model_path).to(self.device)
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=self.all_classes, verbose=False
            )
            print(f"✅ Modelo YOLO inicializado correctamente en {self.device}")
        except Exception as e:
            print(f"❌ Error inicializando YOLO: {e}")
            raise




    def setup_log_file(self):
        try:
            with open(self.log_file, 'w', encoding="utf-8") as f:
                f.write("Timestamp,Frame,Vehiculos_Area,Cars,Trucks,Motorcycles,Personas_Dentro,Personas_Area,Evento,Color_Exacto,Color_General,Tiempo_Acumulado\n")
        except Exception as e:
            print(f"❌ Error creando archivo de log: {e}")



    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x11, y11, x21, y21 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x11, x1_2)
        yi1 = max(y11, y1_2)
        xi2 = min(x21, x2_2)
        yi2 = min(y21, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def center_of(self, box: Tuple) -> Tuple[float, float]:
        """Calcula el centro de un bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def is_inside_polygon(self, point: Tuple, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def compress_image(self, image: np.ndarray) -> np.ndarray:
        """Comprime la imagen para reducir el tamaño del payload"""
        try:
            height, width = image.shape[:2]
            max_width, max_height = self.max_image_size
            
            if width > max_width or height > max_height:
                aspect_ratio = width / height
                if width > max_width:
                    width = max_width
                    height = int(width / aspect_ratio)
                if height > max_height:
                    height = max_height
                    width = int(height * aspect_ratio)
                
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                if self.debug_mode:
                    print(f"📐 Imagen redimensionada a {width}x{height}")
            
            return image
        except Exception as e:
            logger.error(f"Error al comprimir imagen: {e}")
            return image



    def detect_exact_color(self, image: np.ndarray, box: Tuple, object_type: str) -> Tuple[str, str]:
        """
        Detecta el color EXACTO y su categoría general.
        Retorna: (color_exacto, color_general)
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Asegurar que las coordenadas estén dentro de la imagen
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            if x2 <= x1 or y2 <= y1:
                return "desconocido", "desconocido"
            
            # Recortar la región de interés
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return "desconocido", "desconocido"
            
            # Para personas, analizar solo la parte superior (camisa)
            if object_type == 'person':
                # Tomar el 60% superior del bounding box
                person_height = y2 - y1
                shirt_height = int(person_height * 0.6)
                if shirt_height > 0:
                    roi = roi[0:shirt_height, :]
            
            if roi.size == 0:
                return "desconocido", "desconocido"
            
            # Convertir a espacio de color HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Aplicar blur para reducir ruido
            hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
            
            # Calcular histograma de colores
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalizar histogramas
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Encontrar el tono dominante
            dominant_hue = np.argmax(h_hist)
            avg_saturation = np.mean(hsv[:,:,1])
            avg_value = np.mean(hsv[:,:,2])
            
            # Detección especial para colores extremos
            if avg_value < 40 and avg_saturation < 50:
                return "black", "negro"
            
            if avg_value > 200 and avg_saturation < 30:
                return "white", "blanco"
            
            if avg_saturation < 30 and 50 < avg_value < 200:
                return "gray", "gris"
            
            # Buscar el color exacto que mejor coincida
            best_color = "desconocido"
            best_score = 0
            
            for color_name, ranges in self.color_ranges_hsv.items():
                total_pixels = 0
                matched_pixels = 0
                
                if len(ranges) == 2:
                    lower = np.array(ranges[0])
                    upper = np.array(ranges[1])
                    mask = cv2.inRange(hsv, lower, upper)
                    matched_pixels = cv2.countNonZero(mask)
                    total_pixels = roi.shape[0] * roi.shape[1]
                    
                elif len(ranges) == 4:
                    lower1 = np.array(ranges[0])
                    upper1 = np.array(ranges[1])
                    lower2 = np.array(ranges[2])
                    upper2 = np.array(ranges[3])
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                    matched_pixels = cv2.countNonZero(mask)
                    total_pixels = roi.shape[0] * roi.shape[1]
                
                if total_pixels > 0:
                    match_percentage = matched_pixels / total_pixels
                    # Ajustar score considerando saturación y valor
                    h_match = 1.0 - (abs(dominant_hue - np.mean([ranges[0][0], ranges[-1][0] if len(ranges)==2 else ranges[2][0]])) / 180)
                    final_score = match_percentage * 0.7 + h_match * 0.3
                    
                    if final_score > best_score and final_score > 0.2:
                        best_score = final_score
                        best_color = color_name
            
            # Si no se encuentra color específico, intentar con grupos generales
            if best_color == "desconocido":
                # Determinar color general basado en el tono dominante
                if 0 <= dominant_hue <= 15 or 165 <= dominant_hue <= 180:
                    return "red", "rojo"
                elif 16 <= dominant_hue <= 35:
                    return "orange", "naranja"
                elif 36 <= dominant_hue <= 70:
                    return "yellow", "amarillo"
                elif 71 <= dominant_hue <= 85:
                    return "lime", "verde"
                elif 86 <= dominant_hue <= 100:
                    return "green", "verde"
                elif 101 <= dominant_hue <= 130:
                    return "blue", "azul"
                elif 131 <= dominant_hue <= 150:
                    return "purple", "morado"
                elif 151 <= dominant_hue <= 164:
                    return "pink", "rosa"
            
            # Encontrar la categoría general del color exacto
            general_color = "desconocido"
            for group_name, color_list in self.color_groups.items():
                if best_color in color_list:
                    general_color = group_name
                    break
            
            return best_color, general_color
                
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Error detectando color exacto: {e}")
            return "desconocido", "desconocido"



    def get_color_name_es(self, color_en: str) -> str:
        """Traduce el nombre del color de inglés a español"""
        return self.color_names_es.get(color_en.lower(), color_en)



    def get_color_message(self, object_type: str, exact_color: str, general_color: str) -> str:
        """Genera el mensaje de color según el tipo de objeto"""
        if exact_color == "desconocido" or general_color == "desconocido":
            return ""
        


        exact_color_es = self.get_color_name_es(exact_color)
        general_color_es = self.get_color_name_es(general_color)
        
        # Para colores metálicos especiales
        if 'metallic' in exact_color:
            if object_type == 'person':
                return f"con camisa {exact_color_es}"
            else:
                return f"{exact_color_es}"
        
        # Para colores exactos diferentes del general
        if exact_color_es != general_color_es:
            if object_type == 'person':
                return f"con camisa {exact_color_es} ({general_color_es})"
            else:
                return f"{exact_color_es} ({general_color_es})"
        else:
            if object_type == 'person':
                return f"con camisa {exact_color_es}"
            else:
                return f"{exact_color_es}"




    async def send_jarvis(self, base64_img: str, text: str):
        """Envía una imagen a un servidor de manera asíncrona"""
        payload = {
            "my-text": text, 
            "my-file": base64_img, 
            "type": "image/jpg",
            "compressed": "true",
            "quality": str(self.image_quality)
        }
        
        payload_size = len(base64_img) / 1024
        if self.debug_mode:
            print(f"📦 Tamaño del payload: {payload_size:.2f} KB")
        
        if payload_size > 500:
            logger.warning(f"Payload demasiado grande ({payload_size:.2f} KB), enviando solo texto")
            payload = {
                "my-text": f"{text} (Imagen muy grande: {payload_size:.2f} KB)",
                "type": "text"
            }
        
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            try: 
                respuesta = await client.post(
                    "https://72.68.60.254:4000/bot/imgV2/number=120363402589311344@g.us",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                respuesta.raise_for_status()
                logger.info(f"✅ Imagen enviada exitosamente - Estado: {respuesta.status_code}, Tamaño: {payload_size:.2f} KB")
                return respuesta.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 413:
                    logger.warning(f"⚠️ Payload demasiado grande ({payload_size:.2f} KB), reduciendo calidad...")
                logger.error(f"❌ Error HTTP en envío: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"❌ Error en envío: {e}")
                raise



    def send_jarvis_wrapper(self, base64_img: str, text: str, object_id: int):
        """Envía una imagen en un hilo separado con control de frecuencia"""
        current_time = time.time()
        last_time = self.last_sent_time.get(object_id, 0)
        
        if current_time - last_time < self.send_cooldown:
            if self.debug_mode:
                print(f"⏳ Cooldown para objeto {object_id}, esperando...")
            return
        

        def send_async():
            try:
                asyncio.run(self.send_jarvis(base64_img, text))
                self.last_sent_time[object_id] = current_time
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_jarvis(base64_img, text))
                        self.last_sent_time[object_id] = current_time
                    finally:
                        loop.close()
                else:
                    logger.error(f"Envío asíncrono falló: {e}")
            except Exception as e:
                logger.error(f"Envío asíncrono falló: {e}")
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()

    def create_annotated_image(self, frame: np.ndarray, object_type: str, object_id: int, exact_color: str = None) -> np.ndarray:
        """Crea una imagen con el objeto señalado en cuadro AMARILLO"""
        annotated_frame = frame.copy()
        
        if object_id in self.active_tracks:
            track = self.active_tracks[object_id]
            x1, y1, x2, y2 = [int(v) for v in track['box']]
            
            # Color AMARILLO fijo para el recuadro (independiente del color detectado)
            box_color = (0, 255, 255)  # Amarillo en BGR
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Añadir etiqueta con clase, color detectado y tiempo acumulado
            object_name_es = self.class_names_es.get(object_type, object_type.capitalize())
            
            # Mostrar clase y color exacto si está disponible
            if exact_color and exact_color != "desconocido":
                exact_color_es = self.get_color_name_es(exact_color)
                label = f"{object_name_es} - {exact_color_es}"
            else:
                label = f"{object_name_es}"
            
            # Añadir tiempo acumulado si el objeto está dentro del ROI
            if 'entry_time' in track:
                current_time = time.time()
                time_in_roi = int(current_time - track['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                label += f" - {minutes}m {seconds}s"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fondo para el texto
            cv2.rectangle(annotated_frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        (0, 0, 0), 
                        -1)
            
            # Texto
            cv2.putText(annotated_frame, label,
                      (x1, y1 - 5), font, font_scale,
                      (255, 255, 255), thickness)
        
        return annotated_frame



    def get_action_message(self, object_type: str, event: str, exact_color: str = None, 
                          general_color: str = None, time_in_roi: int = 0, total_minutes: int = 0) -> str:
        """Genera el mensaje completo según el tipo de objeto, evento y color"""
        object_name_es = self.class_names_es.get(object_type, object_type.capitalize())
        
        if event == 'entrada':
            if object_name_es in ['Persona', 'Motocicleta']:
                base_message = f"{object_name_es} entró en el Área"
            else:
                base_message = f"{object_name_es} entró al Área"
        elif event == 'salida':
            if total_minutes == 1:
                base_message = f"{object_name_es} duró {total_minutes} minuto en el Área"
            else:
                base_message = f"{object_name_es} duró {total_minutes} minutos en el Área"
        elif event == 'alerta_periodica':
            minutes = time_in_roi // 60
            if minutes == 1:
                base_message = f"{object_name_es} tiene {minutes} minuto en el Área"
            else:
                base_message = f"{object_name_es} tiene {minutes} minutos en el Área"
        else:
            base_message = f"{object_name_es} {event.upper()}"
        
        if exact_color and exact_color != "desconocido":
            color_message = self.get_color_message(object_type, exact_color, general_color)
            if color_message:
                return f"{base_message} {color_message}"
        
        return base_message

    def save_roi_photo(self, frame: np.ndarray, object_type: str, object_id: int, event: str, 
                       exact_color: str = None, general_color: str = None, time_in_roi: int = 0, total_minutes: int = 0):
        """Guarda una foto cuando un objeto entra, sale o se genera una alerta periódica"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            object_name_es = self.class_names_es.get(object_type, object_type.capitalize())
            
            event_dir = os.path.join(self.car_exit_dir, event)
            os.makedirs(event_dir, exist_ok=True)
            
            filename = f"{object_name_es}_{event}_{object_id}_{timestamp}.jpg"
            filepath = os.path.join(event_dir, filename)
            
            annotated_frame = self.create_annotated_image(frame, object_type, object_id, exact_color)
            compressed_frame = self.compress_image(annotated_frame)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
            
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                base64_size_kb = len(imagen_base64) / 1024
                
                if base64_size_kb > 500:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 30]
                    success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
                    if success:
                        imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                        base64_size_kb = len(imagen_base64) / 1024
                
                message = self.get_action_message(object_type, event, exact_color, general_color, time_in_roi, total_minutes)
                self.send_jarvis_wrapper(imagen_base64, message, object_id)
                
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                
                logger.info(f"✅ Foto de {event} guardada: {filename} ({base64_size_kb:.2f} KB)")
                if self.debug_mode:
                    if exact_color and exact_color != "desconocido":
                        exact_color_es = self.get_color_name_es(exact_color)
                        general_color_es = self.get_color_name_es(general_color) if general_color else ""
                        color_info = f" - Color exacto: {exact_color_es}"
                        if general_color_es and general_color_es != exact_color_es:
                            color_info += f" ({general_color_es})"
                        
                        minutes = time_in_roi // 60
                        seconds = time_in_roi % 60
                        time_info = f" - Tiempo: {minutes}m {seconds}s" if time_in_roi > 0 else ""
                        
                        print(f"📸 Foto {event} guardada: {message}{color_info}{time_info} ({base64_size_kb:.2f} KB)")
                
                return True
            return False
        except Exception as e:
            logger.error(f"No se pudo guardar la foto de {event}: {e}")
            return False



    def check_periodic_alerts(self, frame: np.ndarray):
        """Verifica y envía alertas periódicas a los 1, 3, 6, 9... minutos"""
        current_time = time.time()
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            if is_inside:
                # Calcular tiempo dentro del ROI en segundos
                if 'entry_time' not in track:
                    track['entry_time'] = current_time
                    track['last_alert_minute'] = 0
                
                time_in_roi = int(current_time - track['entry_time'])
                current_minute = time_in_roi // 60
                
                # Definir los minutos en los que se debe enviar alerta: 1, 3, 6, 9, 12, 15...
                target_minutes = []
                minute = 1
                while minute <= current_minute:
                    target_minutes.append(minute)
                    minute += 3  # Después de 1, sumar 3 cada vez
                
                # Verificar si hay minutos objetivo que aún no se han enviado
                sent_minutes = self.alert_minutes_sent.get(track_id, [])
                
                for target_minute in target_minutes:
                    if target_minute not in sent_minutes:
                        # Detectar color exacto si no está ya detectado
                        if 'exact_color' not in track or track['exact_color'] == "desconocido":
                            exact_color, general_color = self.detect_exact_color(frame, track['box'], track['class'])
                            track['exact_color'] = exact_color
                            track['general_color'] = general_color
                        
                        # Enviar alerta periódica
                        success = self.save_roi_photo(
                            frame,
                            track['class'],
                            track_id,
                            'alerta_periodica',
                            track.get('exact_color'),
                            track.get('general_color'),
                            time_in_roi,
                            target_minute
                        )
                        
                        if success:
                            # Registrar que hemos enviado alerta para este minuto
                            if track_id not in self.alert_minutes_sent:
                                self.alert_minutes_sent[track_id] = []
                            
                            if target_minute not in self.alert_minutes_sent[track_id]:
                                self.alert_minutes_sent[track_id].append(target_minute)
                            
                            track['last_alert_minute'] = target_minute
                            
                            # Log del evento
                            self._log_periodic_alert(track_id, track['class'], time_in_roi, target_minute,
                                                   track.get('exact_color'), track.get('general_color'))
                            
                            if self.debug_mode:
                                obj_name = self.class_names_es.get(track['class'], track['class'])
                                exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
                                general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
                                color_info = f" ({exact_color_es})"
                                if general_color_es != exact_color_es:
                                    color_info += f" [{general_color_es}]"
                                print(f"⏱️  ALERTA PERIÓDICA: {obj_name}{color_info} tiene {target_minute} minuto{'s' if target_minute > 1 else ''} en el Área")
            else:
                # Si no está dentro, resetear tiempos
                if 'entry_time' in track:
                    del track['entry_time']
                if 'last_alert_minute' in track:
                    del track['last_alert_minute']
                # Remover del registro de alertas si el track ya no está activo
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]



    def _log_periodic_alert(self, track_id: int, obj_type: str, time_in_roi: int, minute: int,
                           exact_color: str = None, general_color: str = None):
        """Registra alerta periódica en el log"""
        ts = datetime.datetime.now().strftime("%Y-%m-d %H:%M:%S")
        obj_name_es = self.class_names_es.get(obj_type, obj_type.capitalize())
        exact_color_es = self.get_color_name_es(exact_color) if exact_color and exact_color != "desconocido" else "desconocido"
        general_color_es = self.get_color_name_es(general_color) if general_color and general_color != "desconocido" else "desconocido"
        minutes = time_in_roi // 60
        seconds = time_in_roi % 60
        
        log_entry = f"{ts},{self.frame_counter},{self.vehiculos_en_area},{self.car_count},{self.truck_count},{self.motorcycle_count},{self.person_count_inside},{self.personas_en_area},ALERTA_{minute}min_{obj_name_es},{exact_color_es},{general_color_es}"
        log_entry += f",{minutes}:{seconds}"
        
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(f"\n{log_entry}")



    def log_detection(self, frame_count: int, flush: bool = False):
        ts = datetime.datetime.now().strftime("%Y-%m-d %H:%M:%S")
        log_entry = f"{ts},{frame_count},{self.vehiculos_en_area},{self.car_count},{self.truck_count},{self.motorcycle_count},{self.person_count_inside},{self.personas_en_area}"
        self._log_buffer.append(log_entry)
        if flush or len(self._log_buffer) >= 60:
            try:
                with open(self.log_file, 'a', encoding="utf-8") as f:
                    f.write("\n".join(self._log_buffer) + "\n")
                self._log_buffer.clear()
            except Exception as e:
                print(f"❌ Error escribiendo en log: {e}")



    def is_near_recent_counted(self, center: Tuple, object_type: str, threshold: int = 50) -> bool:
        if object_type == 'vehicle':
            recent_list = self.recent_counted_vehicles
        else:
            recent_list = self.recent_counted_persons
            
        for counted_id, counted_center, counted_frame in recent_list:
            distance = np.sqrt((center[0] - counted_center[0])**2 + (center[1] - counted_center[1])**2)
            if distance < threshold and (self.frame_counter - counted_frame) < 30:
                return True
        return False
    


    def validate_movement(self, track_id: int, current_pos: Tuple) -> bool:
        if track_id not in self.movement_history:
            self.movement_history[track_id].append(current_pos)
            return True
        
        positions = list(self.movement_history[track_id])
        if len(positions) < 3:
            self.movement_history[track_id].append(current_pos)
            return True
        
        first_pos = positions[0]
        distance = np.sqrt((current_pos[0] - first_pos[0])**2 + (current_pos[1] - first_pos[1])**2)
        
        self.movement_history[track_id].append(current_pos)
        return distance > 10

    def cleanup_undetected_tracks(self, current_detections: list):
        """Elimina tracks que no están siendo detectados en el frame actual"""
        if not current_detections:
            if self.active_tracks and self.debug_mode:
                print(f"⚠️ No hay detecciones, eliminando todos los tracks ({len(self.active_tracks)} tracks)")
            self.active_tracks.clear()
            return
        
        track_ids = list(self.active_tracks.keys())
        unmatched_tracks = []
        
        for track_id in track_ids:
            track = self.active_tracks[track_id]
            track_box = track['box']
            
            has_match = False
            for det in current_detections:
                iou = self.calculate_iou(track_box, det['box'])
                if iou > 0.3:
                    has_match = True
                    break
            
            if not has_match:
                track['frames_without_detection'] = track.get('frames_without_detection', 0) + 1
                
                if track['frames_without_detection'] >= self.max_frames_without_detection:
                    unmatched_tracks.append(track_id)
            else:
                track['frames_without_detection'] = 0
        
        for track_id in unmatched_tracks:
            if self.debug_mode:
                track_info = self.active_tracks[track_id]
                print(f"🗑️ Track {track_id} ({track_info['class']}) eliminado - {self.max_frames_without_detection} frames sin detección")
            self._remove_track(track_id)




    def process_entry_exit_logic(self, frame: np.ndarray):
        """Procesa la lógica de entrada y salida de objetos"""
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        counted_in_frame = []
        person_counted_in_frame = []
        tracks_to_remove = []
        
        self.person_count_inside = 0
        vehicles_inside = 0
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            # Procesar PERSONAS
            if track['class'] == 'person':
                previous_inside = track.get('is_inside', False)
                track['is_inside'] = is_inside
                
                if is_inside and not previous_inside:
                    # Reiniciar flags de alertas cuando entra
                    track['entry_time'] = time.time()
                    track['last_alert_minute'] = 0
                    if track_id in self.alert_minutes_sent:
                        del self.alert_minutes_sent[track_id]
                    
                    # Detectar color exacto al entrar
                    if 'exact_color' not in track or track['exact_color'] == "desconocido":
                        exact_color, general_color = self.detect_exact_color(frame, track['box'], 'person')
                        track['exact_color'] = exact_color
                        track['general_color'] = general_color
                        if self.debug_mode and exact_color != "desconocido":
                            exact_color_es = self.get_color_name_es(exact_color)
                            general_color_es = self.get_color_name_es(general_color) if general_color else ""
                            color_info = f" ({exact_color_es})"
                            if general_color_es and general_color_es != exact_color_es:
                                color_info += f" [{general_color_es}]"
                            print(f"🌈 Persona detectada: color exacto{color_info}")
                    
                    if hasattr(self, 'last_processed_frame'):
                        self.save_roi_photo(
                            self.last_processed_frame, 
                            'person', 
                            track_id, 
                            'entrada',
                            track.get('exact_color'),
                            track.get('general_color')
                        )
                    
                    if self.debug_mode:
                        exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
                        general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
                        color_info = f" ({exact_color_es})"
                        if general_color_es != exact_color_es:
                            color_info += f" [{general_color_es}]"
                        print(f"👤 Persona{color_info} entró en el Área")
                
                elif not is_inside and previous_inside:
                    # Calcular tiempo total en el área
                    total_minutes = 0
                    if 'entry_time' in track:
                        total_time = int(time.time() - track['entry_time'])
                        total_minutes = total_time // 60
                    
                    # Limpiar flags de alertas cuando sale
                    if 'entry_time' in track:
                        del track['entry_time']
                    if 'last_alert_minute' in track:
                        del track['last_alert_minute']
                    if track_id in self.alert_minutes_sent:
                        del self.alert_minutes_sent[track_id]
                    
                    if hasattr(self, 'last_processed_frame'):
                        self.save_roi_photo(
                            self.last_processed_frame, 
                            'person', 
                            track_id, 
                            'salida',
                            track.get('exact_color'),
                            track.get('general_color'),
                            total_time_seconds=0,
                            total_minutes=total_minutes
                        )
                    
                    if self.debug_mode:
                        exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
                        general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
                        color_info = f" ({exact_color_es})"
                        if general_color_es != exact_color_es:
                            color_info += f" [{general_color_es}]"
                        print(f"👤 Persona{color_info} salió del Área - Duró {total_minutes} minutos")
                
                if is_inside:
                    track['has_been_inside'] = True
                    track['frames_out_roi'] = 0
                    track['frames_in_roi'] = track.get('frames_in_roi', 0) + 1
                    self.person_count_inside += 1
                else:
                    track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                    track['frames_in_roi'] = 0
                    
                    frames_in_roi = track.get('total_frames_in_roi', 0)
                    frames_out_roi = track.get('frames_out_roi', 0)
                    
                    if (frames_in_roi >= self.min_time_in_roi and 
                        frames_out_roi >= 2 and
                        track['seen_frames'] >= self.min_track_frames and
                        not track.get('counted', False)):
                        
                        if not self.is_near_recent_counted(current_pos, 'person'):
                            person_counted_in_frame.append(track_id)
                    
                    if track.get('has_been_inside', False) and track['frames_out_roi'] >= self.exit_frames_threshold:
                        tracks_to_remove.append(track_id)
                    
                    elif not track.get('has_been_inside', False) and track['frames_out_roi'] > self.max_frames_out:
                        tracks_to_remove.append(track_id)
                
                continue
            
            # Procesar VEHÍCULOS
            previous_inside = track.get('is_inside', False)
            track['is_inside'] = is_inside
            
            if is_inside and not previous_inside:
                # Reiniciar flags de alertas cuando entra
                track['entry_time'] = time.time()
                track['last_alert_minute'] = 0
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]
                
                # Detectar color exacto al entrar
                if 'exact_color' not in track or track['exact_color'] == "desconocido":
                    exact_color, general_color = self.detect_exact_color(frame, track['box'], track['class'])
                    track['exact_color'] = exact_color
                    track['general_color'] = general_color
                    if self.debug_mode and exact_color != "desconocido":
                        exact_color_es = self.get_color_name_es(exact_color)
                        general_color_es = self.get_color_name_es(general_color) if general_color else ""
                        color_info = f" ({exact_color_es})"
                        if general_color_es and general_color_es != exact_color_es:
                            color_info += f" [{general_color_es}]"
                        print(f"🌈 {self.class_names_es.get(track['class'])} detectado: color exacto{color_info}")
                
                if hasattr(self, 'last_processed_frame'):
                    self.save_roi_photo(
                        self.last_processed_frame, 
                        track['class'], 
                        track_id, 
                        'entrada',
                        track.get('exact_color'),
                        track.get('general_color')
                    )
                
                if self.debug_mode:
                    exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
                    general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
                    color_info = f" ({exact_color_es})"
                    if general_color_es != exact_color_es:
                        color_info += f" [{general_color_es}]"
                    print(f"🚪 {self.class_names_es.get(track['class'])}{color_info} entró al Área")
            
            elif not is_inside and previous_inside:
                # Calcular tiempo total en el área
                total_minutes = 0
                if 'entry_time' in track:
                    total_time = int(time.time() - track['entry_time'])
                    total_minutes = total_time // 60
                
                # Limpiar flags de alertas cuando sale
                if 'entry_time' in track:
                    del track['entry_time']
                if 'last_alert_minute' in track:
                    del track['last_alert_minute']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]
                
                if hasattr(self, 'last_processed_frame'):
                    self.save_roi_photo(
                        self.last_processed_frame, 
                        track['class'], 
                        track_id, 
                        'salida',
                        track.get('exact_color'),
                        track.get('general_color'),
                        total_time_seconds=0,
                        total_minutes=total_minutes
                    )
                
                if self.debug_mode:
                    exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
                    general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
                    color_info = f" ({exact_color_es})"
                    if general_color_es != exact_color_es:
                        color_info += f" [{general_color_es}]"
                    print(f"🚪 {self.class_names_es.get(track['class'])}{color_info} salió del Área - Duró {total_minutes} minutos")
            
            if is_inside:
                track['has_been_inside'] = True
                track['frames_in_roi'] = track.get('frames_in_roi', 0) + 1
                track['frames_out_roi'] = 0
                vehicles_inside += 1
            else:
                track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                track['frames_in_roi'] = 0
                
                frames_in_roi = track.get('frames_in_roi', 0)
                frames_out_roi = track.get('frames_out_roi', 0)
                
                if (frames_in_roi >= self.min_time_in_roi and 
                    frames_out_roi >= 2 and
                    track['seen_frames'] >= self.min_track_frames and
                    not track.get('counted', False)):
                    
                    if not self.is_near_recent_counted(current_pos, 'vehicle'):
                        counted_in_frame.append(track_id)
                
                if track['frames_out_roi'] >= self.exit_frames_threshold:
                    tracks_to_remove.append(track_id)
            
            if not track.get('has_been_inside', False) and track.get('frames_out_roi', 0) >= self.exit_frames_threshold:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
        
        for track_id in counted_in_frame:
            if self._count_vehicle_safe(track_id):
                if track_id in self.active_tracks:
                    self._remove_track(track_id)
        
        for track_id in person_counted_in_frame:
            if self._count_person_safe(track_id):
                if track_id in self.active_tracks:
                    self._remove_track(track_id)
                
        return vehicles_inside

    def _count_vehicle_safe(self, track_id: int) -> bool:
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        if track['class'] == 'person':
            return False
        
        if track.get('counted', False) or track_id in self.counted_tracks:
            return False
        
        if not self.validate_movement(track_id, track['center']):
            if self.debug_mode:
                print(f"⚠️ Vehículo no tiene movimiento válido - ignorando")
            return False
        
        self.counted_tracks.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        
        self.recent_counted_vehicles.append((track_id, track['center'], self.frame_counter))
        
        self.vehiculos_en_area += 1
        self.last_counted_frame = self.frame_counter
        self.last_counted_id = track_id
        
        vehicle_type = track['class']
        if vehicle_type == 'car':
            self.car_count += 1
            type_text = "CARRO"
        elif vehicle_type == 'truck':
            self.truck_count += 1
            type_text = "CAMION"
        elif vehicle_type == 'motorcycle':
            self.motorcycle_count += 1
            type_text = "MOTOCICLETA"
        elif vehicle_type == 'bus':
            self.truck_count += 1
            type_text = "CAMIONETA"
        else:
            type_text = "VEHÍCULO"
        
        exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
        general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
        
        color_info = f" ({exact_color_es})"
        if general_color_es != exact_color_es:
            color_info += f" [{general_color_es}]"
        
        print(f"\n{'='*60}")
        print(f"🎉 {type_text}{color_info} EN AREA!")
        print(f"   Total vehículos en área: {self.vehiculos_en_area} (C:{self.car_count}, T:{self.truck_count}, M:{self.motorcycle_count})")
        print(f"{'='*60}\n")
        
        return True

    def _count_person_safe(self, track_id: int) -> bool:
        """Cuenta una persona como 'en área'"""
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        if track['class'] != 'person':
            return False
        
        if track.get('counted', False) or track_id in self.counted_persons:
            return False
        
        if not self.validate_movement(track_id, track['center']):
            if self.debug_mode:
                print(f"⚠️ Persona no tiene movimiento válido - ignorando")
            return False
        
        self.counted_persons.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        
        self.recent_counted_persons.append((track_id, track['center'], self.frame_counter))
        
        self.personas_en_area += 1
        self.last_counted_frame = self.frame_counter
        self.last_counted_id = track_id
        
        exact_color_es = self.get_color_name_es(track.get('exact_color', 'desconocido'))
        general_color_es = self.get_color_name_es(track.get('general_color', 'desconocido'))
        
        color_info = f" ({exact_color_es})"
        if general_color_es != exact_color_es:
            color_info += f" [{general_color_es}]"
        
        print(f"\n{'='*60}")
        print(f"🎉 PERSONA{color_info} EN AREA!")
        print(f"   Total personas en área: {self.personas_en_area}")
        print(f"{'='*60}\n")
        
        return True

    def _remove_track(self, track_id: int):
        """Elimina un track de forma segura"""
        if track_id in self.active_tracks:
            object_type = self.active_tracks[track_id]['class']
            if self.debug_mode:
                print(f"✅ Track ({object_type}) eliminado completamente")
            del self.active_tracks[track_id]
        
        if track_id in self.track_history:
            del self.track_history[track_id]
        
        if track_id in self.movement_history:
            del self.movement_history[track_id]
        
        if track_id in self.vehicle_cooldown:
            del self.vehicle_cooldown[track_id]
        if track_id in self.person_cooldown:
            del self.person_cooldown[track_id]
        
        # Limpiar también del registro de alertas periódicas
        if track_id in self.alert_minutes_sent:
            del self.alert_minutes_sent[track_id]

    def cleanup_stale_tracks(self):
        """Limpia tracks inactivos"""
        current_frame = self.frame_counter
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last = current_frame - track['last_seen']
            
            if (frames_since_last > 30 or 
                (track.get('frames_out_roi', 0) > 50 and not track.get('has_been_inside', False))):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if self.debug_mode:
                track_type = self.active_tracks[track_id]['class'] if track_id in self.active_tracks else "desconocido"
                print(f"🗑️ {track_type} eliminado (inactivo)")
            self._remove_track(track_id)

    def match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.active_tracks or not detections:
            return [], list(range(len(detections))), list(self.active_tracks.keys())
        
        similarity_matrix = np.zeros((len(self.active_tracks), len(detections)))
        track_ids = list(self.active_tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track_box = self.active_tracks[track_id]['box']
            for j, det in enumerate(detections):
                similarity_matrix[i, j] = self.calculate_iou(track_box, det['box'])
        
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        iou_threshold = 0.3
        
        while True:
            if similarity_matrix.size == 0:
                break
                
            max_iou = np.max(similarity_matrix)
            if max_iou < iou_threshold:
                break
            
            i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            
            matched_pairs.append((track_ids[i], j))
            unmatched_tracks.remove(i)
            unmatched_detections.remove(j)
            
            similarity_matrix[i, :] = 0
            similarity_matrix[:, j] = 0
        
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_track_ids

    def update_tracks(self, detections: list):
        """Actualiza tracks con nuevas detecciones"""
        self.cleanup_undetected_tracks(detections)
        self.cleanup_stale_tracks()
        
        filtered_detections = []
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for det in detections:
            center = det['center']
            distance_to_roi = cv2.pointPolygonTest(roi_polygon_points, (int(center[0]), int(center[1])), True)
            
            if distance_to_roi > -50:
                filtered_detections.append(det)
        
        current_detections = []
        for det in filtered_detections:
            current_detections.append({
                'box': det['box'],
                'class': det['class'],
                'center': det['center'],
                'confidence': det.get('confidence', 0.5)
            })
        
        if self.active_tracks and current_detections:
            matched_pairs, unmatched_detections, unmatched_tracks = self.match_detections_to_tracks(current_detections)
            
            for track_id, det_idx in matched_pairs:
                det = current_detections[det_idx]
                self.active_tracks[track_id].update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1
                })
                self.track_history[track_id].append(det['center'])
            
            for track_id in unmatched_tracks:
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['last_seen'] = self.frame_counter
            
            for det_idx in unmatched_detections:
                det = current_detections[det_idx]
                self._create_new_track(det)
        
        elif current_detections:
            for det in current_detections:
                self._create_new_track(det)

    def _create_new_track(self, detection: Dict):
        new_id = self.next_id
        self.next_id += 1
        
        center = detection['center']
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        is_inside = self.is_inside_polygon(center, roi_polygon_points)
        
        track_data = {
            'class': detection['class'],
            'box': detection['box'],
            'center': center,
            'last_seen': self.frame_counter,
            'seen_frames': 1,
            'counted': False,
            'is_inside': is_inside,
            'has_been_inside': is_inside,
            'frames_in_roi': 1 if is_inside else 0,
            'total_frames_in_roi': 0,
            'frames_out_roi': 0 if is_inside else 1,
            'entry_frame': self.frame_counter if is_inside else None,
            'frames_without_detection': 0,
            'exact_color': "desconocido",
            'general_color': "desconocido"
        }
        
        if is_inside:
            track_data['entry_time'] = time.time()
            track_data['last_alert_minute'] = 0
        
        if detection['class'] == 'person':
            track_data['positions'] = [(center, is_inside)]
            track_data['total_frames_in_roi'] = 1 if is_inside else 0
        
        self.active_tracks[new_id] = track_data
        self.track_history[new_id].append(center)
        
        if self.debug_mode and is_inside:
            obj_name = self.class_names_es.get(detection['class'], detection['class'])
            print(f"🆕 {obj_name} detectado dentro del ROI")

    def draw_detections(self, image: np.ndarray, vehicles_inside: int) -> np.ndarray:
        """Dibuja información en el frame con recuadros AMARILLOS y tiempo acumulado"""
        # Color AMARILLO para todos los recuadros de detección
        BOX_COLOR = (0, 255, 255)  # Amarillo en BGR
        
        # Colores para el ROI
        CLR_ROI = (0, 255, 255)  # Amarillo también para ROI
        
        # Dibujar ROI
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], (0, 255, 255, 100))
        cv2.addWeighted(roi_overlay, 0.3, image, 0.7, 0, image)
        cv2.polylines(image, [self.roi_polygon], isClosed=True, color=CLR_ROI, thickness=3)
        
        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)
        
        # Dibujar tracks con recuadros AMARILLOS
        for tid, obj in list(self.active_tracks.items()):
            if obj.get('counted', False):
                continue
            
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            object_class = obj['class']
            
            # Usar color AMARILLO para todos los recuadros
            color = BOX_COLOR  # Siempre amarillo
            
            class_name_es = self.class_names_es.get(object_class, object_class.capitalize())
            
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Obtener el color detectado (si existe)
            exact_color = obj.get('exact_color', 'desconocido')
            general_color = obj.get('general_color', 'desconocido')
            
            # Construir el texto: clase, color detectado y tiempo acumulado
            if exact_color != "desconocido":
                exact_color_es = self.get_color_name_es(exact_color)
                text = f"{class_name_es} - {exact_color_es}"
            else:
                text = f"{class_name_es}"
            
            # Añadir tiempo acumulado si el objeto está dentro del ROI
            if 'entry_time' in obj:
                current_time = time.time()
                time_in_roi = int(current_time - obj['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                text += f" - {minutes}m {seconds}s"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Fondo para el texto
            cv2.rectangle(image, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         (0, 0, 0), 
                         -1)
            
            # Texto en blanco
            cv2.putText(image, text, 
                       (x1, y1 - 5), 
                       font, font_scale, 
                       (255, 255, 255), 
                       font_thickness)
        
        # Panel de estadísticas (opcional, puedes mantenerlo o quitarlo)
        if self.show_minimal_info:
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            cv2.putText(image, f"Personas en area: {self.person_count_inside}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
            
            cv2.putText(image, f"Vehiculos en area: {vehicles_inside}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return image
    
    

    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Modelo YOLO no inicializado")
        
        if roi is not None: 
            self.roi_polygon = np.array(roi, np.int32)
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        self.frame_counter += 1
        self.last_processed_frame = image.copy()
        
        try:
            # --- LOGICA DE PRENDER/APAGAR ROI ---
            if activate_roi and hasattr(self, 'roi_polygon') and self.roi_polygon is not None:
                # Si está activado, creamos una máscara para que YOLO solo vea el ROI
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [self.roi_polygon], 255)
                inference_image = cv2.bitwise_and(image, image, mask=mask)
            else:
                # Si está desactivado (False), YOLO analiza la imagen completa
                inference_image = image

            # Detección con YOLO (usando inference_image)
            results = self.model.predict(
                inference_image,
                imgsz=640,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.all_classes,
                verbose=False,
                max_det=50
            )
            
            detections = []
            vehicles_detected = 0
            persons_detected = 0
            
            if results and results[0].boxes is not None:
                det = results[0].boxes
                boxes = det.xyxy.cpu().numpy()
                cls = det.cls.cpu().numpy()
                confs = det.conf.cpu().numpy() if det.conf is not None else [0.5] * len(boxes)
                
                for i in range(boxes.shape[0]):
                    cid = int(cls[i])
                    if cid == 0:
                        cname = 'person'
                        persons_detected += 1
                    elif cid == 2:
                        cname = 'car'
                        vehicles_detected += 1
                    elif cid == 3:
                        cname = 'motorcycle'
                        vehicles_detected += 1
                    elif cid == 5:
                        cname = 'bus'
                        vehicles_detected += 1
                    elif cid == 7:
                        cname = 'truck'
                        vehicles_detected += 1
                    else:
                        continue
                    
                    boxg = boxes[i]
                    center = self.center_of(boxg)
                    
                    detections.append({
                        'class': cname,
                        'box': boxg,
                        'center': center,
                        'confidence': confs[i] if i < len(confs) else 0.5
                    })
            
            if self.debug_mode and detections:
                print(f"📊 Frame {self.frame_counter}: {len(detections)} detecciones ({vehicles_detected} vehículos, {persons_detected} personas)")
            
            # Actualizar tracks
            self.update_tracks(detections)
            
            # Procesar lógica de entrada/salida (pasamos la imagen original para visualización)
            vehicles_inside = self.process_entry_exit_logic(image)
            
            # Verificar y enviar alertas periódicas
            self.check_periodic_alerts(image)
            
            # Log periódico
            if self.frame_counter % 30 == 0:
                self.log_detection(self.frame_counter, flush=True)
                
                if self.debug_mode:
                    print(f"\n📈 Resumen Frame {self.frame_counter}:")
                    print(f"   Personas en ROI: {self.person_count_inside}")
                    print(f"   Vehículos en ROI: {vehicles_inside}")
                    print(f"   Vehículos en área: {self.vehiculos_en_area}")
                    print(f"   Personas en área: {self.personas_en_area}")
                    print(f"   Tracks activos: {len(self.active_tracks)}")
            
            # Dibujar resultados sobre la imagen original
            processed_image = self.draw_detections(image.copy(), vehicles_inside)
            
            # Preparar metadatos
            metadata = {
                'frame_number': self.frame_counter,
                'roi_active': activate_roi,
                'vehicles_detected': vehicles_detected,
                'persons_detected': persons_detected,
                'vehicles_in_area': self.vehiculos_en_area,
                'car_count': self.car_count,
                'truck_count': self.truck_count,
                'motorcycle_count': self.motorcycle_count,
                'persons_inside': self.person_count_inside,
                'persons_in_area': self.personas_en_area,
                'vehicles_inside': vehicles_inside,
                'active_tracks': len(self.active_tracks),
                'last_counted_id': self.last_counted_id,
                'last_counted_frame': self.last_counted_frame
            }
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error en procesamiento de frame: {e}")
            import traceback
            traceback.print_exc()
            return image, {'error': str(e)}




    def set_roi(self, roi_points: List[Tuple[int, int]]):
        self.roi_polygon = np.array(roi_points, np.int32)
        print(f"✅ ROI actualizado a {len(roi_points)} puntos")




    def reset_counter(self):
        self.vehiculos_en_area = 0
        self.personas_en_area = 0
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        self.person_count_inside = 0
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.counted_tracks.clear()
        self.recent_counted_vehicles.clear()
        self.vehicle_cooldown.clear()
        self.counted_persons.clear()
        self.recent_counted_persons.clear()
        self.person_cooldown.clear()
        self.active_tracks.clear()
        self.track_history.clear()
        self.movement_history.clear()
        self.sent_entry_photos.clear()
        self.sent_exit_photos.clear()
        self.last_sent_time.clear()
        self.alert_minutes_sent.clear()
        print("🔄 Contadores reiniciados")

    def toggle_minimal_info(self):
        """Activa/desactiva el modo de información mínima"""
        self.show_minimal_info = not self.show_minimal_info
        status = "MINIMAL" if self.show_minimal_info else "COMPLETA"
        print(f"🔧 Modo de información: {status}")

    def get_stats(self) -> Dict[str, Any]:
        vehicles_inside = 0
        
        for track in self.active_tracks.values():
            if track['class'] == 'person':
                continue
            
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
            
            if is_inside:
                vehicles_inside += 1
        
        return {
            'total_vehicles_in_area': self.vehiculos_en_area,
            'total_persons_in_area': self.personas_en_area,
            'car_count': self.car_count,
            'truck_count': self.truck_count,
            'motorcycle_count': self.motorcycle_count,
            'persons_inside': self.person_count_inside,
            'vehicles_inside': vehicles_inside,
            'frame_counter': self.frame_counter,
            'active_tracks': len(self.active_tracks),
            'last_counted_id': self.last_counted_id,
            'last_counted_frame': self.last_counted_frame,
            'roi_points': self.roi_polygon.tolist()
        }


def create_vehicle_processor(**kwargs) -> MultiObjectProcessor:
    return MultiObjectProcessor(**kwargs)