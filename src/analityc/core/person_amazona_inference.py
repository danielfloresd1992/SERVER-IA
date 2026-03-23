import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from collections import defaultdict, deque, Counter
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

# ── Constantes para el clasificador de género / edad (Caffe DNN) ──────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'models', 'classifiers')
_GENDER_PROTO = os.path.join(_MODELS_DIR, 'gender_deploy.prototxt')
_GENDER_MODEL = os.path.join(_MODELS_DIR, 'gender_net.caffemodel')
_AGE_PROTO    = os.path.join(_MODELS_DIR, 'age_deploy.prototxt')
_AGE_MODEL    = os.path.join(_MODELS_DIR, 'age_net.caffemodel')
_MEAN_VALUES  = (78.4263377603, 87.7689143744, 114.895847746)
_AGE_RANGES   = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
_CHILD_AGE_INDICES = {0, 1, 2}  # (0-2), (4-6), (8-12) → Niño


class PersonAmazonas:
    """Procesador para reconocimiento de personal específico"""
    
    # Ruta por defecto del modelo de personas (YOLO estándar COCO)
    _DEFAULT_PERSON_MODEL = r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\models\base\yolo11m.pt"

    def __init__(self,
                client_id: None = None,
                model_path: str = "amazonas.pt",  # Modelo entrenado (interacciones)
                person_model_path: str = None,     # Modelo YOLO para personas
                confidence_threshold: float = 0.7,
                iou_threshold: float = 0.4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                log_file: str = "output/detection_log.txt",
                image_quality: int = 70,
                min_time_in_roi: int = 10,
                max_frames_out: int = 5,
                min_track_frames: int = 3,
                show_minimal_info: bool = True,
                exit_frames_threshold: int = 1,
                max_frames_without_detection: int = 5,
                max_image_size: tuple = (640, 480),
                staff_names_file: str = None,
                shared_model: Any = None,
                shared_person_model: Any = None):
        
        try:
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            self.model_path = model_path
            self.person_model_path = person_model_path or self._DEFAULT_PERSON_MODEL
            self.log_file = log_file
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
            self.personas_en_area = 0
            self.active_tracks = {}
            self.next_id = 1
            self.track_history = defaultdict(lambda: deque(maxlen=30))
            
            # Contadores por empleado
            self.employee_counters = defaultdict(int)

            # Contadores de entradas por categoría
            self.entry_counts = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
            
            # ROI por defecto
            self.roi_polygon = np.array(DEFAULT_ROI if DEFAULT_ROI else [(0, 0), (640, 0), (640, 480), (0, 480)], np.int32)
            
            # Para evitar reconteo
            self.counted_tracks = set()
            self.recent_counted_persons = deque(maxlen=30)
            self.person_cooldown = defaultdict(int)
            
            # Estados de seguimiento
            self.last_counted_frame = 0
            self.last_counted_id = 0
            self.debug_mode = True
            
            # Historial de posiciones para validar movimiento
            self.movement_history = defaultdict(lambda: deque(maxlen=10))

            # Historial de clases por track para estabilizar identidad
            self.class_history_len = 8
            self.class_stability_threshold = 0.6  # proporción para considerar estable
            self.track_class_history = defaultdict(lambda: deque(maxlen=self.class_history_len))
            
            # Nombres del personal (se cargan desde archivo o modelo)
            self.staff_names = {}
            self.load_staff_names(staff_names_file)
            
            # Clases a detectar - SOLO PERSONAL
            self.all_classes = []  # Se determinará después de cargar el modelo
            
            # Para controlar alertas periódicas
            self.alert_minutes_sent = defaultdict(list)
            
            # Para controlar que no se envíen múltiples fotos del mismo evento
            self.sent_entry_photos = defaultdict(lambda: deque(maxlen=2))
            self.sent_exit_photos = defaultdict(lambda: deque(maxlen=2))
            
            # Control de frecuencia de envío
            self.last_sent_time = defaultdict(float)
            self.send_cooldown = 1.0
            
            self.model = None
            self.person_model = None
            self.device = device
            # Identificador del cliente/instancia
            self.client_id = client_id
            print(device)
            # Si se proporciona un modelo compartido, usarlo y evitar re-inicializar
            if shared_model is not None:
                self.model = shared_model
                self.person_model = shared_person_model
                # Cargar solo clases permitidas del modelo compartido
                if hasattr(self.model, 'names') and self.model.names:
                    self.all_classes = [c for c in self._ALLOWED_CLASS_IDS if c in self.model.names]
                    self.staff_names = {cid: self.model.names[cid].replace('_', ' ').title() for cid in self.all_classes}
                # Agregar clase persona del modelo de personas
                self.staff_names[-1] = 'Persona'
            else:
                self._initialize_model()

            # Cache de procesadores por cámara (si se usa una instancia compartida)
            self._camera_processors = {}

            # Estado por cámara (separado)
            # Cada camera_id tendrá su propio conjunto de tracks y contadores
            # camera_states[camera_id] = {
            #    'active_tracks': {}, 'next_id': int, 'track_history': defaultdict(deque),
            #    'movement_history': defaultdict(deque), 'employee_counters': defaultdict(int), ...
            # }
            self.camera_states = {}
            self._state_swap_stack = []
            # Lock to prevent concurrent process_frame calls interfering with state swapping
            self._process_lock = threading.RLock()
            # atributo auxiliar para referencia al último frame procesado
            self.last_processed_frame = None
            
            os.makedirs("output/Personal", exist_ok=True)
            os.makedirs("output/Personal/Entradas", exist_ok=True)
            os.makedirs("output/Personal/Salidas", exist_ok=True)
            os.makedirs("output/Personal/Alertas", exist_ok=True)
            os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
            
            self._log_buffer = []
            self.setup_log_file()

            # ── Clasificador de género / edad (OpenCV DNN) ──
            self._init_gender_age_classifier()
            # Intervalo: clasificar cada N frames por track para no sobrecargar
            self._classify_every_n = 5
            # Umbral de altura relativa para niño (fallback sin modelo de edad)
            self.child_height_ratio = 0.30

            print(f'✅ Modelo de personal inicializado para {client_id}')
            print(f'🖥️  Dispositivo: {self.device}')
            print(f'🎯 Umbral de confianza: {confidence_threshold}')
            print(f'👥 Personal registrado: {len(self.staff_names)} personas')
            print(f'📍 Modo debug: {"ACTIVADO" if self.debug_mode else "DESACTIVADO"}')
        except Exception as e:
            print(e)



    def load_staff_names(self, staff_names_file: str = None):
        """Carga los nombres del personal desde archivo o modelo"""
        try:
            if staff_names_file and os.path.exists(staff_names_file):
                with open(staff_names_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            class_id, name = line.strip().split(':')
                            self.staff_names[int(class_id)] = name
                print(f"📋 Nombres cargados desde archivo: {len(self.staff_names)} empleados")
            else:
                print("ℹ️  No se encontró archivo de nombres, se usarán los del modelo")
        except Exception as e:
            print(f"⚠️  Error cargando nombres: {e}")

    # Clases que realmente queremos detectar del modelo amazonas
    _ALLOWED_CLASS_IDS = [2, 3, 4]  # CLIENTE_TOMA_PRODUCTO_HOMBRE, CLIENTE_TOMA_PRODUCTO_MUJER, ESTANTE

    def _initialize_model(self):
        try:
            # ── Modelo 1: amazonas (interacciones + estante) ──
            print(f"🚀 Inicializando modelo amazonas: {self.model_path}")
            self.model = YOLO(self.model_path).to(self.device)

            if hasattr(self.model, 'names') and self.model.names:
                print(f"✅ Clases disponibles en amazonas: {self.model.names}")
                self.all_classes = [c for c in self._ALLOWED_CLASS_IDS if c in self.model.names]
                self.staff_names = {cid: self.model.names[cid].replace('_', ' ').title() for cid in self.all_classes}
            else:
                self.all_classes = list(self._ALLOWED_CLASS_IDS)
                self.staff_names = {2: 'Cliente Toma Producto Hombre', 3: 'Cliente Toma Producto Mujer', 4: 'Estante'}

            # ── Modelo 2: YOLO estándar (personas COCO clase 0) ──
            print(f"🚀 Inicializando modelo personas: {self.person_model_path}")
            self.person_model = YOLO(self.person_model_path).to(self.device)
            # Usamos class_id -1 internamente para personas (evitar colisión con amazonas)
            self.staff_names[-1] = 'Persona'

            print(f"✅ Detectando clases amazonas: {[self.staff_names.get(c, c) for c in self.all_classes]}")
            print(f"✅ Detectando personas con: {self.person_model_path}")

            # Calentamiento de ambos modelos
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=self.all_classes, verbose=False
            )
            _ = self.person_model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=[0], verbose=False
            )
            print(f"✅ Ambos modelos inicializados")
        except Exception as e:
            print(f"❌ Error inicializando modelos: {e}")
            raise

    def setup_log_file(self):
        try:
            with open(self.log_file, 'w', encoding="utf-8") as f:
                f.write("Timestamp,Frame,Empleado_ID,Empleado_Nombre,Evento,Tiempo_Area,Confianza\n")
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
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def is_inside_polygon(self, point: Tuple, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def compress_image(self, image: np.ndarray) -> np.ndarray:
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
            
            return image
        except Exception as e:
            logger.error(f"Error al comprimir imagen: {e}")
            return image

    def get_staff_name(self, class_id: int, confidence: float = None) -> str:
        """Obtiene el nombre legible del empleado"""
        if class_id in self.staff_names:
            name = self.staff_names[class_id]
            if confidence:
                return f"{name} ({confidence:.2f})"
            return name
        else:
            if confidence:
                return f"Empleado_{class_id} ({confidence:.2f})"
            return f"Empleado_{class_id}"

    def get_staff_display_name(self, class_id: int) -> str:
        """Nombre para mostrar en pantalla"""
        if class_id in self.staff_names:
            return self.staff_names[class_id]
        return f"ID_{class_id}"

    # ── Clasificación por categoría (Hombre / Mujer / Niño) ──────
    _CATEGORY_KEYWORDS = {
        'Hombres': ['hombre', 'hombres', 'man', 'male', 'boy', 'chico'],
        'Mujeres': ['mujer', 'mujeres', 'woman', 'female', 'girl', 'chica'],
        'Niños':   ['niño', 'niña', 'niños', 'child', 'kid', 'infant', 'bebe', 'baby'],
    }

    def _classify_category(self, class_id: int) -> str:
        """Clasifica un class_id del modelo en Hombres/Mujeres/Niños/Personas."""
        name = self.get_staff_display_name(class_id).lower().strip()
        for category, keywords in self._CATEGORY_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                return category
        return 'Personas'

    # ── Clasificador de género / edad con OpenCV DNN ──────────────

    def _init_gender_age_classifier(self):
        """Carga los modelos Caffe de género y edad. Si faltan, desactiva."""
        self._gender_net = None
        self._age_net = None
        self._face_cascade = None

        try:
            if os.path.exists(_GENDER_MODEL) and os.path.exists(_GENDER_PROTO):
                self._gender_net = cv2.dnn.readNet(_GENDER_MODEL, _GENDER_PROTO)
                logger.info("Gender classifier loaded")
            else:
                logger.warning(f"Gender model not found in {_MODELS_DIR}")

            if os.path.exists(_AGE_MODEL) and os.path.exists(_AGE_PROTO):
                self._age_net = cv2.dnn.readNet(_AGE_MODEL, _AGE_PROTO)
                logger.info("Age classifier loaded")
            else:
                logger.warning(f"Age model not found in {_MODELS_DIR}")

            # Haar cascade para detección de rostro (viene con OpenCV)
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            has_gender = self._gender_net is not None
            has_age = self._age_net is not None
            print(f"🧑 Clasificador género: {'OK' if has_gender else 'NO'} | "
                  f"edad: {'OK' if has_age else 'NO'}")
        except Exception as e:
            logger.error(f"Error loading gender/age classifier: {e}")

    def _classify_person(self, frame: np.ndarray, bbox) -> str:
        """Clasifica una persona detectada como Hombres/Mujeres/Niños usando rostro.

        Flujo:
        1. Recorta la persona del frame
        2. Detecta rostro con Haar cascade
        3. Clasifica edad → si <12 años → 'Niños'
        4. Clasifica género → 'Hombres' o 'Mujeres'
        5. Fallback: altura relativa para niños, 'Personas' si no hay rostro
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            frame_h, frame_w = frame.shape[:2]

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            person_h = y2 - y1

            if person_h < 20:
                return 'Personas'

            # Fallback por altura: niño si bbox < 30% del frame
            height_ratio = person_h / frame_h

            # Intentar detectar rostro
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return 'Niños' if height_ratio < self.child_height_ratio else 'Personas'

            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

            if self._face_cascade is None:
                return 'Niños' if height_ratio < self.child_height_ratio else 'Personas'

            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                # Sin rostro detectado → fallback por altura
                return 'Niños' if height_ratio < self.child_height_ratio else 'Personas'

            # Tomar el rostro más grande
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = person_crop[fy:fy + fh, fx:fx + fw]

            if face_crop.size == 0:
                return 'Personas'

            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227), _MEAN_VALUES, swapRB=False
            )

            # ── Edad ──
            if self._age_net is not None:
                self._age_net.setInput(blob)
                age_preds = self._age_net.forward()
                age_idx = age_preds[0].argmax()
                if age_idx in _CHILD_AGE_INDICES:
                    return 'Niños'

            # ── Género ──
            if self._gender_net is not None:
                self._gender_net.setInput(blob)
                gender_preds = self._gender_net.forward()
                # [Male, Female]
                return 'Hombres' if gender_preds[0][0] > gender_preds[0][1] else 'Mujeres'

            # Sin modelos cargados
            return 'Niños' if height_ratio < self.child_height_ratio else 'Personas'

        except Exception as e:
            logger.debug(f"Error classifying person: {e}")
            return 'Personas'

    async def send_alert(self, base64_img: str, text: str):
        """Envía una alerta con imagen"""
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
            logger.warning(f"Payload demasiado grande ({payload_size:.2f} KB)")
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
                logger.info(f"✅ Alerta enviada: {respuesta.status_code}")
                return respuesta.json()
            except Exception as e:
                logger.error(f"❌ Error en envío: {e}")
                raise

    def send_alert_wrapper(self, base64_img: str, text: str, object_id: int):
        current_time = time.time()
        last_time = self.last_sent_time.get(object_id, 0)
        
        if current_time - last_time < self.send_cooldown:
            if self.debug_mode:
                print(f"⏳ Cooldown para objeto {object_id}")
            return
        
        def send_async():
            try:
                asyncio.run(self.send_alert(base64_img, text))
                self.last_sent_time[object_id] = current_time
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_alert(base64_img, text))
                        self.last_sent_time[object_id] = current_time
                    finally:
                        loop.close()
                else:
                    logger.error(f"Envío falló: {e}")
            except Exception as e:
                logger.error(f"Envío falló: {e}")
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()

    def create_annotated_image(self, frame: np.ndarray, staff_id: int, object_id: int, confidence: float = None) -> np.ndarray:
        annotated_frame = frame.copy()
        
        if object_id in self.active_tracks:
            track = self.active_tracks[object_id]
            x1, y1, x2, y2 = [int(v) for v in track['box']]
            
            # Color diferente por empleado (basado en ID)
            color_map = [
                (0, 255, 255),   # Amarillo - ID 0
                (255, 0, 0),     # Azul - ID 1
                (0, 255, 0),     # Verde - ID 2
                (255, 0, 255),   # Magenta - ID 3
                (0, 165, 255),   # Naranja - ID 4
                (255, 255, 0),   # Cyan - ID 5
                (128, 0, 128),   # Púrpura - ID 6
                (255, 192, 203)  # Rosa - ID 7
            ]
            
            color_idx = staff_id % len(color_map)
            box_color = color_map[color_idx]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Etiqueta con nombre y confianza
            staff_name = self.get_staff_name(staff_id, confidence)
            label = f"{staff_name}"
            
            # Añadir tiempo en área si está dentro del ROI
            if 'entry_time' in track:
                current_time = time.time()
                time_in_roi = int(current_time - track['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                label += f" - {minutes}m {seconds}s"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
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

    def get_action_message(self, staff_id: int, event: str, time_in_roi: int = 0, confidence: float = None) -> str:
        staff_name = self.get_staff_display_name(staff_id)
        
        if event == 'entrada':
            return f"👤 {staff_name} entró en el área de oficina"
        elif event == 'salida':
            minutes = time_in_roi // 60
            if minutes == 1:
                return f"👤 {staff_name} salió después de {minutes} minuto"
            else:
                return f"👤 {staff_name} salió después de {minutes} minutos"
        elif event == 'alerta_periodica':
            minutes = time_in_roi // 60
            if minutes == 1:
                return f"⏰ {staff_name} lleva {minutes} minuto en el área"
            else:
                return f"⏰ {staff_name} lleva {minutes} minutos en el área"
        else:
            return f"{staff_name} - {event.upper()}"

    def save_staff_photo(self, frame: np.ndarray, staff_id: int, object_id: int, event: str, 
                         confidence: float = None, time_in_roi: int = 0):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            staff_name_safe = self.get_staff_display_name(staff_id).replace(' ', '_')
            
            event_dir = os.path.join("output/Personal", event.capitalize())
            os.makedirs(event_dir, exist_ok=True)
            
            filename = f"{staff_name_safe}_{event}_{object_id}_{timestamp}.jpg"
            filepath = os.path.join(event_dir, filename)
            
            annotated_frame = self.create_annotated_image(frame, staff_id, object_id, confidence)
            compressed_frame = self.compress_image(annotated_frame)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
            
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                base64_size_kb = len(imagen_base64) / 1024
                
                if base64_size_kb > 500:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
                    success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
                    if success:
                        imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                        base64_size_kb = len(imagen_base64) / 1024
                
                message = self.get_action_message(staff_id, event, time_in_roi, confidence)
                self.send_alert_wrapper(imagen_base64, message, object_id)
                
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                
                # Registrar en log
                self._log_staff_event(staff_id, event, time_in_roi, confidence)
                
                logger.info(f"✅ Foto de {event} guardada: {filename}")
                if self.debug_mode:
                    minutes = time_in_roi // 60
                    seconds = time_in_roi % 60
                    time_info = f" - Tiempo: {minutes}m {seconds}s" if time_in_roi > 0 else ""
                    print(f"📸 {message}{time_info} ({base64_size_kb:.2f} KB)")
                
                return True
            return False
        except Exception as e:
            logger.error(f"No se pudo guardar la foto: {e}")
            return False

    def _log_staff_event(self, staff_id: int, event: str, time_in_roi: int = 0, confidence: float = None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        staff_name = self.get_staff_display_name(staff_id)
        minutes = time_in_roi // 60
        seconds = time_in_roi % 60
        
        log_entry = f"{ts},{self.frame_counter},{staff_id},{staff_name},{event},{minutes}:{seconds}"
        if confidence:
            log_entry += f",{confidence:.2f}"
        else:
            log_entry += ",N/A"
        
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(f"{log_entry}\n")

    def _get_cam_state(self, camera_id: Any) -> Dict[str, Any]:
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = {
                'active_tracks': {},
                'next_id': 1,
                'track_history': defaultdict(lambda: deque(maxlen=30)),
                'movement_history': defaultdict(lambda: deque(maxlen=10)),
                'employee_counters': defaultdict(int),
                'counted_tracks': set(),
                'recent_counted_persons': deque(maxlen=30),
                'person_cooldown': defaultdict(int),
                'alert_minutes_sent': defaultdict(list),
                'sent_entry_photos': defaultdict(lambda: deque(maxlen=2)),
                'sent_exit_photos': defaultdict(lambda: deque(maxlen=2)),
                'last_sent_time': defaultdict(float),
                'last_processed_frame': None,
                'personas_en_area': 0,
                'entry_counts': {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0},
            }
        # Migrar estados que no tengan entry_counts
        state = self.camera_states[camera_id]
        if 'entry_counts' not in state:
            state['entry_counts'] = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
        return state

    def get_camera_processor(self, camera_id: Any):
        """Return or create a per-camera PersonAmazonas instance that shares the heavy model.

        This keeps tracking state isolated per camera while sharing the detection model.
        """
        if camera_id in self._camera_processors:
            return self._camera_processors[camera_id]

        # Crear nueva instancia ligera que comparte ambos modelos
        cam_proc = PersonAmazonas(
            client_id=f"{self.client_id}_{camera_id}",
            model_path=self.model_path,
            person_model_path=self.person_model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            log_file=self.log_file,
            image_quality=self.image_quality,
            min_time_in_roi=self.min_time_in_roi,
            max_frames_out=self.max_frames_out,
            min_track_frames=self.min_track_frames,
            show_minimal_info=self.show_minimal_info,
            exit_frames_threshold=self.exit_frames_threshold,
            max_frames_without_detection=self.max_frames_without_detection,
            max_image_size=self.max_image_size,
            staff_names_file=None,
            shared_model=self.model,
            shared_person_model=self.person_model,
        )

        # Keep camera-specific state separate
        # Copy model-related metadata so camera instance can use class names and filters
        try:
            cam_proc.staff_names = dict(self.staff_names)
            cam_proc.all_classes = list(self.all_classes)
            cam_proc.roi_polygon = np.array(self.roi_polygon)
        except Exception:
            pass

        self._camera_processors[camera_id] = cam_proc
        return cam_proc

    def _push_state(self, cam_state: Dict[str, Any]):
        # Guardar punteros actuales
        names = [
            'active_tracks', 'next_id', 'track_history', 'movement_history', 'employee_counters',
            'counted_tracks', 'recent_counted_persons', 'person_cooldown', 'alert_minutes_sent',
            'sent_entry_photos', 'sent_exit_photos', 'last_sent_time', 'last_processed_frame', 'personas_en_area',
            'entry_counts'
        ]
        saved = {}
        for n in names:
            saved[n] = getattr(self, n, None)
            # asignar el estado de la cámara, si existe la clave
            setattr(self, n, cam_state.get(n))
        self._state_swap_stack.append(saved)

    def _pop_state(self):
        if not self._state_swap_stack:
            return
        saved = self._state_swap_stack.pop()
        for n, v in saved.items():
            setattr(self, n, v)

    def check_periodic_alerts(self, frame: np.ndarray):
        current_time = time.time()
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            if is_inside and track['class'] in self.staff_names:
                if 'entry_time' not in track:
                    track['entry_time'] = current_time
                    track['last_alert_minute'] = 0
                
                time_in_roi = int(current_time - track['entry_time'])
                current_minute = time_in_roi // 60
                
                # Alertas a los 1, 3, 6, 9... minutos
                target_minutes = []
                minute = 1
                while minute <= current_minute:
                    target_minutes.append(minute)
                    minute += 3
                
                sent_minutes = self.alert_minutes_sent.get(track_id, [])
                
                for target_minute in target_minutes:
                    if target_minute not in sent_minutes:
                        success = self.save_staff_photo(
                            frame,
                            track['class'],
                            track_id,
                            'alerta_periodica',
                            track.get('confidence'),
                            time_in_roi
                        )
                        
                        if success:
                            if track_id not in self.alert_minutes_sent:
                                self.alert_minutes_sent[track_id] = []
                            
                            if target_minute not in self.alert_minutes_sent[track_id]:
                                self.alert_minutes_sent[track_id].append(target_minute)
                            
                            track['last_alert_minute'] = target_minute
                            
                            if self.debug_mode:
                                staff_name = self.get_staff_display_name(track['class'])
                                print(f"⏰ ALERTA: {staff_name} lleva {target_minute} minuto{'s' if target_minute > 1 else ''}")
            else:
                if 'entry_time' in track:
                    del track['entry_time']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

    def is_near_recent_counted(self, center: Tuple, threshold: int = 50) -> bool:
        for counted_id, counted_center, counted_frame in self.recent_counted_persons:
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
        if not current_detections:
            if self.active_tracks and self.debug_mode:
                print(f"⚠️ No hay detecciones, limpiando tracks")
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
                staff_name = self.get_staff_display_name(track_info['class'])
                print(f"🗑️ {staff_name} eliminado - {self.max_frames_without_detection} frames sin detección")
            self._remove_track(track_id)

    def process_entry_exit_logic(self, frame: np.ndarray):
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        counted_in_frame = []
        tracks_to_remove = []

        self.person_count_inside = 0
        self.active_by_category = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}

        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)

            previous_inside = track.get('is_inside', False)
            track['is_inside'] = is_inside

            if is_inside and not previous_inside:
                # ENTRADA del empleado
                track['entry_time'] = time.time()
                track['last_alert_minute'] = 0
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

                if hasattr(self, 'last_processed_frame'):
                    self.save_staff_photo(
                        self.last_processed_frame,
                        track['class'],
                        track_id,
                        'entrada',
                        track.get('confidence')
                    )

                if self.debug_mode:
                    staff_name = self.get_staff_display_name(track['class'])
                    confidence = track.get('confidence', 0)
                    print(f"🚪 {staff_name} ({confidence:.2f}) entró al área")

                # Incrementar contador del empleado
                self.employee_counters[track['class']] += 1

                # Incrementar contador por categoría (usa la clasificación del track)
                category = track.get('category', 'Personas')
                self.entry_counts[category] = self.entry_counts.get(category, 0) + 1
            
            elif not is_inside and previous_inside:
                # SALIDA del empleado
                total_minutes = 0
                if 'entry_time' in track:
                    total_time = int(time.time() - track['entry_time'])
                    total_minutes = total_time // 60
                
                if 'entry_time' in track:
                    del track['entry_time']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]
                
                if hasattr(self, 'last_processed_frame'):
                    self.save_staff_photo(
                        self.last_processed_frame, 
                        track['class'], 
                        track_id, 
                        'salida',
                        track.get('confidence'),
                        total_time
                    )
                
                if self.debug_mode:
                    staff_name = self.get_staff_display_name(track['class'])
                    print(f"🚪 {staff_name} salió del área - Duró {total_minutes} minutos")
            
            if is_inside:
                track['has_been_inside'] = True
                track['frames_out_roi'] = 0
                track['frames_in_roi'] = track.get('frames_in_roi', 0) + 1
                self.person_count_inside += 1
                cat = track.get('category', 'Personas')
                self.active_by_category[cat] = self.active_by_category.get(cat, 0) + 1
            else:
                track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                track['frames_in_roi'] = 0
                
                frames_in_roi = track.get('total_frames_in_roi', 0)
                frames_out_roi = track.get('frames_out_roi', 0)
                
                if (frames_in_roi >= self.min_time_in_roi and 
                    frames_out_roi >= 2 and
                    track['seen_frames'] >= self.min_track_frames and
                    not track.get('counted', False)):
                    
                    if not self.is_near_recent_counted(current_pos):
                        counted_in_frame.append(track_id)
                
                if track.get('has_been_inside', False) and track['frames_out_roi'] >= self.exit_frames_threshold:
                    tracks_to_remove.append(track_id)
                
                elif not track.get('has_been_inside', False) and track['frames_out_roi'] > self.max_frames_out:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
        
        for track_id in counted_in_frame:
            if self._count_staff_safe(track_id):
                if track_id in self.active_tracks:
                    self._remove_track(track_id)
        
        return self.person_count_inside

    def _count_staff_safe(self, track_id: int) -> bool:
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        if track.get('counted', False) or track_id in self.counted_tracks:
            return False
        
        if not self.validate_movement(track_id, track['center']):
            if self.debug_mode:
                print(f"⚠️ Movimiento no válido - ignorando")
            return False
        
        self.counted_tracks.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        
        self.recent_counted_persons.append((track_id, track['center'], self.frame_counter))
        
        self.personas_en_area += 1
        self.last_counted_frame = self.frame_counter
        self.last_counted_id = track_id
        
        staff_name = self.get_staff_display_name(track['class'])
        confidence = track.get('confidence', 0)
        
        print(f"\n{'='*60}")
        print(f"🎉 {staff_name} ({confidence:.2f}) EN ÁREA!")
        print(f"   Total personal en área: {self.personas_en_area}")
        print(f"   Contador de {staff_name}: {self.employee_counters[track['class']]}")
        print(f"{'='*60}\n")
        
        return True

    def _remove_track(self, track_id: int):
        if track_id in self.active_tracks:
            staff_id = self.active_tracks[track_id]['class']
            staff_name = self.get_staff_display_name(staff_id)
            if self.debug_mode:
                print(f"✅ {staff_name} eliminado del seguimiento")
            del self.active_tracks[track_id]
        
        if track_id in self.track_history:
            del self.track_history[track_id]
        
        if track_id in self.movement_history:
            del self.movement_history[track_id]
        
        if track_id in self.person_cooldown:
            del self.person_cooldown[track_id]
        
        if track_id in self.alert_minutes_sent:
            del self.alert_minutes_sent[track_id]
        # clean up class history
        if track_id in self.track_class_history:
            try:
                del self.track_class_history[track_id]
            except Exception:
                pass

    def cleanup_stale_tracks(self):
        current_frame = self.frame_counter
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last = current_frame - track['last_seen']
            
            if (frames_since_last > 30 or 
                (track.get('frames_out_roi', 0) > 50 and not track.get('has_been_inside', False))):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if self.debug_mode:
                staff_id = self.active_tracks[track_id]['class'] if track_id in self.active_tracks else None
                staff_name = self.get_staff_display_name(staff_id) if staff_id else "desconocido"
                print(f"🗑️ {staff_name} eliminado (inactivo)")
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
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1,
                    'confidence': max(self.active_tracks[track_id].get('confidence', 0), det['confidence'])
                })
                # Re-clasificar persona si aún es 'Personas' (cada N frames)
                if (self.active_tracks[track_id].get('category', 'Personas') == 'Personas'
                        and self.frame_counter % self._classify_every_n == 0
                        and hasattr(self, 'last_processed_frame')
                        and self.last_processed_frame is not None):
                    new_cat = self._classify_person(self.last_processed_frame, det['box'])
                    if new_cat != 'Personas':
                        self.active_tracks[track_id]['category'] = new_cat
                # Append class observation for stability voting
                self.track_history[track_id].append(det['center'])
                try:
                    self.track_class_history[track_id].append(det['class'])
                except Exception:
                    pass
                # Update stable class if threshold reached
                try:
                    self._update_track_class_stability(track_id)
                except Exception:
                    pass
            
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
        
        # Clasificar persona (género/edad) usando el frame actual
        category = 'Personas'
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
            category = self._classify_person(self.last_processed_frame, detection['box'])

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
            'confidence': detection.get('confidence', 0.5),
            'category': category,
        }
        
        if is_inside:
            track_data['entry_time'] = time.time()
            track_data['last_alert_minute'] = 0
        
        self.active_tracks[new_id] = track_data
        self.track_history[new_id].append(center)
        # Initialize class history for stabilization
        try:
            self.track_class_history[new_id].append(detection['class'])
            self._update_track_class_stability(new_id)
        except Exception:
            pass
        
        if self.debug_mode and is_inside:
            staff_name = self.get_staff_display_name(detection['class'])
            confidence = detection.get('confidence', 0)
            print(f"🆕 {staff_name} ({confidence:.2f}) detectado en ROI")

    def _update_tracks_botsort(self, detections: list):
        """Actualiza tracks usando los IDs estables de BoTSORT.

        A diferencia del matching manual por IoU, BoTSORT ya asigna IDs
        consistentes usando Kalman filter + IoU + apariencia. Solo necesitamos
        sincronizar nuestro dict active_tracks con esos IDs.
        """
        current_botsort_ids = set()

        for det in detections:
            tid = det.get('track_id')
            if tid is None:
                # Sin track ID (frame sin tracking), usar fallback
                self._create_new_track(det)
                continue

            current_botsort_ids.add(tid)

            if tid in self.active_tracks:
                # Track existente: actualizar posición y metadata
                track = self.active_tracks[tid]
                track.update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': track['seen_frames'] + 1,
                    'confidence': det['confidence'],
                    'frames_without_detection': 0,
                })
                self.track_history[tid].append(det['center'])

                # Re-clasificar persona si aún es 'Personas'
                if (track.get('category', 'Personas') == 'Personas'
                        and self.frame_counter % self._classify_every_n == 0
                        and self.last_processed_frame is not None):
                    new_cat = self._classify_person(self.last_processed_frame, det['box'])
                    if new_cat != 'Personas':
                        track['category'] = new_cat
            else:
                # Nuevo track de BoTSORT: crear entrada
                center = det['center']
                roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
                is_inside = self.is_inside_polygon(center, roi_polygon_points)

                category = 'Personas'
                if self.last_processed_frame is not None:
                    category = self._classify_person(self.last_processed_frame, det['box'])

                self.active_tracks[tid] = {
                    'class': det['class'],
                    'box': det['box'],
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
                    'confidence': det.get('confidence', 0.5),
                    'category': category,
                }
                if is_inside:
                    self.active_tracks[tid]['entry_time'] = time.time()
                    self.active_tracks[tid]['last_alert_minute'] = 0

                self.track_history[tid].append(center)

                if self.debug_mode and is_inside:
                    print(f"🆕 [{category}] track_id={tid} ({det.get('confidence', 0):.2f}) en ROI")

        # Limpiar tracks que BoTSORT ya no reporta
        stale_ids = []
        for tid in list(self.active_tracks.keys()):
            if tid not in current_botsort_ids:
                track = self.active_tracks[tid]
                track['frames_without_detection'] = track.get('frames_without_detection', 0) + 1
                if track['frames_without_detection'] >= self.max_frames_without_detection:
                    stale_ids.append(tid)

        for tid in stale_ids:
            self._remove_track(tid)

    def draw_detections(self, image: np.ndarray, persons_inside: int) -> np.ndarray:
        # Dibujar ROI
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], (0, 255, 255, 100))
        cv2.addWeighted(roi_overlay, 0.3, image, 0.7, 0, image)
        cv2.polylines(image, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=3)
        
        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)
        
        # Colores por categoría (personas)
        category_colors = {
            'Hombres': (255, 180, 0),    # Azul claro
            'Mujeres': (180, 0, 255),    # Rosa/Magenta
            'Niños':   (0, 255, 180),    # Verde/Cyan
            'Personas': (0, 255, 255),   # Amarillo
        }

        # IDs de clases detectadas
        _INTERACTION_CLASS_IDS = {2, 3}  # CLIENTE_TOMA_PRODUCTO_HOMBRE, CLIENTE_TOMA_PRODUCTO_MUJER
        _ESTANTE_CLASS_ID = 4

        # Preparar polígono ROI para filtrar dibujo
        roi_pts = self.roi_polygon.reshape((-1, 1, 2)) if self.roi_polygon is not None else None

        # Dibujar detecciones (solo las que están DENTRO del ROI)
        for _tid, obj in list(self.active_tracks.items()):
            if obj.get('counted', False):
                continue

            # Filtrar: solo dibujar si el centro está dentro del ROI
            if roi_pts is not None:
                cx, cy = obj.get('center', (0, 0))
                if cv2.pointPolygonTest(roi_pts, (int(cx), int(cy)), False) < 0:
                    continue

            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            staff_id = obj.get('class', -1)
            category = obj.get('category', 'Personas')
            confidence = obj.get('confidence', 0)

            if staff_id == -1:
                # Persona detectada por modelo YOLO estándar
                color = category_colors.get(category, (0, 255, 0))
                text = f"{category} {confidence:.2f}"
                thickness = 2
            elif staff_id in _INTERACTION_CLASS_IDS:
                # Interacción: persona tomando producto
                color = (0, 0, 255)  # Rojo brillante
                label = self.staff_names.get(staff_id, f'Clase_{staff_id}')
                text = f"{label} {confidence:.2f}"
                thickness = 3
            elif staff_id == _ESTANTE_CLASS_ID:
                color = (200, 200, 200)  # Gris
                text = f"Estante {confidence:.2f}"
                thickness = 2
            else:
                color = category_colors.get(category, (0, 255, 255))
                text = f"{category} {confidence:.2f}"
                thickness = 2

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Añadir tiempo en área si está dentro
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
        
        # Panel de estadísticas con desglose por categoría
        if self.show_minimal_info:
            ec = getattr(self, 'entry_counts', {})
            h_entry = ec.get('Hombres', 0)
            m_entry = ec.get('Mujeres', 0)
            n_entry = ec.get('Niños', 0)
            p_entry = ec.get('Personas', 0)
            total_entries = h_entry + m_entry + n_entry + p_entry

            lines = [
                f"Personas Activas: {persons_inside}",
                f"Hombres: {h_entry}",
                f"Mujeres: {m_entry}",
                f"Ninos: {n_entry}",
                f"Total personas: {total_entries}",
            ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            line_height = 32
            padding = 12
            panel_h = padding + line_height * len(lines) + padding
            panel_w = 340

            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            y = 10 + padding + 20
            for line in lines:
                cv2.putText(image, line, (20, y), font, font_scale, (255, 255, 255), thickness)
                y += line_height

        return image

    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False, camera_id: Any = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Modelo no inicializado")
        
        if roi is not None and isinstance(roi, list) and len(roi) >= 3:
            self.roi_polygon = np.array(roi, np.int32)
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        # Usar estado por cámara para mantener separación de tracks
        cam_state = self._get_cam_state(camera_id)

        # Evitar solapamiento concurrente entre llamadas a process_frame
        with self._process_lock:
            self._push_state(cam_state)

            try:
                self.frame_counter += 1
                self.last_processed_frame = image.copy()

                # Detectar sobre la imagen COMPLETA (sin máscara) para evitar
                # artefactos en los bordes del ROI que generan falsos positivos.
                inference_image = image

                # Preparar polígono ROI para filtrado espacial
                roi_active = activate_roi and hasattr(self, 'roi_polygon') and self.roi_polygon is not None
                roi_poly = self.roi_polygon.reshape((-1, 1, 2)) if roi_active else None

                detections = []
                staff_detected = 0

                # ── Modelo 1: amazonas (interacciones + estante) ──
                track_cls = self.all_classes if self.all_classes else None
                results_amz = self.model.track(
                    inference_image,
                    imgsz=640,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=track_cls,
                    verbose=False,
                    max_det=50,
                    persist=True,
                    tracker="botsort.yaml",
                )

                if results_amz and results_amz[0].boxes is not None:
                    det = results_amz[0].boxes
                    boxes = det.xyxy.cpu().numpy()
                    cls = det.cls.cpu().numpy()
                    confs = det.conf.cpu().numpy() if det.conf is not None else [0.5] * len(boxes)
                    track_ids = det.id.int().cpu().numpy() if det.id is not None else None

                    for i in range(boxes.shape[0]):
                        staff_id = int(cls[i])
                        if staff_id not in self.staff_names:
                            continue
                        box = boxes[i]
                        center = self.center_of(box)
                        confidence = confs[i] if i < len(confs) else 0.5

                        if roi_poly is not None:
                            if cv2.pointPolygonTest(roi_poly, (int(center[0]), int(center[1])), False) < 0:
                                continue

                        tid = int(track_ids[i]) if track_ids is not None else None
                        detections.append({
                            'class': staff_id,
                            'box': box,
                            'center': center,
                            'confidence': confidence,
                            'track_id': tid,
                        })
                        staff_detected += 1

                # ── Modelo 2: personas (YOLO estándar, clase 0 COCO) ──
                if self.person_model is not None:
                    results_per = self.person_model.track(
                        inference_image,
                        imgsz=640,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        classes=[0],
                        verbose=False,
                        max_det=50,
                        persist=True,
                        tracker="botsort.yaml",
                    )

                    # Recopilar boxes de estantes para filtro cruzado
                    estante_boxes = [d['box'] for d in detections if d['class'] == 4]

                    if results_per and results_per[0].boxes is not None:
                        det_p = results_per[0].boxes
                        boxes_p = det_p.xyxy.cpu().numpy()
                        confs_p = det_p.conf.cpu().numpy() if det_p.conf is not None else [0.5] * len(boxes_p)
                        track_ids_p = det_p.id.int().cpu().numpy() if det_p.id is not None else None

                        for i in range(boxes_p.shape[0]):
                            box = boxes_p[i]
                            center = self.center_of(box)
                            confidence = confs_p[i] if i < len(confs_p) else 0.5

                            # Filtro ROI
                            if roi_poly is not None:
                                if cv2.pointPolygonTest(roi_poly, (int(center[0]), int(center[1])), False) < 0:
                                    continue

                            # Filtro cruzado: descartar si se solapa >30% con un estante
                            overlaps_estante = False
                            for eb in estante_boxes:
                                if self.calculate_iou(box, eb) > 0.3:
                                    overlaps_estante = True
                                    break
                            if overlaps_estante:
                                continue

                            # Filtro de aspecto: personas son más altas que anchas
                            bw = box[2] - box[0]
                            bh = box[3] - box[1]
                            if bh < bw * 0.8:
                                continue

                            # Offset de track_id para no colisionar con amazonas
                            tid_p = int(track_ids_p[i]) + 100000 if track_ids_p is not None else None
                            detections.append({
                                'class': -1,  # -1 = persona genérica
                                'box': box,
                                'center': center,
                                'confidence': confidence,
                                'track_id': tid_p,
                            })
                            staff_detected += 1

                if self.debug_mode and detections:
                    print(f"📊 Frame {self.frame_counter}: {len(detections)} detectados ({staff_detected} con track)")

                # Actualizar tracks usando IDs de BoTSORT (estables con Kalman)
                self._update_tracks_botsort(detections)

                # Procesar entrada/salida
                persons_inside = self.process_entry_exit_logic(image)

                # Verificar alertas periódicas
                self.check_periodic_alerts(image)

                # Log periódico
                if self.frame_counter % 30 == 0:
                    if self.debug_mode:
                        print(f"\n📈 Resumen Frame {self.frame_counter}:")
                        print(f"   Empleados en ROI: {persons_inside}")
                        print(f"   Total en área: {self.personas_en_area}")
                        print(f"   Tracks activos: {len(self.active_tracks)}")

                        # Mostrar contadores por empleado
                        for staff_id, count in self.employee_counters.items():
                            if count > 0:
                                staff_name = self.get_staff_display_name(staff_id)
                                print(f"   {staff_name}: {count} veces")

                # Dibujar resultados (sobre el estado por cámara)
                processed_image = self.draw_detections(image.copy(), persons_inside)

                # Metadatos
                ec = getattr(self, 'entry_counts', {})
                abc = getattr(self, 'active_by_category', {})
                metadata = {
                    'frame_number': self.frame_counter,
                    'roi_active': activate_roi,
                    'staff_detected': staff_detected,
                    'persons_inside': persons_inside,
                    'persons_in_area': self.personas_en_area,
                    'active_tracks': len(self.active_tracks),
                    'employee_counters': dict(self.employee_counters),
                    'entry_counts': dict(ec),
                    'active_by_category': dict(abc),
                    'total_entries': sum(ec.values()),
                }

                return processed_image, metadata
            except Exception as e:
                logger.error(f"Error procesando frame: {e}")
                import traceback
                traceback.print_exc()
                return image, {'error': str(e)}
            finally:
                # Guardar último frame procesado en el estado de la cámara y restaurar estado global
                try:
                    cam_state['last_processed_frame'] = self.last_processed_frame
                    cam_state['personas_en_area'] = getattr(self, 'personas_en_area', cam_state.get('personas_en_area', 0))
                    cam_state['entry_counts'] = getattr(self, 'entry_counts', cam_state.get('entry_counts', {}))
                except Exception:
                    pass
                # Restaurar estado global
                self._pop_state()

    def set_roi(self, roi_points: List[Tuple[int, int]]):
        self.roi_polygon = np.array(roi_points, np.int32)
        print(f"✅ ROI actualizado a {len(roi_points)} puntos")

    def reset_counter(self):
        self.personas_en_area = 0
        self.employee_counters.clear()
        self.entry_counts = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.counted_tracks.clear()
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
        self.show_minimal_info = not self.show_minimal_info
        status = "MÍNIMA" if self.show_minimal_info else "COMPLETA"
        print(f"🔧 Información: {status}")

    def get_stats(self) -> Dict[str, Any]:
        persons_inside = 0
        
        for track in self.active_tracks.values():
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
            
            if is_inside:
                persons_inside += 1
        
        ec = getattr(self, 'entry_counts', {})
        return {
            'total_persons_in_area': self.personas_en_area,
            'persons_inside': persons_inside,
            'frame_counter': self.frame_counter,
            'active_tracks': len(self.active_tracks),
            'employee_counters': dict(self.employee_counters),
            'entry_counts': dict(ec),
            'total_entries': sum(ec.values()),
            'staff_names': dict(self.staff_names),
            'roi_points': self.roi_polygon.tolist()
        }

    def _update_track_class_stability(self, track_id: int):
        """Compute majority vote over recent class observations and update track class if stable."""
        history = self.track_class_history.get(track_id)
        if not history:
            return

        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        if len(history) >= 1:
            proportion = count / len(history)
            # require minimum number of observations or proportion
            min_obs = min(3, self.class_history_len)
            if len(history) >= min_obs and proportion >= self.class_stability_threshold:
                # update track's class to stable one
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['class'] = most_common
                    self.active_tracks[track_id]['stable_class'] = most_common


def create_staff_processor(**kwargs) -> PersonAmazonas:
    return PersonAmazonas(**kwargs)