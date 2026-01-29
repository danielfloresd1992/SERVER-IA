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
    """Procesador de múltiples objetos"""

    _shared_model = None
    _shared_model_lock = threading.Lock()
    
    def __init__(self, 
                client_id: None = None,
                model_path: str = "yolov26l.pt",
                confidence_threshold: float = 0.6,
                vehicle_confidence_threshold: float = 0.45,
                person_confidence_threshold: float = 0.6,
                iou_threshold: float = 0.4,
                device: str = 'cpu',
                share_model: bool = True,
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
        self.vehicle_confidence_threshold = vehicle_confidence_threshold
        self.person_confidence_threshold = person_confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.share_model = share_model
        self.client_id = client_id
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
        # ROI de puerta (para entrada/salida de personal)
        self.door_polygon = None
        self.door_active = False
        self.door_alert_cooldown = 2.0
        self.door_last_alert_time = defaultdict(float)
        self.door_class_names = ["door", "puerta"]
        self._door_class_ids = None
        self.door_direction = None
        self.door_direction_active = False
        
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
        

        
        # Para controlar que no se envíen múltiples fotos del mismo evento
        self.sent_entry_photos = defaultdict(lambda: deque(maxlen=2))
        self.sent_exit_photos = defaultdict(lambda: deque(maxlen=2))
        
        # Control de frecuencia de envío para evitar sobrecarga
        self.last_sent_time = defaultdict(float)
        self.send_cooldown = 1.0
        
        # Para controlar alertas periódicas
        self.alert_minutes_sent = defaultdict(list)  # {track_id: [minutos_enviados]}

        # Cola de alertas en metadata para que el cliente las consuma (no envío por red)
        self.pending_alerts = deque(maxlen=500)
        # Si True, no se llamará a `send_jarvis_wrapper` y la alerta quedará en `pending_alerts`
        self.emit_alerts_via_metadata = True
        # Lock para sincronizar acceso a la cola de alertas (puede accederse desde hilos)
        self.pending_alerts_lock = threading.Lock()
        # Máximo de alertas a enviar por frame
        self.max_alerts_per_frame = 1
        self.model = None
        self.device = device
        self._camera_processors = {}
        self._initialize_model()
        
        os.makedirs(self.car_exit_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        self._log_buffer = []
        self.setup_log_file()
        



        print(f'Modelo inicializado para {client_id}')
        print(f'Analisis procesado desde: {self.device}')
        print(f'🔔 Modo: Alertas a los 1, 3, 6, 9... minutos')
        print(f'🚫 Animales ignorados')
        print(f'🎯 Confianza base: {confidence_threshold}')
        print(f'🚗 Confianza vehículos: {vehicle_confidence_threshold}')
        print(f'👤 Confianza personas: {person_confidence_threshold}')
        print(f'⏱️  Alertas: 1 minuto, luego cada 3 minutos')




    def _initialize_model(self):
        try:
            print(f"🚀 Inicializando modelo YOLO en {self.device}...")
            # Reutilizar modelo compartido si existe
            if self.share_model and MultiObjectProcessor._shared_model is not None:
                self.model = MultiObjectProcessor._shared_model
                return

            model = YOLO(self.model_path)
            try:
                model = model.to(self.device)
            except Exception:
                pass
            self.model = model
            if self.share_model:
                with MultiObjectProcessor._shared_model_lock:
                    if MultiObjectProcessor._shared_model is None:
                        MultiObjectProcessor._shared_model = self.model
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=self.all_classes, verbose=False
            )
            print(f"✅ Modelo YOLO inicializado correctamente en {self.device}")
        except Exception as e:
            print(f"❌ Error inicializando YOLO: {e}")
            raise


    def _get_door_class_ids(self) -> List[int]:
        if self._door_class_ids is not None:
            return self._door_class_ids

        ids = []
        try:
            names = getattr(self.model, 'names', None)
            if isinstance(names, dict):
                name_to_id = {str(v).lower(): int(k) for k, v in names.items()}
            elif isinstance(names, list):
                name_to_id = {str(v).lower(): i for i, v in enumerate(names)}
            else:
                name_to_id = {}

            for n in self.door_class_names:
                key = str(n).lower()
                if key in name_to_id:
                    ids.append(name_to_id[key])
        except Exception:
            ids = []

        self._door_class_ids = ids
        return ids


    def _update_door_from_detections(self, door_detections: List[Dict[str, Any]]):
        if not door_detections:
            return

        best = max(door_detections, key=lambda d: d.get('confidence', 0.0))
        x1, y1, x2, y2 = [int(v) for v in best['box']]
        self.door_polygon = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)
        self.door_active = True




    def setup_log_file(self):
        try:
            with open(self.log_file, 'w', encoding="utf-8") as f:
                f.write("Timestamp,Frame,Vehiculos_Area,Cars,Trucks,Motorcycles,Personas_Dentro,Personas_Area,Evento,Tiempo_Acumulado\n")
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

    def create_annotated_image(self, frame: np.ndarray, object_type: str, object_id: int) -> np.ndarray:
        """Crea una imagen con el objeto señalado en cuadro AMARILLO"""
        annotated_frame = frame.copy()
        
        if object_id in self.active_tracks:
            track = self.active_tracks[object_id]
            x1, y1, x2, y2 = [int(v) for v in track['box']]
            
            # Color AMARILLO fijo para el recuadro (independiente del color detectado)
            box_color = (0, 255, 255)  # Amarillo en BGR
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Añadir etiqueta con clase y tiempo acumulado
            object_name_es = self.class_names_es.get(object_type, object_type.capitalize())
            
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



    def get_action_message(self, object_type: str, event: str, time_in_roi: int = 0, total_minutes: int = 0) -> str:
        """Genera el mensaje completo según el tipo de objeto, evento y color"""
        object_name_es = self.class_names_es.get(object_type, object_type.capitalize())
        is_person = object_type == 'person'
        
        if event == 'entrada':
            if is_person:
                base_message = "Persona Entrando al area"
            else:
                base_message = "Vehiculo Entrando al area"
        elif event == 'salida':
            if is_person:
                base_message = "Persona Saliendo del area"
            else:
                base_message = "Vehiculo Saliendo del area"
        elif event == 'alerta_periodica':
            minutes = time_in_roi // 60
            if minutes == 1:
                base_message = f"{object_name_es} tiene {minutes} minuto en el Área"
            else:
                base_message = f"{object_name_es} tiene {minutes} minutos en el Área"
        else:
            base_message = f"{object_name_es} {event.upper()}"
        
        return base_message


    def _get_entity_type(self, object_class: str) -> str:
        return 'person' if object_class == 'person' else 'vehicle'


    def _direction_message(self, entity_type: str, entering: bool) -> str:
        if entity_type == 'person':
            return "Persona Entrando al area" if entering else "Persona Saliendo del area"
        return "Vehiculo Entrando al area" if entering else "Vehiculo Saliendo del area"


    def _side_of_line(self, p1: Tuple[float, float], p2: Tuple[float, float], p: Tuple[float, float]) -> float:
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


    def _enqueue_alert_meta(self, alert_meta: Dict[str, Any]):
        """Encola una alerta en metadata respetando el lock."""
        try:
            if getattr(self, 'pending_alerts', None) is not None:
                with self.pending_alerts_lock:
                    self.pending_alerts.append(alert_meta)
        except Exception as e:
            logger.error(f"Error encolando alerta pendiente: {e}")


    def _process_door_crossings(self, frame: np.ndarray):
        """Detecta cruces de puerta usando un ROI de puerta y encola alertas."""
        if not self.door_active or self.door_polygon is None:
            return

        if self.door_direction_active and self.door_direction and len(self.door_direction) == 2:
            self._process_door_directional()
            return

        roi_polygon_points = self.door_polygon.reshape((-1, 1, 2))
        now = time.time()

        for track_id, track in list(self.active_tracks.items()):
            if track.get('class') not in ('person', 'car', 'truck', 'bus', 'motorcycle'):
                continue

            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            prev_inside = track.get('door_inside', False)
            track['door_inside'] = is_inside

            if is_inside != prev_inside:
                last_ts = self.door_last_alert_time.get(track_id, 0.0)
                if now - last_ts < self.door_alert_cooldown:
                    continue

                self.door_last_alert_time[track_id] = now
                event = 'entrada_puerta' if is_inside else 'salida_puerta'
                entity_type = self._get_entity_type(track.get('class'))
                message = self._direction_message(entity_type, entering=is_inside)

                alert_meta = {
                    'event': event,
                    'object_type': entity_type,
                    'object_id': track_id,
                    'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                    'message': message,
                    'image_path': None,
                    'image_base64': None
                }

                self._enqueue_alert_meta(alert_meta)


    def _process_door_directional(self):
        p1, p2 = self.door_direction
        roi_polygon_points = self.door_polygon.reshape((-1, 1, 2))
        now = time.time()

        for track_id, track in list(self.active_tracks.items()):
            if track.get('class') not in ('person', 'car', 'truck', 'bus', 'motorcycle'):
                continue

            if len(self.track_history.get(track_id, [])) < 2:
                continue

            prev_pos = self.track_history[track_id][-2]
            current_pos = track['center']

            prev_inside = self.is_inside_polygon(prev_pos, roi_polygon_points)
            curr_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            if not (prev_inside or curr_inside):
                continue

            prev_side = self._side_of_line(p1, p2, prev_pos)
            curr_side = self._side_of_line(p1, p2, current_pos)
            if prev_side == 0 or curr_side == 0 or (prev_side > 0) == (curr_side > 0):
                continue

            last_ts = self.door_last_alert_time.get(track_id, 0.0)
            if now - last_ts < self.door_alert_cooldown:
                continue

            self.door_last_alert_time[track_id] = now
            entering = prev_side < 0 and curr_side > 0

            entity_type = self._get_entity_type(track.get('class'))
            message = self._direction_message(entity_type, entering=entering)
            event = 'entrada_puerta' if entering else 'salida_puerta'

            print(f"🚪 Cruce puerta: {message} (id={track_id})")

            alert_meta = {
                'event': event,
                'object_type': entity_type,
                'object_id': track_id,
                'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                'message': message,
                'image_path': None,
                'image_base64': None
            }

            self._enqueue_alert_meta(alert_meta)

    def save_roi_photo(self, frame: np.ndarray, object_type: str, object_id: int, event: str, 
                       time_in_roi: int = 0, total_minutes: int = 0, total_time_seconds: int = 0):
        """Guarda una foto cuando un objeto entra, sale o se genera una alerta periódica"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            object_name_es = self.class_names_es.get(object_type, object_type.capitalize())
            
            event_dir = os.path.join(self.car_exit_dir, event)
            os.makedirs(event_dir, exist_ok=True)
            
            filename = f"{object_name_es}_{event}_{object_id}_{timestamp}.jpg"
            filepath = os.path.join(event_dir, filename)
            
            annotated_frame = self.create_annotated_image(frame, object_type, object_id)
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
                
                message = self.get_action_message(object_type, event, time_in_roi, total_minutes)

                # Construir metadata de alerta que el cliente debe recibir junto al frame
                # Si se pasó total_time_seconds, derivar total_minutes si no se proporcionó
                if total_minutes == 0 and total_time_seconds:
                    total_minutes = total_time_seconds // 60

                alert_meta = {
                    'event': event,  # 'entrada' | 'salida' | 'alerta_periodica'
                    'object_type': object_type,
                    'object_id': object_id,
                    'exact_color': None,
                    'general_color': None,
                    'time_in_roi': time_in_roi,
                    'total_minutes': total_minutes,
                    'total_time_seconds': total_time_seconds,
                    'timestamp': timestamp,
                    'message': message,
                    'image_path': filepath,
                    'image_base64': imagen_base64 if base64_size_kb <= 500 else None
                }

                # Guardar metadata en cola para que el servidor la incluya en el frame/metadata enviado al cliente
                try:
                    if getattr(self, 'pending_alerts', None) is not None:
                        try:
                            with self.pending_alerts_lock:
                                self.pending_alerts.append(alert_meta)
                        except Exception:
                            # Fallback por si algo falla con el lock
                            self.pending_alerts.append(alert_meta)
                except Exception as e:
                    logger.error(f"Error encolando alerta pendiente: {e}")

                # Solo enviar por red si no estamos emitiendo vía metadata
                if not getattr(self, 'emit_alerts_via_metadata', False):
                    self.send_jarvis_wrapper(imagen_base64, message, object_id)
                
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                
                logger.info(f"✅ Foto de {event} guardada: {filename} ({base64_size_kb:.2f} KB)")
                if self.debug_mode:
                    minutes = time_in_roi // 60
                    seconds = time_in_roi % 60
                    time_info = f" - Tiempo: {minutes}m {seconds}s" if time_in_roi > 0 else ""
                    print(f"📸 Foto {event} guardada: {message}{time_info} ({base64_size_kb:.2f} KB)")
                
                return True
            return False
        except Exception as e:
            logger.error(f"No se pudo guardar la foto de {event}: {e}")
            return False



    def check_periodic_alerts(self, frame: np.ndarray):
        """Verifica y envía alertas periódicas cada 5 minutos"""
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
                
                # Definir los minutos en los que se debe enviar alerta: 5, 10, 15, 20...
                target_minutes = []
                minute = 5
                while minute <= current_minute:
                    target_minutes.append(minute)
                    minute += 5
                
                # Verificar si hay minutos objetivo que aún no se han enviado
                sent_minutes = self.alert_minutes_sent.get(track_id, [])
                
                for target_minute in target_minutes:
                    if target_minute not in sent_minutes:
                        # Enviar alerta periódica
                        success = self.save_roi_photo(
                            frame,
                            track['class'],
                            track_id,
                            'alerta_periodica',
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
                            self._log_periodic_alert(track_id, track['class'], time_in_roi, target_minute)
                            
                            if self.debug_mode:
                                obj_name = self.class_names_es.get(track['class'], track['class'])
                                print(f"⏱️  ALERTA PERIÓDICA: {obj_name} tiene {target_minute} minuto{'s' if target_minute > 1 else ''} en el Área")
            else:
                # Si no está dentro, resetear tiempos
                if 'entry_time' in track:
                    del track['entry_time']
                if 'last_alert_minute' in track:
                    del track['last_alert_minute']
                # Remover del registro de alertas si el track ya no está activo
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]



    def _log_periodic_alert(self, track_id: int, obj_type: str, time_in_roi: int, minute: int):
        """Registra alerta periódica en el log"""
        ts = datetime.datetime.now().strftime("%Y-%m-d %H:%M:%S")
        obj_name_es = self.class_names_es.get(obj_type, obj_type.capitalize())
        minutes = time_in_roi // 60
        seconds = time_in_roi % 60

        log_entry = f"{ts},{self.frame_counter},{self.vehiculos_en_area},{self.car_count},{self.truck_count},{self.motorcycle_count},{self.person_count_inside},{self.personas_en_area},ALERTA_{minute}min_{obj_name_es}"
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
                    
                    frame_src = self.last_processed_frame if hasattr(self, 'last_processed_frame') else frame
                    try:
                        with self.pending_alerts_lock:
                            prev_len = len(self.pending_alerts)
                    except Exception:
                        prev_len = None

                    success = self.save_roi_photo(
                        frame_src,
                        'person',
                        track_id,
                        'entrada'
                    )

                    try:
                        with self.pending_alerts_lock:
                            new_len = len(self.pending_alerts)
                    except Exception:
                        new_len = None

                    if not success or (prev_len is not None and new_len is not None and new_len == prev_len):
                        message = self.get_action_message('person', 'entrada')
                        alert_meta = {
                            'event': 'entrada',
                            'object_type': 'person',
                            'object_id': track_id,
                            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                            'message': message,
                            'image_path': None,
                            'image_base64': None
                        }
                        self._enqueue_alert_meta(alert_meta)
                    
                    if self.debug_mode:
                        print(f"👤 Persona entró en el Área")
                
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
                    
                    frame_src = self.last_processed_frame if hasattr(self, 'last_processed_frame') else frame
                    try:
                        with self.pending_alerts_lock:
                            prev_len = len(self.pending_alerts)
                    except Exception:
                        prev_len = None

                    success = self.save_roi_photo(
                        frame_src,
                        'person',
                        track_id,
                        'salida',
                        total_time_seconds=total_time,
                        total_minutes=total_minutes
                    )

                    try:
                        with self.pending_alerts_lock:
                            new_len = len(self.pending_alerts)
                    except Exception:
                        new_len = None

                    if not success or (prev_len is not None and new_len is not None and new_len == prev_len):
                        message = self.get_action_message('person', 'salida', total_minutes=total_minutes)
                        alert_meta = {
                            'event': 'salida',
                            'object_type': 'person',
                            'object_id': track_id,
                            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                            'message': message,
                            'image_path': None,
                            'image_base64': None
                        }
                        self._enqueue_alert_meta(alert_meta)
                    
                    if self.debug_mode:
                        print(f"👤 Persona salió del Área - Duró {total_minutes} minutos")
                
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
                
                frame_src = self.last_processed_frame if hasattr(self, 'last_processed_frame') else frame
                self.save_roi_photo(
                    frame_src,
                    track['class'],
                    track_id,
                    'entrada'
                )
                
                if self.debug_mode:
                    print(f"🚪 {self.class_names_es.get(track['class'])} entró al Área")
            
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
                
                frame_src = self.last_processed_frame if hasattr(self, 'last_processed_frame') else frame
                self.save_roi_photo(
                    frame_src,
                    track['class'],
                    track_id,
                    'salida',
                    total_time_seconds=total_time,
                    total_minutes=total_minutes
                )
                
                if self.debug_mode:
                    print(f"🚪 {self.class_names_es.get(track['class'])} salió del Área - Duró {total_minutes} minutos")
            
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
        
        print(f"\n{'='*60}")
        print(f"🎉 {type_text} EN AREA!")
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
        
        print(f"\n{'='*60}")
        print(f"🎉 PERSONA EN AREA!")
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
            'frames_without_detection': 0
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


    def get_camera_processor(self, camera_id: int):
        """Retorna un procesador aislado por cámara para evitar mezcla de estados."""
        try:
            cam_key = str(camera_id)
        except Exception:
            cam_key = camera_id

        if cam_key in self._camera_processors:
            return self._camera_processors[cam_key]

        proc = MultiObjectProcessor(
            client_id=f"{self.client_id}_{cam_key}" if self.client_id is not None else None,
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            vehicle_confidence_threshold=self.vehicle_confidence_threshold,
            person_confidence_threshold=self.person_confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            share_model=self.share_model,
            log_file=self.log_file,
            car_exit_dir=self.car_exit_dir,
            image_quality=self.image_quality,
            min_time_in_roi=self.min_time_in_roi,
            max_frames_out=self.max_frames_out,
            min_track_frames=self.min_track_frames,
            show_minimal_info=self.show_minimal_info,
            exit_frames_threshold=self.exit_frames_threshold,
            max_frames_without_detection=self.max_frames_without_detection,
            max_image_size=self.max_image_size
        )
        proc.debug_mode = self.debug_mode
        proc.emit_alerts_via_metadata = getattr(self, 'emit_alerts_via_metadata', True)
        proc.max_alerts_per_frame = getattr(self, 'max_alerts_per_frame', 1)
        try:
            proc.roi_polygon = self.roi_polygon.copy()
        except Exception:
            pass

        self._camera_processors[cam_key] = proc
        return proc

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

        if self.door_active and self.door_polygon is not None:
            cv2.polylines(image, [self.door_polygon], isClosed=True, color=(0, 165, 255), thickness=3)

        if self.door_direction_active and self.door_direction and len(self.door_direction) == 2:
            p1, p2 = self.door_direction
            cv2.arrowedLine(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 165, 255), 3, tipLength=0.2)
        
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
            
            # Construir el texto: clase y tiempo acumulado
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
    
    

    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False, door_roi=None, door_activate=False, door_direction=None, door_direction_activate=False) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Modelo YOLO no inicializado")
        
        if roi is not None: 
            self.roi_polygon = np.array(roi, np.int32)
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        if door_roi is not None:
            self.door_polygon = np.array(door_roi, np.int32)
            self.door_active = bool(door_activate)

        self.door_direction_active = bool(door_direction_activate)
        if door_direction is not None:
            try:
                p1, p2 = door_direction
                self.door_direction = (tuple(p1), tuple(p2))
            except Exception:
                self.door_direction = None

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
            door_class_ids = self._get_door_class_ids() if door_activate else []
            yolo_classes = list(self.all_classes)
            for cid in door_class_ids:
                if cid not in yolo_classes:
                    yolo_classes.append(cid)

            effective_min_conf = min(
                self.confidence_threshold,
                self.vehicle_confidence_threshold,
                self.person_confidence_threshold
            )

            results = self.model.predict(
                inference_image,
                imgsz=640,
                conf=effective_min_conf,
                iou=self.iou_threshold,
                classes=yolo_classes,
                verbose=False,
                max_det=50
            )
            
            detections = []
            door_detections = []
            vehicles_detected = 0
            persons_detected = 0
            
            if results and results[0].boxes is not None:
                det = results[0].boxes
                boxes = det.xyxy.cpu().numpy()
                cls = det.cls.cpu().numpy()
                confs = det.conf.cpu().numpy() if det.conf is not None else [0.5] * len(boxes)
                
                for i in range(boxes.shape[0]):
                    cid = int(cls[i])
                    conf = confs[i] if i < len(confs) else 0.5
                    if door_class_ids and cid in door_class_ids:
                        if conf < self.confidence_threshold:
                            continue
                        door_detections.append({
                            'class': 'door',
                            'box': boxes[i],
                            'center': self.center_of(boxes[i]),
                            'confidence': conf
                        })
                        continue
                    if cid == 0:
                        if conf < self.person_confidence_threshold:
                            continue
                        cname = 'person'
                        persons_detected += 1
                    elif cid == 2:
                        if conf < self.vehicle_confidence_threshold:
                            continue
                        cname = 'car'
                        vehicles_detected += 1
                    elif cid == 3:
                        if conf < self.vehicle_confidence_threshold:
                            continue
                        cname = 'motorcycle'
                        vehicles_detected += 1
                    elif cid == 5:
                        if conf < self.vehicle_confidence_threshold:
                            continue
                        cname = 'bus'
                        vehicles_detected += 1
                    elif cid == 7:
                        if conf < self.vehicle_confidence_threshold:
                            continue
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
                        'confidence': conf
                    })

            if door_activate and door_roi is None:
                self._update_door_from_detections(door_detections)
            
            if self.debug_mode and detections:
                print(f"📊 Frame {self.frame_counter}: {len(detections)} detecciones ({vehicles_detected} vehículos, {persons_detected} personas)")
            
            # Actualizar tracks
            self.update_tracks(detections)
            
            # Procesar lógica de entrada/salida (pasamos la imagen original para visualización)
            vehicles_inside = self.process_entry_exit_logic(image)
            
            # Verificar y enviar alertas periódicas
            self.check_periodic_alerts(image)

            # Verificar cruces de puerta (entrada/salida de personal)
            self._process_door_crossings(image)
            
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
                'door_active': self.door_active,
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
            
            # Extraer alertas pendientes y adjuntarlas al metadata (se vacía la cola)
            alerts = []
            try:
                if getattr(self, 'pending_alerts', None) is not None:
                    with self.pending_alerts_lock:
                        max_alerts = int(getattr(self, 'max_alerts_per_frame', 1) or 1)
                        for _ in range(max_alerts):
                            if not self.pending_alerts:
                                break
                            alerts.append(self.pending_alerts.popleft())
            except Exception as e:
                logger.error(f"Error extrayendo alertas pendientes: {e}")

            metadata['alerts'] = alerts

            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error en procesamiento de frame: {e}")
            import traceback
            traceback.print_exc()
            return image, {'error': str(e), 'alerts': []}




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