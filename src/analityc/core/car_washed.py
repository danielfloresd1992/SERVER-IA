import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from typing import Tuple, Dict, Any, List, Optional
import asyncio
import threading
import base64
import httpx
import datetime
import os
import json
import uuid
from ..config.config import DEFAULT_ROI

logger = logging.getLogger(__name__)


class VehicleProcessor:
    """Procesador de vehículos optimizado para lavado de autos dentro del ROI"""
    
    def __init__(self, 
                client_id: Optional[str] = None,
                model_path: str = "yolo12m.pt",
                confidence_threshold: float = 0.5,
                iou_threshold: float = 0.5,
                device: str = 'cpu',
                log_file: str = "output/detection_log.txt",
                car_wash_dir: str = "output/Car_Wash",
                image_quality: int = 60,
                min_wash_time: int = 30,
                min_track_frames: int = 10,
                min_detection_area: int = 1000,
                max_frames_missing: int = 10,
                motion_threshold: int = 15,
                server_url: Optional[str] = None,
                component_key: str = "vehicle_wash_detector",
                type_inference: str = "vehicle_wash_detection",
                max_image_size: tuple = (640, 480)):
        
        # Configuración de conexión al servidor
        self.client_id = client_id or str(uuid.uuid4())
        self.component_key = component_key
        self.type_inference = type_inference
        self.server_url = server_url
        
        # Configuración de detección
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.log_file = log_file
        self.car_wash_dir = car_wash_dir
        self.image_quality = image_quality
        self.max_image_size = max_image_size
        
        # Configuración de lavado
        self.min_wash_time = min_wash_time
        self.min_track_frames = min_track_frames
        self.min_detection_area = min_detection_area
        self.max_frames_missing = max_frames_missing
        self.motion_threshold = motion_threshold
        
        # Estado interno
        self.frame_counter = 0
        self.autos_lavados = 0
        self.active_tracks = {}
        self.next_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
        # Contadores por tipo de vehículo (simplificados)
        self.vehicle_count = 0  # Para carros, camiones, camionetas, autobuses
        self.motorcycle_count = 0  # Solo motocicletas
        
        # ROI por defecto
        self.roi_polygon = np.array(DEFAULT_ROI, np.int32)
        
        # Para evitar reconteo
        self.counted_tracks = set()
        self.washing_tracks = {}  # Tracks que están siendo lavados
        self.recent_counted_vehicles = deque(maxlen=30)
        
        # Estados de seguimiento
        self.debug_mode = True
        
        # Historial de posiciones para validar lavado
        self.wash_history = defaultdict(lambda: deque(maxlen=30))
        
        # Para verificar que está dentro del ROI
        self.roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        # Historial de movimiento para validar tracks
        self.movement_history = defaultdict(lambda: deque(maxlen=10))
        
        # Historial de confianza para validar tracks
        self.confidence_history = defaultdict(lambda: deque(maxlen=5))
        
        # Filtro de ruido: solo clases de vehículos
        self.valid_classes = ['vehicle', 'motorcycle']
        self.min_vehicle_area = 1500  # Área mínima para vehículos
        
        # Detección de lavado basada en tiempo en ROI
        self.wash_start_time = {}  # frame_counter cuando comenzó el lavado por track_id
        
        # Historial de estados de tracks para detectar cambios
        self.previous_frame_tracks = {}
        
        self.model = None
        self.device = device
        self._initialize_model()
        
        os.makedirs(self.car_wash_dir, exist_ok=True)
        self._log_buffer = []
        self.setup_log_file()
        
        print(f'Modelo inicializado para {self.client_id}')
        print(f'Analisis procesado desde: {self.device}')
        print(f'ROI configurado con {len(self.roi_polygon)} puntos')
        print(f'Umbral de confianza: {confidence_threshold}, Área mínima: {min_detection_area}')
        print('⚠️ Modo sin detección de personas - Solo vehículos')
        print(f'🔗 Component Key: {self.component_key}')
        print(f'🎯 Type Inference: {self.type_inference}')
        if server_url:
            print(f'🌐 Server URL: {server_url}')
    
    def _initialize_model(self):
        try:
            print(f"🚀 Inicializando modelo YOLO con tracker en {self.device}...")
            self.model = YOLO(self.model_path).to(self.device)
            
            # Cargar configuración del tracker
            tracker_config = {
                'tracker_type': 'botsort',
                'track_high_thresh': 0.5,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.6,
                'match_thresh': 0.8,
                'track_buffer': 30,
                'frame_rate': 30
            }
            
            # Verificar que el modelo soporta tracking
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.track(
                dummy_input, 
                imgsz=320, 
                device=self.device,
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                verbose=False,
                persist=True
            )
            print(f"✅ Modelo YOLO con tracker inicializado correctamente en {self.device}")
            print("📋 Categorías detectadas: VEHÍCULOS (carros, camiones, camionetas, autobuses), MOTOCICLETAS")
        except Exception as e:
            print(f"❌ Error inicializando YOLO tracker: {e}")
            raise

    def setup_log_file(self):
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write("Timestamp,Frame,Total_Lavados,Vehiculos,Motocicletas,Wash_Status\n")

    def log_detection(self, frame_count: int, wash_status: str = "", flush: bool = False):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{ts},{frame_count},{self.autos_lavados},{self.vehicle_count},{self.motorcycle_count},{wash_status}"
        self._log_buffer.append(log_entry)
        if flush or len(self._log_buffer) >= 60:
            with open(self.log_file, 'a', encoding="utf-8") as f:
                f.write("\n".join(self._log_buffer) + "\n")
            self._log_buffer.clear()

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

    def distance_between(self, point1: Tuple, point2: Tuple) -> float:
        # Asegurarse de que son valores escalares
        if isinstance(point1, np.ndarray):
            point1 = tuple(point1.tolist())
        if isinstance(point2, np.ndarray):
            point2 = tuple(point2.tolist())
        
        # Convertir a float para evitar problemas
        x1, y1 = float(point1[0]), float(point1[1])
        x2, y2 = float(point2[0]), float(point2[1])
        
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def is_inside_roi(self, point: Tuple) -> bool:
        """Verifica si un punto está dentro del ROI"""
        # Asegurarse de que el punto es un tuple de escalares
        if isinstance(point, np.ndarray):
            point = tuple(point.tolist())
        
        x, y = float(point[0]), float(point[1])
        return cv2.pointPolygonTest(self.roi_polygon_points, (int(x), int(y)), False) >= 0

    def calculate_box_area(self, box: Tuple) -> float:
        """Calcula el área de un bounding box"""
        x1, y1, x2, y2 = box
        return float((x2 - x1) * (y2 - y1))

    def is_valid_detection(self, detection: Dict) -> bool:
        """Valida si una detección es válida basada en clase, área y confianza"""
        # Verificar clase válida (solo vehículos)
        if detection['class'] not in self.valid_classes:
            return False
        
        # Verificar confianza mínima
        if detection['confidence'] < self.confidence_threshold:
            return False
        
        # Verificar área mínima para vehículos
        box_area = self.calculate_box_area(detection['box'])
        return box_area >= self.min_vehicle_area

    def filter_detections_by_roi(self, detections: List[Dict]) -> List[Dict]:
        """Filtra las detecciones para quedarse solo con las válidas dentro del ROI"""
        filtered = []
        for det in detections:
            # Validar detección (solo vehículos)
            if not self.is_valid_detection(det):
                if self.debug_mode and det['confidence'] > 0.3:
                    print(f"❌ Detección inválida: {det['class']} conf:{det['confidence']:.2f}")
                continue
            
            # Verificar si está dentro del ROI
            if self.is_inside_roi(det['center']):
                filtered.append(det)
            elif self.debug_mode:
                print(f"⚠️ Detección fuera del ROI: {det['class']} en {det['center']}")
        
        return filtered

    def has_significant_movement(self, track_id: int, current_pos: Tuple) -> bool:
        """Verifica si un track ha tenido movimiento significativo"""
        if track_id not in self.movement_history:
            self.movement_history[track_id].append(current_pos)
            return True
        
        # Asegurarse de que current_pos no es un array
        if isinstance(current_pos, np.ndarray):
            current_pos = tuple(current_pos.tolist())
        
        # Calcular distancia desde la primera posición
        positions = list(self.movement_history[track_id])
        if len(positions) < 3:
            self.movement_history[track_id].append(current_pos)
            return True
        
        # Asegurarse de que first_pos no es un array
        first_pos = positions[0]
        if isinstance(first_pos, np.ndarray):
            first_pos = tuple(first_pos.tolist())
        
        distance = self.distance_between(first_pos, current_pos)
        
        # Si no se ha movido lo suficiente, podría ser falso positivo
        if distance < self.motion_threshold and len(positions) > 5:
            if self.debug_mode:
                print(f"⚠️ Track {track_id} sin movimiento significativo: {distance:.1f} píxeles")
            return False
        
        self.movement_history[track_id].append(current_pos)
        return True

    def validate_track_confidence(self, track_id: int, confidence: float) -> bool:
        """Valida que un track tenga confianza consistente"""
        self.confidence_history[track_id].append(confidence)
        
        # Si tenemos suficiente historial, verificar consistencia
        if len(self.confidence_history[track_id]) >= 3:
            confidences = list(self.confidence_history[track_id])
            # Asegurarse de que son valores escalares
            confidences = [float(c) for c in confidences]
            avg_confidence = np.mean(confidences)
            if avg_confidence < self.confidence_threshold * 0.8:  # 80% del umbral
                if self.debug_mode:
                    print(f"⚠️ Track {track_id} con baja confianza promedio: {avg_confidence:.2f}")
                return False
        
        return True

    def detect_washing_action(self, detections: List[Dict]) -> List[int]:
        """Detecta cuando un vehículo está siendo lavado dentro del ROI (basado en tiempo)"""
        washing_vehicles = []
        
        # Filtrar solo vehículos dentro del ROI
        vehicles = [d for d in detections if d['class'] in self.valid_classes 
                   and self.is_inside_roi(d['center'])]
        
        if self.debug_mode and vehicles:
            print(f"🚗 Vehículos en ROI: {len(vehicles)}")
        
        for vehicle in vehicles:
            vehicle_center = vehicle['center']
            vehicle_id = vehicle.get('track_id', None)
            
            # Verificar que el vehículo esté dentro del ROI y sea un track válido
            if not self.is_inside_roi(vehicle_center) or vehicle_id is None:
                continue
            
            # Verificar que el track del vehículo sea válido
            if vehicle_id not in self.active_tracks:
                continue
            
            # Verificar movimiento mínimo (para evitar contar vehículos en tránsito)
            if self.has_significant_movement(vehicle_id, vehicle_center):
                # Si se mueve mucho, probablemente no está siendo lavado
                if vehicle_id in self.wash_start_time:
                    # Resetear el tiempo de lavado si empieza a moverse
                    del self.wash_start_time[vehicle_id]
                    if self.debug_mode:
                        print(f"⏹️ Track {vehicle_id} se movió - resetear lavado")
                continue
            
            # Si está relativamente quieto, puede estar siendo lavado
            if vehicle_id not in self.wash_start_time:
                # Iniciar temporizador de lavado
                self.wash_start_time[vehicle_id] = {
                    'start_frame': self.frame_counter,
                    'vehicle_info': vehicle,
                    'last_seen': self.frame_counter
                }
                if self.debug_mode:
                    print(f"⏱️ Iniciando temporizador lavado para track {vehicle_id}")
            
            # Actualizar tiempo de lavado
            self.wash_start_time[vehicle_id]['last_seen'] = self.frame_counter
            wash_time = self.frame_counter - self.wash_start_time[vehicle_id]['start_frame']
            
            # Verificar si ha pasado el tiempo mínimo para considerar lavado
            if (wash_time >= self.min_wash_time and 
                vehicle_id not in self.counted_tracks and
                self.active_tracks[vehicle_id]['seen_frames'] >= self.min_track_frames):
                
                washing_vehicles.append(vehicle_id)
        
        # Limpiar temporizadores de lavado inactivos o que hayan salido del ROI
        tracks_to_remove = []
        for track_id, wash_info in self.wash_start_time.items():
            # Verificar si el vehículo sigue dentro del ROI
            if track_id in self.active_tracks:
                vehicle_center = self.active_tracks[track_id]['center']
                if not self.is_inside_roi(vehicle_center):
                    tracks_to_remove.append(track_id)
                    continue
            
            # Verificar inactividad (más de 60 frames sin ser visto)
            if self.frame_counter - wash_info.get('last_seen', 0) > 60:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.wash_start_time:
                del self.wash_start_time[track_id]
                if self.debug_mode:
                    print(f"🧹 Temporizador lavado {track_id} eliminado")
        
        return washing_vehicles

    def _count_wash_vehicle(self, track_id: int) -> bool:
        """Cuenta un vehículo lavado con todas las verificaciones y toma foto inmediatamente"""
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        # Verificar que no haya sido contado antes
        if track.get('counted', False) or track_id in self.counted_tracks:
            return False
        
        # Verificar que no se haya enviado foto ya
        if track.get('photo_sent', False):
            return False
        
        # Verificar que el vehículo esté dentro del ROI al momento de contar
        if not self.is_inside_roi(track['center']):
            if self.debug_mode:
                print(f"⚠️ Vehículo {track_id} fuera del ROI - no se cuenta")
            return False
        
        # Verificar confianza consistente
        if not self.validate_track_confidence(track_id, track.get('avg_confidence', 0.5)):
            return False
        
        # Verificar frames mínimos detectados
        if track['seen_frames'] < self.min_track_frames:
            if self.debug_mode:
                print(f"⚠️ Track {track_id} con pocos frames: {track['seen_frames']}")
            return False
        
        # Marcar como contado
        self.counted_tracks.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        track['wash_detected'] = True
        track['photo_sent'] = True  # Marcar que ya se envió la foto
        
        # Actualizar contadores
        self.autos_lavados += 1
        
        # Actualizar contador por tipo (solo 2 categorías)
        vehicle_type = track['class']
        if vehicle_type == 'vehicle':
            self.vehicle_count += 1
            type_text = "VEHÍCULO"
        elif vehicle_type == 'motorcycle':
            self.motorcycle_count += 1
            type_text = "MOTOCICLETA"
        else:
            type_text = "VEHÍCULO"
        
        # Mostrar mensaje de éxito
        print(f"\n{'='*60}")
        print(f"🧼 {type_text} LAVADO DETECTADO DENTRO DEL ROI!")
        print(f"   ID: {track_id}")
        print(f"   Posición: {track['center']}")
        wash_start = self.wash_start_time.get(track_id, {}).get('start_frame', 0)
        print(f"   Tiempo estacionado: {self.frame_counter - wash_start} frames")
        print(f"   Frames detectados: {track['seen_frames']}")
        print(f"   Confianza promedio: {track.get('avg_confidence', 0):.2f}")
        print(f"   Total lavados: {self.autos_lavados} (Vehículos:{self.vehicle_count}, Motos:{self.motorcycle_count})")
        print(f"{'='*60}\n")
        
        # TOMAR Y ENVIAR FOTO INMEDIATAMENTE
        if hasattr(self, 'last_processed_frame'):
            # 1. Guardar en disco y enviar a Jarvis
            photo_saved = self.save_wash_photo(self.last_processed_frame, track['class'], track_id)
            
            # 2. Enviar al servidor principal (si está configurado)
            if self.server_url:
                specific_data = {
                    "wash_event": {
                        "track_id": int(track_id),
                        "vehicle_type": track['class'],
                        "wash_time_frames": int(self.frame_counter - self.wash_start_time.get(track_id, {}).get('start_frame', 0)),
                        "detection_time": datetime.datetime.now().isoformat(),
                        "photo_saved": bool(photo_saved)
                    }
                }
                payload = self.create_server_payload(
                    self.last_processed_frame,
                    self.get_stats(),
                    "wash_detected",
                    specific_data
                )
                if payload:
                    # Enviar de manera síncrona para asegurar que se envía inmediatamente
                    self.send_to_server_wrapper(payload)
        
        # Eliminar del temporizador de lavado
        if track_id in self.wash_start_time:
            del self.wash_start_time[track_id]
        
        return True

    def update_tracks_with_tracker(self, tracked_vehicles: Dict[int, Dict]):
        """
        Actualiza los tracks usando la información del tracker de YOLO
        """
        # Eliminar tracks que ya no están siendo trackeados
        current_track_ids = set(tracked_vehicles.keys())
        tracks_to_remove = []
        
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_track_ids:
                frames_missing = self.frame_counter - self.active_tracks[track_id]['last_seen']
                if frames_missing > self.max_frames_missing:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
            if self.debug_mode:
                print(f"🗑️ Track {track_id} eliminado (no detectado por tracker)")
        
        # Actualizar o crear tracks
        for track_id, vehicle_info in tracked_vehicles.items():
            center = vehicle_info['center']
            if isinstance(center, np.ndarray):
                center = tuple(center.tolist())
            
            if track_id in self.active_tracks:
                # Actualizar track existente
                self.active_tracks[track_id].update({
                    'box': vehicle_info['box'],
                    'center': center,
                    'last_seen': self.frame_counter,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1,
                    'class': vehicle_info['class'],
                    'avg_confidence': (self.active_tracks[track_id].get('avg_confidence', 0) * 
                                     (self.active_tracks[track_id]['seen_frames'] - 1) + 
                                     vehicle_info['confidence']) / self.active_tracks[track_id]['seen_frames']
                })
            else:
                # Crear nuevo track
                self.active_tracks[track_id] = {
                    'class': vehicle_info['class'],
                    'box': vehicle_info['box'],
                    'center': center,
                    'last_seen': self.frame_counter,
                    'seen_frames': 1,
                    'counted': False,
                    'wash_detected': False,
                    'photo_sent': False,
                    'inside_roi': True,
                    'avg_confidence': vehicle_info['confidence']
                }
                if self.debug_mode:
                    print(f"🆕 Track {track_id} ({vehicle_info['class']}) creado por tracker")
            
            # Actualizar historial
            self.track_history[track_id].append(center)

    def cleanup_tracks(self):
        """Limpia tracks inválidos, inactivos o fuera del ROI"""
        tracks_to_remove = []
        
        for track_id, track in list(self.active_tracks.items()):
            # Verificar si está fuera del ROI
            if not self.is_inside_roi(track['center']):
                # Incrementar contador de frames fuera del ROI
                track['frames_outside_roi'] = track.get('frames_outside_roi', 0) + 1
                if track['frames_outside_roi'] > 15:  # 15 frames fuera = eliminar
                    tracks_to_remove.append(track_id)
                continue
            
            # Resetear contador si está dentro
            track['frames_outside_roi'] = 0
            
            # Verificar confianza promedio
            if (track.get('avg_confidence', 0) < self.confidence_threshold * 0.7 and 
                track['seen_frames'] > 5):
                tracks_to_remove.append(track_id)
                if self.debug_mode:
                    print(f"⚠️ Track {track_id} eliminado (baja confianza: {track.get('avg_confidence', 0):.2f})")
                continue
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
            if self.debug_mode:
                print(f"🗑️ Track {track_id} eliminado (limpieza)")

    def _remove_track(self, track_id: int):
        """Elimina un track de forma segura"""
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.wash_start_time:
            del self.wash_start_time[track_id]
        if track_id in self.movement_history:
            del self.movement_history[track_id]
        if track_id in self.confidence_history:
            del self.confidence_history[track_id]

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

    def convert_to_serializable(self, obj):
        """Convierte objetos numpy a tipos serializables para JSON"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        else:
            return obj

    def create_server_payload(self, frame: np.ndarray, metadata: Dict[str, Any], 
                            event_type: str = "detection", 
                            specific_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Crea el payload para enviar al servidor según la estructura especificada.
        """
        try:
            # Comprimir y convertir imagen a base64
            compressed_frame = self.compress_image(frame)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
            
            if not success:
                logger.error("Error al codificar imagen")
                return None
                
            imagen_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Crear header con información del frame
            header = {
                "frame_number": self.frame_counter,
                "timestamp": datetime.datetime.now().isoformat(),
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
                "event_type": event_type,
                "processing_time": float(metadata.get('processing_time', 0)),
                "roi_points": self.convert_to_serializable(self.roi_polygon),
                "vehicle_events": metadata.get('vehicle_events', [])
            }
            
            # Crear data_result con resultados de inferencia
            data_result = {
                "detections": [],
                "statistics": {
                    "vehicles_washed": int(self.autos_lavados),
                    "vehicles_detected": int(metadata.get('detections_in_roi', 0)),
                    "vehicle_count": int(self.vehicle_count),
                    "motorcycle_count": int(self.motorcycle_count),
                    "valid_tracks_in_roi": int(metadata.get('valid_tracks_in_roi', 0)),
                    "washing_now": int(metadata.get('washing_now', 0)),
                    "total_tracks": int(len(self.active_tracks))
                },
                "tracks": [],
                "washing_status": [],
                "events": metadata.get('vehicle_events', [])
            }
            
            # Añadir información de tracks activos
            for track_id, track in self.active_tracks.items():
                if self.is_inside_roi(track['center']):
                    track_info = {
                        "track_id": int(track_id),
                        "class": track['class'],
                        "bbox": self.convert_to_serializable(track['box']),
                        "center": self.convert_to_serializable(track['center']),
                        "seen_frames": int(track.get('seen_frames', 0)),
                        "avg_confidence": float(track.get('avg_confidence', 0)),
                        "wash_detected": bool(track.get('wash_detected', False)),
                        "counted": bool(track.get('counted', False)),
                        "wash_start_time": self.convert_to_serializable(self.wash_start_time.get(track_id, {}).get('start_frame'))
                    }
                    data_result["tracks"].append(track_info)
            
            # Añadir información de lavado en progreso
            for track_id, wash_info in self.wash_start_time.items():
                if track_id in self.active_tracks:
                    wash_status = {
                        "track_id": int(track_id),
                        "wash_start_frame": self.convert_to_serializable(wash_info.get('start_frame')),
                        "wash_time_frames": int(self.frame_counter - wash_info.get('start_frame', 0)),
                        "vehicle_class": self.active_tracks[track_id]['class'],
                        "progress_percentage": float(min(100, (self.frame_counter - wash_info.get('start_frame', 0)) / self.min_wash_time * 100))
                    }
                    data_result["washing_status"].append(wash_status)
            
            # Añadir datos específicos si existen
            if specific_data:
                data_result.update(self.convert_to_serializable(specific_data))
            
            # Convertir todo a tipos serializables
            header = self.convert_to_serializable(header)
            data_result = self.convert_to_serializable(data_result)
            
            # Crear payload completo
            payload = {
                "id_connection": self.client_id,
                "component_key": self.component_key,
                "type_inference": self.type_inference,
                "data": {
                    "header": header,
                    "image": imagen_base64,
                    "data_result": data_result
                }
            }
            
            return payload
            
        except Exception as e:
            logger.error(f"Error creando payload: {e}")
            return None

    async def send_to_server(self, payload: Dict[str, Any]) -> bool:
        """
        Envía el payload al servidor de manera asíncrona.
        """
        if not self.server_url:
            logger.warning("No hay URL de servidor configurada")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.server_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Datos enviados al servidor - Estado: {response.status_code}")
                    return True
                else:
                    logger.error(f"❌ Error enviando datos: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error en conexión al servidor: {e}")
            return False

    def send_to_server_wrapper(self, payload: Dict[str, Any]):
        """
        Envía datos al servidor en un hilo separado.
        """
        if not payload:
            return
            
        def send_async():
            try:
                asyncio.run(self.send_to_server(payload))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_to_server(payload))
                    finally:
                        loop.close()
                else:
                    logger.error(f"Envío asíncrono falló: {e}")
            except Exception as e:
                logger.error(f"Envío asíncrono falló: {e}")
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()

    async def send_jarvis(self, base64_img: str, text: str):
        """Envía una imagen a un servidor de manera asíncrona"""
        payload = {"my-text": text, "my-file": base64_img, "type": "image/jpg"}
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            try: 
                respuesta = await client.post(
                    "https://72.68.60.254:4000/bot/imgV2/number=120363402589311344@g.us",
                    json=payload
                )
                respuesta.raise_for_status()
                logger.info(f"Imagen enviada exitosamente - Estado: {respuesta.status_code}")
                return respuesta.json()
            except Exception as e:
                logger.error(f"Error en envío: {e}")
                raise

    def send_jarvis_wrapper(self, base64_img: str, text: str):
        """Envía una imagen en un hilo separado"""
        def send_async():
            try:
                asyncio.run(self.send_jarvis(base64_img, text))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_jarvis(base64_img, text))
                    finally:
                        loop.close()
                else:
                    logger.error(f"Envío asíncrono falló: {e}")
            except Exception as e:
                logger.error(f"Envío asíncrono falló: {e}")
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()

    def save_wash_photo(self, frame: np.ndarray, vehicle_type: str, vehicle_id: int):
        """Guarda una foto del vehículo siendo lavado dentro del ROI"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Usar nombres de archivo más simples
        if vehicle_type == 'motorcycle':
            filename = f"moto_lavada_{vehicle_id}_{timestamp}.jpg"
            type_text = "MOTOCICLETA"
        else:
            filename = f"vehiculo_lavado_{vehicle_id}_{timestamp}.jpg"
            type_text = "VEHÍCULO"
            
        filepath = os.path.join(self.car_wash_dir, filename)
        try:
            # Dibujar ROI en la imagen antes de guardar
            frame_with_roi = frame.copy()
            cv2.polylines(frame_with_roi, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=3)
            
            # Resaltar el vehículo lavado en la foto
            if vehicle_id in self.active_tracks:
                track = self.active_tracks[vehicle_id]
                x1, y1, x2, y2 = [int(v) for v in track['box']]
                cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Rectángulo verde grueso
                cv2.putText(frame_with_roi, f"LAVADO: {type_text} ID:{vehicle_id}", 
                          (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', frame_with_roi, encode_params)
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                # Enviar a Jarvis de manera inmediata
                self.send_jarvis_wrapper(imagen_base64, 
                    f'🧼 {type_text} lavado detectado - ID: {vehicle_id} - DENTRO DEL ROI')
                # Guardar en disco
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                logger.info(f"Lavado detectado y guardado: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"No se pudo guardar la foto: {e}")
            return False

    def draw_detections(self, image: np.ndarray) -> np.ndarray:
        """Dibuja información en el frame (solo vehículos dentro del ROI)"""
        CLR_VEHICLE = (0, 165, 255)  # Color para vehículos (carros, camiones, etc.)
        CLR_MOTORCYCLE = (0, 255, 255)  # Color para motocicletas
        CLR_ROI = (0, 255, 255)
        CLR_WASHING = (255, 0, 255)
        CLR_WAITING = (255, 165, 0)  # Color para vehículos esperando lavado
        
        # Crear una copia de la imagen para trabajar
        processed_image = image.copy()
        
        # Dibujar primero las detecciones y tracks
        for tid, obj in self.active_tracks.items():
            # Verificar si está dentro del ROI
            if not self.is_inside_roi(obj['center']):
                continue
                
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            vehicle_class = obj['class']
            
            # Determinar color basado en estado de lavado y tipo de vehículo
            if obj.get('wash_detected', False):
                color = CLR_WASHING
                status = "LAVADO"
            elif tid in self.wash_start_time:
                wash_time = self.frame_counter - self.wash_start_time[tid]['start_frame']
                color = CLR_WAITING
                status = f"LAVANDO ({wash_time}/{self.min_wash_time})"
            else:
                if vehicle_class == 'vehicle':
                    color = CLR_VEHICLE
                elif vehicle_class == 'motorcycle':
                    color = CLR_MOTORCYCLE
                else:
                    color = (255, 255, 255)
                status = "EN ROI"
            
            # Dibujar bounding box
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar centro
            center_x, center_y = int(obj['center'][0]), int(obj['center'][1])
            cv2.circle(processed_image, (center_x, center_y), 4, color, -1)
            
            # Etiqueta (solo 2 categorías)
            vehicle_label = "VEHÍCULO" if vehicle_class == 'vehicle' else "MOTO"
            label = f"ID:{tid} {vehicle_label} {status}"
            cv2.putText(processed_image, label, (x1, max(y1 - 8, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Información adicional
            info_text = f"F:{obj['seen_frames']} C:{obj.get('avg_confidence', 0):.2f}"
            cv2.putText(processed_image, info_text, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Ahora dibujar el ROI - primero el relleno con transparencia
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], (0, 255, 255, 50))
        cv2.addWeighted(roi_overlay, 0.2, processed_image, 0.8, 0, processed_image)  # Reducir opacidad
        
        # Luego el contorno del ROI
        cv2.polylines(processed_image, [self.roi_polygon], isClosed=True, color=CLR_ROI, thickness=3)
        
        # Dibujar puntos de control del ROI (opcional)
        for x, y in self.roi_polygon:
            cv2.circle(processed_image, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(processed_image, (x, y), 8, (255, 255, 255), 2)
        
        # Calcular estadísticas para el panel
        valid_tracks = 0
        waiting_tracks = 0
        
        for track_id, track in self.active_tracks.items():
            if self.is_inside_roi(track['center']):
                valid_tracks += 1
                if track_id in self.wash_start_time:
                    waiting_tracks += 1
        
        # Panel de estadísticas (dibujar al final para que esté encima de todo)
        overlay = processed_image.copy()
        cv2.rectangle(overlay, (5, 5), (500, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, processed_image, 0.3, 0, processed_image)
        
        # Textos del panel (simplificado)
        texts = [
            (f"🧼 VEHÍCULOS LAVADOS: {self.autos_lavados}", (15, 35), 1.0, (0, 255, 0), 3),
            (f"🚗 Vehículos: {self.vehicle_count}", (15, 75), 0.7, CLR_VEHICLE, 2),
            (f"🏍️ Motocicletas: {self.motorcycle_count}", (15, 105), 0.7, CLR_MOTORCYCLE, 2),
            (f"✅ Vehículos en ROI: {valid_tracks}", (15, 135), 0.5, (0, 255, 0), 1),
            (f"⏱️ Esperando lavado: {waiting_tracks}", (15, 155), 0.5, CLR_WAITING, 1),
            (f"🔧 Lavando ahora: {len([tid for tid in self.wash_start_time if tid in self.active_tracks])}", (15, 175), 0.5, CLR_WASHING, 1),
            (f"Frame: {self.frame_counter}", (15, 195), 0.5, (200, 200, 200), 1),
            (f"Umbral: {self.confidence_threshold} | Área mín: {self.min_vehicle_area}px", (15, 215), 0.4, (200, 200, 200), 1)
        ]
        
        for text, pos, size, color, thickness in texts:
            cv2.putText(processed_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        
        return processed_image

    def process_frame(self, image: np.ndarray, roi=None, send_to_server: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa un frame usando el tracker integrado de YOLO para evitar duplicaciones
        Solo detecta vehículos, no personas
        """
        import time
        start_time = time.time()
        
        if self.model is None:
            raise RuntimeError("Modelo YOLO no inicializado")
        
        if roi is not None: 
            self.roi_polygon = np.array(roi, np.int32)
            self.roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        self.frame_counter += 1
        self.last_processed_frame = image.copy()
        
        # Inicializar lista de eventos de vehículos para este frame
        vehicle_events = []
        
        try:
            # Detección con YOLO TRACKING - Usar el tracker integrado
            results = self.model.track(
                image,
                persist=True,  # IMPORTANTE: mantener persistencia entre frames
                imgsz=640,
                conf=self.confidence_threshold,
                iou=0.45,  # IOU más alto para NMS más agresivo
                classes=[2, 3, 5, 7],  # Solo vehículos: car, motorcycle, bus, truck
                verbose=False,
                max_det=10,  # Reducir máximo de detecciones
                tracker="botsort.yaml"  # Usar tracker integrado
            )
            
            detections = []
            tracked_vehicles = {}
            
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                for i in range(boxes.shape[0]):
                    cid = int(cls[i])
                    
                    # Simplificar categorías: todo es "vehicle" excepto motocicletas
                    if cid == 2 or cid == 5 or cid == 7:  # car, bus, truck -> VEHICLE
                        cname = 'vehicle'
                    elif cid == 3:  # motorcycle -> MOTORCYCLE
                        cname = 'motorcycle'
                    else:
                        continue  # Solo vehículos
                    
                    box = boxes[i]
                    center = self.center_of(box)
                    track_id = int(track_ids[i])
                    
                    # Verificar si está dentro del ROI
                    if not self.is_inside_roi(center):
                        continue
                    
                    # Verificar área mínima
                    area = self.calculate_box_area(box)
                    if area < self.min_vehicle_area:
                        continue
                    
                    # Usar el track_id proporcionado por YOLO
                    detections.append({
                        'class': cname,
                        'box': box,
                        'center': center,
                        'confidence': float(confs[i]),
                        'track_id': track_id,
                        'area': area
                    })
                    
                    # Guardar información del vehículo trackeado
                    tracked_vehicles[track_id] = {
                        'class': cname,
                        'box': box,
                        'center': center,
                        'confidence': float(confs[i]),
                        'area': area
                    }
            
            # Actualizar tracks activos con la información del tracker
            self.update_tracks_with_tracker(tracked_vehicles)
            
            if self.debug_mode and detections:
                vehicle_count = len([d for d in detections if d['class'] == 'vehicle'])
                motorcycle_count = len([d for d in detections if d['class'] == 'motorcycle'])
                print(f"📊 Frame {self.frame_counter}: {len(detections)} vehículos trackeados en ROI")
                print(f"   - Vehículos: {vehicle_count}")
                print(f"   - Motocicletas: {motorcycle_count}")
            
            # Detectar acciones de lavado
            washing_vehicles = self.detect_washing_action(detections)
            
            # Procesar vehículos lavados
            for track_id in washing_vehicles:
                if self._count_wash_vehicle(track_id):
                    if track_id in self.active_tracks:
                        track = self.active_tracks[track_id]
                        event = {
                            'vehicle_id': int(track_id),
                            'timestamp': datetime.datetime.now().isoformat(),
                            'message': 'lavado',
                            'vehicle_type': track['class'],
                            'position': [float(coord) for coord in track['center']],
                            'confidence': float(track.get('avg_confidence', 0)),
                            'frame_number': int(self.frame_counter),
                            'wash_time': int(self.frame_counter - self.wash_start_time.get(track_id, {}).get('start_frame', 0))
                        }
                        vehicle_events.append(event)
                        
                        wash_status = f"Lavado_ID{track_id}"
                        self.log_detection(self.frame_counter, wash_status, flush=False)
            
            # Log periódico
            if self.frame_counter % 30 == 0:
                self.log_detection(self.frame_counter, "Periodic", flush=True)
                
                if self.debug_mode:
                    valid_tracks = len([tid for tid, track in self.active_tracks.items() 
                                       if self.is_inside_roi(track['center'])])
                    washing_now = len([tid for tid in self.wash_start_time if tid in self.active_tracks])
                    
                    print(f"\n📈 Resumen Frame {self.frame_counter}:")
                    print(f"   Tracks activos en ROI: {valid_tracks}")
                    print(f"   Vehículos siendo lavados: {washing_now}")
                    print(f"   Total lavados: {self.autos_lavados}")
                    print(f"   - Vehículos lavados: {self.vehicle_count}")
                    print(f"   - Motocicletas lavadas: {self.motorcycle_count}")
            
            # Dibujar resultados
            processed_image = self.draw_detections(image.copy())
            
            # Calcular tiempo de procesamiento
            processing_time = time.time() - start_time
            
            # Preparar metadatos
            valid_tracks = len([tid for tid, track in self.active_tracks.items() 
                               if self.is_inside_roi(track['center'])])
            washing_now = len([tid for tid in self.wash_start_time if tid in self.active_tracks])
            
            metadata = {
                'frame_number': int(self.frame_counter),
                'detections_in_roi': int(len(detections)),
                'vehicles_washed': int(self.autos_lavados),
                'vehicle_count': int(self.vehicle_count),
                'motorcycle_count': int(self.motorcycle_count),
                'valid_tracks_in_roi': int(valid_tracks),
                'total_tracks': int(len(self.active_tracks)),
                'washing_now': int(washing_now),
                'processing_time': float(processing_time),
                'vehicle_events': vehicle_events
            }
            
            # Enviar datos al servidor si está configurado
            if send_to_server and self.server_url:
                payload = self.create_server_payload(processed_image, metadata)
                if payload:
                    if vehicle_events:
                        payload['data']['header']['vehicle_events'] = vehicle_events
                        payload['data']['data_result']['events'] = vehicle_events
                    self.send_to_server_wrapper(payload)
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error en procesamiento de frame: {e}")
            import traceback
            traceback.print_exc()
            return image, {'error': str(e)}

    def reset_counter(self):
        """Reinicia todos los contadores y limpia tracks"""
        self.autos_lavados = 0
        self.vehicle_count = 0
        self.motorcycle_count = 0
        self.counted_tracks.clear()
        self.wash_start_time.clear()
        self.active_tracks.clear()
        self.track_history.clear()
        self.wash_history.clear()
        self.movement_history.clear()
        self.confidence_history.clear()
        print("🔄 Contadores de lavado reiniciados y tracks limpiados")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas actuales (solo vehículos dentro del ROI)"""
        valid_tracks = 0
        for track_id, track in self.active_tracks.items():
            if self.is_inside_roi(track['center']):
                valid_tracks += 1
        
        washing_now = len([tid for tid in self.wash_start_time if tid in self.active_tracks])
        
        return {
            'total_vehicles_washed': int(self.autos_lavados),
            'vehicle_count': int(self.vehicle_count),
            'motorcycle_count': int(self.motorcycle_count),
            'frame_counter': int(self.frame_counter),
            'valid_tracks_in_roi': int(valid_tracks),
            'total_tracks': int(len(self.active_tracks)),
            'washing_now': int(washing_now),
            'roi_points': self.convert_to_serializable(self.roi_polygon)
        }


def create_vehicle_processor(**kwargs) -> VehicleProcessor:
    """Crea y retorna una instancia configurada de VehicleProcessor"""
    return VehicleProcessor(**kwargs)