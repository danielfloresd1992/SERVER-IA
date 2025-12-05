import cv2
import numpy as np
import torch
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


class VehicleProcessor:
    """Procesador de vehículos optimizado para entradas y salidas"""
    
    def __init__(self, 
                client_id: None,
                model_path: str = "yolo11x.pt",
                confidence_threshold: float = 0.3,
                iou_threshold: float = 0.4,
                device: str = 'cpu',
                log_file: str = "output/detection_log.txt",
                car_exit_dir: str = "output/Car_Exit",
                image_quality: int = 60,
                min_time_in_roi: int = 10,  # Frames mínimos dentro para contar salida
                max_frames_out: int = 5,    # Frames máximos fuera para eliminar
                min_track_frames: int = 3):  # Frames mínimos de seguimiento para considerar vehículo válido
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.log_file = log_file
        self.car_exit_dir = car_exit_dir
        self.image_quality = image_quality
        
        # Configuración de tiempos
        self.min_time_in_roi = min_time_in_roi
        self.max_frames_out = max_frames_out
        self.min_track_frames = min_track_frames
        
        # Estado interno
        self.frame_counter = 0
        self.autos_lavados = 0
        self.active_tracks = {}
        self.next_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
        # Contadores por tipo de vehículo
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        
        # ROI por defecto
        self.roi_polygon = np.array(DEFAULT_ROI, np.int32)
        
        # Para evitar reconteo
        self.counted_tracks = set()
        self.recent_counted_vehicles = deque(maxlen=30)
        self.vehicle_cooldown = defaultdict(int)
        
        # Estados de seguimiento
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.debug_mode = True
        
        # Historial de posiciones para validar movimiento
        self.movement_history = defaultdict(lambda: deque(maxlen=10))
        
        self.model = None
        self.device = device
        self._initialize_model()
        
        os.makedirs(self.car_exit_dir, exist_ok=True)
        self._log_buffer = []
        self.setup_log_file()
        
        print(f'Modelo inicializado para {client_id}')
        print(f'Analisis procesado desde: {self.device}')
    
    
    

    def _initialize_model(self):
        try:
            print(f"🚀 Inicializando modelo YOLO en {self.device}...")
            self.model = YOLO(self.model_path).to(self.device)
            # Solo clases de vehículos: coche(2), moto(3), camión(7), autobús(5)
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=[2, 3, 5, 7], verbose=False
            )
            print(f"✅ Modelo YOLO inicializado correctamente en {self.device}")
        except Exception as e:
            print(f"❌ Error inicializando YOLO: {e}")
            raise

    def setup_log_file(self):
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write("Timestamp,Frame,Total_Vehicles,Cars,Trucks,Motorcycles\n")

    def log_detection(self, frame_count: int, flush: bool = False):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{ts},{frame_count},{self.autos_lavados},{self.car_count},{self.truck_count},{self.motorcycle_count}"
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

    def is_inside_polygon(self, point: Tuple, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def is_near_recent_counted(self, center: Tuple, threshold: int = 50) -> bool:
        """Verifica si un punto está cerca de un vehículo recién contado"""
        for counted_id, counted_center, counted_frame in self.recent_counted_vehicles:
            # Calcular distancia euclidiana
            distance = np.sqrt((center[0] - counted_center[0])**2 + (center[1] - counted_center[1])**2)
            if distance < threshold and (self.frame_counter - counted_frame) < 30:
                return True
        return False

    def validate_movement(self, track_id: int, current_pos: Tuple) -> bool:
        """Valida que el vehículo se esté moviendo (para evitar falsos positivos)"""
        if track_id not in self.movement_history:
            self.movement_history[track_id].append(current_pos)
            return True
        
        # Calcular distancia recorrida
        positions = list(self.movement_history[track_id])
        if len(positions) < 3:
            self.movement_history[track_id].append(current_pos)
            return True
        
        # Verificar movimiento significativo
        first_pos = positions[0]
        distance = np.sqrt((current_pos[0] - first_pos[0])**2 + (current_pos[1] - first_pos[1])**2)
        
        self.movement_history[track_id].append(current_pos)
        return distance > 10  # Mínimo 10 píxeles de movimiento

    def process_entry_exit_logic(self):
        """Procesa la lógica de entrada y salida de vehículos"""
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        counted_in_frame = []
        
        for track_id, track in list(self.active_tracks.items()):
            # Saltar si ya fue contado
            if track.get('counted', False):
                continue
            
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            # Actualizar historial de posiciones
            track['positions'] = track.get('positions', []) + [(current_pos, is_inside)]
            if len(track['positions']) > 20:
                track['positions'] = track['positions'][-20:]
            
            # CASO 1: ENTRADA al ROI
            if is_inside and not track.get('has_been_inside', False):
                track['has_been_inside'] = True
                track['entry_frame'] = self.frame_counter
                track['frames_in_roi'] = 1
                track['frames_out_roi'] = 0
                
                if self.debug_mode:
                    print(f"🚪 Track {track_id} ({track['class']}) ENTRÓ al ROI en frame {self.frame_counter}")
            
            # CASO 2: DENTRO del ROI
            elif is_inside and track.get('has_been_inside', False):
                track['frames_in_roi'] = track.get('frames_in_roi', 0) + 1
                track['frames_out_roi'] = 0
                
                # Si lleva mucho tiempo dentro sin salir, podría ser un vehículo estacionado
                if track['frames_in_roi'] > 100:
                    if self.debug_mode:
                        print(f"⚠️ Track {track_id} lleva {track['frames_in_roi']} frames dentro - posible estacionado")
            
            # CASO 3: SALIDA del ROI
            elif not is_inside and track.get('has_been_inside', False):
                track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                track['frames_in_roi'] = track.get('frames_in_roi', 0)
                
                # Verificar condiciones para contar salida
                frames_in_roi = track.get('frames_in_roi', 0)
                frames_out_roi = track.get('frames_out_roi', 0)
                
                # Solo contar si estuvo suficiente tiempo dentro y ahora está fuera
                if (frames_in_roi >= self.min_time_in_roi and 
                    frames_out_roi >= 2 and
                    track['seen_frames'] >= self.min_track_frames):
                    
                    # Verificar que no sea un reconteo
                    if not self.is_near_recent_counted(current_pos):
                        counted_in_frame.append(track_id)
                    else:
                        if self.debug_mode:
                            print(f"⚠️ Track {track_id} cerca de vehículo recién contado - ignorando")
            
            # CASO 4: FUERA del ROI (nunca entró)
            else:
                track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                
                # Si nunca entró y lleva muchos frames fuera, eliminarlo
                if track['frames_out_roi'] > self.max_frames_out:
                    if self.debug_mode:
                        print(f"🗑️ Track {track_id} eliminado - nunca entró al ROI")
                    self._remove_track(track_id)
        
        # Procesar vehículos contados
        for track_id in counted_in_frame:
            if self._count_vehicle_safe(track_id):
                self._remove_track(track_id)

    def _count_vehicle_safe(self, track_id: int) -> bool:
        """Cuenta un vehículo con todas las verificaciones de seguridad"""
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        # Verificar que no haya sido contado antes
        if track.get('counted', False) or track_id in self.counted_tracks:
            return False
        
        # Verificar movimiento válido
        if not self.validate_movement(track_id, track['center']):
            if self.debug_mode:
                print(f"⚠️ Track {track_id} no tiene movimiento válido - ignorando")
            return False
        
        # Marcar como contado
        self.counted_tracks.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        
        # Registrar en recientemente contados
        self.recent_counted_vehicles.append((track_id, track['center'], self.frame_counter))
        
        # Actualizar contadores
        self.autos_lavados += 1
        self.last_counted_frame = self.frame_counter
        self.last_counted_id = track_id
        
        # Actualizar contador por tipo
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
            self.truck_count += 1  # Contar buses como camiones
            type_text = "AUTOBÚS"
        else:
            type_text = "VEHÍCULO"
        
        # Mostrar mensaje de éxito
        print(f"\n{'='*60}")
        print(f"🎉 {type_text} CONTADO!")
        print(f"   ID: {track_id}")
        print(f"   Frames en ROI: {track.get('frames_in_roi', 0)}")
        print(f"   Total: {self.autos_lavados} (C:{self.car_count}, T:{self.truck_count}, M:{self.motorcycle_count})")
        print(f"{'='*60}\n")
        
        # Guardar foto
        if hasattr(self, 'last_processed_frame'):
            self.save_exit_photo(self.last_processed_frame, track['class'], track_id)
        
        return True

    def _remove_track(self, track_id: int):
        """Elimina un track de forma segura"""
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.movement_history:
            del self.movement_history[track_id]

    def cleanup_stale_tracks(self):
        """Limpia tracks inactivos"""
        current_frame = self.frame_counter
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last = current_frame - track['last_seen']
            
            # Eliminar si está inactivo o si lleva mucho tiempo sin moverse
            if (frames_since_last > 30 or 
                (track.get('frames_out_roi', 0) > 50 and not track.get('has_been_inside', False))):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
            if self.debug_mode:
                print(f"🗑️ Track {track_id} eliminado (inactivo)")

    def match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Empareja detecciones con tracks existentes"""
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
        # Limpiar tracks antiguos
        self.cleanup_stale_tracks()
        
        # Filtrar detecciones dentro o cerca del ROI
        filtered_detections = []
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for det in detections:
            center = det['center']
            
            # Solo considerar vehículos dentro o cerca del ROI
            distance_to_roi = cv2.pointPolygonTest(roi_polygon_points, (int(center[0]), int(center[1])), True)
            
            # Aceptar si está dentro o a menos de 50 píxeles del ROI
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
        
        # Emparejar con tracks existentes
        if self.active_tracks and current_detections:
            matched_pairs, unmatched_detections, unmatched_tracks = self.match_detections_to_tracks(current_detections)
            
            # Actualizar tracks emparejados
            for track_id, det_idx in matched_pairs:
                det = current_detections[det_idx]
                self.active_tracks[track_id].update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1
                })
                self.track_history[track_id].append(det['center'])
            
            # Marcar tracks no emparejados como vistos
            for track_id in unmatched_tracks:
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['last_seen'] = self.frame_counter
            
            # Crear nuevos tracks para detecciones no emparejadas
            for det_idx in unmatched_detections:
                det = current_detections[det_idx]
                self._create_new_track(det)
        
        # Si no hay tracks existentes, crear nuevos
        elif current_detections:
            for det in current_detections:
                self._create_new_track(det)

    def _create_new_track(self, detection: Dict):
        """Crea un nuevo track a partir de una detección"""
        new_id = self.next_id
        self.next_id += 1
        
        center = detection['center']
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        is_inside = self.is_inside_polygon(center, roi_polygon_points)
        
        self.active_tracks[new_id] = {
            'class': detection['class'],
            'box': detection['box'],
            'center': center,
            'last_seen': self.frame_counter,
            'seen_frames': 1,
            'counted': False,
            'has_been_inside': is_inside,
            'frames_in_roi': 1 if is_inside else 0,
            'frames_out_roi': 0 if is_inside else 1,
            'entry_frame': self.frame_counter if is_inside else None
        }
        self.track_history[new_id].append(center)
        
        if self.debug_mode and is_inside:
            print(f"🆕 Nuevo track {new_id} ({detection['class']}) creado dentro del ROI")

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

    def save_exit_photo(self, frame: np.ndarray, vehicle_type: str, vehicle_id: int):
        """Guarda una foto del vehículo lavado"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{vehicle_type}_lavado_{vehicle_id}_{timestamp}.jpg"
        filepath = os.path.join(self.car_exit_dir, filename)
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                self.send_jarvis_wrapper(imagen_base64, 
                    f'{vehicle_type.capitalize()} lavado #{self.autos_lavados} - ID: {vehicle_id}')
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                logger.info(f"Vehículo lavado guardado: {filename}")
            return True
        except Exception as e:
            logger.error(f"No se pudo guardar la foto: {e}")
            return False

    def draw_detections(self, image: np.ndarray) -> np.ndarray:
        """Dibuja información en el frame"""
        CLR_CAR = (0, 165, 255)
        CLR_TRUCK = (255, 0, 0)
        CLR_MOTORCYCLE = (0, 255, 255)
        CLR_BUS = (255, 165, 0)
        CLR_ROI = (0, 255, 255)
        CLR_INSIDE = (0, 255, 0)
        CLR_OUTSIDE = (255, 165, 0)
        CLR_ENTERING = (255, 255, 0)
        CLR_EXITING = (255, 0, 255)
        
        # Dibujar ROI
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], (0, 255, 255, 100))
        cv2.addWeighted(roi_overlay, 0.3, image, 0.7, 0, image)
        cv2.polylines(image, [self.roi_polygon], isClosed=True, color=CLR_ROI, thickness=3)
        
        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)
        
        # Dibujar tracks
        for tid, obj in list(self.active_tracks.items()):
            if obj.get('counted', False):
                continue
            
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            vehicle_class = obj['class']
            
            # Determinar color según estado
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(obj['center'], roi_polygon_points)
            
            if vehicle_class == 'car':
                base_color = CLR_CAR
            elif vehicle_class == 'truck':
                base_color = CLR_TRUCK
            elif vehicle_class == 'motorcycle':
                base_color = CLR_MOTORCYCLE
            elif vehicle_class == 'bus':
                base_color = CLR_BUS
            else:
                base_color = (255, 255, 255)
            
            if is_inside:
                if obj.get('has_been_inside', False):
                    color = CLR_INSIDE
                    status = "DENTRO"
                else:
                    color = CLR_ENTERING
                    status = "ENTRANDO"
            else:
                if obj.get('has_been_inside', False):
                    color = CLR_EXITING
                    status = "SALIENDO"
                else:
                    color = CLR_OUTSIDE
                    status = "FUERA"
            
            # Dibujar bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar centro
            center_x, center_y = int(obj['center'][0]), int(obj['center'][1])
            cv2.circle(image, (center_x, center_y), 6, color, -1)
            cv2.circle(image, (center_x, center_y), 6, (255, 255, 255), 1)
            
            # Etiqueta
            label = f"ID:{tid} {vehicle_class.upper()} {status}"
            cv2.putText(image, label, (x1, max(y1 - 8, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Información adicional
            info_text = f"D:{obj.get('frames_in_roi', 0)} F:{obj.get('seen_frames', 0)}"
            cv2.putText(image, info_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Panel de estadísticas
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (500, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        cv2.putText(image, f"🚗 VEHÍCULOS LAVADOS: {self.autos_lavados}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        if self.last_counted_frame > 0:
            frames_since_last = self.frame_counter - self.last_counted_frame
            cv2.putText(image, f"Último: ID {self.last_counted_id} ({frames_since_last}f)", 
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(image, f"🚘 Carros: {self.car_count}", (15, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_CAR, 2)
        
        cv2.putText(image, f"🚚 Camiones: {self.truck_count}", (15, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_TRUCK, 2)
        
        cv2.putText(image, f"🏍️ Motos: {self.motorcycle_count}", (15, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_MOTORCYCLE, 2)
        
        # Contadores de estado
        inside_count = 0
        entering_count = 0
        exiting_count = 0
        
        for track in self.active_tracks.values():
            if track.get('counted', False):
                continue
            
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
            
            if is_inside:
                if track.get('has_been_inside', False):
                    inside_count += 1
                else:
                    entering_count += 1
            else:
                if track.get('has_been_inside', False):
                    exiting_count += 1
        
        status_text = f"ENTRANDO: {entering_count} | DENTRO: {inside_count} | SALIENDO: {exiting_count}"
        cv2.putText(image, status_text, (15, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(image, f"Frame: {self.frame_counter} | Tracks: {len(self.active_tracks)}", (15, 205), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image

    def process_frame(self, image: np.ndarray, roi=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa un frame y retorna la imagen procesada + metadatos
        """
        if self.model is None:
            raise RuntimeError("Modelo YOLO no inicializado")
        
        if roi is not None: 
            self.roi_polygon = np.array(roi, np.int32)
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        self.frame_counter += 1
        self.last_processed_frame = image.copy()
        
        try:
            # Detección con YOLO - Solo vehículos
            results = self.model.predict(
                image,
                imgsz=640,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                verbose=False,
                max_det=30
            )
            
            detections = []
            if results and results[0].boxes is not None:
                det = results[0].boxes
                boxes = det.xyxy.cpu().numpy()
                cls = det.cls.cpu().numpy()
                confs = det.conf.cpu().numpy() if det.conf is not None else [0.5] * len(boxes)
                
                for i in range(boxes.shape[0]):
                    cid = int(cls[i])
                    if cid == 2:
                        cname = 'car'
                    elif cid == 3:
                        cname = 'motorcycle'
                    elif cid == 5:
                        cname = 'bus'
                    elif cid == 7:
                        cname = 'truck'
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
                print(f"📊 Frame {self.frame_counter}: {len(detections)} detecciones")
            
            # Actualizar tracks
            self.update_tracks(detections)
            
            # Procesar lógica de entrada/salida
            self.process_entry_exit_logic()
            
            # Log periódico
            if self.frame_counter % 30 == 0:
                self.log_detection(self.frame_counter, flush=True)
                
                # Debug info
                entering_count = 0
                inside_count = 0
                exiting_count = 0
                
                for track in self.active_tracks.values():
                    if track.get('counted', False):
                        continue
                    
                    roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
                    is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
                    
                    if is_inside:
                        if track.get('has_been_inside', False):
                            inside_count += 1
                        else:
                            entering_count += 1
                    else:
                        if track.get('has_been_inside', False):
                            exiting_count += 1
                
                print(f"\n📈 Resumen Frame {self.frame_counter}:")
                print(f"   Entrando: {entering_count}, Dentro: {inside_count}, Saliendo: {exiting_count}")
                print(f"   Total lavados: {self.autos_lavados}")
                print(f"   C:{self.car_count}, T:{self.truck_count}, M:{self.motorcycle_count}")
            
            # Dibujar resultados
            processed_image = self.draw_detections(image.copy())
            
            # Preparar metadatos
            entering_count = 0
            inside_count = 0
            exiting_count = 0
            
            for track in self.active_tracks.values():
                if track.get('counted', False):
                    continue
                
                roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
                is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
                
                if is_inside:
                    if track.get('has_been_inside', False):
                        inside_count += 1
                    else:
                        entering_count += 1
                else:
                    if track.get('has_been_inside', False):
                        exiting_count += 1
            
            metadata = {
                'frame_number': self.frame_counter,
                'vehicles_detected': len(detections),
                'vehicles_washed': self.autos_lavados,
                'car_count': self.car_count,
                'truck_count': self.truck_count,
                'motorcycle_count': self.motorcycle_count,
                'active_tracks': len(self.active_tracks),
                'entering_vehicles': entering_count,
                'inside_vehicles': inside_count,
                'exiting_vehicles': exiting_count,
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
        """Establece una nueva región de interés (ROI)"""
        self.roi_polygon = np.array(roi_points, np.int32)
        print(f"✅ ROI actualizado a {len(roi_points)} puntos")

    def reset_counter(self):
        """Reinicia todos los contadores de vehículos"""
        self.autos_lavados = 0
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.counted_tracks.clear()
        self.recent_counted_vehicles.clear()
        self.vehicle_cooldown.clear()
        self.active_tracks.clear()
        self.track_history.clear()
        self.movement_history.clear()
        print("🔄 Contadores de vehículos reiniciados")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas actuales del procesador"""
        entering_count = 0
        inside_count = 0
        exiting_count = 0
        
        for track in self.active_tracks.values():
            if track.get('counted', False):
                continue
            
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
            
            if is_inside:
                if track.get('has_been_inside', False):
                    inside_count += 1
                else:
                    entering_count += 1
            else:
                if track.get('has_been_inside', False):
                    exiting_count += 1
        
        return {
            'total_vehicles_washed': self.autos_lavados,
            'car_count': self.car_count,
            'truck_count': self.truck_count,
            'motorcycle_count': self.motorcycle_count,
            'frame_counter': self.frame_counter,
            'active_tracks': len(self.active_tracks),
            'entering_vehicles': entering_count,
            'inside_vehicles': inside_count,
            'exiting_vehicles': exiting_count,
            'last_counted_id': self.last_counted_id,
            'last_counted_frame': self.last_counted_frame,
            'roi_points': self.roi_polygon.tolist()
        }


def create_vehicle_processor(**kwargs) -> VehicleProcessor:
    """Crea y retorna una instancia configurada de VehicleProcessor"""
    return VehicleProcessor(**kwargs)