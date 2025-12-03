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
    """Procesador de vehículos desacoplado para uso en endpoints/sockets"""
    
    def __init__(self, 
                 model_path: str = "yolo11n.pt",
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.4,
                 device: str = "auto",
                 log_file: str = "output/detection_log.txt",
                 car_exit_dir: str = "output/Car_Exit",
                 image_quality: int = 60):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.log_file = log_file
        self.car_exit_dir = car_exit_dir
        self.image_quality = image_quality
        
        # Configuración de tracking - AJUSTADO PARA MEJOR DETECCIÓN DE SALIDA
        self.max_frame_gap = 30
        self.min_consec_frames = 2  # Reducido para mayor sensibilidad
        self.min_time_inside = 3    # Reducido para mayor sensibilidad
        
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
        
        self.image_compression_level = 9
        self._log_buffer = []
        
        # Estado para debug
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.debug_mode = True
        
        self.model = None
        self.device = self._setup_device(device)
        self._initialize_model()
        
        os.makedirs(self.car_exit_dir, exist_ok=True)
        self.setup_log_file()
    
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                print("✅ CUDA funcionando - Usando GPU")
                return "cuda:0"
            else:
                print("⚠️ CUDA no disponible - Usando CPU")
                return "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                print(f"⚠️ {device} solicitado pero no disponible - Usando CPU")
                return "cpu"
            else:
                print(f"✅ Usando {device}")
                return device

    def _initialize_model(self):
        try:
            print(f"🚀 Inicializando modelo YOLO en {self.device}...")
            self.model = YOLO(self.model_path).to(self.device)
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_input, imgsz=320, device=self.device,
                classes=[2, 3, 7], verbose=False
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

    def count_vehicle_exit(self, track_id: int, frame_idx: int) -> bool:
        """Cuenta un vehículo cuando sale del ROI - VERSIÓN MEJORADA"""
        if track_id not in self.active_tracks:
            if self.debug_mode:
                print(f"❌ Track {track_id} no encontrado para conteo")
            return False
        
        track = self.active_tracks[track_id]
        
        if self.debug_mode:
            print(f"\n🔍 ANALIZANDO CONTEO para Track {track_id}:")
            print(f"   Clase: {track['class']}")
            print(f"   Estado: {track.get('state', 'unknown')}")
            print(f"   Ya contado?: {track.get('counted_exit', False)}")
            print(f"   Frames vistos: {track['seen_frames']}")
            print(f"   Frames dentro ROI: {track.get('frames_inside_roi', 0)}")
            print(f"   Entry frame: {track.get('entry_frame', 'N/A')}")
        
        if track.get('counted_exit', False):
            if self.debug_mode:
                print(f"⚠️ Track {track_id} ya fue contado anteriormente")
            return False
        
        if track['class'] not in ['car', 'truck', 'motorcycle']:
            if self.debug_mode:
                print(f"⚠️ Track {track_id} no es vehículo válido: {track['class']}")
            return False
        
        # Reducir el mínimo de frames consecutivos requeridos para mayor sensibilidad
        if track['seen_frames'] < 2:  # Reducido para mejor detección
            if self.debug_mode:
                print(f"⚠️ Track {track_id} no tiene frames suficientes: {track['seen_frames']} < 2")
            return False
        
        if 'entry_frame' not in track:
            if self.debug_mode:
                print(f"⚠️ Track {track_id} nunca entró al ROI (entry_frame no existe)")
            return False
        
        frames_inside = track.get('frames_inside_roi', 0)
        # Reducir el tiempo mínimo dentro del ROI para mayor sensibilidad
        if frames_inside < 3:  # Reducido para mejor detección
            if self.debug_mode:
                print(f"⚠️ Track {track_id} no estuvo suficiente tiempo en ROI: {frames_inside} < 3")
            return False
        
        self.autos_lavados += 1
        self.last_counted_frame = frame_idx
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
        else:
            type_text = "VEHÍCULO"
        
        track['counted_exit'] = True
        track['exit_frame'] = frame_idx
        
        print(f"\n{'='*60}")
        print(f"🎉🎉🎉 {type_text} CONTADO EXITOSAMENTE!")
        print(f"   ID: {track_id}")
        print(f"   Tipo: {track['class']}")
        print(f"   Frames dentro ROI: {frames_inside}")
        print(f"   Frames totales: {track['seen_frames']}")
        print(f"   Total vehículos: {self.autos_lavados}")
        print(f"   Estadísticas: Carros={self.car_count}, Camiones={self.truck_count}, Motos={self.motorcycle_count}")
        print(f"{'='*60}\n")
        
        if hasattr(self, 'last_processed_frame'):
            self.save_exit_photo(self.last_processed_frame, track['class'], track_id)
        
        return True

    def cleanup_stale_tracks(self, frame_idx: int) -> int:
        """Elimina tracks que no se han visto en varios frames - VERSIÓN MENOS AGRESIVA"""
        to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last_seen = frame_idx - track['last_seen']
            
            if track.get('counted_exit', False):
                if frames_since_last_seen > 5:
                    to_remove.append(track_id)
                continue
            
            # Solo eliminar tracks que nunca entraron al ROI y llevan mucho tiempo
            if track.get('entry_frame') is None and frames_since_last_seen > self.max_frame_gap * 2:
                to_remove.append(track_id)
            # Eliminar tracks que llevan mucho tiempo sin verse
            elif frames_since_last_seen > self.max_frame_gap * 3:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
            if self.debug_mode:
                print(f"🗑️ Track {track_id} eliminado por inactividad")
        
        return len(to_remove)
    
    def match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Empareja detecciones con tracks activos usando IOU"""
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
    
    def update_tracks(self, detections: list, frame_idx: int):
        """Actualiza los tracks con nuevas detecciones"""
        removed_count = self.cleanup_stale_tracks(frame_idx)
        if removed_count > 0 and self.debug_mode:
            print(f"🔧 Limpieza: {removed_count} tracks eliminados")
        
        current_detections = []
        for det in detections:
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
                    'last_seen': frame_idx,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1
                })
                self.track_history[track_id].append(det['center'])
            
            for track_id in unmatched_tracks:
                self.active_tracks[track_id]['last_seen'] = frame_idx
            
            for det_idx in unmatched_detections:
                det = current_detections[det_idx]
                new_id = self.next_id
                self.next_id += 1
                
                self.active_tracks[new_id] = {
                    'class': det['class'],
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': frame_idx,
                    'seen_frames': 1,
                    'state': 'outside',
                    'counted_exit': False,
                    'frames_inside_roi': 0,
                    'consecutive_inside': 0,
                    'consecutive_outside': 0,
                    'entry_frame': None,
                    'exit_frame': None
                }
                self.track_history[new_id].append(det['center'])
                
        elif current_detections:
            for det in current_detections:
                new_id = self.next_id
                self.next_id += 1
                
                self.active_tracks[new_id] = {
                    'class': det['class'],
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': frame_idx,
                    'seen_frames': 1,
                    'state': 'outside',
                    'counted_exit': False,
                    'frames_inside_roi': 0,
                    'consecutive_inside': 0,
                    'consecutive_outside': 0,
                    'entry_frame': None,
                    'exit_frame': None
                }
                self.track_history[new_id].append(det['center'])
        
        return self.active_tracks

    def update_vehicle_state(self, frame_idx: int):
        """Actualiza el estado de los vehículos - VERSIÓN CORREGIDA PARA MEJOR DETECCIÓN DE SALIDA"""
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        exit_candidates = []
        
        for tid, obj in list(self.active_tracks.items()):
            if obj.get('counted_exit', False):
                continue
            
            current_pos = obj['center']
            is_currently_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            # Guardar el estado anterior
            prev_state = obj.get('state', 'outside')
            
            # Actualizar contadores consecutivos
            if is_currently_inside:
                obj['consecutive_inside'] = obj.get('consecutive_inside', 0) + 1
                obj['consecutive_outside'] = 0
                
                # Actualizar frames dentro ROI
                obj['frames_inside_roi'] = obj.get('frames_inside_roi', 0) + 1
                
                # Determinar si acaba de entrar
                if prev_state == 'outside' and obj['consecutive_inside'] >= 2:
                    obj['state'] = 'inside'
                    if obj.get('entry_frame') is None:
                        obj['entry_frame'] = frame_idx
                    if self.debug_mode:
                        print(f"🚗 Track {tid} ({obj['class']}) ENTRÓ al ROI en frame {frame_idx}")
                        print(f"   Centro: {current_pos}")
                        print(f"   Consecutive inside: {obj['consecutive_inside']}")
                elif prev_state == 'inside':
                    obj['state'] = 'inside'
            else:
                obj['consecutive_outside'] = obj.get('consecutive_outside', 0) + 1
                obj['consecutive_inside'] = 0
                
                # Determinar si acaba de salir
                if prev_state == 'inside' and obj['consecutive_outside'] >= 2:  # Aumentado a 2 para mayor estabilidad
                    obj['state'] = 'outside'
                    # Verificar si debe ser contado
                    frames_inside = obj.get('frames_inside_roi', 0)
                    if frames_inside >= self.min_time_inside:
                        exit_candidates.append(tid)
                        if self.debug_mode:
                            print(f"🚪 Track {tid} ({obj['class']}) SALIÓ del ROI en frame {frame_idx}")
                            print(f"   Frames dentro: {frames_inside}")
                            print(f"   Centro: {current_pos}")
                            print(f"   Consecutive outside: {obj['consecutive_outside']}")
                elif prev_state == 'outside':
                    obj['state'] = 'outside'
            
            # Debug: mostrar cambio de estado
            if self.debug_mode and prev_state != obj['state']:
                print(f"🔄 Track {tid} cambió de {prev_state} a {obj['state']}")
        
        # Procesar vehículos que salieron
        for tid in exit_candidates:
            if tid in self.active_tracks:
                if self.count_vehicle_exit(tid, frame_idx):
                    # Marcar para eliminación inmediata
                    self.active_tracks[tid]['marked_for_removal'] = True
                elif self.debug_mode:
                    print(f"❌ Conteo FALLIDO para track {tid}")
        
        # Eliminar tracks contados inmediatamente
        for tid in list(self.active_tracks.keys()):
            if self.active_tracks[tid].get('marked_for_removal', False):
                del self.active_tracks[tid]
                if tid in self.track_history:
                    del self.track_history[tid]
                if self.debug_mode:
                    print(f"🗑️ Track {tid} eliminado después de conteo")

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
        """Guarda una foto del vehículo lavado y la envía"""
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
        """Dibuja SOLO tracks que están dentro del ROI"""
        CLR_CAR = (0, 165, 255)
        CLR_TRUCK = (255, 0, 0)
        CLR_MOTORCYCLE = (0, 255, 255)
        CLR_ROI = (0, 255, 255)
        CLR_VERTEX = (255, 0, 0)
        CLR_INSIDE = (0, 255, 0)
        CLR_COUNTED = (255, 0, 255)
        
        # Dibujar ROI
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], (0, 255, 255, 100))
        cv2.addWeighted(roi_overlay, 0.3, image, 0.7, 0, image)
        cv2.polylines(image, [self.roi_polygon], isClosed=True, color=CLR_ROI, thickness=3)
        
        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 8, CLR_VERTEX, -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)
        
        # SOLO dibujar tracks que están DENTRO del ROI
        for tid, obj in list(self.active_tracks.items()):
            if obj.get('state') != 'inside':
                continue  # Solo dibujar tracks dentro
            
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            vehicle_class = obj['class']
            frames_inside = obj.get('frames_inside_roi', 0)
            consecutive_inside = obj.get('consecutive_inside', 0)
            
            if vehicle_class == 'car':
                color = CLR_CAR
                type_text = "CAR"
            elif vehicle_class == 'truck':
                color = CLR_TRUCK
                type_text = "TRUCK"
            elif vehicle_class == 'motorcycle':
                color = CLR_MOTORCYCLE
                type_text = "MOTO"
            else:
                color = (128, 128, 128)
                type_text = vehicle_class.upper()
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            cv2.circle(image, (int(obj['center'][0]), int(obj['center'][1])), 6, color, -1)
            cv2.circle(image, (int(obj['center'][0]), int(obj['center'][1])), 6, (255, 255, 255), 1)
            
            label = f"ID:{tid} {type_text} DENTRO"
            cv2.putText(image, label, (x1, max(y1 - 8, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            time_text = f"D:{frames_inside}f C:{consecutive_inside}"
            cv2.putText(image, time_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Dibujar HUD con estadísticas
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (500, 200), (0, 0, 0), -1)
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
        
        inside_count = len([t for t in self.active_tracks.values() if t.get('state') == 'inside'])
        cv2.putText(image, f"Tracks DENTRO: {inside_count}", (15, 185), 
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
            # Realizar detección con YOLO
            results = self.model.track(
                image,
                imgsz=640,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[2, 3, 7],
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                max_det=50
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
                    elif cid == 7:
                        cname = 'truck'
                    else:
                        continue
                    
                    boxg = boxes[i]
                    center = self.center_of(boxg)
                    
                    # Solo procesar detecciones que están DENTRO del ROI
                    roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
                    if self.is_inside_polygon(center, roi_polygon_points):
                        detections.append({
                            'class': cname,
                            'box': boxg,
                            'center': center,
                            'confidence': confs[i] if i < len(confs) else 0.5
                        })
            
            if self.debug_mode and detections:
                print(f"📊 Frame {self.frame_counter}: {len(detections)} detecciones DENTRO del ROI")
            
            # Actualizar tracks
            active_tracks = self.update_tracks(detections, self.frame_counter)
            
            # Actualizar estado de vehículos
            self.update_vehicle_state(self.frame_counter)
            
            # Verificación adicional: si un track lleva mucho tiempo dentro, forzar conteo
            for tid, obj in list(self.active_tracks.items()):
                if obj.get('state') == 'inside' and not obj.get('counted_exit', False):
                    frames_inside = obj.get('frames_inside_roi', 0)
                    if frames_inside > 30:  # Si lleva más de 30 frames dentro, forzar conteo
                        if self.debug_mode:
                            print(f"⚠️ Track {tid} lleva mucho tiempo dentro ({frames_inside}f), forzando conteo")
                        self.count_vehicle_exit(tid, self.frame_counter)
            
            # Debug periódico
            if self.debug_mode and self.frame_counter % 15 == 0:
                self.debug_track_states()
            
            # Dibujar detecciones
            processed_image = self.draw_detections(image.copy())
            
            # Log del frame
            self.log_detection(self.frame_counter, flush=(self.frame_counter % 30 == 0))
            
            # Preparar metadatos
            inside_tracks = [tid for tid, obj in self.active_tracks.items() 
                           if obj.get('state') == 'inside']
            
            metadata = {
                'frame_number': self.frame_counter,
                'vehicles_detected': len(detections),
                'vehicles_washed': self.autos_lavados,
                'car_count': self.car_count,
                'truck_count': self.truck_count,
                'motorcycle_count': self.motorcycle_count,
                'active_tracks': len(self.active_tracks),
                'inside_tracks': len(inside_tracks),
                'last_counted_id': self.last_counted_id,
                'last_counted_frame': self.last_counted_frame,
                'debug_info': {
                    'min_time_inside': self.min_time_inside,
                    'min_consec_frames': self.min_consec_frames
                }
            }
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error en procesamiento de frame: {e}")
            import traceback
            traceback.print_exc()
            return image, {'error': str(e)}
    
    def debug_track_states(self):
        """Muestra el estado actual de todos los tracks"""
        print(f"\n📋 ESTADO DE TRACKS - Frame {self.frame_counter}")
        print(f"{'='*60}")
        for tid, obj in sorted(self.active_tracks.items()):
            state = obj.get('state', 'unknown')
            counted = obj.get('counted_exit', False)
            frames_inside = obj.get('frames_inside_roi', 0)
            entry_frame = obj.get('entry_frame', 'N/A')
            print(f"Track {tid}: {obj['class']} | Estado: {state} | Contado: {counted} | Frames dentro: {frames_inside} | Entry: {entry_frame}")
        print(f"{'='*60}\n")
    
    def debug_tracking_info(self):
        """Muestra información de debug sobre el tracking"""
        print(f"\n{'='*60}")
        print(f"📊 DEBUG TRACKING INFO - Frame: {self.frame_counter}")
        print(f"{'='*60}")
        
        inside_count = 0
        for track in self.active_tracks.values():
            if track.get('state') == 'inside':
                inside_count += 1
        
        print(f"🔢 Tracks activos: {len(self.active_tracks)}")
        print(f"📥 Tracks dentro ROI: {inside_count}")
        print(f"🚗 Total vehículos lavados: {self.autos_lavados}")
        print(f"📈 Estadísticas: Carros={self.car_count}, Camiones={self.truck_count}, Motos={self.motorcycle_count}")
        
        if inside_count > 0:
            print(f"\n📋 DETALLE DE TRACKS DENTRO DEL ROI:")
            for tid, track in sorted(self.active_tracks.items()):
                if track.get('state') == 'inside':
                    frames_inside = track.get('frames_inside_roi', 0)
                    consecutive_inside = track.get('consecutive_inside', 0)
                    print(f"  ID {tid} ({track['class']}):")
                    print(f"    Frames dentro: {frames_inside}")
                    print(f"    Cons. dentro: {consecutive_inside}")
                    print(f"    Frames totales: {track['seen_frames']}")
                    print(f"    Entry frame: {track.get('entry_frame', 'N/A')}")
        
        print(f"{'='*60}\n")

    def set_roi(self, roi_points: List[Tuple[int, int]]):
        """Establece una nueva región de interés (ROI)"""
        self.roi_polygon = np.array(roi_points, np.int32)
        print(f"✅ ROI actualizado a {len(roi_points)} puntos")
        print(f"📍 ROI points: {roi_points}")

    def reset_counter(self):
        """Reinicia todos los contadores de vehículos"""
        self.autos_lavados = 0
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        self.last_counted_frame = 0
        self.last_counted_id = 0
        print("🔄 Contadores de vehículos reiniciados")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas actuales del procesador"""
        inside_count = len([t for t in self.active_tracks.values() if t.get('state') == 'inside'])
        
        return {
            'total_vehicles_washed': self.autos_lavados,
            'car_count': self.car_count,
            'truck_count': self.truck_count,
            'motorcycle_count': self.motorcycle_count,
            'frame_counter': self.frame_counter,
            'active_tracks': len(self.active_tracks),
            'inside_tracks': inside_count,
            'last_counted_id': self.last_counted_id,
            'last_counted_frame': self.last_counted_frame,
            'roi_points': self.roi_polygon.tolist()
        }

    def force_count_vehicle(self, track_id: int):
        """Fuerza el conteo de un vehículo específico (para debugging)"""
        if track_id in self.active_tracks:
            print(f"🔧 FORZANDO CONTEO MANUAL del track {track_id}")
            if self.count_vehicle_exit(track_id, self.frame_counter):
                print(f"✅ Conteo forzado EXITOSO para track {track_id}")
            else:
                print(f"❌ Conteo forzado FALLIDO para track {track_id}")
        else:
            print(f"❌ Track {track_id} no encontrado en active_tracks")

    def toggle_debug_mode(self):
        """Activa/desactiva el modo debug"""
        self.debug_mode = not self.debug_mode
        print(f"🔧 Modo debug {'ACTIVADO' if self.debug_mode else 'DESACTIVADO'}")


def create_vehicle_processor(**kwargs) -> VehicleProcessor:
    """Crea y retorna una instancia configurada de VehicleProcessor"""
    return VehicleProcessor(**kwargs)