import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from typing import Tuple, Dict, Any, Optional, List
import asyncio
import threading
import base64
import httpx
import json
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
                 roi_config_file: str = "roi_config.json",
                 log_file: str = "detection_log.txt",
                 car_exit_dir: str = "Car Exit",
                 image_quality: int = 60):
        """
        Inicializa el procesador de vehículos
        
        Args:
            model_path: Ruta al modelo YOLO
            confidence_threshold: Umbral de confianza para detecciones
            iou_threshold: Umbral IoU para NMS
            device: Dispositivo para inferencia ('auto', 'cpu', 'cuda:0')
            roi_config_file: Ruta al archivo de configuración del ROI
            log_file: Ruta al archivo de log de detecciones
            car_exit_dir: Directorio para guardar fotos de autos lavados
            image_quality: Calidad de la imagen para guardar (0-100)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.roi_config_file = roi_config_file
        self.log_file = log_file
        self.car_exit_dir = car_exit_dir
        self.image_quality = image_quality
        
        # Configuración de tracking (parámetros optimizados)
        self.max_frame_gap = 30
        self.min_consec_frames = 1
        self.frames_outside_to_remove = 5
        self.min_time_inside = 2
        
        # Estado interno
        self.frame_counter = 0
        self.autos_lavados = 0
        self.active_tracks = {}
        self.removed_tracks = set()
        self.next_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=15))
        self.counted_ids = set()
        
        # ROI por defecto (puede ser configurado)
        self.roi_polygon = np.array(DEFAULT_ROI, np.int32)
        
        # Cargar configuración del ROI si existe
        self.load_roi_config()
        
        # Configuración de compresión de imagen
        self.image_compression_level = 9
        
        # Buffer para logs
        self._log_buffer = []
        
        # Inicializar modelo
        self.model = None
        self.device = self._setup_device(device)
        self._initialize_model()
        
        # Crear directorios necesarios
        os.makedirs(self.car_exit_dir, exist_ok=True)
        self.setup_log_file()
    
    def _setup_device(self, device: str) -> str:
        """Configura el dispositivo de inferencia y muestra mensaje"""
        if device == "auto":
            if torch.cuda.is_available():
                cuda_device = "cuda:0"
                logger.info("✅ CUDA funcionando - Usando GPU")
                print("✅ CUDA funcionando - Usando GPU")
                return cuda_device
            else:
                logger.info("⚠️ CUDA no disponible - Usando CPU")
                print("⚠️ CUDA no disponible - Usando CPU")
                return "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(f"⚠️ {device} solicitado pero no disponible - Usando CPU")
                print(f"⚠️ {device} solicitado pero no disponible - Usando CPU")
                return "cpu"
            else:
                if device.startswith("cuda"):
                    logger.info(f"✅ CUDA funcionando - Usando {device}")
                    print(f"✅ CUDA funcionando - Usando {device}")
                else:
                    logger.info(f"✅ Usando CPU")
                    print(f"✅ Usando CPU")
                return device

    def _initialize_model(self):
        """Inicializa el modelo YOLO con configuración optimizada"""
        try:
            logger.info(f"Inicializando modelo YOLO en {self.device}")
            print(f"🚀 Inicializando modelo YOLO en {self.device}...")
            
            self.model = YOLO(self.model_path).to(self.device)
            
            # Warmup del modelo con configuración optimizada
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_input, 
                imgsz=320, 
                device=self.device,
                classes=[2, 7],  # car=2, truck=7
                verbose=False
            )
            
            device_type = "GPU" if self.device.startswith("cuda") else "CPU"
            logger.info(f"✅ Modelo YOLO inicializado correctamente en {device_type}")
            print(f"✅ Modelo YOLO inicializado correctamente en {device_type}")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando YOLO: {e}")
            print(f"❌ Error inicializando YOLO: {e}")
            raise
    
    def load_roi_config(self):
        """Carga la configuración del ROI desde archivo JSON"""
        if os.path.exists(self.roi_config_file):
            try:
                with open(self.roi_config_file, 'r') as f:
                    config = json.load(f)
                    self.roi_polygon = np.array(config.get('roi_polygon', []), np.int32)
                    logger.info(f"Configuración del ROI cargada desde {self.roi_config_file}")
            except Exception as e:
                logger.error(f"No se pudo cargar la configuración: {e}")
        else:
            logger.info("Usando configuración por defecto del ROI")
    
    def save_roi_config(self):
        """Guarda la configuración actual del ROI en archivo JSON"""
        try:
            config = {
                'roi_polygon': self.roi_polygon.tolist(),
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            with open(self.roi_config_file, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuración guardada en {self.roi_config_file}")
            return True
        except Exception as e:
            logger.error(f"No se pudo guardar la configuración: {e}")
            return False
    
    def setup_log_file(self):
        """Configura el archivo de log"""
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write("Timestamp,Frame,Autos_Lavados\n")
    
    def log_detection(self, frame_count: int, autos_lavados: int, flush: bool = False):
        """Registra una detección en el log"""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_buffer.append(f"{ts},{frame_count},{autos_lavados}")
        
        if flush or len(self._log_buffer) >= 60:
            with open(self.log_file, 'a', encoding="utf-8") as f:
                f.write("\n".join(self._log_buffer) + "\n")
            self._log_buffer.clear()
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calcula Intersection over Union entre dos bounding boxes"""
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
        """Verifica si un punto está dentro del polígono"""
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0
    
    def check_exit_direction(self, track_id: int) -> bool:
        """Verifica la dirección de salida del vehículo con algoritmo mejorado"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return True  # Si no hay historial suficiente, asumir que está saliendo
        
        history = list(self.track_history[track_id])
        # Tomar los últimos 5 puntos del historial
        recent_points = history[-min(5, len(history)):]
        
        if len(recent_points) < 2:
            return True
        
        # Calcular dirección general (eje Y - vertical)
        first_y = recent_points[0][1]
        last_y = recent_points[-1][1]
        
        # Si se está moviendo hacia abajo en el frame (aumenta Y), está saliendo
        y_movement = last_y - first_y
        
        # Umbral para considerar que está saliendo (positivo = hacia abajo)
        return y_movement > 5
    
    def match_detections_to_tracks(self, detections: list) -> tuple:
        """Empareja detecciones con tracks existentes - VERSIÓN CORREGIDA"""
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.active_tracks.keys())
        
        # Si no hay detecciones o tracks, retornar vacío
        if not detections or not self.active_tracks:
            return matched_pairs, unmatched_detections, unmatched_tracks
        
        # Crear matriz de costos IoU
        cost_matrix = []
        for det_idx, det in enumerate(detections):
            row = []
            for track_id in unmatched_tracks:
                track = self.active_tracks[track_id]
                iou = self.calculate_iou(det['box'], track['box'])
                row.append(iou)
            cost_matrix.append(row)
        
        # Emparejamiento greedy mejorado
        for det_idx in range(len(detections)):
            best_iou = 0.3  # Umbral mínimo
            best_track_idx = -1
            
            for track_idx, track_id in enumerate(unmatched_tracks):
                if cost_matrix[det_idx][track_idx] > best_iou:
                    best_iou = cost_matrix[det_idx][track_idx]
                    best_track_idx = track_idx
            
            if best_track_idx != -1:
                track_id = unmatched_tracks[best_track_idx]
                matched_pairs.append((track_id, det_idx))
                unmatched_tracks.pop(best_track_idx)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def cleanup_stale_tracks(self, frame_idx: int):
        """Limpieza agresiva de tracks antiguos o perdidos"""
        tracks_to_remove = []
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            # Condición 1: Vehículo fuera del área por demasiados frames
            if not is_inside:
                frames_outside = frame_idx - track.get('last_inside_frame', frame_idx)
                if frames_outside >= self.frames_outside_to_remove:
                    tracks_to_remove.append(track_id)
                    logger.info(f"Eliminando ID {track_id} - Fuera del área por {frames_outside} frames")
                    continue
            
            # Condición 2: No detectado por mucho tiempo
            gap = frame_idx - track['last_seen']
            if gap > self.max_frame_gap:
                tracks_to_remove.append(track_id)
                logger.info(f"Eliminando ID {track_id} - No detectado por {gap} frames")
                continue
        
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                if not self.active_tracks[track_id].get('counted_exit', False):
                    self.removed_tracks.add(track_id)
                del self.active_tracks[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
        
        return len(tracks_to_remove)
    
    def update_tracks(self, detections: list, frame_idx: int):
        """Actualiza los tracks con nuevas detecciones - VERSIÓN MEJORADA"""
        self.cleanup_stale_tracks(frame_idx)
        
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
            
            # Actualizar tracks emparejados
            for track_id, det_idx in matched_pairs:
                det = current_detections[det_idx]
                self.active_tracks[track_id].update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': frame_idx,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1,
                    'class': det['class']
                })
                self.track_history[track_id].append(det['center'])
            
            # Manejar tracks no emparejados
            for track_id in unmatched_tracks:
                self.active_tracks[track_id]['last_seen'] = frame_idx
            
            # Crear nuevos tracks para detecciones no emparejadas
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
                    'counted': False,
                    'counted_exit': False,
                    'first_detected': frame_idx,
                    'last_inside_frame': frame_idx
                }
                self.track_history[new_id].append(det['center'])
                
        elif current_detections:
            # Crear tracks para todas las detecciones si no hay tracks activos
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
                    'counted': False,
                    'counted_exit': False,
                    'first_detected': frame_idx,
                    'last_inside_frame': frame_idx
                }
                self.track_history[new_id].append(det['center'])
        
        return self.active_tracks
    
    def update_vehicle_state(self, frame_idx: int):
        """Actualiza el estado de los vehículos y cuenta lavados con lógica mejorada"""
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for tid, obj in list(self.active_tracks.items()):
            current_pos = obj['center']
            prev_state = obj.get('state') 
            confirmed = (obj['seen_frames'] >= self.min_consec_frames)
            
            if not confirmed:
                continue
                
            is_currently_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            new_state = 'inside' if is_currently_inside else 'outside'
            
            if new_state == 'inside':
                obj['last_inside_frame'] = frame_idx
            
            if prev_state is not None and prev_state != new_state:
                logger.info(f"🚗 ID {tid} cambió de {prev_state} a {new_state} en frame {frame_idx}")
                
                if prev_state == 'inside' and new_state == 'outside':
                    time_inside = frame_idx - obj.get('first_inside_frame', frame_idx)
                    is_exiting = self.check_exit_direction(tid)
                    
                    if (not obj.get('counted_exit', False) and 
                        time_inside >= self.min_time_inside and 
                        obj['class'] == 'car' and
                        is_exiting):
                        
                        self.autos_lavados += 1
                        obj['counted_exit'] = True
                        logger.info(f"🎉 AUTO LAVADO CONTADO - Total: {self.autos_lavados} (ID: {tid})")
                        
                        # Guardar foto del auto lavado
                        if hasattr(self, 'last_processed_frame'):
                            self.save_exit_photo(self.last_processed_frame, obj['class'], tid)
            
            if new_state == 'inside' and prev_state == 'outside':
                obj['first_inside_frame'] = frame_idx
            
            obj['state'] = new_state
    
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
        """Guarda una foto del auto lavado y la envía"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{vehicle_type}_lavado_{vehicle_id}_{timestamp}.jpg"
        filepath = os.path.join(self.car_exit_dir, filename)
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                self.send_jarvis_wrapper(imagen_base64, f'Auto lavado #{self.autos_lavados} - ID: {vehicle_id}')
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                logger.info(f"Auto lavado guardado: {filename}")
            return True
        except Exception as e:
            logger.error(f"No se pudo guardar la foto: {e}")
            return False
    
    def draw_detections(self, image: np.ndarray) -> np.ndarray:
        """Dibuja las detecciones y tracks en la imagen con información mejorada"""
        # Colores
        CLR_TRACK = (0, 165, 255)       # Naranja
        CLR_CONFIRMED = (0, 255, 0)     # Verde
        CLR_ROI = (0, 255, 255)         # Amarillo
        CLR_EXIT = (0, 0, 255)          # Rojo
        CLR_VERTEX = (255, 0, 0)        # Azul para vértices
        
        # Dibujar ROI
        cv2.polylines(image, [self.roi_polygon], isClosed=True, color=CLR_ROI, thickness=3)
        
        # Dibujar vértices del ROI
        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 6, CLR_VERTEX, -1)
        
        # Dibujar tracks activos
        for tid, obj in list(self.active_tracks.items()):
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            
            if obj.get('state') == 'inside':
                color = CLR_CONFIRMED
                state_text = "DENTRO"
            elif obj.get('counted_exit', False):
                color = CLR_EXIT
                state_text = "LAVADO"
            else:
                color = CLR_TRACK
                state_text = "FUERA"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            if tid in self.track_history:
                history = self.track_history[tid]
                for i in range(1, len(history)):
                    pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                    pt2 = (int(history[i][0]), int(history[i][1]))
                    cv2.line(image, pt1, pt2, color, 2)
            
            label = f"ID:{tid} {state_text}"
            cv2.putText(image, label, (x1, max(y1 - 8, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(image, (int(obj['center'][0]), int(obj['center'][1])), 4, color, -1)
            
            # Información de debug
            debug_text = f"({int(obj['center'][0])},{int(obj['center'][1])})"
            cv2.putText(image, debug_text, (x1, y1 - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dibujar HUD mejorado
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        cv2.putText(image, f"AUTOS LAVADOS: {self.autos_lavados}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        cv2.putText(image, f"Tracks activos: {len(self.active_tracks)}", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.putText(image, f"Frame: {self.frame_counter}", (15, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(image, f"Device: {self.device}", (15, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image
    
    def process_frame(self, image: np.ndarray,  roi) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa un frame y retorna la imagen procesada + metadatos
        
        Args:
            image: Frame de entrada (BGR format)
            
        Returns:
            tuple: (imagen_procesada, metadatos)
        """
        if self.model is None:
            raise RuntimeError("Modelo YOLO no inicializado")
        
        if not roi is None: self.roi_polygon = np.array(roi, np.int32)

        self.frame_counter += 1
        self.last_processed_frame = image.copy()  # Guardar para posibles fotos
        
        try:
            # Realizar detección con YOLO optimizado
            results = self.model.track(
                image,
                imgsz=640,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[2, 7],  # car=2, truck=7
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            detections = []
            if results and results[0].boxes is not None:
                det = results[0].boxes
                boxes = det.xyxy.cpu().numpy()
                cls = det.cls.cpu().numpy()
                confs = det.conf.cpu().numpy() if det.conf is not None else [0.5] * len(boxes)
                
                for i in range(boxes.shape[0]):
                    cid = int(cls[i])
                    cname = 'car' if cid == 2 else ('truck' if cid == 7 else str(cid))
                    boxg = boxes[i]
                    center = self.center_of(boxg)
                    
                    detections.append({
                        'class': cname,
                        'box': boxg,
                        'center': center,
                        'confidence': confs[i] if i < len(confs) else 0.5
                    })

            # Actualizar tracks
            active_tracks = self.update_tracks(detections, self.frame_counter)
            
            # Actualizar estado de vehículos
            self.update_vehicle_state(self.frame_counter)
            
            # Dibujar detecciones
            processed_image = self.draw_detections(image.copy())
            
            # Log del frame
            self.log_detection(self.frame_counter, self.autos_lavados, flush=(self.frame_counter % 60 == 0))
            
            # Preparar metadatos
            metadata = {
                'frame_number': self.frame_counter,
                'vehicles_detected': len(active_tracks),
                'vehicles_washed': self.autos_lavados,
                'active_tracks': len(self.active_tracks),
                'timestamp': self.frame_counter,
                'device': self.device,
                'total_tracks_created': self.next_id - 1,
                'roi_points': self.roi_polygon.tolist(),
                'detections_in_frame': len(detections)
            }
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error en procesamiento de frame: {e}")
            return image, {'error': str(e)}
    
    def set_roi(self, polygon_coords: np.ndarray):
        """
        Configura la región de interés (ROI)
        
        Args:
            polygon_coords: Array numpy con coordenadas del polígono [[x1,y1], [x2,y2], ...]
        """
        self.roi_polygon = polygon_coords.astype(np.int32)
        logger.info(f"ROI actualizado: {len(polygon_coords)} puntos")
        self.save_roi_config()
    
    def reset_counter(self):
        """Reinicia el contador de autos lavados"""
        self.autos_lavados = 0
        logger.info("Contador de autos lavados reiniciado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas actuales"""
        return {
            'total_washed': self.autos_lavados,
            'active_tracks': len(self.active_tracks),
            'frame_counter': self.frame_counter,
            'total_tracks_created': self.next_id - 1,
            'device': self.device,
            'roi_config_file': self.roi_config_file,
            'log_file': self.log_file,
            'car_exit_dir': self.car_exit_dir
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Retorna información de debug para troubleshooting"""
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        tracks_info = {}
        
        for tid, obj in self.active_tracks.items():
            current_pos = obj['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            frames_outside = self.frame_counter - obj.get('last_inside_frame', self.frame_counter)
            
            tracks_info[tid] = {
                'class': obj['class'],
                'state': obj.get('state', 'unknown'),
                'position': current_pos,
                'is_inside_roi': is_inside,
                'frames_outside': frames_outside,
                'seen_frames': obj['seen_frames'],
                'counted_exit': obj.get('counted_exit', False)
            }
        
        return {
            'frame_counter': self.frame_counter,
            'active_tracks_count': len(self.active_tracks),
            'tracks_info': tracks_info,
            'roi_polygon': self.roi_polygon.tolist(),
            'model_device': self.device,
            'autos_lavados': self.autos_lavados
        }
    
    def debug_tracking_state(self):
        """Muestra el estado actual del tracking para debugging"""
        print(f"\n=== DEBUG TRACKING STATE ===")
        print(f"Frame: {self.frame_counter}")
        print(f"Active tracks: {len(self.active_tracks)}")
        print(f"Autos lavados: {self.autos_lavados}")
        
        for track_id, track in self.active_tracks.items():
            print(f"ID {track_id}: state={track.get('state')}, "
                  f"seen_frames={track['seen_frames']}, "
                  f"counted_exit={track.get('counted_exit', False)}, "
                  f"position={track['center']}")
        
        print("============================\n")

    def reset_all(self):
        """Reinicia completamente el estado del procesador"""
        self.frame_counter = 0
        self.autos_lavados = 0
        self.active_tracks.clear()
        self.removed_tracks.clear()
        self.next_id = 1
        self.track_history.clear()
        self.counted_ids.clear()
        self._log_buffer.clear()
        
        logger.info("Estado del procesador reiniciado completamente")

# Función de utilidad para crear instancia del procesador
def create_vehicle_processor(**kwargs) -> VehicleProcessor:
    """Crea y retorna una instancia configurada de VehicleProcessor"""
    return VehicleProcessor(**kwargs)