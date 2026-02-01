import cv2
import numpy as np
import time
import base64
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple

class BasePerimeter:
    def __init__(self, client_id: str, model_path: str, device: str = 'cpu'):
        self.client_id = client_id
        self.model_path = model_path
        self.device = device
        
        # Inicializar modelo
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.model = None

        # Estado del sistema
        self.frame_counter = 0
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.active_tracks = {}  # {track_id: {'class_id': int, 'centroid': (x,y), 'in_roi': bool, ...}}
        
        # Contadores acumulativos
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        self.person_count_inside = 0  # Personas que han entrado (acumulativo)
        
        # Contadores de estado actual
        self.vehiculos_en_area = 0
        self.personas_en_area = 0
        self.vehicles_inside = 0  # Vehículos actualmente dentro del ROI
        
        self.last_counted_id = 0
        self.last_counted_frame = 0
        
        # Configuración de zonas
        self.roi_polygon = None
        self.door_polygon = None
        self.door_direction = None
        self.roi_active = False
        self.door_active = False
        self.door_direction_active = False # Nueva bandera
        
        # Alertas del frame actual
        self.current_alerts = []
        
        # Mapeo de clases (COCO default para vehículos y personas)
        self.class_names = {
            0: 'person', 
            1: 'bicycle',
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck',
            16: 'dog'
        }
        
        self.class_translations = {
            'person': 'Persona',
            'car': 'Carro',
            'truck': 'Camion',
            'motorcycle': 'Motocicleta',
            'bus': 'Autobus',
            'bicycle': 'Bicicleta',
            'dog': 'Perro'
        }

        # Configuración de track
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5

    def process_frame(self, 
                      image: np.ndarray, 
                      roi: Any = None,
                      activate_roi: bool = False, 
                      door_roi: Any = None,
                      door_activate: bool = False, 
                      door_direction: Any = None,
                      door_direction_activate: bool = False,
                      camera_id: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        self.frame_counter += 1
        self.current_alerts = [] # Limpiar alertas del frame anterior
        self.roi_active = activate_roi
        self.door_active = door_activate
        self.door_direction_active = door_direction_activate
        

        # 1. Parsear Geometrías
        self._parse_geometries(roi, door_roi, door_direction)

        # 2. Inferencia y Tracking
        if self.model is None:
            return image, self._build_metadata(0, 0, [])

        # Filtrar clases de interés (personas + vehículos + perros)
        results = self.model.track(
            image, 
            persist=True, 
            classes=[0, 1, 2, 3, 5, 7, 16], 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        vehicles_detected = 0
        persons_detected = 0
        current_frame_tracks = set()

        # 3. Procesar Detecciones
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                current_frame_tracks.add(track_id)
                class_name = self.class_names.get(cls_id, 'unknown')
                
                # Contar detecciones en este frame
                if class_name == 'person':
                    persons_detected += 1
                elif class_name in ['car', 'truck', 'motorcycle', 'bus', 'bicycle']:
                    vehicles_detected += 1

                # Calcular centroide
                centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                
                # Actualizar historial
                self.track_history[track_id].append(centroid)
                
                # Lógica de estados (ROI y Puerta)
                self._update_track_state(track_id, cls_id, centroid, image, box)

                # Dibujar bounding box y label
                self._draw_detection(image, box, track_id, cls_id, conf)

        # 4. Limpieza de tracks perdidos
        self._cleanup_tracks(current_frame_tracks)
        
        # 5. Calcular totales actuales en área
        self._update_area_counters()

        # 6. Dibujar zonas
        self._draw_zones(image)

        # 7. Construir Metadata
        metadata = self._build_metadata(vehicles_detected, persons_detected, results[0].boxes if results else None)

        return image, metadata
    


    def _parse_geometries(self, roi, door_roi, door_direction):
        """Convierte las entradas de listas/diccionarios a numpy arrays"""
        # Parse ROI even if not active, for visualization
        self.roi_polygon = self._to_numpy_poly(roi)
        # Parse Door ROI even if not active, for visualization
        self.door_polygon = self._to_numpy_poly(door_roi)
        
        self.door_direction = None
        # Parse Door Direction even if not active, for visualization
        if door_direction is not None:
             # Asumimos que door_direction es una lista de 2 puntos [[x1,y1], [x2,y2]]
             try:
                 pts = np.array(door_direction, dtype=np.int32)
                 if pts.shape == (2, 2):
                     self.door_direction = pts
             except Exception:
                 pass

    def _to_numpy_poly(self, points):
        if not points:
            return None
        try:
            # Maneja lista de dicts [{'x':.., 'y':..}] o lista de listas [[x,y]]
            if isinstance(points, list):
                if len(points) > 0 and isinstance(points[0], dict):
                    pts = [[p['x'], p['y']] for p in points]
                else:
                    pts = points
                return np.array(pts, np.int32).reshape((-1, 1, 2))
        except Exception:
            return None
        return None

    def _update_track_state(self, track_id, cls_id, centroid, image, box):
        class_name = self.class_names.get(cls_id, 'unknown')
        
        # Inicializar track si es nuevo
        if track_id not in self.active_tracks:
            self.active_tracks[track_id] = {
                'class_id': cls_id,
                'in_roi': False,
                'in_door': False,
                'history': deque(maxlen=30),
                'processed_entry': False, # Flag para evitar múltiples alertas
                'processed_exit': False
            }
        
        track_data = self.active_tracks[track_id]
        prev_in_roi = track_data['in_roi']
        is_in_roi = False
        
        # --- Lógica ROI Principal ---
        if self.roi_polygon is not None and self.roi_active:
            # pointPolygonTest: >0 inside, <0 outside, =0 edge
            dist = cv2.pointPolygonTest(self.roi_polygon, (float(centroid[0]), float(centroid[1])), False)
            is_in_roi = dist >= 0
            
            track_data['in_roi'] = is_in_roi
            
            # Detectar cambios de estado
            if not prev_in_roi and is_in_roi:
                # Entrada
                self._trigger_alert('Entrada', class_name, image, box)
                self._increment_counters(class_name, 'entry')
                
            elif prev_in_roi and not is_in_roi:
                # Salida
                self._trigger_alert('Salida', class_name, image, box)
                # Opcional: decrementar o manejar lógica de salida
                
        # --- Lógica ROI Puerta y Dirección ---
        if self.door_polygon is not None and self.door_active:
            dist_door = cv2.pointPolygonTest(self.door_polygon, (float(centroid[0]), float(centroid[1])), False)
            is_in_door = dist_door >= 0

            prev_in_door = track_data.get('in_door', False)
            track_data['in_door'] = is_in_door

            direction_label = None

            if self.door_direction is not None and self.door_direction_active and len(self.track_history[track_id]) > 1:
                prev_point = self.track_history[track_id][-2]
                curr_point = centroid
                move_vec = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
                dir_vec = (self.door_direction[1][0] - self.door_direction[0][0],
                           self.door_direction[1][1] - self.door_direction[0][1])
                dot = move_vec[0] * dir_vec[0] + move_vec[1] * dir_vec[1]


               
                if is_in_door:
                    # Dentro de la puerta: usa el sentido de movimiento
                    if dot > 0:
                        direction_label = "Entrada Puerta"
                    elif dot < 0:
                        direction_label = "Salida Puerta"
                else:
                    # Salió del polígono: si viene de dentro, decide según el último movimiento
                    if prev_in_door:
                        if dot > 0:
                            direction_label = "Entrada Puerta"
                        elif dot < 0:
                            direction_label = "Salida Puerta"
                        # dot == 0 se resolverá con fallback de transición más abajo

            # Fallback si no hay línea de dirección o dot = 0
            if direction_label is None:
                if is_in_door and not prev_in_door:
                    direction_label = "Entrada Puerta"
                elif (not is_in_door) and prev_in_door:
                    direction_label = "Salida Puerta"

            if direction_label is not None:
                current_time = time.time()
                last_alert = track_data.get('last_door_alert', 0)
                if current_time - last_alert > 2.0:  # Cooldown 2 segundos
                    self._trigger_alert(direction_label, class_name, image, box)
                    track_data['last_door_alert'] = current_time


    def _check_line_crossing(self, p1, p2, l1, l2):
        """Verifica si el segmento p1-p2 intercepta el segmento l1-l2"""
        # Simple implementación usando ccw
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)

    def _cross_product(self, A, B, P):
        """Producto cruz 2D para determinar lado de la línea AB donde está P"""
        return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])

    def _trigger_alert(self, event_type, class_name, image, box):
        # Recortar objeto para la alerta
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((10, 10, 3), np.uint8)

        # Codificar imagen recortada
        _, buffer = cv2.imencode('.jpg', crop)
        crop_base64 = base64.b64encode(buffer).decode('utf-8') if buffer is not None else ""

        # Imagen completa con el bbox resaltado en amarillo
        full_img = image.copy()
        cv2.rectangle(full_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(full_img, class_name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _, full_buf = cv2.imencode('.jpg', full_img)
        full_base64 = base64.b64encode(full_buf).decode('utf-8') if full_buf is not None else ""
        
        translated_class = self.class_translations.get(class_name, class_name)
        
        alert = {
            'event_type': event_type, # 'Entrada', 'Salida', 'Cruce Puerta'
            'class_name': translated_class,
            'timestamp': time.time(),
            'crop_image': crop_base64,
            'image_base64': full_base64,
            'description': self._format_description(event_type, translated_class)
        }
        self.current_alerts.append(alert)

    def _format_description(self, event_type: str, translated_class: str) -> str:
        """Devuelve la descripción de alerta según el evento"""
        if event_type == 'Entrada':
            return f"Entrada de {translated_class} en el perímetro"
        if event_type == 'Salida':
            return f"Salida de {translated_class} del perímetro"
        if event_type == 'Entrada Puerta':
            return f"Entrada de {translated_class} en puerta del perímetro"
        if event_type == 'Salida Puerta':
            return f"Salida de {translated_class} en puerta del perímetro"
        return f"{translated_class} {event_type.lower()}"

    def _increment_counters(self, class_name, event_type):
        if event_type == 'entry':
            if class_name == 'person':
                self.person_count_inside += 1
            if class_name == 'car':
                self.car_count += 1
            elif class_name == 'truck':
                self.truck_count += 1
            elif class_name == 'motorcycle':
                self.motorcycle_count += 1
                
        # Guardar IDs contados recientemente
        self.last_counted_frame = self.frame_counter
        # Nota: Idealmente guardaríamos el ID del track que causó el conteo
        
    def _update_area_counters(self):
        """Recalcula cuántos objetos hay actualmente dentro del ROI"""
        self.personas_en_area = sum(1 for t in self.active_tracks.values() 
                                   if t['in_roi'] and self.class_names.get(t['class_id']) == 'person')
        self.vehicles_inside = sum(1 for t in self.active_tracks.values() 
                                  if t['in_roi'] and self.class_names.get(t['class_id']) in ['car', 'truck', 'motorcycle', 'bus'])
        
        # Actualizar acumulativo de vehículos en area (nombre heredado que a veces confunde con 'vehicles_inside')
        self.vehiculos_en_area = self.vehicles_inside 

    def _cleanup_tracks(self, current_frame_track_ids):
        """Elimina tracks que ya no están presentes"""
        active_ids = list(self.active_tracks.keys())
        for tid in active_ids:
            if tid not in current_frame_track_ids:
                del self.active_tracks[tid]

    def _draw_detection(self, image, box, track_id, cls_id, conf):
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = self.class_names.get(cls_id, '')
        color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id} {class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_zones(self, image):
        if self.roi_polygon is not None:
             color = (0, 255, 255) if self.roi_active else (128, 128, 128)
             cv2.polylines(image, [self.roi_polygon], True, color, 2)
        
        if self.door_polygon is not None:
             # Blue for door ROI (BGR format: 255, 0, 0)
             color = (255, 0, 0) if self.door_active else (128, 128, 128)
             cv2.polylines(image, [self.door_polygon], True, color, 2)
             
        if self.door_direction is not None:
             pt1 = tuple(self.door_direction[0])
             pt2 = tuple(self.door_direction[1])
             # Red for direction with arrow (BGR format: 0, 0, 255)
             color = (0, 0, 255) if self.door_direction_active else (128, 128, 128)
             cv2.arrowedLine(image, pt1, pt2, color, 3, tipLength=0.1)

    def _build_metadata(self, vehicles_detected, persons_detected, results_boxes):
        return {
            'frame_number': self.frame_counter,
            'roi_active': self.roi_active,
            'door_active': self.door_active,
            'vehicles_detected': vehicles_detected,
            'persons_detected': persons_detected,
            'vehicles_in_area': self.vehiculos_en_area,
            'car_count': self.car_count,
            'truck_count': self.truck_count,
            'motorcycle_count': self.motorcycle_count,
            'persons_inside': self.person_count_inside,
            'persons_in_area': self.personas_en_area,
            'vehicles_inside': self.vehicles_inside,
            'active_tracks': len(self.active_tracks),
            'last_counted_id': self.last_counted_id,
            'last_counted_frame': self.last_counted_frame,
            'alerts': self.current_alerts # Lista de alertas generadas en este frame
        }

    # Método para extender o liberar recursos
    def cleanup(self):
        pass
