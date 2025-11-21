import json
import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import datetime
import logging
import atexit

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# =========================
# CONFIGURACIÓN GLOBAL
# =========================

# Variables globales para la imagen y el estado de la ventana
window_open = False
current_image = None
image_lock = threading.Lock()

# Variables globales para YOLO
model = None
vehicle_tracker = None
autos_lavados = 0
frame_counter = 0

# ROI (Región de interés) - ajustable con mouse
ROI_POLYGON_COORDS = np.array([
    [100, 150],
    [900, 150],
    [900, 500],
    [100, 500]
], np.int32)

# Parámetros de YOLO y tracking
CONFIDENCE_THRESHOLD = 0.3
IOU_NMS = 0.4
MAX_FRAME_GAP = 30
MIN_CONSEC_FRAMES = 1
FRAMES_OUTSIDE_TO_REMOVE = 5
MIN_TIME_INSIDE = 2

# Colores
CLR_TRACK = (0, 165, 255)       # Naranja: Fuera del polígono
CLR_CONFIRMED = (0, 255, 0)     # Verde: Dentro del polígono
CLR_ROI = (0, 255, 255)         # Amarillo para el polígono
CLR_VERTEX = (255, 0, 0)        # Azul para los vértices arrastrables
CLR_EXIT = (0, 0, 255)          # Rojo para vehículos saliendo

# Variables para el arrastre del polígono
dragging_vertex_index = -1
drag_offset = 10

# Configuración de GPU
DEVICE_INDEX = 0
USE_HALF = False  # Usar False para mayor compatibilidad

# =========================
# SISTEMA DE TRACKING
# =========================

class VehicleTracker:
    def __init__(self, max_frame_gap=30, min_consec_frames=1, iou_threshold=0.5, frames_outside_to_remove=5):
        self.max_frame_gap = max_frame_gap
        self.min_consec_frames = min_consec_frames
        self.iou_threshold = iou_threshold
        self.frames_outside_to_remove = frames_outside_to_remove
        
        self.active_tracks = {}
        self.removed_tracks = set()
        self.next_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=15))
        self.counted_ids = set()
        self.frame_counter = 0
        
    def calculate_iou(self, box1, box2):
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
    
    def check_exit_direction(self, track_id):
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return True
        
        history = list(self.track_history[track_id])
        recent_points = history[-min(5, len(history)):]
        
        if len(recent_points) < 2:
            return True
        
        first_y = recent_points[0][1]
        last_y = recent_points[-1][1]
        y_movement = last_y - first_y
        
        return y_movement > 5
    
    def match_detections_to_tracks(self, detections):
        matched_pairs = []
        unmatched_detections = []
        unmatched_tracks = list(self.active_tracks.keys())
        
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            for track_id in unmatched_tracks:
                track = self.active_tracks[track_id]
                iou = self.calculate_iou(det['box'], track['box'])
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                matched_pairs.append((best_track_id, det_idx))
                unmatched_tracks.remove(best_track_id)
            else:
                unmatched_detections.append(det_idx)
                
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def cleanup_stale_tracks(self, frame_idx, roi_polygon):
        tracks_to_remove = []
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = cv2.pointPolygonTest(roi_polygon, (int(current_pos[0]), int(current_pos[1])), False) >= 0
            
            if not is_inside:
                frames_outside = frame_idx - track.get('last_inside_frame', frame_idx)
                if frames_outside >= self.frames_outside_to_remove:
                    tracks_to_remove.append(track_id)
                    continue
            
            gap = frame_idx - track['last_seen']
            if gap > self.max_frame_gap:
                tracks_to_remove.append(track_id)
                continue
            
            if self.frame_counter % 10 == 0:
                if not self.verify_vehicle_presence(track_id, roi_polygon):
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                if not self.active_tracks[track_id].get('counted_exit', False):
                    self.removed_tracks.add(track_id)
                del self.active_tracks[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
        
        return len(tracks_to_remove)
    
    def verify_vehicle_presence(self, track_id, roi_polygon):
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        current_pos = track['center']
        
        return cv2.pointPolygonTest(roi_polygon, (int(current_pos[0]), int(current_pos[1])), False) >= 0
    
    def update_tracks(self, detections, frame_idx, roi_polygon):
        self.frame_counter = frame_idx
        
        removed_count = self.cleanup_stale_tracks(frame_idx, roi_polygon)
        
        current_detections = []
        
        for det in detections:
            current_detections.append({
                'box': det['box'],
                'class': det['class'],
                'center': det['center'],
                'confidence': det.get('confidence', 0.5)
            })
        
        if self.active_tracks:
            matched_pairs, unmatched_detections, unmatched_tracks = self.match_detections_to_tracks(current_detections)
            
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
                    'counted': False,
                    'counted_exit': False,
                    'first_detected': frame_idx,
                    'last_inside_frame': frame_idx
                }
                self.track_history[new_id].append(det['center'])
                
        else:
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

# =========================
# INICIALIZACIÓN DE MODELOS
# =========================

def initialize_yolo():
    """Inicializa el modelo YOLO y el tracker"""
    global model, vehicle_tracker
    
    try:
        if not torch.cuda.is_available():
            device = 'cpu'
            logger.info("Usando CPU para inferencia")
        else:
            device = f'cuda:{DEVICE_INDEX}'
            logger.info(f"Usando GPU {DEVICE_INDEX} para inferencia")

        # Cargar modelo YOLO
        model = YOLO("yolo11n.pt").to(device)
        
        # Warmup del modelo
        dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = model.predict(dummy_input, imgsz=320, device=device, half=USE_HALF, verbose=False)
        
        # Inicializar tracker
        vehicle_tracker = VehicleTracker(
            max_frame_gap=MAX_FRAME_GAP,
            min_consec_frames=MIN_CONSEC_FRAMES,
            iou_threshold=0.3,
            frames_outside_to_remove=FRAMES_OUTSIDE_TO_REMOVE
        )
        
        logger.info("✅ Modelo YOLO y tracker inicializados correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inicializando YOLO: {e}")
        return False

# =========================
# FUNCIONES DE PROCESAMIENTO
# =========================

def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

def update_vehicle_state(objects, frame_idx, polygon):
    global autos_lavados
    
    for tid, obj in list(objects.items()):
        current_pos = obj['center']
        prev_state = obj.get('state') 
        confirmed = (obj['seen_frames'] >= MIN_CONSEC_FRAMES)
        
        if not confirmed:
            continue
            
        is_currently_inside = is_inside_polygon(current_pos, polygon)
        new_state = 'inside' if is_currently_inside else 'outside'
        
        if new_state == 'inside':
            obj['last_inside_frame'] = frame_idx
        
        if prev_state is not None and prev_state != new_state:
            logger.info(f"🚗 ID {tid} cambió de {prev_state} a {new_state} en frame {frame_idx}")
            
            if prev_state == 'inside' and new_state == 'outside':
                time_inside = frame_idx - obj.get('first_inside_frame', frame_idx)
                is_exiting = vehicle_tracker.check_exit_direction(tid)
                
                if (not obj.get('counted_exit', False) and 
                    time_inside >= MIN_TIME_INSIDE and 
                    obj['class'] == 'car' and
                    is_exiting):
                    
                    autos_lavados += 1
                    obj['counted_exit'] = True
                    logger.info(f"🎉 AUTO LAVADO CONTADO - Total: {autos_lavados} (ID: {tid}, Tiempo dentro: {time_inside} frames)")
                else:
                    logger.info(f"⏳ Auto no contado - ID: {tid}, Razón: tiempo={time_inside}, clase={obj['class']}, ya_contado={obj.get('counted_exit', False)}, saliendo={is_exiting}")
        
        if new_state == 'inside' and prev_state == 'outside':
            obj['first_inside_frame'] = frame_idx
            logger.info(f"➡️ ID {tid} ENTRÓ al área de lavado")
        
        obj['state'] = new_state

def draw_detections(image, active_tracks, roi_polygon):
    """Dibuja las detecciones y tracks en la imagen"""
    global autos_lavados
    
    # Dibujar ROI
    cv2.polylines(image, [roi_polygon], isClosed=True, color=CLR_ROI, thickness=3)
    for x, y in roi_polygon:
        cv2.circle(image, (x, y), 6, CLR_VERTEX, -1)
    
    # Dibujar tracks activos
    for tid, obj in list(active_tracks.items()):
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
        
        if tid in vehicle_tracker.track_history:
            history = vehicle_tracker.track_history[tid]
            for i in range(1, len(history)):
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                cv2.line(image, pt1, pt2, color, 2)
        
        label = f"ID:{tid} {state_text}"
        cv2.putText(image, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(image, (int(obj['center'][0]), int(obj['center'][1])), 4, color, -1)
    
    # Dibujar HUD
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (350, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    cv2.putText(image, f"AUTOS LAVADOS: {autos_lavados}", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    cv2.putText(image, f"Tracks activos: {len(active_tracks)}", (15, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.putText(image, f"Frame: {frame_counter}", (15, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return image

def process_frame_with_yolo(image):
    """Procesa un frame con YOLO y tracking"""
    global frame_counter, vehicle_tracker
    
    if model is None or vehicle_tracker is None:
        return image
    
    frame_counter += 1
    
    try:
        # Realizar detección con YOLO
        results = model.track(
            image,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_NMS,
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
            
            for i in range(boxes.shape[0]):
                cid = int(cls[i])
                cname = 'car' if cid == 2 else ('truck' if cid == 7 else str(cid))
                boxg = boxes[i]
                center = center_of(boxg)
                
                detections.append({
                    'class': cname,
                    'box': boxg,
                    'center': center
                })

        # Actualizar tracks
        ROI_POLYGON_POINTS = ROI_POLYGON_COORDS.reshape((-1, 1, 2))
        active_tracks = vehicle_tracker.update_tracks(detections, frame_counter, ROI_POLYGON_POINTS)
        
        # Actualizar estado de vehículos
        update_vehicle_state(active_tracks, frame_counter, ROI_POLYGON_POINTS)
        
        # Dibujar detecciones
        processed_image = draw_detections(image.copy(), active_tracks, ROI_POLYGON_COORDS)
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Error en procesamiento YOLO: {e}")
        return image

# =========================
# MANEJO DE VENTANA OPENCV
# =========================

def mouse_callback(event, x, y, flags, param):
    """Callback para arrastrar los vértices del ROI"""
    global dragging_vertex_index, ROI_POLYGON_COORDS
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (vx, vy) in enumerate(ROI_POLYGON_COORDS):
            if abs(x - vx) < drag_offset and abs(y - vy) < drag_offset:
                dragging_vertex_index = i
                break
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging_vertex_index != -1:
        ROI_POLYGON_COORDS[dragging_vertex_index] = [x, y]
        
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_vertex_index = -1

def show_image_window():
    """Función principal que muestra la ventana con análisis en tiempo real"""
    global window_open, current_image
    
    window_name = "Análisis de Vehículos - Sistema de Lavado"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    window_open = True
    
    # Inicializar YOLO
    if not initialize_yolo():
        logger.error("No se pudo inicializar YOLO. Cerrando ventana...")
        window_open = False
        return
    
    logger.info("Ventana de análisis iniciada - Presiona 'q' para salir, 'c' para reiniciar contador, 'r' para resetear ROI")
    
    while window_open:
        with image_lock:
            if current_image is not None:
                try:
                    # Procesar frame con YOLO
                    processed_frame = process_frame_with_yolo(current_image)
                    
                    # Mostrar frame procesado
                    cv2.imshow(window_name, processed_frame)
                    
                except Exception as e:
                    logger.error(f"Error procesando frame: {e}")
                    cv2.imshow(window_name, current_image)
            else:
                # Mostrar pantalla de espera
                black_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_image, "ESPERANDO IMAGEN...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(black_image, "Conecta un cliente WebSocket", (30, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow(window_name, black_image)
        
        # Control de teclado
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            global autos_lavados
            autos_lavados = 0
            logger.info("Contador de autos lavados reiniciado")
        elif key == ord('r'):
            # Resetear ROI a posición por defecto
            ROI_POLYGON_COORDS[:] = np.array([[100, 150], [900, 150], [900, 500], [100, 500]])
            logger.info("ROI reiniciado a posición por defecto")
    
    cv2.destroyWindow(window_name)
    window_open = False
    logger.info("Ventana de análisis cerrada")

# =========================
# ENDPOINTS FASTAPI
# =========================

def process_image_from_json(image_data):
    """Procesa la imagen desde el JSON y la prepara para mostrar"""
    global current_image
    
    try:
        # Extraer la cadena base64 (eliminar el prefijo si existe)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decodificar base64 a bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convertir bytes a array numpy
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Decodificar la imagen con OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            with image_lock:
                current_image = img
            return True
        else:
            logger.error("No se pudo decodificar la imagen")
            return False
            
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        return False

@app.get('/')
def init_server():
    return {"status": "active", "message": "Servidor de análisis de vehículos funcionando"}

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    global window_open
    
    await websocket.accept()
    logger.info("Cliente WebSocket conectado")
    
    # Inicializar YOLO si no está inicializado
    if model is None:
        initialize_yolo()
    
    # Iniciar la ventana de análisis si no está abierta
    if not window_open:
        window_thread = threading.Thread(target=show_image_window, daemon=True)
        window_thread.start()
        logger.info("Ventana de análisis iniciada")
    
    try:
        while True:
            raw_message = await websocket.receive_text()
            
            if not raw_message.strip():
                logger.info('Mensaje vacío recibido')
                await websocket.send_text("Error: Mensaje vacío")
                continue
            
            try:
                data = json.loads(raw_message)
                logger.info(f"JSON recibido - Timestamp: {data.get('header', {}).get('timestamp')}")
                logger.info(f"Tamaño: {data.get('header', {}).get('size')} bytes")
                logger.info(f"Formato: {data.get('header', {}).get('format')}")
            
                # Procesar la imagen
                image_data = data.get('image', '')
                if image_data:
                    success = process_image_from_json(image_data)
                    if success:
                        await websocket.send_text("Imagen recibida y procesada correctamente")
                    else:
                        await websocket.send_text("Error: No se pudo procesar la imagen")
                else:
                    await websocket.send_text("Error: No hay datos de imagen en el JSON")
             
            except json.JSONDecodeError as e:
                logger.error(f"Error JSON: {e}")
                
                # Si no es JSON, procesar como texto plano
                if '\n' in raw_message:
                    logger.info('Mensaje multilínea recibido')
                    await websocket.send_text("Mensaje multilínea recibido")
                else:
                    logger.info(f'Texto plano recibido: {raw_message[:100]}...')
                    await websocket.send_text("Texto plano recibido")
                
    except WebSocketDisconnect:
        logger.info('Cliente WebSocket desconectado')
    except Exception as e:
        logger.error(f"Error general en WebSocket: {e}")
    finally:
        await websocket.close()

# =========================
# MANEJO DE CIERRE
# =========================

@atexit.register
def cleanup():
    global window_open
    window_open = False
    cv2.destroyAllWindows()
    logger.info("Limpieza realizada - Servidor cerrado")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor de análisis de vehículos...")
    uvicorn.run(app, host="0.0.0.0", port=9000)