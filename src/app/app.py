import json
import base64
import numpy as np
import cv2
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
import logging
from concurrent.futures import ThreadPoolExecutor
from ..analityc.core.Perimetrales import MultiObjectProcessor
from ..analityc.core.base_perimeter import BasePerimeter
from ..analityc.core.car_washed import VehicleProcessor
from ..analityc.core.perimetrales_multicam import PerimetralesMultiCam
from ..analityc.core.botsort_wrapper import BoTSORTWrapper
from ..analityc.core.person_amazona_inference import PersonAmazonas
from ..analityc.config.config import get_config
from ..analityc.core.hardware_available import device_hardware
import time
import msgpack

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool para procesamiento pesado
# Incremento moderado de workers para paralelizar inferencia y codificación
executor = ThreadPoolExecutor(max_workers=8)

# Diccionario para manejar múltiples clientes
active_connections = {}
connection_lock = asyncio.Lock()

# Cola global de frames para procesamiento secuencial
frame_queue: asyncio.Queue = asyncio.Queue()
# Tarea worker global (iniciada en la primera conexión)
worker_task: Optional[asyncio.Task] = None
# Pending per-client: guarda el último future encolado por cliente (se reemplaza al llegar uno nuevo)
pending_frames: Dict[str, Any] = {}
# Control de tasa por cliente (segundos mínimo entre encolados)
CLIENT_MIN_INTERVAL = 0.2  # segundos (5 FPS)
last_enqueue_time: Dict[str, float] = {}


def _get_device_default() -> str:
    """Obtiene el dispositivo de inferencia (cuda:N o cpu) de forma segura."""
    dev = getattr(device_hardware, 'device_default', 'cpu')
    if isinstance(dev, dict):
        return dev.get('gpu_use', 'cpu')
    return dev if isinstance(dev, str) else 'cpu'


def _get_first_gpu() -> str:
    gpus = getattr(device_hardware, 'gpu_tuple', [])
    if isinstance(gpus, (list, tuple)) and len(gpus) > 0:
        gpu0 = gpus[0]
        if isinstance(gpu0, dict):
            return gpu0.get('gpu_use', _get_device_default())
    return _get_device_default()


async def _frame_worker():
    """Worker que procesa frames uno a la vez en orden de llegada.

    Cada item en la cola es una tupla:
    (future, processor, img, roi, roi_activate, door_roi, door_activate, door_direction, door_direction_activate,
     pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id, request, websocket, client_id)
    El worker ejecuta el procesamiento en el `executor` y completa el `future` con el
    dict de respuesta listo para enviar por WS.
    """
    global frame_queue
    while True:
        item = await frame_queue.get()
        try:
            future, processor, img, roi, roi_activate, door_roi, door_activate, door_direction, door_direction_activate, pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id, request, websocket, client_id = item
        except Exception:
            frame_queue.task_done()
            continue

        loop = asyncio.get_event_loop()
        start_time = time.time()
        try:
            qsize = frame_queue.qsize()
        except Exception:
            qsize = -1
        # logger.info(f"Worker: dequeued frame client={client_id} camera={camera_id} queue_size={qsize}")
        try:
            # Procesamiento sincrónico en threadpool
            processed_img, metadata = await loop.run_in_executor(
                executor, process_image_sync, processor, img, roi, roi_activate, door_roi, door_activate, door_direction, door_direction_activate, pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id
            )

            # Codificar imagen también en threadpool
            def _encode_jpeg(image, quality=70):
                return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])

            success, encoded_image = await loop.run_in_executor(executor, _encode_jpeg, processed_img, 70)
            if not success:
                raise ValueError('Error en la codificación de salida')

            processed_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

            processing_time = round(time.time() - start_time, 3)
            try:
                alerts_dbg = metadata.get('alerts', []) if isinstance(metadata, dict) else []
                #print(f"🔔 alerts enviados: {len(alerts_dbg)}")
            except Exception:
                pass

            response = {
                'camera_id': camera_id,
                'status': 'success',
                'metadata': metadata,
                'processed_image': f'data:image/jpeg;base64,{processed_base64}',
                'processing_time': processing_time
            }
            # logger.info(f"Worker: done client={client_id} camera={camera_id} time={processing_time}s")
            # Completar el future para el handler
            if not future.done():
                future.set_result(response)
            # limpiar pending_frames si corresponde
            try:
                cur = pending_frames.get(client_id)
                if cur is not None and cur[0] is future:
                    pending_frames.pop(client_id, None)
            except Exception:
                pass
        except Exception as e:
            if not future.done():
                future.set_result({'status': 'error', 'message': str(e)})
            try:
                # limpiar pending_frames en caso de error
                cur = pending_frames.get(client_id)
                if cur is not None and cur[0] is future:
                    pending_frames.pop(client_id, None)
            except Exception:
                pass
        finally:
            frame_queue.task_done()




def process_image_sync(processor, img, roi, activate_roi, door_roi=None, door_activate=False,
                       door_direction=None, door_direction_activate=False, pay_roi=None, withdraw_roi=None,
                       pay_roi_activate=True, withdraw_roi_activate=True, camera_id: int = 1):
    """Función sincrónica para procesamiento de imágenes.

    Soporta procesadores existentes (`process_frame(img, roi, activate_roi)`) y
    `PerimetralesMultiCam` (que expone `process_frame(frame, camera_id)`).
    """

    try:
        # Si es el procesador multicámara usamos su API y dibujamos resultados
        if isinstance(processor, PerimetralesMultiCam):
            tracks = processor.process_frame(
                img,
                camera_id,
                pay_roi=pay_roi,
                withdraw_roi=withdraw_roi,
                pay_roi_activate=pay_roi_activate,
                withdraw_roi_activate=withdraw_roi_activate
            )
            for t in tracks:
                x1, y1, x2, y2 = t.get('bbox', (0, 0, 0, 0))
                gid = t.get('global_id', 0)
                conf = t.get('conf', 0.0)
                cam_id = t.get('camera_id', camera_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'ID:{gid} CAM:{cam_id} {conf:.2f}'
                y_text = max(10, y1 - 6)
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                rect_tl = (x1, max(0, y_text - text_h - baseline))
                rect_br = (x1 + text_w, y_text + baseline)
                # Fondo amarillo (BGR) y texto en negro
                cv2.rectangle(img, rect_tl, rect_br, (0, 255, 255), -1)
                cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            metadata = {'tracks': tracks}
            return img, metadata

        # Si es PersonAmazonas, usar o crear un procesador por cámara que comparta el modelo
        if isinstance(processor, PersonAmazonas):
            try:
                cam_proc = processor.get_camera_processor(camera_id)
                return cam_proc.process_frame(img, roi, activate_roi, camera_id)
            except Exception:
                # fallback to original processor
                return processor.process_frame(img, roi, activate_roi, camera_id)

        # Si es MultiObjectProcessor, aislar estado por cámara
        if isinstance(processor, MultiObjectProcessor):
            try:
                cam_proc = processor.get_camera_processor(camera_id)
                return cam_proc.process_frame(img, roi, activate_roi, door_roi, door_activate, door_direction, door_direction_activate)
            except Exception:
                return processor.process_frame(img, roi, activate_roi, door_roi, door_activate, door_direction, door_direction_activate)

        # Procesadores legacy
        return processor.process_frame(img, roi, activate_roi, door_roi, door_activate, door_direction, door_direction_activate)
    except Exception as e:
        logger.error(f"Error en procesamiento: {e}")
        raise



@app.websocket('/ws/{type_inference}')
async def websocket_endpoint(websocket: WebSocket, type_inference: str):
    await websocket.accept()
    client_id = f'client_{id(websocket)}'
    processor = None

    # 1. Inicialización y Registro
    try:
        config = get_config()
        print(type_inference)


        # Mapeo de procesadores para evitar múltiples ifs
        if type_inference == 'Lavado':  
            processor =  VehicleProcessor(
                client_id=client_id,
                model_path=config["model_path"],
                confidence_threshold=config["confidence_threshold"],
                iou_threshold=config["iou_threshold"],
                device=device_hardware.gpu_tuple[0]['gpu_use']

            )
        elif type_inference == 'Perimetrales':
            # client_id: str, model_path: str, device: str = 'cpu'
            processor = BasePerimeter(
                client_id=client_id,
                model_path='yolo12l.pt',
               # confidence_threshold=config["confidence_threshold"],
                #iou_threshold=config["iou_threshold"],
                device=device_hardware.gpu_tuple[0]['gpu_use']
            )
        elif type_inference == 'PerimetralesMultiCam':
            processor = PerimetralesMultiCam(
                model_path=config.get("model_path", "yolo26l.pt"),
                device=device_hardware.gpu_tuple[0]['gpu_use'],        
                debug=False
            )
        elif type_inference == 'PerimetralesBoTSORT':
            processor = BoTSORTWrapper(
                model_path=config.get("model_path", "yolo26l.pt"),
                device=device_hardware.gpu_tuple[0]['gpu_use'],
                match_thresh=0.45,
                debug=False
            )
        elif type_inference == 'Personal de Amazonas':
            processor = PersonAmazonas(
                client_id=client_id,
                model_path='best.pt',
                confidence_threshold=config["confidence_threshold"],
                iou_threshold=config["iou_threshold"],
                device=device_hardware.gpu_tuple[0]['gpu_use']
            )


        async with connection_lock:
            active_connections[client_id] = {
                'websocket': websocket,
                'processor': processor,
                'last_active': time.time()
            }
        


        print('dato enviado ☻☻☻☻')        

        # Enviar mensaje de bienvenida
        await websocket.send_text(json.dumps({
            'id_connection': id(websocket),
            'event': 'conection_init',
            'type_inference': type_inference,
            'data': {'roy': False}
        }))


        # 2. Bucle Principal
        WEBSOCKET_TIMEOUT = 30.0
        while True:
            # Receive either binary msgpack or text JSON depending on client
            recv = await asyncio.wait_for(websocket.receive(), timeout=WEBSOCKET_TIMEOUT)
            incoming_is_binary = False
            if recv.get('bytes') is not None:
                incoming_is_binary = True
                try:
                    request = msgpack.unpackb(recv['bytes'], raw=False)
                except Exception as e:
                    raise ValueError(f"Msgpack unpack error: {e}")
            else:
                try:
                    txt = recv.get('text') or ''
                    request = json.loads(txt)
                except Exception as e:
                    raise ValueError(f"JSON parse error: {e}")

            data = request['data']
        
            try:
                start_time = time.time()

                # --- PROCESAMIENTO ENCOLADO (cola global, procesado secuencial) ---
                # Validar campos requeridos y asignar variables
              


                image_data = data['image']


                # ROI PRINCIPAL
                roi = data.get('roi_coordinates', '')
                roi_activate = data['roi_activate']



                # ROI DE PUERTA
                door_roi = data.get('door_roi_coordinates', None)
                door_activate = data.get('door_roi_activate', False)

                # Dirección de puerta (línea)
                door_direction = data.get('door_direction', None)
                door_direction_activate = data.get('door_direction_activate', False)



                # Compatibilidad: algunos clientes envían la línea en door_direction_activate
                if isinstance(door_direction_activate, (list, tuple)) and door_direction is None:
                    door_direction = door_direction_activate
                    door_direction_activate = True
                pay_roi = data.get('pay_roi', None)
                withdraw_roi = data.get('withdraw_roi', None)
                pay_roi_activate = data.get('pay_roi_activate', True)
                withdraw_roi_activate = data.get('withdraw_roi_activate', True)
                # camera_id puede venir como int o como string (p.ej. uuid). Intentar convertir
                # a int, pero si falla mantener el valor original (evitar ValueError).
                camera_id_raw = data.get('camera_id', 1)
                try:
                    camera_id = int(camera_id_raw)
                except (TypeError, ValueError):
                    camera_id = camera_id_raw

                # Decodificar imagen (soporta: data URI base64 str, base64 str, raw bytes)
                image_bytes = None
                if isinstance(image_data, (bytes, bytearray)):
                    image_bytes = bytes(image_data)
                elif isinstance(image_data, str):
                    # data URI like 'data:image/jpeg;base64,...'
                    if image_data.startswith('data:') and ',' in image_data:
                        image_data = image_data.split(',', 1)[1]
                    try:
                        image_bytes = base64.b64decode(image_data)
                    except Exception:
                        # Fallback: treat as UTF-8 encoded string
                        image_bytes = image_data.encode('utf-8')
                else:
                    # Otros tipos (p.ej. list) — forzar a bytes via str()
                    image_bytes = str(image_data).encode('utf-8')

                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Imagen corrupta o formato inválido")

                # Asegurar que el worker esté en marcha
                global worker_task
                if worker_task is None:
                    worker_task = asyncio.create_task(_frame_worker())

                # Throttle simple: evitar encolar frames demasiados frecuentes por cliente
                now_enqueue = time.time()
                last_t = last_enqueue_time.get(client_id, 0.0)
                last_enqueue_time[client_id] = now_enqueue

                # Crear future que el worker completará
                loop = asyncio.get_event_loop()
                future = loop.create_future()

                # Si ya hay un pending para este cliente, marcarlo como "replaced" y reemplazarlo
                prev = pending_frames.get(client_id)
                if prev is not None:
                    try:
                        prev_future = prev[0]
                        if not prev_future.done():
                            prev_future.set_result({'status': 'dropped', 'reason': 'replaced_by_newer_frame'})
                            logger.info(f"Dropped pending frame for client={client_id} replaced by newer frame")
                    except Exception:
                        pass

                # Si el cliente está enviando frames más rápido que CLIENT_MIN_INTERVAL,
                # aún aceptamos la nueva (reemplaza la anterior). Esto mantiene sólo la última.

                # Registrar nuevo pending y encolar
                pending_frames[client_id] = (future, processor, img, roi, roi_activate, door_roi, door_activate, door_direction, door_direction_activate, pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id, request, websocket)
                await frame_queue.put((future, processor, img, roi, roi_activate, door_roi, door_activate, door_direction, door_direction_activate, pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id, request, websocket, client_id))

                # Esperar resultado del worker (respuesta ya lista para enviar)
                result = await future
                # Adjuntar y enviar — mantener el mismo formato de entrada
                request['data'] = result
                if websocket.client_state != WebSocketState.CONNECTED:
                    raise WebSocketDisconnect()

                if incoming_is_binary:
                    try:
                        packed = msgpack.packb(request, use_bin_type=True)
                        await websocket.send_bytes(packed)
                    except Exception as e:
                        logger.error(f"Error enviando msgpack: {e}")
                        if websocket.client_state == WebSocketState.CONNECTED:
                            # Fallback a JSON
                            await websocket.send_json(request)
                else:
                    await websocket.send_json(request)


                # Actualizar actividad
                async with connection_lock:
                    if client_id in active_connections:
                        active_connections[client_id]['last_active'] = time.time()


            # --- MANEJO DE ERRORES DEL BUCLE ---
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"status": "ping"}))
                except: break

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Captura errores de datos, JSON mal formado o claves faltantes (como component_id)
                logger.error(f"Error de validación: {e}")
                request['data'] = {"status": "error", "message": str(e)}
                await websocket.send_json(request)


            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"Error crítico en {client_id}: {e}")
    
    finally:
        # 3. Limpieza única al cerrar
        async with connection_lock:
            active_connections.pop(client_id, None)
        if processor and hasattr(processor, 'cleanup'):
            processor.cleanup()
        logger.info(f"Cliente {client_id} desconectado.")







@app.get('/')
def init_server():
    return {"status": "active", "connections": len(active_connections)}




@app.get('/health')
def health_check():
    """Endpoint para verificar estado del servidor"""
    return {
        "status": "active",
        "connections": len(active_connections),
        "thread_pool": {
            "active_threads": executor._work_queue.qsize() if hasattr(executor, '_work_queue') else "unknown"
        }
    }