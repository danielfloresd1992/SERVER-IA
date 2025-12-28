import json
import base64
import numpy as np
import cv2
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
from concurrent.futures import ThreadPoolExecutor
from ..analityc.core.Perimetrales import MultiObjectProcessor
from ..analityc.core.car_washed import VehicleProcessor
from ..analityc.config.config import get_config
from ..analityc.core.hardware_available import device_hardware
import time

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
executor = ThreadPoolExecutor(max_workers=4)

# Diccionario para manejar múltiples clientes
active_connections = {}
connection_lock = asyncio.Lock()



def process_image_sync(processor, img, roi, activate_roi):
    """Función sincrónica para procesamiento de imágenes"""
    try:
        return processor.process_frame(img, roi, activate_roi)
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
        # Mapeo de procesadores para evitar múltiples ifs
        processors = {
            'Lavado': VehicleProcessor,
            'Perimetrales': MultiObjectProcessor
        }
        
        processor_class = processors.get(type_inference)
        if processor_class:
            processor = processor_class(
                client_id=client_id,
                model_path=config["model_path"],
                confidence_threshold=config["confidence_threshold"],
                iou_threshold=config["iou_threshold"],
                device=device_hardware.device_default['gpu_use']
            )

        async with connection_lock:
            active_connections[client_id] = {
                'websocket': websocket,
                'processor': processor,
                'last_active': time.time()
            }
        

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
            raw_message = await asyncio.wait_for(websocket.receive_text(), timeout=WEBSOCKET_TIMEOUT)
            request = json.loads(raw_message)
            data = request['data']

            try:
                start_time = time.time()
                
                # --- PROCESAMIENTO LINEAL ---
                # Si cualquiera de estas líneas falla, salta al 'except (json.JSONDecodeError, KeyError, ...)'
                
                # Validar campos requeridos y asignar variables
                image_data = data['image']

                roi = data.get('roi_coordinates', '')
                roi_activate = data['roi_activate']
           


                # Decodificar imagen
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise ValueError("Imagen corrupta o formato inválido")

                # Inferencia (Thread Pool)20
                loop = asyncio.get_event_loop()
                processed_img, metadata = await loop.run_in_executor(
                    executor, process_image_sync, processor, img, roi, roi_activate
                )

                # Codificar y Enviar
                success, encoded_image = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    raise ValueError("Error en la codificación de salida")

                processed_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')


                request['data'] = {
                        "status": "success",
                        "metadata": metadata,
                        "processed_image": f"data:image/jpeg;base64,{processed_base64}",
                        "processing_time": round(time.time() - start_time, 3)
                    }
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