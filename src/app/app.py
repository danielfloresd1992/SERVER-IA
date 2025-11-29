import json
import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import logging
from ..analityc.core.inference_video import VehicleProcessor
from ..analityc.config.config import get_config




app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Variable global para controlar la ventana de OpenCV
window_open = False
current_image = None

logger = logging.getLogger(__name__)




async def startup_event():
    """Inicializa el procesador al iniciar la aplicación"""
    global vehicle_processor
    try:
        config = get_config()
        vehicle_processor = VehicleProcessor(
            model_path=config["model_path"],
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=config["device"]
        )
        logger.info("✅ VehicleProcessor inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando VehicleProcessor: {e}")
        


@app.get('/')
def init_server():
    return {"status": "active"}


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para procesamiento en tiempo real"""
    global vehicle_processor
    
    await websocket.accept()
    logger.info("Cliente WebSocket conectado")
    
    await startup_event()

    if vehicle_processor is None:
        await websocket.send_text("Error: Procesador no inicializado")
        await websocket.close()
        return
    
    try:
        while True:
            raw_message = await websocket.receive_text()
            
            if not raw_message.strip():
                await websocket.send_text("Error: Mensaje vacío")
                continue
            
            try:
                data = json.loads(raw_message)
                logger.info(f"JSON recibido - Timestamp: {data.get('header', {}).get('timestamp')}")
                
                # Procesar la imagen
                image_data = data.get('image', '')
                if image_data:
                    # Decodificar base64
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # Procesar frame
                        processed_img, metadata = vehicle_processor.process_frame(img)
                        
                        # Codificar resultado como JPG
                        success, encoded_image = cv2.imencode('.jpg', processed_img, 
                                                             [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        if success:
                            processed_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                            response = {
                                "status": "success",
                                "metadata": metadata,
                                "processed_image": f"data:image/jpeg;base64,{processed_base64}"
                            }
                            await websocket.send_text(json.dumps(response))
                            print('procesamiento activo')
                        else:
                            await websocket.send_text(json.dumps({
                                "status": "error", 
                                "message": "Error codificando imagen"
                            }))
                            print('Error codificando imagen')
                    else:
                        await websocket.send_text(json.dumps({
                            "status": "error", 
                            "message": "Error decodificando imagen"
                        }))
                        print('Error decodificando imagen')
                else:
                    await websocket.send_text(json.dumps({
                        "status": "error", 
                        "message": "No hay datos de imagen"
                    }))
                    print('No hay datos de imagen')
                 
            except json.JSONDecodeError as e:
                logger.error(f"Error JSON: {e}")
                await websocket.send_text(json.dumps({
                    "status": "error", 
                    "message": f"Error en formato JSON: {str(e)}"
                }))
                print(f"Error en formato JSON: {str(e)}")
                


    except WebSocketDisconnect:
        logger.info('Cliente WebSocket desconectado')
    except Exception as e:
        logger.error(f"Error general en WebSocket: {e}")
    finally:
        await websocket.close()



        

# Manejo de cierre limpio
import atexit

@atexit.register
def cleanup():
    global window_open
    window_open = False
    cv2.destroyAllWindows()
    print("Limpieza realizada")

