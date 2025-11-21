import json
import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
import time

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

def show_image_window():
    """Función para mostrar la ventana de OpenCV en un hilo separado"""
    global window_open, current_image
    
    window_name = "Imagen desde WebSocket"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_open = True
    
    while window_open:
        if current_image is not None:
            try:
                cv2.imshow(window_name, current_image)
                # Espera 30ms y verifica si se presiona 'q' para salir
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error mostrando imagen: {e}")
                break
        time.sleep(0.1)
    
    cv2.destroyWindow(window_name)
    current_image = None

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
            print(f"Imagen decodificada - Dimensiones: {img.shape}")
            current_image = img
            return True
        else:
            print("Error: No se pudo decodificar la imagen")
            return False
            
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return False

@app.get('/')
def init_server():
    return {"status": "active"}




@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    global window_open
    
    await websocket.accept()
    
    # Iniciar la ventana de OpenCV si no está abierta
    if not window_open:
        window_thread = threading.Thread(target=show_image_window, daemon=True)
        window_thread.start()
        print("Ventana de OpenCV iniciada")
    
    try:
        while True:
            raw_message = await websocket.receive_text()
            
            if not raw_message.strip():
                print('Mensaje vacío recibido')
                await websocket.send_text("Error: Mensaje vacío")
                continue
            
            try:
                data = json.loads(raw_message)
                print(f"JSON recibido - Timestamp: {data.get('header', {}).get('timestamp')}")
                print(f"Tamaño: {data.get('header', {}).get('size')} bytes")
                print(f"Formato: {data.get('header', {}).get('format')}")
            
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
                print(f"Error JSON: {e}")
                
                # Si no es JSON, procesar como texto plano
                if '\n' in raw_message:
                    print('Mensaje multilínea recibido')
                    await websocket.send_text("Mensaje multilínea recibido")
                else:
                    print(f'Texto plano recibido: {raw_message[:100]}...')  # Mostrar solo primeros 100 caracteres
                    await websocket.send_text("Texto plano recibido")
                



    except WebSocketDisconnect:
        print('Cliente desconectado')
    except Exception as e:
        print(f"Error general: {e}")
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

