import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    '*',
    ### "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=['*'],  # Permite todos los métodos
    allow_headers=['*'],
)



@app.get('/', )
def init_sever():
    return { "status": "active" }



@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_message = await websocket.receive_text()
            
            if not raw_message.strip():
                print('empty string')
                continue
            
            if '\n' in raw_message:
                print('logica a')
            
            else:
            
                print(raw_message)
                
        
            
            
           
    
    except WebSocketDisconnect:
        print(f'cliente desconectado')
    finally:
        await websocket.close()