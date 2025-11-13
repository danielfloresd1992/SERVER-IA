from fastapi import FastAPI, WebSocket, WebSocketDisconnect


app = FastAPI()


@app.get('/', )
def init_sever():
    return { "status": "active" }



@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            pass
    
    
    except WebSocketDisconnect:
        print(f'cliente desconectado')