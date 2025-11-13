import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server as UvicornServer
import asyncio
import threading
import time
from .app import app


class Server_services:
    
    def __init__(self, port=8500):
        self.port = port
        self.server = None
        self.config = Config(
            app=None,
            host='0.0.0.0',
            port=self.port,
            log_level='info'
        )
         
        
        
    def start(self):
        try:
            self.thread = threading.Thread(target= self._run_server)
            self.thread.daemon = True
            self.should_exit = False
            self.thread.start()
    
        except Exception as e:
            print(f'Error al iniciar el servidor: {e}')
        
        
        
    def stop(self):
        if self.server:
            # Usar asyncio para detener el servidor correctamente
            if self.server.should_exit is not None:
                self.server.should_exit = True
            time.sleep(2)  # Dar tiempo para que se detenga

            
            
            
    def _run_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Importar la app dentro del hilo para evitar problemas
            
            from .app import app
            self.config.app = app
            self.server = UvicornServer(self.config )
            loop.run_until_complete(self.server.serve())
          
          
            
        except Exception as e:
            print(f'Error en el servidor: {e}')
        finally:
            loop.close()