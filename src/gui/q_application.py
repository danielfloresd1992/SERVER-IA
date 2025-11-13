import sys
from PySide6.QtWidgets import QApplication
from  PySide6.QtCore import QObject



class AppSingletonGui(QObject):
    
    _instance = None
    _app = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppSingletonGui, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, sys_argv=None):
       
        if cls._app is None:
            if sys_argv is None:
                sys_argv = sys.argv
            cls._app = QApplication(sys_argv)
            
            # Configuración básica de la aplicación
            cls._app.setQuitOnLastWindowClosed(True)
            
            cls._initialized = True
        return cls._app