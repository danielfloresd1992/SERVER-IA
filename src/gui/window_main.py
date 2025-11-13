from PySide6.QtWidgets import QMainWindow, QPushButton,QWidget, QVBoxLayout,QHBoxLayout



class Main_Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.set_ui()
        self.set_ui()
        
        
    def set_ui(self):
        self.resize(800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        
        layout = QHBoxLayout(central_widget)
        self.btn_server_init = QPushButton('Iniciar servidor')
        self.btn_server_stop = QPushButton('Detener servidor')
        layout.addWidget(self.btn_server_init)
        layout.addWidget(self.btn_server_stop)