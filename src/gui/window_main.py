from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from ..analityc.core.hardware_available import device_hardware


def _get_device_default() -> str:
    dev = getattr(device_hardware, 'device_default', 'cpu')
    if isinstance(dev, dict):
        return dev.get('gpu_use', 'cpu')
    return dev if isinstance(dev, str) else 'cpu'

class Main_Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.set_ui()
        
    def set_ui(self):
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal único
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # Sección de información de hardware
        title = QLabel('Hardware Available')
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        cpu_info = QLabel(f'CPU: {device_hardware.cpu}')
        memory_gb = device_hardware.memory.total / (1024**3)
        memory_info = QLabel(f'Memory: {memory_gb:.2f} GB')
        cuda_available = QLabel(f"CUDA Available: {'Yes' if device_hardware.cuda_available else 'No'}")

        main_layout.addWidget(title)
        main_layout.addWidget(cpu_info)
        main_layout.addWidget(memory_info)
        main_layout.addWidget(cuda_available)

        # Sección de GPUs
        title_gpu = QLabel('Installed GPUs:')
        title_gpu.setStyleSheet("font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(title_gpu)

        if len(device_hardware.gpu_tuple) > 0:
            for index, device in enumerate(device_hardware.gpu_tuple):
                gpus_label = QLabel(f'{device["gpu_use"]}, {device["gpu_name"]}')
                data_gpu = device_hardware.get_gpu_info(index)
                
                tooltip = (
                    f"<b>GPU {index} - {data_gpu['name']}</b><br>"
                    f"Memory total: {data_gpu['memory_total_gb']:.1f} GB<br>"
                    f"Compute capability: {data_gpu['compute_capability']}<br>"
                    f"Multiprocessors: {data_gpu['multiprocessor_count']}"
                )
                
                gpus_label.setToolTip(tooltip)
                main_layout.addWidget(gpus_label)
        else:
            no_gpu_label = QLabel('No GPUs available (Not detected)')
            main_layout.addWidget(no_gpu_label)
            
        device_default = QLabel(f"Default Device: {_get_device_default() if device_hardware.cuda_available else 'cpu'}")
        main_layout.addWidget(device_default)

        # Sección de botones
        buttons_layout = QHBoxLayout()
        
        self.btn_server_init = QPushButton('Iniciar servidor')
        self.btn_server_stop = QPushButton('Detener servidor')
        
        buttons_layout.addWidget(self.btn_server_init)
        buttons_layout.addWidget(self.btn_server_stop)
        

        # Agregar el layout de botones al layout principal
        main_layout.addLayout(buttons_layout)
        main_layout.addStretch()  # Empuja todo hacia arriba