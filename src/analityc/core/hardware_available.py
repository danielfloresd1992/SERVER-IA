import psutil
import torch
import cpuinfo


class Device_hardware:
    
    ''' INFO HARDWARE AVAILABLE '''
    cpu = None
    memory = None
    ''' INFOR FOR IA  '''
    device_default = 'cpu'
    cuda_available = False
    gpu_available = False
    gpu_tuple = []

    def __init__(self):
        self.cpu = cpuinfo.get_cpu_info()['brand_raw']
        self.memory = psutil.virtual_memory()
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            n = torch.cuda.device_count()
            if n > 0: 
                list = []
                for i in range(n):
                    device = {}
                    device['gpu_use'] = f'cuda:{i}'
                    device['gpu_name'] = torch.cuda.get_device_name(i)
                    list.append(device)

                self.gpu_tuple = tuple(list)
                self.gpu_available = True
                self.device_default = self.gpu_tuple[0]['gpu_use']
            else:
                self.gpu_tuple = []
                self.gpu_available = False



    def seletec_divice(self, number: int):
        if len(self.gpu_tuple) < 1:
            return None
        if number >= len(self.gpu_tuple):
            return None
        self.device_default = self.gpu_tuple[number]['gpu_use']




    def get_gpu_info(self, device_index: int = None):
        """Obtiene información detallada de una GPU específica o de todas"""
        if not self.gpu_available:
            return 'No GPUs available'
        
        if device_index is not None:
            if device_index < 0 or device_index >= len(self.gpu_tuple):
                return f'GPU index {device_index} out of range'
            
            device = self.gpu_tuple[device_index]
            print(device)
            properties = torch.cuda.get_device_properties(device_index)
            
            return {
                'device': device['gpu_use'],
                'name': device['gpu_name'],
                'memory_total_gb': properties.total_memory / (1024**3),
                'mealmory_tot_mb': properties.total_memory / (1024**2),
                'compute_capability': f"{properties.major}.{properties.minor}",
                'multiprocessor_count': properties.multi_processor_count
            }
        else:
            # Retornar información de todas las GPUs
            gpus_info = []
            for i in range(len(self.gpu_tuple)):
                gpus_info.append(self.get_gpu_info(i))
            return gpus_info
        

        

device_hardware = Device_hardware()