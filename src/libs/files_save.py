import cv2
from pathlib import Path
import os
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]  # sube hasta la raíz del proyecto
image_output = project_root / 'debugger' / 'ouput'


print(image_output)


def save_frame_png(frame, path = image_output) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        """Guarda un frame como imagen PNG en la ruta especificada."""
        print(image_output)
        name_file = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.png'
        full_path = os.path.join(path, name_file)
        
        success = cv2.imwrite(full_path, frame)
        print(full_path)
        if success:
            print(f'Img save in : {full_path}')
        else:
            print('Error saving image')
        
    except Exception as e:
        print(f'Error saving image: {e}')
    
    
    