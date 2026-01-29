import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from src.analityc.core.perimetrales_multicam import PerimetralesMultiCam


def find_videos(folder):
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    return files


def mean_v_of_crop(img, bbox):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 255
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def main():
    cam_folder = r"C:\Users\Sistema-1\Documents\camaras"
    vids = find_videos(cam_folder)
    if len(vids) < 2:
        print("Debe haber al menos 2 vídeos en:", cam_folder)
        print("Encontrados:", vids)
        return

    v1, v2 = vids[0], vids[1]
    print("Usando videos:", v1, v2)

    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)

    # Intentar usar un modelo local pequeño si existe para acelerar la prueba
    # Buscar yolo12m en ubicaciones comunes: raíz del proyecto o models/base/
    candidates = [
        os.path.join(PROJECT_ROOT, 'yolo26m.pt'),
        os.path.join(PROJECT_ROOT, 'models', 'base', 'yolo26m.pt'),
        os.path.join(PROJECT_ROOT, 'models', 'yolo26m.pt')
    ]
    chosen = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break

    if chosen:
        print('Usando modelo local:', chosen)
        pm = PerimetralesMultiCam(model_path=chosen, debug=True, device='cpu')
    else:
        pm = PerimetralesMultiCam(debug=True, device='cpu')

    if pm.model is None:
        print("Aviso: YOLO no está inicializado (pm.model is None). Instala 'ultralytics' y/o coloca un modelo yolo para pruebas.")
        # We'll still try, but process_frame devolverá [] si no hay modelo.

    ids_seen = {1: set(), 2: set()}
    timeline = []

    # Alternar lectura: leer un frame de cada cámara en cada iteración
    frame_idx = 0
    MAX_FRAMES = 600
    LOG_ALL_DETECTIONS = True
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 and not ret2:
            break

        if ret1:
            res1 = pm.process_frame(f1, camera_id=1)
            for d in res1:
                v = mean_v_of_crop(f1, d['bbox'])
                if LOG_ALL_DETECTIONS or v < 90:  # umbral heurístico para camisa negra
                    ids_seen[1].add(d['global_id'])
                    timeline.append((frame_idx, 1, d['global_id'], v))

        if ret2:
            res2 = pm.process_frame(f2, camera_id=2)
            for d in res2:
                v = mean_v_of_crop(f2, d['bbox'])
                if LOG_ALL_DETECTIONS or v < 90:
                    ids_seen[2].add(d['global_id'])
                    timeline.append((frame_idx, 2, d['global_id'], v))

        frame_idx += 1
        if frame_idx >= MAX_FRAMES:
            print(f"Reached MAX_FRAMES={MAX_FRAMES}, stopping early.")
            break
        # avoid spinning too fast
        time.sleep(0.01)

    cap1.release()
    cap2.release()

    print("IDs detectados por camara:")
    print("Camara 1:", ids_seen[1])
    print("Camara 2:", ids_seen[2])

    cross = ids_seen[1].intersection(ids_seen[2])
    if cross:
        print("IDs comunes entre camaras (coincidencia encontrada):", cross)
    else:
        print("No se encontraron ids comunes entre las cámaras usando el criterio de camisa negra.")

    # save timeline
    out = os.path.join(PROJECT_ROOT, 'multicam_test_output.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('frame_idx,cam,global_id,V\n')
        for t in timeline:
            f.write(f"{t[0]},{t[1]},{t[2]},{t[3]:.2f}\n")

    print('Timeline guardado en', out)


if __name__ == '__main__':
    main()
