import os
import sys
import time
import argparse
import csv
import cv2
import numpy as np

PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from src.analityc.core.perimetrales_multicam import PerimetralesMultiCam

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def find_videos(folder):
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    return files


def run_once(videos, params, max_frames=120, device='cpu'):
    cap = [cv2.VideoCapture(v) for v in videos]
    pm = PerimetralesMultiCam(
        model_path=params.get('model_path', 'yolo12m.pt'),
        device=device,
        match_threshold=params['match_threshold'],
        orb_weight=params['orb_weight'],
        max_hist_samples=params['max_hist_samples'],
        force_reid=params['force_reid'],
        debug=False
    )

    ids_seen = {i+1: set() for i in range(len(videos))}
    frame_idx = 0
    while frame_idx < max_frames:
        rets = [c.read() for c in cap]
        if not any(r for r, _ in rets):
            break
        for i, (ret, frame) in enumerate(rets):
            if not ret:
                continue
            res = pm.process_frame(frame, camera_id=i+1)
            for d in res:
                ids_seen[i+1].add(d['global_id'])
        frame_idx += 1

    for c in cap:
        c.release()

    # compute common ids intersection across all cams
    inter = set.intersection(*[s for s in ids_seen.values()]) if ids_seen else set()
    return ids_seen, inter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', default=r"C:\Users\Sistema-1\Documents\camaras", help='Folder with camera videos')
    parser.add_argument('--out', default='multicam_grid_results.csv')
    parser.add_argument('--full', action='store_true', help='Run full grid (slow)')
    args = parser.parse_args()

    vids = find_videos(args.videos)
    if len(vids) < 2:
        print('Need at least 2 videos in', args.videos)
        return

    # decidir dispositivo
    device = 'cpu'
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                device = 'cuda'
        except Exception:
            device = 'cpu'
    print('Dispositivo de inferencia:', device)

    # grid definitions
    match_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40] if args.full else [0.25, 0.30]
    orb_weights = [0.0, 0.3, 0.6] if args.full else [0.0, 0.3]
    max_hist_samples_list = [3, 5, 10, 20] if args.full else [5, 10]
    force_reid_list = [False, True] if args.full else [False]

    # model candidate
    candidates = [
        os.path.join(PROJECT_ROOT, 'yolo26m.pt'),
        os.path.join(PROJECT_ROOT, 'models', 'base', 'yolo26m.pt'),
        os.path.join(PROJECT_ROOT, 'models', 'yolo26m.pt')
    ]
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
    if model_path is None:
        print('No local yolo12m found; will let class download or use default (may be slow).')

    out_rows = []
    total = len(match_thresholds) * len(orb_weights) * len(max_hist_samples_list) * len(force_reid_list)
    idx = 0
    for mt in match_thresholds:
        for ow in orb_weights:
            for mh in max_hist_samples_list:
                for fr in force_reid_list:
                    idx += 1
                    params = {'match_threshold': mt, 'orb_weight': ow, 'max_hist_samples': mh, 'force_reid': fr}
                    if model_path:
                        params['model_path'] = model_path
                    print(f'[{idx}/{total}] Testing mt={mt} ow={ow} mh={mh} fr={fr}')
                    start = time.time()
                    ids_seen, inter = run_once(vids[:2], params, max_frames=120, device=device)
                    elapsed = time.time() - start
                    row = {
                        'match_threshold': mt,
                        'orb_weight': ow,
                        'max_hist_samples': mh,
                        'force_reid': fr,
                        'ids_cam1': ';'.join(map(str, sorted(ids_seen[1]))),
                        'ids_cam2': ';'.join(map(str, sorted(ids_seen[2]))),
                        'intersection_count': len(inter),
                        'intersection_ids': ';'.join(map(str, sorted(inter))),
                        'elapsed_s': round(elapsed, 2)
                    }
                    out_rows.append(row)

    # save CSV
    keys = ['match_threshold', 'orb_weight', 'max_hist_samples', 'force_reid', 'ids_cam1', 'ids_cam2', 'intersection_count', 'intersection_ids', 'elapsed_s']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print('Grid done. Results saved to', args.out)


if __name__ == '__main__':
    main()
