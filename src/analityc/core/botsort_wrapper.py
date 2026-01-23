import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment

try:
    import torch
    import torchreid
    TORCHREID_AVAILABLE = True
except Exception:
    TORCHREID_AVAILABLE = False

from .perimetrales_multicam import PerimetralesMultiCam


class BoTSORTWrapper:
    """Wrapper simplificado estilo BoTSORT + OSNet (si está disponible).

    Funcionamiento:
    - Usa un detector (se espera integrar YOLO desde caller): recibir detections
      como lista de (x1,y1,x2,y2,conf).
    - Extrae feature de apariencia con OSNet (torchreid) si está disponible,
      si no, usa HS descriptor del `PerimetralesMultiCam`.
    - Empareja detecciones con tracks usando coste combinado (1 - cos_sim) + (1 - IoU).
    - Mantiene `track_id` estable y expone `process_frame(frame, camera_id)`.
    """

    def __init__(self, model_path: str = "yolo12m.pt", device: str = "cpu", match_thresh: float = 0.45, debug: bool = False):
        self.debug = debug
        self.device = device
        self.match_thresh = match_thresh

        # Reuse HS+ORB utilities from PerimetralesMultiCam for fallback
        self.appear = PerimetralesMultiCam(model_path=model_path, device=device, match_threshold=match_thresh, debug=debug)

        # Try to setup OSNet extractor using local checkpoints (models/osnet/)
        self.reid_extractor = None
        self.reid_device = 'cuda' if (TORCHREID_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        if TORCHREID_AVAILABLE:
            try:
                # Prefer local Market1501 or Duke checkpoints if present
                ckpt_dir = os.path.join(os.getcwd(), 'models', 'osnet')
                market_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_market1501.pt')
                duke_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_dukemtmcreid.pt')
                chosen = None
                if os.path.exists(market_ckpt):
                    chosen = market_ckpt
                elif os.path.exists(duke_ckpt):
                    chosen = duke_ckpt

                # If torchreid FeatureExtractor is available and user didn't provide checkpoint,
                # try to use the high-level extractor which may auto-download imagenet weights.
                try:
                    if chosen is None:
                        self.reid_extractor = torchreid.utils.FeatureExtractor(
                            model_name='osnet_x1_0',
                            model_path=None,
                            device=self.reid_device
                        )
                    else:
                        # Build a simple OSNet model and load checkpoint (filter classifier keys)
                        from torchreid.reid.models import osnet as osnet_model
                        net = osnet_model.osnet_x1_0(pretrained=False)
                        ck = torch.load(chosen, map_location='cpu')
                        if isinstance(ck, dict):
                            if 'state_dict' in ck:
                                state = ck['state_dict']
                            elif 'model' in ck:
                                state = ck['model']
                            else:
                                state = ck
                        else:
                            state = ck
                        new_state = {}
                        for k,v in state.items():
                            nk = k.replace('module.', '')
                            if nk.startswith('classifier.') or nk.startswith('fc.'):
                                continue
                            new_state[nk] = v
                        net.load_state_dict(new_state, strict=False)
                        net = net.to(self.reid_device).eval()

                        # create a lightweight extractor wrapper around the torch model
                        def _extractor(img_bgr):
                            try:
                                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                h, w = 256, 128
                                resized = cv2.resize(img_rgb, (w, h))
                                tensor = torch.from_numpy(resized.astype('float32') / 255.0).permute(2,0,1).unsqueeze(0)
                                tensor = tensor.to(self.reid_device)
                                with torch.no_grad():
                                    feat = net(tensor)
                                feat = feat.cpu().numpy().ravel().astype('float32')
                                norm = np.linalg.norm(feat)
                                if norm > 0:
                                    feat = feat / norm
                                return feat
                            except Exception:
                                return None

                        self.reid_extractor = _extractor
                except Exception:
                    self.reid_extractor = None

                if self.debug and self.reid_extractor is not None:
                    print(f'OSNet extractor initialized on {self.reid_device}, checkpoint={chosen}')
            except Exception:
                self.reid_extractor = None

        # Tracks: id -> dict
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        self.max_age = 60.0  # seconds before deleting a track

    @staticmethod
    def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    def _extract_reid(self, frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
        x1,y1,x2,y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Prefer OSNet features via torchreid if available
        if self.reid_extractor is not None:
            try:
                feats = self.reid_extractor(crop)
                if isinstance(feats, list) and len(feats) > 0:
                    f = feats[0].cpu().numpy().ravel().astype(np.float32)
                    # L2 normalize
                    n = np.linalg.norm(f)
                    if n > 0:
                        f = f / n
                    return f
            except Exception:
                if self.debug:
                    print('Warning: OSNet extraction failed, falling back to HS descriptor')

        # Fallback to HS descriptor if OSNet not available or fails
        desc = self.appear._compute_hs_descriptor(crop)
        # ensure L2-normalized
        n = np.linalg.norm(desc)
        if n > 0:
            desc = desc / n
        return desc

    def _build_cost_matrix(self, detections: List[Tuple[int,int,int,int,float]], features: List[Optional[np.ndarray]]) -> np.ndarray:
        n, m = len(detections), len(self.tracks)
        if m == 0:
            return np.zeros((n,0), dtype=np.float32)
        cost = np.ones((n,m), dtype=np.float32)
        track_ids = list(self.tracks.keys())
        for i, det in enumerate(detections):
            bx = det[:4]
            for j, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                iou_score = self.iou(bx, tr['bbox'])
                # appearance similarity (cosine) if features available
                ap_score = 0.0
                if features[i] is not None and tr.get('feature') is not None:
                    a = features[i]
                    b = tr['feature']
                    denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)
                    if denom > 0:
                        ap_score = float(np.dot(a,b)/denom)
                # combine: prefer high iou and high appearance
                combined = 0.5 * iou_score + 0.5 * ap_score
                # cost for assignment (lower better)
                cost[i,j] = 1.0 - combined
        return cost

    def process_frame(self, frame: np.ndarray, camera_id: int) -> List[Dict[str,Any]]:
        """Run detection via internal YOLO (from PerimetralesMultiCam model) and perform tracking.
        Returns list of tracks with stable ids.
        """
        # Use the underlying detector in PerimetralesMultiCam (self.appear.model)
        detections = []  # list of (x1,y1,x2,y2,conf)
        if self.appear.model is not None:
            try:
                preds = self.appear.model.predict(frame, imgsz=640, device=self.device, classes=[0], verbose=False)
                if preds and len(preds) > 0:
                    r = preds[0]
                    if hasattr(r, 'boxes'):
                        boxes = getattr(r, 'boxes')
                        try:
                            xyxy = boxes.xyxy.cpu().numpy()
                            confs = boxes.conf.cpu().numpy()
                        except Exception:
                            xyxy = np.array(boxes.xyxy)
                            confs = np.array(boxes.conf)
                        for b, c in zip(xyxy, confs):
                            x1,y1,x2,y2 = map(int, b[:4])
                            detections.append((x1,y1,x2,y2,float(c)))
            except Exception:
                detections = []

        # Extract features for detections
        features = [self._extract_reid(frame, det[:4]) for det in detections]

        # Build cost matrix and assign
        cost = self._build_cost_matrix(detections, features)
        assigned = {}
        unassigned_dets = list(range(len(detections)))
        unassigned_trks = list(self.tracks.keys())
        if cost.size > 0 and cost.shape[1] > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            track_ids = list(self.tracks.keys())
            for r,c in zip(row_ind, col_ind):
                if cost[r,c] < (1.0 - self.match_thresh):
                    tid = track_ids[c]
                    assigned[r] = tid
                    if r in unassigned_dets: unassigned_dets.remove(r)
                    if tid in unassigned_trks: unassigned_trks.remove(tid)

        # Update assigned tracks
        now = time.time()
        for det_idx, tid in assigned.items():
            x1,y1,x2,y2,conf = detections[det_idx]
            self.tracks[tid]['bbox'] = (x1,y1,x2,y2)
            if features[det_idx] is not None:
                self.tracks[tid]['feature'] = features[det_idx]
            self.tracks[tid]['last_seen'] = now
            self.tracks[tid]['camera_id'] = camera_id

        # Create new tracks for unassigned detections
        for di in unassigned_dets:
            x1,y1,x2,y2,conf = detections[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox': (x1,y1,x2,y2),
                'feature': features[di],
                'last_seen': now,
                'camera_id': camera_id
            }

        # Prune old tracks
        to_del = []
        for tid, tr in list(self.tracks.items()):
            if now - tr['last_seen'] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        # Build output list
        out = []
        for tid, tr in self.tracks.items():
            out.append({'global_id': tid, 'bbox': tr['bbox'], 'conf': None, 'camera_id': tr.get('camera_id', camera_id)})

        return out

    def process_detections(self, frame: np.ndarray, detections: List[Tuple[int,int,int,int,float]], camera_id: int) -> List[Dict[str,Any]]:
        """Procesa una lista de detecciones ya calculadas (x1,y1,x2,y2,conf).
        Esto es útil para pruebas sin depender del detector interno.
        """
        # Extract features for detections
        features = [self._extract_reid(frame, det[:4]) for det in detections]

        # Build cost matrix and assign
        cost = self._build_cost_matrix(detections, features)
        assigned = {}
        unassigned_dets = list(range(len(detections)))
        unassigned_trks = list(self.tracks.keys())
        if cost.size > 0 and cost.shape[1] > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            track_ids = list(self.tracks.keys())
            for r,c in zip(row_ind, col_ind):
                if cost[r,c] < (1.0 - self.match_thresh):
                    tid = track_ids[c]
                    assigned[r] = tid
                    if r in unassigned_dets: unassigned_dets.remove(r)
                    if tid in unassigned_trks: unassigned_trks.remove(tid)

        # Update assigned tracks
        now = time.time()
        for det_idx, tid in assigned.items():
            x1,y1,x2,y2,conf = detections[det_idx]
            self.tracks[tid]['bbox'] = (x1,y1,x2,y2)
            if features[det_idx] is not None:
                self.tracks[tid]['feature'] = features[det_idx]
            self.tracks[tid]['last_seen'] = now
            self.tracks[tid]['camera_id'] = camera_id

        # Create new tracks for unassigned detections
        for di in unassigned_dets:
            x1,y1,x2,y2,conf = detections[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox': (x1,y1,x2,y2),
                'feature': features[di],
                'last_seen': now,
                'camera_id': camera_id
            }

        # Prune old tracks
        to_del = []
        for tid, tr in list(self.tracks.items()):
            if now - tr['last_seen'] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        # Build output list
        out = []
        for tid, tr in self.tracks.items():
            out.append({'global_id': tid, 'bbox': tr['bbox'], 'conf': None, 'camera_id': tr.get('camera_id', camera_id)})

        return out
