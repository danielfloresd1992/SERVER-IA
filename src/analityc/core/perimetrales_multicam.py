import os
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
    import torchreid
    TORCHREID_AVAILABLE = True
except Exception:
    TORCHREID_AVAILABLE = False


class PerimetralesMultiCam:
    """Procesador multicámara que mantiene IDs entre cámaras usando descriptors de apariencia.

    Enfoque ligero: crea un descriptor HSV (canal H) por persona y compara con el registro global
    usando correlación de histogramas. Si la similitud supera `match_threshold` se reutiliza el ID.
    """

    def __init__(self,
                 model_path: str = "yolo12m.pt",
                 device: str = "cpu",
                 match_threshold: float = 0.45,
                 orb_weight: float = 0.4,
                 debug: bool = False):
        self.model_path = model_path
        self.device = device
        self.match_threshold = match_threshold
        # peso para combinar HS (coseno) y ORB (match ratio). final = (1-orb_weight)*hs + orb_weight*orb
        self.orb_weight = float(max(0.0, min(1.0, orb_weight)))
        self.debug = debug

        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(self.model_path).to(self.device)
            except Exception:
                self.model = None

        # Registro global de identidades: global_id -> {'hist': hist, 'last_seen': ts, 'camera_id': cam}
        self.registry: Dict[int, Dict[str, Any]] = {}
        self.next_global_id = 1

        # Parámetros de histograma (canal H y S)
        self.h_bins = 30
        self.s_bins = 32
        self.h_range = [0, 180]
        self.s_range = [0, 256]

        # ORB descriptor
        self.orb_max_kp = 200
        try:
            self.orb = cv2.ORB_create(nfeatures=self.orb_max_kp)
        except Exception:
            self.orb = None
        # BFMatcher for ORB (Hamming)
        try:
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        except Exception:
            self.bf = None

        # Último frame/time tracking
        self.frame_counter = 0

        # ReID extractor (OSNet) if available
        self.reid_extractor = None
        self.reid_device = 'cuda' if (TORCHREID_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        if TORCHREID_AVAILABLE:
            try:
                ckpt_dir = os.path.join(os.getcwd(), 'models', 'osnet')
                market_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_market1501.pt')
                duke_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_dukemtmcreid.pt')
                chosen = None
                if os.path.exists(market_ckpt):
                    chosen = market_ckpt
                elif os.path.exists(duke_ckpt):
                    chosen = duke_ckpt

                if chosen is None:
                    # use torchreid high-level extractor (may download imagenet weights)
                    try:
                        self.reid_extractor = torchreid.utils.FeatureExtractor(
                            model_name='osnet_x1_0',
                            model_path=None,
                            device=self.reid_device
                        )
                    except Exception:
                        self.reid_extractor = None
                else:
                    # build OSNet and load checkpoint without the classifier layer
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
                if self.debug and self.reid_extractor is not None:
                    print(f'PerimetralesMultiCam: OSNet extractor initialized on {self.reid_device}, checkpoint={chosen}')
            except Exception:
                self.reid_extractor = None

    def _compute_hs_descriptor(self, crop: np.ndarray) -> np.ndarray:
        """Calcula un descriptor HS (histograma 2D HxS) y lo normaliza L2."""
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins], self.h_range + self.s_range)
            hist = hist.flatten().astype(np.float32)
            # Normalizar para comparación coseno (L2)
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            return hist
        except Exception:
            return np.zeros((self.h_bins * self.s_bins,), dtype=np.float32)

    def _extract_reid(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extrae feature de re-identificación (OSNet) si está disponible, devuelve vector L2-normalizado."""
        if self.reid_extractor is None:
            return None
        try:
            # if torchreid.utils.FeatureExtractor, it accepts PIL or numpy BGR; our wrapper handles both
            feats = None
            try:
                feats = self.reid_extractor(crop)
            except Exception:
                # maybe extractor is a callable wrapper returning numpy vector
                feats = self.reid_extractor(crop)
            if feats is None:
                return None
            if isinstance(feats, list) and len(feats) > 0:
                f = feats[0].cpu().numpy().ravel().astype(np.float32)
            elif isinstance(feats, np.ndarray):
                f = feats.ravel().astype(np.float32)
            else:
                # unknown type
                return None
            norm = np.linalg.norm(f)
            if norm > 0:
                f = f / norm
            return f
        except Exception:
            return None

    def _compute_orb_descriptor(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Calcula descriptores ORB y devuelve array de descriptors (N x 32 uint8) o None."""
        if self.orb is None:
            return None
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            kps, des = self.orb.detectAndCompute(gray, None)
            return des  # may be None if no keypoints
        except Exception:
            return None

    def _orb_match_score(self, des_query: Optional[np.ndarray], des_ref: Optional[np.ndarray]) -> float:
        """Calcula una puntuación normalizada [0,1] basada en matches ORB entre query y ref.
        Usa kNN y ratio test; score = good_matches / max(1, min(num_kp_query, num_kp_ref)).
        """
        if self.bf is None or des_query is None or des_ref is None:
            return 0.0
        try:
            # kNN matches
            matches = self.bf.knnMatch(des_query, des_ref, k=2)
            good = 0
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good += 1
            denom = max(1, min(len(des_query), len(des_ref)))
            return float(good) / float(denom)
        except Exception:
            return 0.0

    def _match_registry(self, desc: np.ndarray, des_orb: Optional[np.ndarray] = None) -> Tuple[Optional[int], float]:
        """Compara `desc` con el registro global usando similaridad coseno.
        Retorna (best_id, best_score) donde score en [0,1].
        """
        best_id = None
        best_score = -1.0
        dnorm = np.linalg.norm(desc)
        if dnorm == 0:
            return None, -1.0
        for gid, info in self.registry.items():
            ref = info.get('hist')
            if ref is None:
                continue
            try:
                hs_score = float(np.dot(ref, desc) / (np.linalg.norm(ref) * dnorm + 1e-8))
            except Exception:
                hs_score = 0.0
            # ORB score if available
            ref_orb = info.get('orb')
            orb_score = self._orb_match_score(des_orb, ref_orb) if des_orb is not None else 0.0
            # combinar
            score = (1.0 - self.orb_weight) * hs_score + self.orb_weight * orb_score
            if score > best_score:
                best_score = score
                best_id = gid
        return best_id, best_score

    def _update_registry(self, gid: int, hist: np.ndarray, camera_id: int):
        info = self.registry.get(gid)
        now = time.time()
        if info is None:
            self.registry[gid] = {'hist': hist, 'last_seen': now, 'camera_id': camera_id}
        else:
            # suavizado simple del descriptor
            info['hist'] = 0.7 * info['hist'] + 0.3 * hist
            info['last_seen'] = now
            info['camera_id'] = camera_id

    def process_frame(self, frame: np.ndarray, camera_id: int) -> List[Dict[str, Any]]:
        """Procesa un frame de la cámara `camera_id` y retorna lista de tracks:
        [{'global_id': int, 'bbox': (x1,y1,x2,y2), 'conf': float, 'camera_id': int}]

        - Ejecuta detección con YOLO (si está disponible) para la clase `person` (0).
        - Para cada detección de persona calcula descriptor y hace match con el registro global.
        - Si no hay match suficiente, asigna un nuevo `global_id`.
        """
        self.frame_counter += 1
        results = []

        detections: List[Tuple[int, int, int, int, float]] = []  # bbox + conf

        if self.model is not None:
            try:
                preds = self.model.predict(frame, imgsz=640, device=self.device, classes=[0], verbose=False)
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
                            x1, y1, x2, y2 = map(int, b[:4])
                            detections.append((x1, y1, x2, y2, float(c)))
            except Exception:
                detections = []

        # Si no hay modelo, devolvemos vacío
        if not detections:
            return []

        for (x1, y1, x2, y2, conf) in detections:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2c - x1c <= 0 or y2c - y1c <= 0:
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            # Try OSNet feature first
            reid_feat = self._extract_reid(crop)
            desc = self._compute_hs_descriptor(crop)
            des_orb = self._compute_orb_descriptor(crop)

            # If we have reid features, prefer them for matching
            if reid_feat is not None:
                # compare against registry 'reid' vectors if present
                best_id, best_score = self._match_registry(desc, des_orb)
                # override score if reid matches exist
                best_id_reid = None
                best_score_reid = -1.0
                for gid, info in self.registry.items():
                    ref = info.get('reid')
                    if ref is None:
                        continue
                    try:
                        s = float(np.dot(ref, reid_feat) / (np.linalg.norm(ref) * np.linalg.norm(reid_feat) + 1e-8))
                    except Exception:
                        s = 0.0
                    if s > best_score_reid:
                        best_score_reid = s
                        best_id_reid = gid
                # if reid score is confident, use it
                if best_id_reid is not None and best_score_reid >= self.match_threshold:
                    best_id = best_id_reid
                    best_score = best_score_reid
            else:
                best_id, best_score = self._match_registry(desc, des_orb)
            if best_id is not None and best_score >= self.match_threshold:
                gid = best_id
                self._update_registry(gid, desc, camera_id)
                # actualizar ORB y reid también
                if des_orb is not None:
                    self.registry[gid]['orb'] = des_orb
                if reid_feat is not None:
                    self.registry[gid]['reid'] = reid_feat
                if self.debug:
                    print(f"Matched to existing ID {gid} (score={best_score:.3f}) from camera {camera_id}")
            else:
                gid = self.next_global_id
                self.next_global_id += 1
                # almacenar HS y ORB
                self._update_registry(gid, desc, camera_id)
                if des_orb is not None:
                    self.registry[gid]['orb'] = des_orb
                if reid_feat is not None:
                    self.registry[gid]['reid'] = reid_feat
                if self.debug:
                    print(f"Registered new ID {gid} on camera {camera_id} (best_score={best_score:.3f})")

            results.append({
                'global_id': gid,
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'camera_id': camera_id
            })

        return results
