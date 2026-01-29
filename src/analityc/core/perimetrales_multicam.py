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

    PRESETS = {
        # Preset recomendado por estabilidad/latencia (según grid local)
        'balanced': {
            'match_threshold': 0.20,
            'orb_weight': 0.3,
            'max_hist_samples': 10,
            'max_reid_samples': 15,
            'reid_weight': 0.8,
            'reid_min_score': 0.40,
            'local_iou_threshold': 0.30,
            'cross_camera_margin': 0.05,
            'force_reid': True
        },
        # Preset más estricto (menos falsas asociaciones)
        'strict': {
            'match_threshold': 0.35,
            'orb_weight': 0.3,
            'max_hist_samples': 10,
            'max_reid_samples': 10,
            'reid_weight': 0.7,
            'reid_min_score': 0.55,
            'local_iou_threshold': 0.35,
            'cross_camera_margin': 0.03,
            'force_reid': False
        }
    }

    def __init__(self,
                 model_path: str = "yolo26m.pt",
                 device: str = "cpu",
                 match_threshold: float = 0.30,
                 orb_weight: float = 0.3,
                 max_hist_samples: int = 10,
                 max_reid_samples: int = 10,
                 reid_weight: float = 0.5,
                 reid_min_score: float = 0.45,
                 local_iou_threshold: float = 0.3,
                 cross_camera_margin: float = 0.05,
                 force_reid: bool = True,
                 debug: bool = False,
                 registry_ttl: float = 3600.0,
                 pay_match_threshold: float = 0.6,
                 pending_pay_ttl: float = 1800.0,
                 preset: str = "balanced"):
        # aplicar preset si existe
        if isinstance(preset, str) and preset in self.PRESETS:
            p = self.PRESETS[preset]
            match_threshold = p.get('match_threshold', match_threshold)
            orb_weight = p.get('orb_weight', orb_weight)
            max_hist_samples = p.get('max_hist_samples', max_hist_samples)
            max_reid_samples = p.get('max_reid_samples', max_reid_samples)
            reid_weight = p.get('reid_weight', reid_weight)
            reid_min_score = p.get('reid_min_score', reid_min_score)
            local_iou_threshold = p.get('local_iou_threshold', local_iou_threshold)
            cross_camera_margin = p.get('cross_camera_margin', cross_camera_margin)
            force_reid = p.get('force_reid', force_reid)
        self.preset = preset
        self.model_path = model_path
        self.device = device
        self.match_threshold = match_threshold
        # peso para combinar HS (coseno) y ORB (match ratio). final = (1-orb_weight)*hs + orb_weight*orb
        self.orb_weight = float(max(0.0, min(1.0, orb_weight)))
        # umbral IoU para reutilizar track local y evitar crear nuevos IDs en la misma cámara
        self.local_iou_threshold = float(max(0.0, min(1.0, local_iou_threshold)))
        # margen extra para permitir match entre cámaras (reduce threshold si no se ha visto en esa cámara)
        self.cross_camera_margin = float(max(0.0, min(0.2, cross_camera_margin)))
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
        # Número máximo de muestras de hist/orb por identity para mantener variaciones entre cámaras
        self.max_hist_samples = int(max(1, max_hist_samples))
        # Número máximo de muestras de reid por identity
        self.max_reid_samples = int(max(1, max_reid_samples))
        # Peso de reid cuando está disponible
        self.reid_weight = float(max(0.0, min(1.0, reid_weight)))
        # Score mínimo de reid para aceptar match cross-camera
        self.reid_min_score = float(max(0.0, min(1.0, reid_min_score)))
        self.force_reid = bool(force_reid)
        # Tiempo (s) para mantener un global_id activo aunque la persona esté fuera de rango
        self.registry_ttl = float(registry_ttl)
        # Matching entre zona de pago y retiro
        self.pay_match_threshold = float(max(0.0, min(1.0, pay_match_threshold)))
        self.pending_pay_ttl = float(max(1.0, pending_pay_ttl))
        # Pending pagos: lista de {global_id, camera_id, ts, hist}
        self.pending_pay: List[Dict[str, Any]] = []

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
        # Estado por cámara: para cada camera_id guardamos tracks locales y contador
        # camera_states[camera_id] = {
        #   'tracks': { local_id: {'bbox':(x1,y1,x2,y2), 'global_id': gid, 'last_seen': ts} },
        #   'next_local_id': int
        # }
        self.camera_states: Dict[Any, Dict[str, Any]] = {}
        # tiempo en segundos para considerar un track local como expirado
        self.local_track_ttl = 2.0

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
                    ck = torch.load(chosen, map_location='cpu', weights_only=True)
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

        # If user requested to force OSNet initialization, try again even if TORCHREID_AVAILABLE was False
        if self.reid_extractor is None and self.force_reid:
            try:
                import importlib
                torch_mod = importlib.import_module('torch')
                try:
                    torchreid_mod = importlib.import_module('torchreid')
                    tr_available = True
                except Exception:
                    torchreid_mod = None
                    tr_available = False

                ckpt_dir = os.path.join(os.getcwd(), 'models', 'osnet')
                market_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_market1501.pt')
                duke_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_dukemtmcreid.pt')
                chosen = None
                if os.path.exists(market_ckpt):
                    chosen = market_ckpt
                elif os.path.exists(duke_ckpt):
                    chosen = duke_ckpt

                if chosen is None and tr_available:
                    try:
                        self.reid_extractor = torchreid_mod.utils.FeatureExtractor(
                            model_name='osnet_x1_0',
                            model_path=None,
                            device='cuda' if torch_mod.cuda.is_available() else 'cpu'
                        )
                    except Exception:
                        self.reid_extractor = None
                elif chosen is not None:
                    from torchreid.reid.models import osnet as osnet_model
                    net = osnet_model.osnet_x1_0(pretrained=False)
                    ck = torch_mod.load(chosen, map_location='cpu', weights_only=True)
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
                    dev = 'cuda' if torch_mod.cuda.is_available() else 'cpu'
                    net = net.to(dev).eval()

                    def _extractor_force(img_bgr):
                        try:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            h, w = 256, 128
                            resized = cv2.resize(img_rgb, (w, h))
                            tensor = torch_mod.from_numpy(resized.astype('float32') / 255.0).permute(2,0,1).unsqueeze(0)
                            tensor = tensor.to(dev)
                            with torch_mod.no_grad():
                                feat = net(tensor)
                            feat = feat.cpu().numpy().ravel().astype('float32')
                            norm = np.linalg.norm(feat)
                            if norm > 0:
                                feat = feat / norm
                            return feat
                        except Exception:
                            return None

                    self.reid_extractor = _extractor_force

                if self.debug and self.reid_extractor is not None:
                    print(f'PerimetralesMultiCam: forced OSNet extractor initialized, checkpoint={chosen}')
            except Exception:
                # forced attempt failed; keep reid_extractor as None
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

    

    def _iou(self, boxA, boxB) -> float:
        (xA1, yA1, xA2, yA2) = boxA
        (xB1, yB1, xB2, yB2) = boxB
        x_left = max(xA1, xB1)
        y_top = max(yA1, yB1)
        x_right = min(xA2, xB2)
        y_bottom = min(yA2, yB2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        inter = (x_right - x_left) * (y_bottom - y_top)
        areaA = max(0, xA2 - xA1) * max(0, yA2 - yA1)
        areaB = max(0, xB2 - xB1) * max(0, yB2 - yB1)
        union = areaA + areaB - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

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

    def _point_in_poly(self, point: Tuple[float, float], polygon: Optional[np.ndarray]) -> bool:
        if polygon is None or len(polygon) < 3:
            return False
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def _center_of(self, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _cleanup_pending_pay(self):
        if not self.pending_pay:
            return
        now = time.time()
        self.pending_pay = [p for p in self.pending_pay if (now - p.get('ts', now)) <= self.pending_pay_ttl]

    def _match_pending_pay(self, desc: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
        best = None
        best_score = -1.0
        dnorm = np.linalg.norm(desc)
        if dnorm == 0 or not self.pending_pay:
            return None, best_score
        for p in self.pending_pay:
            h = p.get('hist')
            if h is None:
                continue
            try:
                score = float(np.dot(h, desc) / (np.linalg.norm(h) * dnorm + 1e-8))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best = p
        return best, best_score

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

    def _match_registry(self, desc: np.ndarray, des_orb: Optional[np.ndarray] = None,
                        reid_feat: Optional[np.ndarray] = None) -> Tuple[Optional[int], float, float]:
        """Compara `desc` con el registro global usando similaridad coseno.
        Retorna (best_id, best_score) donde score en [0,1].
        """
        best_id = None
        best_score = -1.0
        best_reid = 0.0
        dnorm = np.linalg.norm(desc)
        if dnorm == 0:
            return None, -1.0, 0.0
        for gid, info in self.registry.items():
            # try multiple hist samples if available (helps cross-camera variation)
            hists = info.get('hists')
            if hists is None or len(hists) == 0:
                continue
            # best per-gid scores
            best_hs = 0.0
            for ref in hists:
                try:
                    s = float(np.dot(ref, desc) / (np.linalg.norm(ref) * dnorm + 1e-8))
                except Exception:
                    s = 0.0
                if s > best_hs:
                    best_hs = s

            # ORB: try all stored ORB samples
            orb_score = 0.0
            if des_orb is not None:
                ref_orbs = info.get('orbs', [])
                for ro in ref_orbs:
                    orb_score = max(orb_score, self._orb_match_score(des_orb, ro))

            # ReID: try all stored reid samples
            reid_score = 0.0
            if reid_feat is not None:
                ref_reids = info.get('reids', [])
                for rr in ref_reids:
                    try:
                        s = float(np.dot(rr, reid_feat) / (np.linalg.norm(rr) * np.linalg.norm(reid_feat) + 1e-8))
                    except Exception:
                        s = 0.0
                    if s > reid_score:
                        reid_score = s

            base_score = (1.0 - self.orb_weight) * best_hs + self.orb_weight * orb_score
            if reid_feat is not None:
                score = (1.0 - self.reid_weight) * base_score + self.reid_weight * reid_score
            else:
                score = base_score
            if score > best_score:
                best_score = score
                best_reid = reid_score
                best_id = gid
        return best_id, best_score, best_reid

    def _update_registry(self, gid: int, hist: np.ndarray, camera_id: int):
        info = self.registry.get(gid)
        now = time.time()
        if info is None:
            self.registry[gid] = {
                'hists': [hist.copy()],
                'hist': hist.copy(),
                'orbs': [],
                'reid': None,
                'reids': [],
                'last_seen': now,
                'cameras': set([camera_id])
            }
        else:
            # append sample and cap stored samples to preserve cross-camera variations
            hists = info.get('hists', [])
            hists.append(hist.copy())
            if len(hists) > self.max_hist_samples:
                hists = hists[-self.max_hist_samples:]
            info['hists'] = hists
            # maintain an averaged histogram for quick reference (smoothed)
            try:
                avg = np.mean(np.stack(hists, axis=0), axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                info['hist'] = avg
            except Exception:
                info['hist'] = hist.copy()
            info['last_seen'] = now
            cams = info.get('cameras', set())
            cams.add(camera_id)
            info['cameras'] = cams

    def cleanup_registry(self):
        """Remove registry entries not seen for longer than `registry_ttl` seconds.

        Keeps global IDs for `registry_ttl` seconds after last seen to preserve identity
        even if the person leaves the monitored area.
        """
        if self.registry_ttl is None or self.registry_ttl <= 0:
            return
        now = time.time()
        to_delete = []
        for gid, info in list(self.registry.items()):
            last = info.get('last_seen', now)
            if now - last > self.registry_ttl:
                to_delete.append(gid)
        for gid in to_delete:
            if self.debug:
                info = self.registry.get(gid, {})
                try:
                    ls = info.get('last_seen')
                except Exception:
                    ls = None
                print(f'PerimetralesMultiCam: expiring registry id={gid} last_seen={ls}')
            self.registry.pop(gid, None)

    def process_frame(self, frame: np.ndarray, camera_id: int,
                      pay_roi: Optional[List[Tuple[int, int]]] = None,
                      withdraw_roi: Optional[List[Tuple[int, int]]] = None,
                      pay_roi_activate: bool = True,
                      withdraw_roi_activate: bool = True) -> List[Dict[str, Any]]:
        """Procesa un frame de la cámara `camera_id` y retorna lista de tracks:
        [{'global_id': int, 'bbox': (x1,y1,x2,y2), 'conf': float, 'camera_id': int}]

        - Ejecuta detección con YOLO (si está disponible) para la clase `person` (0).
        - Para cada detección de persona calcula descriptor y hace match con el registro global.
        - Si no hay match suficiente, asigna un nuevo `global_id`.
        """
        # limpiar entradas del registro que hayan expirado antes de procesar
        try:
            self.cleanup_registry()
        except Exception:
            pass

        self.frame_counter += 1
        results = []
        result_index_by_local = {}

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

        # ---- Estado por cámara ----
        cam_state = self.camera_states.get(camera_id)
        if cam_state is None:
            cam_state = {
                'tracks': {},
                'next_local_id': 1,
                'pay_roi': None,
                'withdraw_roi': None,
                'pay_roi_active': True,
                'withdraw_roi_active': True
            }
            self.camera_states[camera_id] = cam_state

        cam_state['pay_roi_active'] = bool(pay_roi_activate)
        cam_state['withdraw_roi_active'] = bool(withdraw_roi_activate)

        if pay_roi is not None:
            cam_state['pay_roi'] = np.array(pay_roi, np.int32)
        if withdraw_roi is not None:
            cam_state['withdraw_roi'] = np.array(withdraw_roi, np.int32)

        tracks = cam_state['tracks']
        assigned_tracks = set()
        assigned_dets = set()
        used_global_ids = set()
        now_ts = time.time()
        active_gids = set()
        for tinfo in tracks.values():
            if now_ts - tinfo.get('last_seen', now_ts) <= self.local_track_ttl:
                active_gids.add(tinfo.get('global_id'))

        # Pass 1: reutilizar global_id de tracks locales (estabilidad intra-cámara)
        for di, (x1, y1, x2, y2, conf) in enumerate(detections):
            dbb = (x1, y1, x2, y2)
            best_tid = None
            best_iou = 0.0
            for tid, tinfo in tracks.items():
                if tid in assigned_tracks:
                    continue
                i = self._iou(dbb, tinfo['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_tid is not None and best_iou >= self.local_iou_threshold:
                gid = tracks[best_tid]['global_id']
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2c - x1c <= 0 or y2c - y1c <= 0:
                    continue
                crop = frame[y1c:y2c, x1c:x2c]
                reid_feat = self._extract_reid(crop)
                desc = self._compute_hs_descriptor(crop)
                des_orb = self._compute_orb_descriptor(crop)

                # Si el gid ya fue usado por otra detección en este frame, reasignar uno nuevo
                if gid in used_global_ids:
                    gid = self.next_global_id
                    self.next_global_id += 1
                    tracks[best_tid]['global_id'] = gid

                # actualizar registro global con la evidencia actual
                self._update_registry(gid, desc, camera_id)
                if des_orb is not None:
                    orbs = self.registry[gid].get('orbs', [])
                    orbs.append(des_orb)
                    if len(orbs) > self.max_hist_samples:
                        orbs = orbs[-self.max_hist_samples:]
                    self.registry[gid]['orbs'] = orbs
                    self.registry[gid]['orb'] = des_orb
                if reid_feat is not None:
                    reids = self.registry[gid].get('reids', [])
                    reids.append(reid_feat)
                    if len(reids) > self.max_reid_samples:
                        reids = reids[-self.max_reid_samples:]
                    self.registry[gid]['reids'] = reids
                    self.registry[gid]['reid'] = reid_feat

                tracks[best_tid]['bbox'] = dbb
                tracks[best_tid]['last_seen'] = time.time()
                tracks[best_tid]['last_desc'] = desc

                results.append({
                    'global_id': gid,
                    'bbox': dbb,
                    'conf': conf,
                    'camera_id': camera_id,
                    'local_id': best_tid
                })
                result_index_by_local[best_tid] = len(results) - 1
                used_global_ids.add(gid)
                assigned_tracks.add(best_tid)
                assigned_dets.add(di)

        # Pass 2: detecciones nuevas -> matching con registro global
        for di, (x1, y1, x2, y2, conf) in enumerate(detections):
            if di in assigned_dets:
                continue
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2c - x1c <= 0 or y2c - y1c <= 0:
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            reid_feat = self._extract_reid(crop)
            desc = self._compute_hs_descriptor(crop)
            des_orb = self._compute_orb_descriptor(crop)

            # If we have reid features, prefer them for matching
            best_id, best_score, best_reid = self._match_registry(desc, des_orb, reid_feat)

            # Evitar reusar un global_id ya usado en este frame o en otro track activo
            if best_id is not None and (best_id in used_global_ids or best_id in active_gids):
                best_id = None
                best_score = -1.0

            # Ajustar threshold si el ID viene de otra cámara
            effective_threshold = self.match_threshold
            accept_by_reid = False
            if best_id is not None:
                info = self.registry.get(best_id, {})
                cams = info.get('cameras', set())
                if camera_id not in cams:
                    effective_threshold = max(0.0, self.match_threshold - self.cross_camera_margin)
                    if reid_feat is not None and best_reid >= self.reid_min_score:
                        accept_by_reid = True
                    elif reid_feat is not None and best_reid < self.reid_min_score:
                        best_id = None
                        best_score = -1.0

            # Determinar gid y actualizar registro
            if best_id is not None and (best_score >= effective_threshold or accept_by_reid):
                gid = best_id
                self._update_registry(gid, desc, camera_id)
                if des_orb is not None:
                    orbs = self.registry[gid].get('orbs', [])
                    orbs.append(des_orb)
                    if len(orbs) > self.max_hist_samples:
                        orbs = orbs[-self.max_hist_samples:]
                    self.registry[gid]['orbs'] = orbs
                    self.registry[gid]['orb'] = des_orb
                if reid_feat is not None:
                    reids = self.registry[gid].get('reids', [])
                    reids.append(reid_feat)
                    if len(reids) > self.max_reid_samples:
                        reids = reids[-self.max_reid_samples:]
                    self.registry[gid]['reids'] = reids
                    self.registry[gid]['reid'] = reid_feat
                if self.debug:
                    print(f"Matched to existing ID {gid} (score={best_score:.3f}) from camera {camera_id}")
            else:
                gid = self.next_global_id
                self.next_global_id += 1
                self._update_registry(gid, desc, camera_id)
                if des_orb is not None:
                    self.registry[gid].setdefault('orbs', []).append(des_orb)
                    self.registry[gid]['orb'] = des_orb
                if reid_feat is not None:
                    self.registry[gid]['reids'] = [reid_feat]
                    self.registry[gid]['reid'] = reid_feat
                if self.debug:
                    print(f"Registered new ID {gid} on camera {camera_id} (best_score={best_score:.3f})")

            used_global_ids.add(gid)

            nid = cam_state['next_local_id']
            cam_state['next_local_id'] += 1
            tracks[nid] = {
                'bbox': (x1, y1, x2, y2),
                'global_id': gid,
                'last_seen': time.time(),
                'last_desc': desc
            }

            results.append({
                'global_id': gid,
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'camera_id': camera_id,
                'local_id': nid
            })
            result_index_by_local[nid] = len(results) - 1

        # Actualizar lógica de zonas (pay/withdraw)
        pay_poly = cam_state.get('pay_roi') if cam_state.get('pay_roi_active', True) else None
        withdraw_poly = cam_state.get('withdraw_roi') if cam_state.get('withdraw_roi_active', True) else None
        if pay_poly is not None or withdraw_poly is not None:
            self._cleanup_pending_pay()
            for local_id, idx in result_index_by_local.items():
                tinfo = tracks.get(local_id)
                if tinfo is None:
                    continue
                center = self._center_of(tinfo['bbox'])
                in_pay = self._point_in_poly(center, pay_poly)
                in_withdraw = self._point_in_poly(center, withdraw_poly)

                new_zone = None
                if in_pay:
                    new_zone = 'pay_roi'
                elif in_withdraw:
                    new_zone = 'withdraw_roi'

                prev_zone = tinfo.get('zone')
                zone_event = None
                if new_zone != prev_zone:
                    if new_zone == 'pay_roi':
                        zone_event = 'pagando'
                    elif new_zone == 'withdraw_roi':
                        zone_event = 'retirando'
                    elif prev_zone == 'pay_roi':
                        zone_event = 'salio_pagando'
                    elif prev_zone == 'withdraw_roi':
                        zone_event = 'salio_retirando'

                tinfo['zone'] = new_zone
                results[idx]['zone'] = new_zone
                results[idx]['zone_event'] = zone_event

                # Matching entre zona de pago y retiro
                if zone_event == 'pagando':
                    desc = tinfo.get('last_desc')
                    if desc is not None:
                        self.pending_pay.append({
                            'global_id': tinfo.get('global_id'),
                            'camera_id': camera_id,
                            'ts': time.time(),
                            'hist': desc
                        })
                elif zone_event == 'retirando':
                    desc = tinfo.get('last_desc')
                    if desc is not None:
                        match, score = self._match_pending_pay(desc)
                        if match is not None and score >= self.pay_match_threshold:
                            total_seconds = int(time.time() - match.get('ts', time.time()))
                            results[idx]['pay_match_id'] = match.get('global_id')
                            results[idx]['pay_match_score'] = float(score)
                            results[idx]['total_seconds'] = total_seconds
                            results[idx]['alert'] = {
                                'event': 'pay_withdraw_match',
                                'message': 'Cliente emparejado entre Caja y Retiro',
                                'pay_global_id': match.get('global_id'),
                                'withdraw_global_id': tinfo.get('global_id'),
                                'score': float(score),
                                'total_seconds': total_seconds
                            }

        # Limpiar tracks locales expirados
        now = time.time()
        to_delete = []
        for tid, tinfo in list(tracks.items()):
            if now - tinfo.get('last_seen', now) > self.local_track_ttl:
                to_delete.append(tid)
        for tid in to_delete:
            tracks.pop(tid, None)

        # Devolver resultados con local_id rellenado (y global_id/camera_id preservados)
        return results
