"""
HummusVLMProcess — Detección de eventos en mostrador via YOLO + LLaVA VLM.
==========================================================================

Reemplaza el modelo YOLO personalizado de eventos por un clasificador VLM
(LLaVA-1.5-7B 4-bit) que analiza crops de personas detenidas en zonas ROI.

Eventos detectados:
  A) "Toma de Orden"    → empleado entrega ticket al cliente   (zona_caja)
  B) "Entrega de Comida" → cliente retira bandeja naranja       (zona_entrega)

Métrica derivada:
  tiempo_espera = timestamp(B) − timestamp(A)  por track_id

Arquitectura:
  - YOLO (persona) + ByteTrack → tracking estable por ID
  - Dos polígonos ROI (zona_caja, zona_entrega)
  - Detección de "persona detenida" (centroide < 30px en 3s)
  - LLaVA 4-bit en hilo dedicado → clasificación Yes/No del crop
  - Resultados VLM se recogen en frames subsiguientes (no bloquea)

Compatibilidad:
  - Misma interfaz que HummusProcess: process_frame() → (image, metadata)
  - Formato de alertas compatible con AlertsSidebar del cliente
  - Integrable en app.py como tipo de inferencia "HummusVLM"
"""

import base64
import csv
import datetime
import logging
import os
import queue
import threading
import time
import unicodedata
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Imports pesados diferidos: torch, transformers, PIL se cargan solo cuando
# se instancia _VLMSingleton por primera vez. Esto permite que el servidor
# arranque sin transformers/bitsandbytes si nadie usa "HummusVLM".
torch = None          # type: ignore
Image = None          # type: ignore
AutoProcessor = None  # type: ignore
BitsAndBytesConfig = None       # type: ignore
LlavaForConditionalGeneration = None  # type: ignore


def _ensure_vlm_deps() -> None:
    """Importa torch, transformers y PIL la primera vez que se necesitan."""
    global torch, Image, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    if torch is not None:
        return
    import torch as _torch
    torch = _torch
    from PIL import Image as _Image
    Image = _Image
    from transformers import (
        AutoProcessor as _AP,
        BitsAndBytesConfig as _BNB,
        LlavaForConditionalGeneration as _LLaVA,
    )
    AutoProcessor = _AP
    BitsAndBytesConfig = _BNB
    LlavaForConditionalGeneration = _LLaVA

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Rutas por defecto
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PERSON_MODEL_PATH = (
    r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\models\base\yolo12l.pt"
)
DEFAULT_VLM_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_ORDER_SCREENSHOT_DIR = (
    r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\output\Toma_de_orden_hummus"
)
DEFAULT_DELIVERY_SCREENSHOT_DIR = (
    r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\output\Entrega_de_plato_hummus"
)
DEFAULT_ALERT_CSV_PATH = (
    r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\output\hummus_vlm_alertas.csv"
)

# Zonas por defecto (píxeles, para resolución ~960×540).
# Recalibrar según la cámara real.
DEFAULT_ZONA_CAJA = [[50, 330], [290, 330], [290, 530], [50, 530]]
DEFAULT_ZONA_ENTREGA = [[300, 150], [720, 150], [720, 390], [300, 390]]

# Prompts binarios para LLaVA
PROMPT_CAJA = (
    "In this overhead image of a fast-food counter, is an employee's hand "
    "extending a small piece of paper (receipt/ticket) toward the customer's "
    "hand? Answer only Yes or No."
)
PROMPT_ENTREGA = (
    "In this overhead image of a fast-food counter, is the customer picking "
    "up or holding a large orange tray with food? Answer only Yes or No."
)


# ═════════════════════════════════════════════════════════════════════════════
# VLM SINGLETON — LLaVA-1.5 7B 4-bit
# ═════════════════════════════════════════════════════════════════════════════

class _VLMSingleton:
    """
    Singleton que carga LLaVA una sola vez y corre inferencia en un hilo
    dedicado. Compartido entre todas las instancias de HummusVLMProcess.
    """

    _instance: Optional["_VLMSingleton"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str = DEFAULT_VLM_MODEL) -> "_VLMSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name: str) -> None:
        _ensure_vlm_deps()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._request_q: queue.Queue = queue.Queue(maxsize=8)
        self._results: Dict[int, Optional[bool]] = {}
        self._results_lock = threading.Lock()
        self._next_id = 0
        self._running = True

        logger.info("VLM: cargando %s en 4-bit …", model_name)
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=qcfg,
            device_map={"": 0} if self.device == "cuda" else "auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        vram = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        logger.info("VLM listo — VRAM: %.0f MB", vram)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, crop: np.ndarray, prompt: str) -> int:
        req_id = self._next_id
        self._next_id += 1
        try:
            self._request_q.put_nowait((req_id, crop.copy(), prompt))
        except queue.Full:
            logger.warning("VLM queue llena — descartando req %d", req_id)
            with self._results_lock:
                self._results[req_id] = None
        return req_id

    def poll(self, req_id: int) -> Optional[bool]:
        with self._results_lock:
            return self._results.pop(req_id, None)

    def _worker(self) -> None:
        while self._running:
            item = self._request_q.get()
            if item is None:
                break
            req_id, crop, prompt = item
            try:
                result = self._infer(crop, prompt)
            except Exception as exc:
                logger.error("VLM inference error: %s", exc)
                result = False
            with self._results_lock:
                self._results[req_id] = result

    def _infer(self, crop: np.ndarray, prompt: str) -> bool:
        with torch.no_grad():
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text_prompt, images=pil, return_tensors="pt"
            ).to(self.device, torch.float16)

            out_ids = self.model.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )
            generated = self.processor.decode(
                out_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip().lower()

            torch.cuda.empty_cache()

        token = generated.split()[0] if generated else ""
        is_yes = token.startswith(("yes", "sí", "si"))
        logger.debug("VLM → '%s' → %s", generated, is_yes)
        return is_yes


# ═════════════════════════════════════════════════════════════════════════════
# ESTADO POR PERSONA RASTREADA
# ═════════════════════════════════════════════════════════════════════════════

class _TrackState:
    __slots__ = (
        "zone", "t_entry", "stopped", "centroids", "bbox",
        "vlm_cooldown_until", "pending_req", "pending_zone",
    )

    def __init__(self, maxlen: int = 150) -> None:
        self.zone: Optional[str] = None
        self.t_entry: float = 0.0
        self.stopped: bool = False
        self.centroids: deque = deque(maxlen=maxlen)
        self.bbox: Optional[np.ndarray] = None
        self.vlm_cooldown_until: float = 0.0
        self.pending_req: Optional[int] = None
        self.pending_zone: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

class HummusVLMProcess:
    """
    Detector de eventos 'Toma de orden' y 'Entrega de plato' mediante
    YOLO (personas) + LLaVA VLM (clasificación visual de crops).

    Interfaz compatible con HummusProcess:
        process_frame(image, roi, send_to_server) → (processed_image, metadata)
    """

    _DISPLAY_NAMES = {
        "order": "Toma de orden",
        "delivery": "Entrega de plato",
    }
    _ZONE_COLORS = {
        "caja": (0, 200, 255),      # naranja (BGR)
        "entrega": (0, 255, 128),    # verde (BGR)
    }
    _ZONE_EVENT_MAP = {
        "caja": "order",
        "entrega": "delivery",
    }

    def __init__(
        self,
        client_id: Optional[str] = None,
        person_model_path: str = DEFAULT_PERSON_MODEL_PATH,
        vlm_model: str = DEFAULT_VLM_MODEL,
        # ── YOLO ─────────────────────────────────────────────────────────
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        imgsz: int = 640,
        # ── Zonas ROI (píxeles) ──────────────────────────────────────────
        zona_caja: Optional[List[List[int]]] = None,
        zona_entrega: Optional[List[List[int]]] = None,
        # ── Detección de parada ──────────────────────────────────────────
        stop_threshold_px: float = 30.0,
        stop_duration_s: float = 3.0,
        centroid_history_maxlen: int = 150,
        # ── VLM ──────────────────────────────────────────────────────────
        vlm_cooldown_s: float = 10.0,
        vlm_no_cooldown_s: float = 2.0,
        crop_pad_ratio: float = 0.4,
        prompt_caja: str = PROMPT_CAJA,
        prompt_entrega: str = PROMPT_ENTREGA,
        # ── Tracker ──────────────────────────────────────────────────────
        tracker: str = "trackers/botsort_reid.yaml",
        min_person_confidence: float = 0.3,
        min_person_area: int = 900,
        # ── Output ───────────────────────────────────────────────────────
        order_screenshot_dir: str = DEFAULT_ORDER_SCREENSHOT_DIR,
        delivery_screenshot_dir: str = DEFAULT_DELIVERY_SCREENSHOT_DIR,
        alert_csv_path: str = DEFAULT_ALERT_CSV_PATH,
        image_quality: int = 85,
        component_key: str = "hummus_vlm",
        type_inference: str = "hummus_vlm",
        server_url: Optional[str] = None,
    ) -> None:
        import uuid as _uuid

        self.client_id = client_id or str(_uuid.uuid4())
        self.component_key = component_key
        self.type_inference = type_inference
        self.server_url = server_url
        self.device = device
        self.image_quality = image_quality
        self.imgsz = imgsz

        # ── YOLO persona ─────────────────────────────────────────────────
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_person_confidence = min_person_confidence
        self.min_person_area = min_person_area

        resolved = self._resolve_path(person_model_path, DEFAULT_PERSON_MODEL_PATH)
        logger.info("Cargando YOLO personas: %s", resolved)
        self.person_model = YOLO(resolved).to(device)

        if tracker and not os.path.isabs(tracker):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            self.tracker = os.path.join(base_dir, tracker)
        else:
            self.tracker = tracker

        # ── VLM (singleton) ──────────────────────────────────────────────
        self.vlm = _VLMSingleton.get(vlm_model)
        self.vlm_cooldown_s = vlm_cooldown_s
        self.vlm_no_cooldown_s = vlm_no_cooldown_s
        self.crop_pad_ratio = crop_pad_ratio
        self.prompts = {"caja": prompt_caja, "entrega": prompt_entrega}

        # ── Zonas ────────────────────────────────────────────────────────
        zc = zona_caja if zona_caja else DEFAULT_ZONA_CAJA
        ze = zona_entrega if zona_entrega else DEFAULT_ZONA_ENTREGA
        self.zone_polys: Dict[str, np.ndarray] = {
            "caja": np.array(zc, dtype=np.int32),
            "entrega": np.array(ze, dtype=np.int32),
        }

        # ── Detección de parada ──────────────────────────────────────────
        self.stop_threshold_px = stop_threshold_px
        self.stop_duration_s = stop_duration_s
        self.centroid_maxlen = centroid_history_maxlen

        # ── Estado de tracking ───────────────────────────────────────────
        self.tracks: Dict[int, _TrackState] = {}
        self.pending_vlm: Dict[int, Tuple[int, str, float]] = {}  # req_id → (tid, zone, t)

        # ── Contadores ───────────────────────────────────────────────────
        self.frame_counter = 0
        self.order_count = 0
        self.delivery_count = 0
        self.total_events = 0

        # ── Tiempos de espera ────────────────────────────────────────────
        self.order_timestamps: Dict[int, float] = {}  # tid → timestamp de orden
        self.wait_times: List[float] = []

        # ── Historial de eventos (evita duplicados por track) ────────────
        self.event_history: Dict[int, set] = {}  # tid → {"order", "delivery"}

        # ── Screenshots ──────────────────────────────────────────────────
        self.order_screenshot_dir = order_screenshot_dir
        self.delivery_screenshot_dir = delivery_screenshot_dir
        self.alert_csv_path = alert_csv_path
        self.saved_screenshot_keys: set = set()
        self.saved_screenshot_paths: Dict[Tuple[str, int], str] = {}

        # Crear carpetas
        for d in (order_screenshot_dir, delivery_screenshot_dir):
            os.makedirs(d, exist_ok=True)
        self._setup_csv()

        # ── ROI global (opcional, del cliente) ───────────────────────────
        self.global_roi: Optional[np.ndarray] = None

        logger.info(
            "HummusVLMProcess listo — device=%s | stop=%.0fpx/%.1fs | vlm_cooldown=%.1fs",
            device, stop_threshold_px, stop_duration_s, vlm_cooldown_s,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(path: str, fallback: str) -> str:
        if path and os.path.isfile(path):
            return path
        if not os.path.isabs(path):
            base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            candidate = os.path.join(base, path)
            if os.path.isfile(candidate):
                return candidate
        if fallback and os.path.isfile(fallback):
            return fallback
        raise FileNotFoundError(f"Modelo no encontrado: {path} (fallback: {fallback})")

    def _point_in_zone(self, pt: Tuple[float, float]) -> Optional[str]:
        px, py = int(pt[0]), int(pt[1])
        for name, poly in self.zone_polys.items():
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (px, py), False) >= 0:
                return name
        return None

    def _point_in_global_roi(self, pt: Tuple[float, float]) -> bool:
        if self.global_roi is None:
            return True
        return cv2.pointPolygonTest(
            self.global_roi.reshape(-1, 1, 2),
            (int(pt[0]), int(pt[1])), False
        ) >= 0

    def _is_stopped(self, state: _TrackState, now: float) -> bool:
        if not state.centroids:
            return False
        cutoff = now - self.stop_duration_s
        recent = [(t, cx, cy) for t, cx, cy in state.centroids if t >= cutoff]
        if not recent or (now - recent[0][0]) < self.stop_duration_s * 0.8:
            return False
        cx0, cy0 = recent[-1][1], recent[-1][2]
        max_d = max(
            ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
            for _, cx, cy in recent
        )
        return max_d < self.stop_threshold_px

    def _crop_bbox(
        self, bbox: np.ndarray, shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        px, py = w * self.crop_pad_ratio, h * self.crop_pad_ratio
        H, W = shape[:2]
        return (
            int(max(0, x1 - px)),
            int(max(0, y1 - py)),
            int(min(W, x2 + px)),
            int(min(H, y2 + py)),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Screenshots & CSV
    # ─────────────────────────────────────────────────────────────────────

    def _setup_csv(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.alert_csv_path), exist_ok=True)
            if not os.path.isfile(self.alert_csv_path):
                with open(self.alert_csv_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        "timestamp", "event", "track_id", "wait_time_s",
                    ])
        except Exception as exc:
            logger.warning("No se pudo inicializar CSV: %s", exc)

    def _append_csv(self, event: str, tid: int, wait_time: Optional[float]) -> None:
        try:
            ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
            with open(self.alert_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    ts, event, tid, f"{wait_time:.1f}" if wait_time else "",
                ])
        except Exception as exc:
            logger.warning("CSV write error: %s", exc)

    def _save_screenshot(
        self, frame: np.ndarray, event: str, tid: int,
    ) -> str:
        key = (event, tid)
        if key in self.saved_screenshot_keys:
            return self.saved_screenshot_paths.get(key, "")

        if event == "order":
            target_dir, prefix = self.order_screenshot_dir, "toma_de_orden"
        else:
            target_dir, prefix = self.delivery_screenshot_dir, "entrega_de_plato"

        os.makedirs(target_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_id{tid}_{ts}.jpg"
        path = os.path.join(target_dir, filename)

        ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        self.saved_screenshot_keys.add(key)
        if ok:
            self.saved_screenshot_paths[key] = path
            logger.info("Screenshot: %s", path)
            return path
        return ""

    # ─────────────────────────────────────────────────────────────────────
    # Visualización
    # ─────────────────────────────────────────────────────────────────────

    def _draw(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()

        # Zonas semi-transparentes
        for name, poly in self.zone_polys.items():
            c = self._ZONE_COLORS.get(name, (255, 255, 255))
            ov = out.copy()
            cv2.fillPoly(ov, [poly], c)
            cv2.addWeighted(ov, 0.12, out, 0.88, 0, out)
            cv2.polylines(out, [poly], True, c, 2)
            mx = int(poly[:, 0].mean()) - 30
            my = int(poly[:, 1].min()) - 10
            cv2.putText(out, name.upper(), (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        # ROI global
        if self.global_roi is not None:
            cv2.polylines(out, [self.global_roi], True, (0, 255, 255), 1)

        # Tracks
        for tid, s in self.tracks.items():
            if s.bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in s.bbox]
            c = (0, 255, 0) if s.stopped else (180, 180, 180)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            lbl = f"ID:{tid}"
            if s.zone:
                lbl += f" [{s.zone}]"
            if s.stopped:
                lbl += " STOP"
            cv2.putText(out, lbl, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 2)

        # HUD
        hud = out.copy()
        cv2.rectangle(hud, (5, 5), (340, 130), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.55, out, 0.45, 0, out)
        for text, pos in [
            (f"Frame: {self.frame_counter}", (15, 28)),
            (f"Tomas de orden: {self.order_count}", (15, 53)),
            (f"Entregas de plato: {self.delivery_count}", (15, 78)),
            (f"Total eventos: {self.total_events}", (15, 103)),
            (f"Espera prom: {self._avg_wait():.1f}s" if self.wait_times else "", (15, 123)),
        ]:
            if text:
                cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return out

    def _avg_wait(self) -> float:
        return sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0.0

    # ─────────────────────────────────────────────────────────────────────
    # PROCESO PRINCIPAL DEL FRAME
    # ─────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        image: np.ndarray,
        roi=None,
        send_to_server: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa un frame. Compatible con HummusProcess.

        Returns:
            (processed_image, metadata_dict)
        """
        self.frame_counter += 1
        now = time.time()
        events: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []

        # ROI global del cliente (opcional)
        if roi is not None and isinstance(roi, (list, np.ndarray)) and len(roi) >= 3:
            self.global_roi = np.array(roi, np.int32)

        # ── 1. YOLO persona + ByteTrack ──────────────────────────────────
        results = self.person_model.track(
            image,
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            tracker=self.tracker,
            verbose=False,
        )

        alive: set = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            ids = (
                boxes.id.cpu().numpy()
                if boxes.id is not None
                else [None] * len(xyxy)
            )

            for i in range(len(xyxy)):
                # Solo personas
                if int(cls_ids[i]) != 0:
                    continue
                if float(confs[i]) < self.min_person_confidence:
                    continue

                tid = int(ids[i]) if ids[i] is not None else None
                if tid is None:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_person_area:
                    continue

                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

                # Filtrar por ROI global si está activo
                if not self._point_in_global_roi((cx, cy)):
                    continue

                alive.add(tid)

                # ── 2. Actualizar estado del track ───────────────────────
                if tid not in self.tracks:
                    self.tracks[tid] = _TrackState(self.centroid_maxlen)

                state = self.tracks[tid]
                state.centroids.append((now, cx, cy))
                state.bbox = xyxy[i]

                # Determinar zona
                zone = self._point_in_zone((cx, cy))
                if zone != state.zone:
                    state.zone = zone
                    state.t_entry = now
                    state.stopped = False
                    state.centroids.clear()
                    state.centroids.append((now, cx, cy))
                    continue

                if zone is None:
                    state.stopped = False
                    continue

                # ── 3. Detectar parada ───────────────────────────────────
                state.stopped = self._is_stopped(state, now)

                if not state.stopped:
                    continue

                # ── 4. Enviar crop al VLM si aplica ──────────────────────
                event_label = self._ZONE_EVENT_MAP.get(zone)
                if event_label and event_label in self.event_history.get(tid, set()):
                    continue  # ya registrado para este track

                can_query = (
                    now >= state.vlm_cooldown_until
                    and state.pending_req is None
                )
                if can_query:
                    cb = self._crop_bbox(xyxy[i], image.shape)
                    crop = image[cb[1]:cb[3], cb[0]:cb[2]]
                    if crop.size > 0:
                        prompt = self.prompts.get(zone, "")
                        if prompt:
                            req_id = self.vlm.submit(crop, prompt)
                            state.pending_req = req_id
                            state.pending_zone = zone
                            self.pending_vlm[req_id] = (tid, zone, now)

        # ── 5. Recoger resultados VLM ────────────────────────────────────
        resolved = []
        for req_id, (tid, zone, t_sub) in self.pending_vlm.items():
            result = self.vlm.poll(req_id)
            if result is None:
                continue  # aún pendiente
            resolved.append(req_id)

            # Limpiar pending
            if tid in self.tracks:
                self.tracks[tid].pending_req = None
                self.tracks[tid].pending_zone = None

            event_label = self._ZONE_EVENT_MAP.get(zone)
            if not event_label:
                continue

            # Verificar que no se haya registrado mientras esperábamos
            if event_label in self.event_history.get(tid, set()):
                continue

            if result is True:
                # ── EVENTO CONFIRMADO ────────────────────────────────
                self.event_history.setdefault(tid, set()).add(event_label)

                if event_label == "order":
                    self.order_count += 1
                    self.order_timestamps[tid] = now
                elif event_label == "delivery":
                    self.delivery_count += 1
                self.total_events += 1

                # Tiempo de espera
                wait_time: Optional[float] = None
                if event_label == "delivery" and tid in self.order_timestamps:
                    wait_time = round(now - self.order_timestamps.pop(tid), 1)
                    self.wait_times.append(wait_time)
                    logger.info("Tiempo de espera ID %d: %.1f s", tid, wait_time)

                # Screenshot
                screenshot_path = self._save_screenshot(image, event_label, tid)

                # Crop base64 para la alerta
                crop_b64 = ""
                if screenshot_path and os.path.isfile(screenshot_path):
                    try:
                        with open(screenshot_path, "rb") as f:
                            crop_b64 = base64.b64encode(f.read()).decode("utf-8")
                    except Exception:
                        pass

                if not crop_b64:
                    state = self.tracks.get(tid)
                    if state is not None and state.bbox is not None:
                        cb = self._crop_bbox(state.bbox, image.shape)
                        region = image[cb[1]:cb[3], cb[0]:cb[2]]
                        if region.size > 0:
                            _, buf = cv2.imencode(".jpg", region, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            if buf is not None:
                                crop_b64 = base64.b64encode(buf).decode("utf-8")

                display_name = self._DISPLAY_NAMES.get(event_label, event_label)

                evt = {
                    "event": event_label,
                    "track_id": tid,
                    "global_id": tid,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "frame": self.frame_counter,
                    "wait_time_s": wait_time,
                }
                events.append(evt)

                alert = {
                    "event_type": display_name,
                    "class_name": display_name,
                    "description": (
                        f"{display_name} detectada (ID: {tid})"
                        + (f" — espera: {wait_time}s" if wait_time else "")
                    ),
                    "timestamp": now,
                    "crop_image": crop_b64,
                    "screenshot_path": screenshot_path,
                }
                alerts.append(alert)

                # CSV
                self._append_csv(event_label, tid, wait_time)

                # Cooldown largo
                if tid in self.tracks:
                    self.tracks[tid].vlm_cooldown_until = now + self.vlm_cooldown_s

                logger.info(
                    "EVENTO: %s | ID: %d | frame: %d%s",
                    display_name, tid, self.frame_counter,
                    f" | espera: {wait_time}s" if wait_time else "",
                )
            else:
                # VLM dijo "No" → cooldown corto
                if tid in self.tracks:
                    self.tracks[tid].vlm_cooldown_until = now + self.vlm_no_cooldown_s

        for req_id in resolved:
            del self.pending_vlm[req_id]

        # ── 6. Limpiar tracks viejos ─────────────────────────────────────
        if self.frame_counter % 150 == 0:
            stale = [
                k for k, v in self.tracks.items()
                if k not in alive
                and v.centroids
                and (now - v.centroids[-1][0]) > 30.0
            ]
            for k in stale:
                del self.tracks[k]
                self.event_history.pop(k, None)
                self.order_timestamps.pop(k, None)

        # ── 7. Dibujar ──────────────────────────────────────────────────
        processed = self._draw(image)

        # ── 8. Metadata ─────────────────────────────────────────────────
        metadata: Dict[str, Any] = {
            "events": events,
            "frame": self.frame_counter,
            "order_count": self.order_count,
            "delivery_count": self.delivery_count,
            "total_events": self.total_events,
            "orders_in_frame": sum(1 for e in events if e["event"] == "order"),
            "deliveries_in_frame": sum(1 for e in events if e["event"] == "delivery"),
            "frame_events_total": len(events),
            "tracks_in_roi": len(alive),
            "unique_tracks_total": len(self.event_history),
            "avg_wait_time_s": round(self._avg_wait(), 1) if self.wait_times else None,
            "alerts": alerts,
        }

        return processed, metadata

    # ─────────────────────────────────────────────────────────────────────
    # Payload para servidor (compatibilidad con HummusProcess)
    # ─────────────────────────────────────────────────────────────────────

    def create_server_payload(
        self,
        frame: np.ndarray,
        metadata: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            )
            if not ok:
                return None
            return {
                "id_connection": self.client_id,
                "component_key": self.component_key,
                "type_inference": self.type_inference,
                "data": {
                    "header": {
                        "frame_number": self.frame_counter,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "event_type": "hummus_vlm_events",
                    },
                    "image": base64.b64encode(buf).decode("utf-8"),
                    "data_result": {
                        "statistics": metadata,
                        "events": events,
                    },
                },
            }
        except Exception as exc:
            logger.error("Error creando payload: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Reset y estadísticas
    # ─────────────────────────────────────────────────────────────────────

    def reset_counter(self) -> None:
        self.order_count = 0
        self.delivery_count = 0
        self.total_events = 0
        self.frame_counter = 0
        self.tracks.clear()
        self.pending_vlm.clear()
        self.event_history.clear()
        self.order_timestamps.clear()
        self.wait_times.clear()
        self.saved_screenshot_keys.clear()
        self.saved_screenshot_paths.clear()
        logger.info("Contadores reiniciados")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "order_count": self.order_count,
            "delivery_count": self.delivery_count,
            "total_events": self.total_events,
            "frame_counter": self.frame_counter,
            "avg_wait_time_s": round(self._avg_wait(), 1) if self.wait_times else None,
            "active_tracks": len(self.tracks),
        }
