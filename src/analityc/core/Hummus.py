import asyncio
import base64
import csv
import datetime
import logging
import os
import threading
import uuid
import unicodedata
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from ..config.config import DEFAULT_ROI


logger = logging.getLogger(__name__)


# ── Rutas por defecto ─────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH              = r"C:\\Users\\Sistema-1\\Desktop\\ELDE\\SERVER-IA PERIMETRALES\\models\\base\\1080.pt"
DEFAULT_PERSON_MODEL_PATH       = r"C:\\Users\\Sistema-1\\Desktop\\ELDE\\SERVER-IA PERIMETRALES\\models\\base\\yolo12l.pt"
DEFAULT_ORDER_SCREENSHOT_DIR    = r"C:\\Users\\Sistema-1\\Desktop\\ELDE\\SERVER-IA PERIMETRALES\\output\\Toma_de_orden_hummus"
DEFAULT_DELIVERY_SCREENSHOT_DIR = r"C:\\Users\\Sistema-1\\Desktop\\ELDE\\SERVER-IA PERIMETRALES\\output\\Entrega_de_plato_hummus"
DEFAULT_ALERT_CSV_PATH          = r"C:\\Users\\Sistema-1\\Desktop\\ELDE\\SERVER-IA PERIMETRALES\\output\\hummus_alertas.csv"
DEFAULT_ALERT_PENDING_CSV_PATH  = r"C:\\Users\\Sistema-1\\Desktop\\ELDE\\SERVER-IA PERIMETRALES\\output\\hummus_alertas_pending.csv"


# ─────────────────────────────────────────────────────────────────────────────
# MEJORAS ANTI FALSOS POSITIVOS – resumen
#
# 1. CONFIRMACIÓN TEMPORAL  : evento solo se registra si aparece N frames
#    consecutivos dentro de una ventana deslizante.
# 2. UMBRAL DE CONFIANZA ADAPTATIVO POR CLASE.
# 3. VOTACIÓN DE CONFIANZA ACUMULADA: promedio de confianza en la ventana.
# 4. REQUISITO ESTRICTO DE PERSONA ASOCIADA (configurable).
# 5. COOLDOWN EXTENDIDO Y POR EVENTO+TRACK.
# 6. ÁREA MÍNIMA DE DETECCIÓN DE EVENTO.
# 7. ASPECT RATIO GUARD.
# 8. NMS INTER-CLASE MANUAL: suprime solapamientos entre eventos del mismo frame.
# 9. NORMALIZACIÓN DE TRACK ROBUSTECIDA.
# 10. LOGGING DE FALSOS POSITIVOS POTENCIALES para auditoría.
# ─────────────────────────────────────────────────────────────────────────────


class EventConfirmationBuffer:
    """
    Ventana deslizante de confirmación temporal para un par (global_id, label).

    Un evento se confirma cuando:
      - aparece en `required_frames` de los últimos `window_size` frames
      - la confianza media de esas apariciones supera `confidence_threshold`
    """

    __slots__ = ("window_size", "required_frames", "confidence_threshold", "_hits")

    def __init__(
        self,
        window_size: int = 5,
        required_frames: int = 3,
        confidence_threshold: float = 0.60,
    ) -> None:
        self.window_size = window_size
        self.required_frames = required_frames
        self.confidence_threshold = confidence_threshold
        # deque de (frame_number, confidence)
        self._hits: deque = deque(maxlen=window_size)

    def add(self, frame_number: int, confidence: float) -> None:
        self._hits.append((frame_number, confidence))

    def is_confirmed(self, current_frame: int) -> bool:
        recent = [(f, c) for f, c in self._hits if (current_frame - f) < self.window_size]
        if len(recent) < self.required_frames:
            return False
        avg_conf = sum(c for _, c in recent) / len(recent)
        return avg_conf >= self.confidence_threshold

    def reset(self) -> None:
        self._hits.clear()


class HummusProcess:
    """
    Detector de eventos 'Toma de orden' y 'Entrega de plato' mediante YOLO.

    Versión v3 – Cambios respecto a v2:
      • Eliminada toda la integración con Excel (openpyxl).
      • Umbrales de confianza reducidos para mejorar recall en ambas clases.
      • Limpieza de imports y variables huérfanas.
      • Constantes de color/display centralizadas.
      • Pequeñas optimizaciones: early-return en bucles, acceso a dict más seguro.
    """

    # Mapa de colores para visualización (BGR)
    _DRAW_COLORS: Dict[str, Tuple[int, int, int]] = {
        "order":    (0, 200, 255),
        "delivery": (0, 255, 128),
        "person":   (255, 200, 0),
    }
    # Nombres de display por label interno
    _DISPLAY_NAMES: Dict[str, str] = {
        "order":    "Toma de orden",
        "delivery": "Entrega de plato",
    }

    def __init__(
        self,
        client_id: Optional[str] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        person_model_path: str = DEFAULT_PERSON_MODEL_PATH,
        # ── Umbrales globales (bajados para mejorar recall) ──────────────────
        confidence_threshold: float = 0.25,          # antes 0.5
        iou_threshold: float = 0.45,
        person_confidence_threshold: Optional[float] = None,
        person_iou_threshold: Optional[float] = None,
        device: str = "cpu",
        log_file: str = "output/hummus_log.csv",
        image_quality: int = 85,
        server_url: Optional[str] = None,
        component_key: str = "hummus_inference",
        type_inference: str = "hummus",
        max_image_size: Tuple[int, int] = (640, 640),
        tracker: str = "trackers/botsort_reid.yaml",
        id_reset_frames: int = 900,
        min_person_confidence: float = 0.3,
        min_event_confidence: Optional[float] = None,
        min_person_area: int = 900,
        max_event_person_distance: float = 120.0,
        min_event_person_iou: float = 0.08,
        appearance_match_threshold: float = 0.85,
        max_reid_distance: float = 80.0,
        appearance_update_alpha: float = 0.35,
        event_cooldown_frames: int = 90,
        order_cooldown_frames: Optional[int] = None,
        delivery_cooldown_frames: Optional[int] = 300,
        min_track_age_frames: int = 5,
        order_screenshot_dir: str = DEFAULT_ORDER_SCREENSHOT_DIR,
        delivery_screenshot_dir: str = DEFAULT_DELIVERY_SCREENSHOT_DIR,
        alert_csv_path: str = DEFAULT_ALERT_CSV_PATH,
        # ── Confirmación temporal ────────────────────────────────────────────
        temporal_confirmation_frames: int = 1,
        temporal_window_size: int = 1,
        order_confirmation_frames: Optional[int] = 1,
        order_window_size: Optional[int] = 1,
        delivery_confirmation_frames: Optional[int] = None,
        delivery_window_size: Optional[int] = None,
        # ── Umbrales por clase (bajados para mejorar recall) ─────────────────
        order_confidence_threshold: Optional[float] = 0.15,    # antes 0.01
        delivery_confidence_threshold: Optional[float] = 0.20, # nuevo, explícito
        confidence_vote_threshold: float = 0.01,
        # ── Filtros anti FP ──────────────────────────────────────────────────
        require_person_match: bool = False,
        min_event_area: int = 1,
        event_aspect_ratio_range: Tuple[float, float] = (0.01, 10.0),
        event_nms_iou_threshold: float = 0.9,
        max_events_per_frame: int = 10,
        suppress_confidence_drop: bool = False,
        confidence_drop_threshold: float = 1.0,
    ) -> None:
        self.client_id = client_id or str(uuid.uuid4())
        self.component_key = component_key
        self.type_inference = type_inference
        self.server_url = server_url

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = self._resolve_model_path(model_path, DEFAULT_MODEL_PATH, "principal")
        self.person_model_path = self._resolve_model_path(
            person_model_path, DEFAULT_PERSON_MODEL_PATH, "personas"
        )
        self.person_confidence_threshold = (
            person_confidence_threshold
            if person_confidence_threshold is not None
            else confidence_threshold
        )
        self.person_iou_threshold = (
            person_iou_threshold if person_iou_threshold is not None else iou_threshold
        )
        self.image_quality = image_quality
        self.max_image_size = max_image_size
        self.log_file = log_file
        self.device = device

        if tracker and not os.path.isabs(tracker):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            self.tracker = os.path.join(base_dir, tracker)
        else:
            self.tracker = tracker

        self.id_reset_frames = id_reset_frames
        self.min_person_confidence = min_person_confidence
        self.min_event_confidence = (
            min_event_confidence if min_event_confidence is not None else confidence_threshold
        )
        self.min_person_area = min_person_area
        self.max_event_person_distance = max_event_person_distance
        self.min_event_person_iou = min_event_person_iou
        self.appearance_match_threshold = appearance_match_threshold
        self.max_reid_distance = max_reid_distance
        self.appearance_update_alpha = appearance_update_alpha

        # Cooldowns: específico por tipo > global
        self.event_cooldown_frames = event_cooldown_frames
        self.order_cooldown_frames = (
            order_cooldown_frames if order_cooldown_frames is not None else event_cooldown_frames
        )
        self.delivery_cooldown_frames = (
            delivery_cooldown_frames
            if delivery_cooldown_frames is not None
            else event_cooldown_frames
        )

        self.min_track_age_frames = min_track_age_frames
        self.order_screenshot_dir = order_screenshot_dir
        self.delivery_screenshot_dir = delivery_screenshot_dir
        self.alert_csv_path = alert_csv_path
        self.alert_pending_csv_path = DEFAULT_ALERT_PENDING_CSV_PATH

        # ── Parámetros anti FP ────────────────────────────────────────────────
        self.temporal_confirmation_frames = max(1, temporal_confirmation_frames)
        self.temporal_window_size = max(
            self.temporal_confirmation_frames + 1, temporal_window_size
        )

        self.order_confirmation_frames = max(
            1,
            order_confirmation_frames
            if order_confirmation_frames is not None
            else self.temporal_confirmation_frames,
        )
        self.order_window_size = max(
            self.order_confirmation_frames,
            order_window_size if order_window_size is not None else self.temporal_window_size,
        )
        self.delivery_confirmation_frames = max(
            1,
            delivery_confirmation_frames
            if delivery_confirmation_frames is not None
            else self.temporal_confirmation_frames,
        )
        self.delivery_window_size = max(
            self.delivery_confirmation_frames + 1,
            delivery_window_size
            if delivery_window_size is not None
            else self.temporal_window_size,
        )

        self.order_confidence_threshold = (
            order_confidence_threshold if order_confidence_threshold is not None else 0.15
        )
        self.delivery_confidence_threshold = (
            delivery_confidence_threshold
            if delivery_confidence_threshold is not None
            else self.min_event_confidence
        )
        self.confidence_vote_threshold = confidence_vote_threshold
        self.require_person_match = require_person_match
        self.min_event_area = min_event_area
        self.event_aspect_ratio_range = event_aspect_ratio_range
        self.event_nms_iou_threshold = event_nms_iou_threshold
        self.max_events_per_frame = max_events_per_frame
        self.suppress_confidence_drop = suppress_confidence_drop
        self.confidence_drop_threshold = confidence_drop_threshold
        self._order_relax_filters = True

        # Buffers de confirmación temporal: {(global_id, label): EventConfirmationBuffer}
        self._confirmation_buffers: Dict[Tuple[Any, str], EventConfirmationBuffer] = {}
        # Última confianza por (global_id, label) para detectar caídas
        self._last_confidence: Dict[Tuple[Any, str], float] = {}

        # Contadores
        self.frame_counter = 0
        self.order_count = 0
        self.delivery_count = 0
        self.total_events = 0
        self.frame_events: List[Dict[str, Any]] = []

        self.roi_polygon = np.array(DEFAULT_ROI, np.int32)
        self.roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))

        self.event_history: Dict[Any, set] = {}
        self.unique_track_ids: set = set()
        self.track_id_map: Dict[int, int] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.next_global_id = 1
        self.last_processed_frame: Optional[np.ndarray] = None
        self.global_id_last_seen: Dict[int, int] = {}
        self.global_id_last_center: Dict[int, Tuple[float, float]] = {}
        self.global_id_features: Dict[int, np.ndarray] = {}
        self.event_last_seen: Dict[Tuple[Any, str], int] = {}
        self.global_id_first_seen: Dict[int, int] = {}
        self.saved_screenshot_keys: set = set()
        self.saved_screenshot_paths: Dict[Tuple[str, Any], str] = {}
        self.active_persons: Dict[int, Tuple[float, float]] = {}
        self.has_person_class = False

        # Deduplicación espacial: label -> [(frame, cx, cy)]
        self._spatial_event_log: Dict[str, List[Tuple[int, float, float]]] = {
            "order": [], "delivery": []
        }
        self._spatial_dedup_distance = 150.0
        self._spatial_dedup_frames = 300

        self.target_labels = {"toma de orden": "order", "entrega de plato": "delivery"}
        self.person_labels = {"person", "persona", "person/persona"}
        self.person_labels_normalized = {self._normalize_label(x) for x in self.person_labels}

        self.model: Optional[YOLO] = None
        self.person_model: Optional[YOLO] = None
        self._initialize_model()
        self.setup_log_file()

        logger.info(
            "HummusProcess v3 listo. "
            "conf_global=%.2f | order_conf=%.2f | delivery_conf=%.2f | "
            "confirmación=%d/%d frames | require_person=%s | min_event_area=%d",
            self.confidence_threshold,
            self.order_confidence_threshold,
            self.delivery_confidence_threshold,
            self.temporal_confirmation_frames,
            self.temporal_window_size,
            self.require_person_match,
            self.min_event_area,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Infraestructura
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_model_path(
        self, model_path: str, fallback_path: str, model_kind: str
    ) -> str:
        candidates: List[str] = []
        if model_path:
            candidates.append(model_path)
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        if model_path and not os.path.isabs(model_path):
            candidates.append(os.path.join(base_dir, model_path))
            candidates.append(os.path.join(base_dir, "models", "base", model_path))

        seen: set = set()
        for candidate in candidates:
            normalized = os.path.abspath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            if os.path.isfile(normalized):
                return normalized

        fallback = os.path.abspath(fallback_path) if fallback_path else ""
        if fallback and os.path.isfile(fallback):
            logger.warning(
                "Modelo %s no encontrado en '%s'. Usando fallback '%s'.",
                model_kind, model_path, fallback,
            )
            return fallback

        tried = ", ".join(os.path.abspath(p) for p in candidates) or str(model_path)
        raise FileNotFoundError(
            f"No se encontró el modelo {model_kind}. "
            f"Rutas probadas: {tried}. Fallback ausente: {fallback or 'N/A'}"
        )

    def _initialize_model(self) -> None:
        try:
            self.model = YOLO(self.model_path).to(self.device)
            self.person_model = YOLO(self.person_model_path).to(self.device)
            person_names = {
                self._normalize_label(name)
                for name in self.person_model.names.values()
            }
            self.has_person_class = any(
                name in self.person_labels_normalized for name in person_names
            )
            logger.info("Modelos YOLO cargados en '%s'", self.device)
        except Exception as exc:
            logger.error("No se pudo cargar el modelo: %s", exc)
            raise

    def setup_log_file(self) -> None:
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write("timestamp,frame,order,delivery,total,event\n")
            except Exception:
                logger.warning("No se pudo inicializar el log %s", self.log_file)
        try:
            os.makedirs(self.order_screenshot_dir, exist_ok=True)
            os.makedirs(self.delivery_screenshot_dir, exist_ok=True)
        except Exception as exc:
            logger.warning(
                "No se pudieron inicializar carpetas de screenshots: %s", exc
            )
        self._setup_alert_csv()

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _setup_alert_csv(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.alert_csv_path), exist_ok=True)
            if not os.path.isfile(self.alert_csv_path):
                with open(self.alert_csv_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        ["timestamp", "event", "order_id", "delivery_id"]
                    )
        except Exception as exc:
            logger.warning(
                "No se pudo inicializar CSV de alertas %s: %s", self.alert_csv_path, exc
            )

    def _build_alert_row(self, event: Dict[str, Any]) -> Optional[List[str]]:
        event_name = str(event.get("event", "")).strip().lower()
        if event_name not in {"order", "delivery"}:
            return None
        timestamp = self._format_timestamp_millis()
        event_id = str(event.get("global_id", event.get("track_id", "na")))
        if event_name == "order":
            return [timestamp, "order", event_id, ""]
        return [timestamp, "delivery", "", event_id]

    def _append_row_to_csv_file(self, file_path: str, row: List[str]) -> None:
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _flush_pending_alerts(self) -> None:
        if not os.path.isfile(self.alert_pending_csv_path):
            return
        try:
            with open(self.alert_pending_csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if not rows:
                os.remove(self.alert_pending_csv_path)
                return
            with open(self.alert_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            os.remove(self.alert_pending_csv_path)
        except PermissionError:
            return
        except Exception as exc:
            logger.warning("No se pudo volcar CSV pendiente de alertas: %s", exc)

    def _append_alert_to_csv(self, event: Dict[str, Any]) -> None:
        csv_row = self._build_alert_row(event)
        if csv_row is None:
            return
        self._flush_pending_alerts()
        try:
            self._append_row_to_csv_file(self.alert_csv_path, csv_row)
        except PermissionError:
            try:
                self._append_row_to_csv_file(self.alert_pending_csv_path, csv_row)
                logger.warning(
                    "CSV principal bloqueado. Alerta guardada en pendiente: %s",
                    self.alert_pending_csv_path,
                )
            except Exception as exc:
                logger.warning(
                    "No se pudo registrar alerta en CSV pendiente: %s", exc
                )
        except Exception as exc:
            logger.warning("No se pudo registrar alerta en CSV principal: %s", exc)

    # ── Screenshots ───────────────────────────────────────────────────────────

    def _save_event_screenshot(
        self, frame: np.ndarray, event: Dict[str, Any]
    ) -> Optional[str]:
        event_name = str(event.get("event", "")).strip().lower()
        if event_name == "order":
            target_dir, prefix = self.order_screenshot_dir, "toma_de_orden"
        elif event_name == "delivery":
            target_dir, prefix = self.delivery_screenshot_dir, "entrega_de_plato"
        else:
            return None

        os.makedirs(target_dir, exist_ok=True)
        track_value = event.get("global_id", event.get("track_id", "na"))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_frame{self.frame_counter}_id{track_value}_{timestamp}.jpg"
        file_path = os.path.join(target_dir, filename)

        ok = cv2.imwrite(
            file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(self.image_quality)]
        )
        if ok:
            logger.info("Screenshot guardado (%s): %s", event_name, file_path)
            return file_path
        logger.warning("No se pudo guardar screenshot (%s): %s", event_name, file_path)
        return None

    # ── Timestamps ────────────────────────────────────────────────────────────

    def _format_timestamp_millis(self, raw_timestamp: Any = None) -> str:
        if raw_timestamp is None:
            dt = datetime.datetime.now()
        elif isinstance(raw_timestamp, datetime.datetime):
            dt = raw_timestamp
        elif raw_timestamp:
            raw_str = str(raw_timestamp).strip()
            for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S"):
                try:
                    dt = datetime.datetime.strptime(raw_str, fmt)
                    break
                except ValueError:
                    pass
            else:
                try:
                    dt = datetime.datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.datetime.now()
        else:
            dt = datetime.datetime.now()
        return dt.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]

    # ──────────────────────────────────────────────────────────────────────────
    # Geometría / utilidades
    # ──────────────────────────────────────────────────────────────────────────

    def is_inside_roi(self, point: Tuple[float, float]) -> bool:
        return (
            cv2.pointPolygonTest(
                self.roi_polygon_points,
                (int(point[0]), int(point[1])),
                False,
            )
            >= 0
        )

    def compress_image(self, image: np.ndarray) -> np.ndarray:
        try:
            h, w = image.shape[:2]
            max_w, max_h = self.max_image_size
            if w > max_w or h > max_h:
                aspect = w / h
                if w > max_w:
                    w, h = max_w, int(max_w / aspect)
                if h > max_h:
                    h, w = max_h, int(max_h * aspect)
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        return image

    def convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [self.convert_to_serializable(x) for x in obj]
            return type(obj)(converted)
        return obj

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = unicodedata.normalize("NFKD", label)
        normalized = "".join(
            ch for ch in normalized if not unicodedata.combining(ch)
        )
        return " ".join(normalized.replace("_", " ").strip().lower().split())

    @staticmethod
    def _box_area(box: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    @staticmethod
    def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = (
            max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
            + max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
            - inter
        )
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _center_distance(
        c1: Tuple[float, float], c2: Tuple[float, float]
    ) -> float:
        return float(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Filtros anti falsos positivos
    # ──────────────────────────────────────────────────────────────────────────

    def _passes_geometry_filters(self, box: np.ndarray, label: str) -> bool:
        """Filtra por área mínima y aspect-ratio (relajado para 'order')."""
        if label == "order" and self._order_relax_filters:
            return True
        area = self._box_area(box)
        if area < self.min_event_area:
            logger.debug(
                "FP-GUARD area: label=%s area=%.0f < min=%d → descartado",
                label, area, self.min_event_area,
            )
            return False
        x1, y1, x2, y2 = box
        h = float(y2 - y1)
        if h <= 0:
            return False
        ratio = float(x2 - x1) / h
        min_r, max_r = self.event_aspect_ratio_range
        if not (min_r <= ratio <= max_r):
            logger.debug(
                "FP-GUARD aspect: label=%s ratio=%.2f not in [%.2f, %.2f] → descartado",
                label, ratio, min_r, max_r,
            )
            return False
        return True

    def _is_spatially_duplicate(
        self, label: str, center: Tuple[float, float]
    ) -> bool:
        """True si ya se registró un evento del mismo tipo cerca de esta posición."""
        log = self._spatial_event_log.get(label, [])
        cutoff = self.frame_counter - self._spatial_dedup_frames
        # Purgar entradas antiguas in-place
        self._spatial_event_log[label] = [
            (f, cx, cy) for f, cx, cy in log if f >= cutoff
        ]
        cx, cy = center
        return any(
            ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 < self._spatial_dedup_distance
            for _, ex, ey in self._spatial_event_log[label]
        )

    def _passes_confidence_threshold(self, label: str, confidence: float) -> bool:
        """Umbral de confianza específico por clase."""
        if label == "order" and self._order_relax_filters:
            return True
        threshold = (
            self.order_confidence_threshold
            if label == "order"
            else self.delivery_confidence_threshold
            if label == "delivery"
            else self.min_event_confidence
        )
        if confidence < threshold:
            logger.debug(
                "FP-GUARD conf: label=%s conf=%.3f < threshold=%.3f → descartado",
                label, confidence, threshold,
            )
            return False
        return True

    def _apply_event_nms(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """NMS inter-evento para el mismo frame, ordenado por confianza."""
        if len(candidates) <= 1:
            return candidates
        candidates_sorted = sorted(
            candidates, key=lambda d: d["confidence"], reverse=True
        )
        kept: List[Dict[str, Any]] = []
        suppressed: set = set()
        for i, det in enumerate(candidates_sorted):
            if i in suppressed:
                continue
            kept.append(det)
            for j in range(i + 1, len(candidates_sorted)):
                if j in suppressed:
                    continue
                if (
                    self._box_iou(det["box"], candidates_sorted[j]["box"])
                    > self.event_nms_iou_threshold
                ):
                    suppressed.add(j)
                    logger.debug(
                        "FP-GUARD NMS: suprimido candidato j=%d (solapamiento con i=%d)", j, i
                    )
        return kept

    def _get_confirmation_buffer(
        self, global_id: Any, label: str
    ) -> EventConfirmationBuffer:
        key = (global_id, label)
        if key not in self._confirmation_buffers:
            if label == "order":
                req, win = self.order_confirmation_frames, self.order_window_size
            elif label == "delivery":
                req, win = self.delivery_confirmation_frames, self.delivery_window_size
            else:
                req, win = self.temporal_confirmation_frames, self.temporal_window_size
            self._confirmation_buffers[key] = EventConfirmationBuffer(
                window_size=win,
                required_frames=req,
                confidence_threshold=self.confidence_vote_threshold,
            )
        return self._confirmation_buffers[key]

    def _update_confirmation(
        self, global_id: Any, label: str, confidence: float
    ) -> bool:
        """
        Actualiza el buffer de confirmación temporal.
        Devuelve True si el evento ha sido confirmado con suficientes frames
        consecutivos y confianza acumulada adecuada.
        """
        # Ya fue registrado → no re-confirmar
        if global_id is not None and label in self.event_history.get(global_id, set()):
            return False

        buf = self._get_confirmation_buffer(global_id, label)

        if self.suppress_confidence_drop:
            key = (global_id, label)
            prev_conf = self._last_confidence.get(key)
            if (
                prev_conf is not None
                and (prev_conf - confidence) > self.confidence_drop_threshold
            ):
                logger.debug(
                    "FP-GUARD conf-drop: id=%s label=%s prev=%.3f cur=%.3f → reset buffer",
                    global_id, label, prev_conf, confidence,
                )
                buf.reset()
            self._last_confidence[key] = confidence

        buf.add(self.frame_counter, confidence)
        confirmed = buf.is_confirmed(self.frame_counter)
        if not confirmed:
            logger.debug(
                "FP-GUARD temporal: id=%s label=%s frame=%d → pendiente confirmación",
                global_id, label, self.frame_counter,
            )
        return confirmed

    def _get_cooldown_frames(self, label: str) -> int:
        if label == "order":
            return self.order_cooldown_frames
        if label == "delivery":
            return self.delivery_cooldown_frames
        return self.event_cooldown_frames

    # ──────────────────────────────────────────────────────────────────────────
    # ReID y tracking
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_appearance_signature(
        self, image: np.ndarray, box: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            h_img, w_img = image.shape[:2]
            x1, x2 = max(0, x1), min(w_img, x2)
            y1, y2 = max(0, y1), min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            ch = crop.shape[0]
            upper = crop[: int(ch * 0.55), :]
            lower = crop[int(ch * 0.55) :, :]

            def _hist(region: np.ndarray) -> np.ndarray:
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                return cv2.normalize(
                    cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]), None
                ).flatten()

            def _lab_mean(region: np.ndarray) -> np.ndarray:
                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                return np.array(cv2.mean(lab)[:3], dtype=np.float32) / 255.0

            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            skin_ratio = (
                float(np.count_nonzero(skin_mask)) / skin_mask.size
                if skin_mask.size
                else 0.0
            )
            skin_mean_bgr = cv2.mean(crop, mask=skin_mask)[:3]
            skin_patch = np.full((4, 4, 3), skin_mean_bgr, dtype=np.uint8)
            skin_lab = cv2.cvtColor(skin_patch, cv2.COLOR_BGR2LAB)
            skin_mean = (np.array(cv2.mean(skin_lab)[:3], dtype=np.float32) / 255.0)

            signature = np.concatenate(
                [
                    _hist(upper), _hist(lower),
                    _lab_mean(upper), _lab_mean(lower),
                    np.array([skin_ratio, skin_mean[0], skin_mean[1], skin_mean[2]]),
                ]
            ).astype(np.float32)
            norm = np.linalg.norm(signature)
            if norm > 0:
                signature /= norm
            return signature
        except Exception:
            return None

    def _compare_features(self, f1: np.ndarray, f2: np.ndarray) -> float:
        denom = float(np.linalg.norm(f1) * np.linalg.norm(f2))
        if denom <= 0:
            return 0.0
        return float(np.dot(f1, f2) / denom)

    def _update_global_features(
        self,
        global_id: int,
        feature: Optional[np.ndarray],
        center: Tuple[float, float],
    ) -> None:
        self.global_id_last_seen[global_id] = self.frame_counter
        self.global_id_last_center[global_id] = center
        self.global_id_first_seen.setdefault(global_id, self.frame_counter)
        if feature is None:
            return
        existing = self.global_id_features.get(global_id)
        if existing is None:
            self.global_id_features[global_id] = feature
        else:
            alpha = self.appearance_update_alpha
            blended = (1 - alpha) * existing + alpha * feature
            norm = np.linalg.norm(blended)
            if norm > 0:
                blended /= norm
            self.global_id_features[global_id] = blended

    def _assign_global_id(
        self,
        track_id: Optional[int],
        center: Tuple[float, float],
        feature: Optional[np.ndarray],
        used_ids: set,
    ) -> Optional[int]:
        # Re-usar mapping existente
        if track_id is not None:
            existing = self.track_id_map.get(track_id)
            if existing is not None:
                self.track_last_seen[track_id] = self.frame_counter
                self._update_global_features(existing, feature, center)
                used_ids.add(existing)
                return existing

        # Buscar por ReID
        best_id, best_score = None, 0.0
        for gid, feat in self.global_id_features.items():
            if gid in used_ids:
                continue
            last_seen = self.global_id_last_seen.get(gid, 0)
            if (
                self.id_reset_frames > 0
                and (self.frame_counter - last_seen) > self.id_reset_frames
            ):
                continue
            last_center = self.global_id_last_center.get(gid)
            if (
                last_center is None
                or self._center_distance(center, last_center) > self.max_reid_distance
            ):
                continue
            if feature is None or feat is None:
                continue
            score = self._compare_features(feature, feat)
            if score > best_score:
                best_score, best_id = score, gid

        if best_id is not None and best_score >= self.appearance_match_threshold:
            if track_id is not None:
                self.track_id_map[track_id] = best_id
                self.track_last_seen[track_id] = self.frame_counter
            self._update_global_features(best_id, feature, center)
            used_ids.add(best_id)
            return best_id

        if track_id is None:
            return None

        # Nuevo ID global
        global_id = self.next_global_id
        self.next_global_id += 1
        self.track_id_map[track_id] = global_id
        self.track_last_seen[track_id] = self.frame_counter
        self._update_global_features(global_id, feature, center)
        used_ids.add(global_id)
        return global_id

    def _release_track(self, track_id: int) -> None:
        global_id = self.track_id_map.pop(track_id, None)
        self.track_last_seen.pop(track_id, None)
        if global_id is not None:
            self.global_id_last_seen[global_id] = self.frame_counter

    def _expire_stale_tracks(self) -> None:
        if self.id_reset_frames <= 0:
            return
        stale_tracks = [
            tid
            for tid, last in self.track_last_seen.items()
            if (self.frame_counter - last) > self.id_reset_frames
        ]
        for tid in stale_tracks:
            self._release_track(tid)

        stale_global = [
            gid
            for gid, last in self.global_id_last_seen.items()
            if (self.frame_counter - last) > self.id_reset_frames
        ]
        for gid in stale_global:
            self.global_id_last_seen.pop(gid, None)
            self.global_id_last_center.pop(gid, None)
            self.global_id_features.pop(gid, None)
            self.global_id_first_seen.pop(gid, None)
            self.event_history.pop(gid, None)
            self.event_last_seen.pop((gid, "order"), None)
            self.event_last_seen.pop((gid, "delivery"), None)
            self._confirmation_buffers.pop((gid, "order"), None)
            self._confirmation_buffers.pop((gid, "delivery"), None)
            self._last_confidence.pop((gid, "order"), None)
            self._last_confidence.pop((gid, "delivery"), None)

    def _is_track_stable(self, global_id: Optional[int]) -> bool:
        if global_id is None or self._order_relax_filters:
            return True
        first_seen = self.global_id_first_seen.get(global_id)
        if first_seen is None:
            return False
        return (self.frame_counter - first_seen) >= self.min_track_age_frames

    # ──────────────────────────────────────────────────────────────────────────
    # Registro de eventos
    # ──────────────────────────────────────────────────────────────────────────

    def _register_event(
        self,
        label: str,
        track_id: Optional[int],
        center: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        cooldown = self._get_cooldown_frames(label)
        if cooldown > 0 and track_id is not None:
            last_seen = self.event_last_seen.get((track_id, label))
            if last_seen is not None and (self.frame_counter - last_seen) <= cooldown:
                return None

        history_key: Any
        if track_id is not None:
            history_key = track_id
        else:
            cx, cy = int(center[0]), int(center[1])
            history_key = f"no_track:{cx}:{cy}"

        if label in self.event_history.get(history_key, set()):
            return None
        self.event_history.setdefault(history_key, set()).add(label)

        if track_id is not None:
            self.event_last_seen[(track_id, label)] = self.frame_counter

        if label == "order":
            self.order_count += 1
        elif label == "delivery":
            self.delivery_count += 1
        self.total_events += 1

        return {
            "event": label,
            "track_id": track_id,
            "center": [float(center[0]), float(center[1])],
            "timestamp": datetime.datetime.now().isoformat(),
            "frame": int(self.frame_counter),
        }

    def _store_frame_counts(
        self,
        frame_number: int,
        orders: int,
        deliveries: int,
        total: int,
        events: List[Dict[str, Any]],
        tracks_in_roi: int,
    ) -> None:
        self.frame_events.append(
            {
                "frame": frame_number,
                "timestamp": datetime.datetime.now().isoformat(),
                "orders": orders,
                "deliveries": deliveries,
                "total": total,
                "tracks_in_roi": tracks_in_roi,
                "events": events,
            }
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Visualización
    # ──────────────────────────────────────────────────────────────────────────

    def _draw(
        self, image: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            label = det["label"]
            color = self._DRAW_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            display_text = self._DISPLAY_NAMES.get(label, "Persona")
            display_id = det.get("global_id") or det.get("track_id")
            if display_id is not None:
                display_text += f" ID:{display_id}"
            cv2.putText(
                out, display_text,
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        # ROI overlay
        overlay = out.copy()
        cv2.fillPoly(overlay, [self.roi_polygon], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.polylines(out, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        # HUD
        hud = out.copy()
        cv2.rectangle(hud, (5, 5), (320, 120), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.55, out, 0.45, 0, out)
        for text, pos in [
            (f"Frame: {self.frame_counter}", (15, 30)),
            (f"Tomas de orden: {self.order_count}", (15, 55)),
            (f"Entregas de plato: {self.delivery_count}", (15, 80)),
            (f"Total eventos: {self.total_events}", (15, 105)),
        ]:
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Red / servidor
    # ──────────────────────────────────────────────────────────────────────────

    def create_server_payload(
        self,
        frame: np.ndarray,
        metadata: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        try:
            compressed = self.compress_image(frame)
            ok, buf = cv2.imencode(
                ".jpg", compressed, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            )
            if not ok:
                return None
            return {
                "id_connection": self.client_id,
                "component_key": self.component_key,
                "type_inference": self.type_inference,
                "data": {
                    "header": {
                        "frame_number": int(self.frame_counter),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "event_type": "hummus_events",
                        "roi_points": self.convert_to_serializable(self.roi_polygon),
                    },
                    "image": base64.b64encode(buf).decode("utf-8"),
                    "data_result": {
                        "statistics": self.convert_to_serializable(metadata),
                        "events": self.convert_to_serializable(events),
                    },
                },
            }
        except Exception as exc:
            logger.error("Error creando payload: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # PROCESO PRINCIPAL DEL FRAME
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        image: np.ndarray,
        roi=None,
        send_to_server: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.frame_counter += 1
        if roi is not None and isinstance(roi, (list, np.ndarray)) and len(roi) >= 3:
            self.roi_polygon = np.array(roi, np.int32)
            self.roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))

        self.last_processed_frame = image.copy()
        detections: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        track_ids_in_roi: set = set()
        active_persons_this_frame: Dict[int, Tuple[float, float]] = {}
        persons_in_roi: List[Dict[str, Any]] = []
        used_global_ids: set = set()

        try:
            # ── 1. Detección de personas ───────────────────────────────────
            person_results = self.person_model.track(
                image,
                persist=True,
                conf=self.person_confidence_threshold,
                iou=self.person_iou_threshold,
                imgsz=640,
                tracker=self.tracker,
                verbose=False,
            )

            if person_results and person_results[0].boxes is not None:
                p_boxes = person_results[0].boxes
                p_xyxy = p_boxes.xyxy.cpu().numpy()
                p_cls  = p_boxes.cls.cpu().numpy()
                p_conf = p_boxes.conf.cpu().numpy()
                p_ids  = (
                    p_boxes.id.cpu().numpy()
                    if p_boxes.id is not None
                    else [None] * len(p_xyxy)
                )

                for i in range(len(p_xyxy)):
                    cname = self._normalize_label(
                        str(self.person_model.names[int(p_cls[i])])
                    )
                    if cname not in self.person_labels_normalized:
                        continue
                    if float(p_conf[i]) < self.min_person_confidence:
                        continue
                    if self._box_area(p_xyxy[i]) < self.min_person_area:
                        continue

                    center: Tuple[float, float] = (
                        (p_xyxy[i][0] + p_xyxy[i][2]) * 0.5,
                        (p_xyxy[i][1] + p_xyxy[i][3]) * 0.5,
                    )
                    if not self.is_inside_roi(center):
                        raw_tid = int(p_ids[i]) if p_ids[i] is not None else None
                        if raw_tid is not None and raw_tid in self.track_id_map:
                            self._release_track(raw_tid)
                        continue

                    feature = self._extract_appearance_signature(image, p_xyxy[i])
                    raw_track_id = int(p_ids[i]) if p_ids[i] is not None else None
                    global_id = self._assign_global_id(
                        raw_track_id, center, feature, used_global_ids
                    )

                    detections.append(
                        {
                            "label": "person",
                            "box": p_xyxy[i],
                            "center": center,
                            "confidence": float(p_conf[i]),
                            "track_id": raw_track_id,
                            "global_id": global_id,
                        }
                    )
                    if global_id is not None:
                        track_ids_in_roi.add(global_id)
                        self.unique_track_ids.add(global_id)
                        active_persons_this_frame[global_id] = center
                        persons_in_roi.append(
                            {
                                "global_id": global_id,
                                "box": p_xyxy[i],
                                "center": center,
                                "feature": feature,
                            }
                        )

            # ── 2. Detección de eventos ────────────────────────────────────
            event_results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=640,
                verbose=False,
            )

            raw_event_candidates: List[Dict[str, Any]] = []

            if event_results and event_results[0].boxes is not None:
                e_boxes = event_results[0].boxes
                e_xyxy = e_boxes.xyxy.cpu().numpy()
                e_cls  = e_boxes.cls.cpu().numpy()
                e_conf = e_boxes.conf.cpu().numpy()

                for i in range(len(e_xyxy)):
                    cname = self._normalize_label(
                        str(self.model.names[int(e_cls[i])])
                    )
                    if cname not in self.target_labels:
                        continue
                    label = self.target_labels[cname]
                    confidence = float(e_conf[i])

                    # FILTRO 1: umbral por clase
                    if not self._passes_confidence_threshold(label, confidence):
                        continue

                    x1, y1, x2, y2 = e_xyxy[i]
                    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                    if not self.is_inside_roi(center):
                        continue

                    # FILTRO 2: área y aspect-ratio
                    if not self._passes_geometry_filters(e_xyxy[i], label):
                        continue

                    raw_event_candidates.append(
                        {
                            "label": label,
                            "box": e_xyxy[i],
                            "center": center,
                            "confidence": confidence,
                            "track_id": None,
                        }
                    )

            # FILTRO 3: NMS inter-evento
            event_candidates = self._apply_event_nms(raw_event_candidates)

            # FILTRO 4: límite por frame
            if len(event_candidates) > self.max_events_per_frame:
                event_candidates = event_candidates[: self.max_events_per_frame]
                logger.debug(
                    "FP-GUARD max_events_per_frame: recortado a %d",
                    self.max_events_per_frame,
                )

            for det in event_candidates:
                label      = det["label"]
                confidence = det["confidence"]
                center     = det["center"]

                # FILTRO 5: persona obligatoria (si activado)
                if self.require_person_match and not persons_in_roi:
                    logger.debug(
                        "FP-GUARD no-person: label=%s frame=%d → descartado",
                        label, self.frame_counter,
                    )
                    continue

                # Asociar la persona más cercana/solapada
                assigned_id: Optional[int] = None
                if persons_in_roi:
                    best_iou_id, best_iou_val = None, 0.0
                    best_dist_id, best_dist_val = None, float("inf")
                    for person in persons_in_roi:
                        iou  = self._box_iou(det["box"], person["box"])
                        dist = self._center_distance(center, person["center"])
                        if iou > best_iou_val:
                            best_iou_val, best_iou_id = iou, person["global_id"]
                        if dist < best_dist_val:
                            best_dist_val, best_dist_id = dist, person["global_id"]

                    match_by_iou  = best_iou_val  >= self.min_event_person_iou
                    match_by_dist = best_dist_val <= self.max_event_person_distance
                    if not (match_by_iou or match_by_dist):
                        logger.debug(
                            "FP-GUARD person-dist: label=%s iou=%.2f dist=%.1f → descartado",
                            label, best_iou_val, best_dist_val,
                        )
                        continue
                    assigned_id = best_iou_id if match_by_iou else best_dist_id

                # Early-exit: evento ya registrado para este ID
                if assigned_id is not None and label in self.event_history.get(
                    assigned_id, set()
                ):
                    continue

                # FILTRO ESPACIAL: duplicado cercano reciente
                if self._is_spatially_duplicate(label, center):
                    logger.debug(
                        "FP-GUARD spatial-dedup: label=%s center=(%.0f,%.0f) → duplicado",
                        label, center[0], center[1],
                    )
                    continue

                # FILTRO 6: madurez del track
                if not self._is_track_stable(assigned_id):
                    logger.debug(
                        "FP-GUARD track-age: id=%s label=%s → inmaduro",
                        assigned_id, label,
                    )
                    continue

                # FILTRO 7: confirmación temporal + votación de confianza
                if not self._update_confirmation(assigned_id, label, confidence):
                    continue

                # ── Evento confirmado ──────────────────────────────────────
                det["global_id"] = assigned_id
                detections.append(det)

                screenshot_key  = (label, assigned_id)
                screenshot_path = self.saved_screenshot_paths.get(screenshot_key, "")
                if screenshot_key not in self.saved_screenshot_keys:
                    try:
                        saved = self._save_event_screenshot(
                            image,
                            {"event": label, "global_id": assigned_id, "track_id": None},
                        )
                        if saved:
                            screenshot_path = saved
                            self.saved_screenshot_paths[screenshot_key] = saved
                        self.saved_screenshot_keys.add(screenshot_key)
                    except Exception as exc:
                        logger.warning(
                            "No se pudo guardar screenshot del evento %s: %s", label, exc
                        )

                evt = self._register_event(label, assigned_id, center)
                if evt:
                    evt["global_id"]       = assigned_id
                    evt["source_track_id"] = None
                    evt["screenshot_path"] = screenshot_path
                    events.append(evt)

                    # Resetear buffer para evitar re-disparos
                    self._confirmation_buffers.pop((assigned_id, label), None)

                    # Registrar posición para deduplicación espacial
                    self._spatial_event_log.setdefault(label, []).append(
                        (self.frame_counter, center[0], center[1])
                    )
                    # Marcar todas las personas activas con este label
                    for pid in active_persons_this_frame:
                        self.event_history.setdefault(pid, set()).add(label)

            self.active_persons = active_persons_this_frame
            self._expire_stale_tracks()

        except Exception as exc:
            logger.error("Error en inferencia (frame %d): %s", self.frame_counter, exc)

        processed = self._draw(image, detections)

        orders_in_frame     = sum(1 for e in events if e["event"] == "order")
        deliveries_in_frame = sum(1 for e in events if e["event"] == "delivery")
        total_frame_events  = orders_in_frame + deliveries_in_frame
        self._store_frame_counts(
            self.frame_counter,
            orders_in_frame,
            deliveries_in_frame,
            total_frame_events,
            events,
            len(track_ids_in_roi),
        )

        # Construir alertas para el cliente
        alerts_for_client: List[Dict[str, Any]] = []
        for evt in events:
            event_label  = evt.get("event", "")
            display_name = self._DISPLAY_NAMES.get(event_label, event_label)
            crop_b64     = ""
            screenshot_path = evt.get("screenshot_path", "")

            # Intentar leer screenshot guardado
            if screenshot_path and os.path.isfile(screenshot_path):
                try:
                    with open(screenshot_path, "rb") as img_file:
                        crop_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                except Exception as exc:
                    logger.warning(
                        "No se pudo leer screenshot para alerta: %s", exc
                    )

            # Fallback: crop del frame procesado alrededor del centro del evento
            if not crop_b64 and image is not None:
                try:
                    evt_center = evt.get("center", [])
                    if len(evt_center) == 2:
                        cx, cy   = int(evt_center[0]), int(evt_center[1])
                        h_img, w_img = image.shape[:2]
                        crop_size    = 80
                        crop = image[
                            max(0, cy - crop_size) : min(h_img, cy + crop_size),
                            max(0, cx - crop_size) : min(w_img, cx + crop_size),
                        ]
                        if crop.size > 0:
                            _, buf = cv2.imencode(
                                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75]
                            )
                            if buf is not None:
                                crop_b64 = base64.b64encode(buf).decode("utf-8")
                except Exception as exc:
                    logger.warning(
                        "No se pudo generar crop para alerta: %s", exc
                    )

            # Timestamp del archivo o momento actual
            timestamp_value = datetime.datetime.now().timestamp()
            if screenshot_path and os.path.isfile(screenshot_path):
                try:
                    timestamp_value = float(os.path.getmtime(screenshot_path))
                except Exception:
                    pass

            alerts_for_client.append(
                {
                    "event_type":      display_name,
                    "class_name":      display_name,
                    "description":     (
                        f"{display_name} detectada "
                        f"(ID: {evt.get('global_id', evt.get('track_id', '?'))})"
                    ),
                    "timestamp":       timestamp_value,
                    "crop_image":      crop_b64,
                    "screenshot_path": screenshot_path or "",
                }
            )

        metadata: Dict[str, Any] = {
            "frame":               int(self.frame_counter),
            "order_count":         int(self.order_count),
            "delivery_count":      int(self.delivery_count),
            "total_events":        int(self.total_events),
            "orders_in_frame":     int(orders_in_frame),
            "deliveries_in_frame": int(deliveries_in_frame),
            "frame_events_total":  int(total_frame_events),
            "tracks_in_roi":       int(len(track_ids_in_roi)),
            "unique_tracks_total": int(len(self.unique_track_ids)),
            "roi_points":          self.convert_to_serializable(self.roi_polygon),
            "alerts":              alerts_for_client,
        }

        # Persistir alertas en CSV
        for evt in events:
            self._append_alert_to_csv(evt)

        # Log de texto
        if events and self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    for evt in events:
                        ts = self._format_timestamp_millis(evt.get("timestamp"))
                        f.write(
                            f"{ts},{evt['frame']},{self.order_count},"
                            f"{self.delivery_count},{self.total_events},{evt['event']}\n"
                        )
            except Exception:
                pass

        return processed, {"events": events, **metadata}

    # ──────────────────────────────────────────────────────────────────────────
    # Reset y estadísticas
    # ──────────────────────────────────────────────────────────────────────────

    def reset_counter(self) -> None:
        """Reinicia todos los contadores y estructuras de tracking."""
        self.order_count = 0
        self.delivery_count = 0
        self.total_events = 0
        self.frame_counter = 0
        self.next_global_id = 1
        self.event_history.clear()
        self.unique_track_ids.clear()
        self.track_id_map.clear()
        self.track_last_seen.clear()
        self.global_id_last_seen.clear()
        self.global_id_last_center.clear()
        self.global_id_features.clear()
        self.event_last_seen.clear()
        self.global_id_first_seen.clear()
        self.saved_screenshot_keys.clear()
        self.saved_screenshot_paths.clear()
        self._confirmation_buffers.clear()
        self._last_confidence.clear()
        self._spatial_event_log = {"order": [], "delivery": []}
        self.frame_events.clear()
        logger.info("Contadores reiniciados")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna métricas actuales del proceso."""
        return {
            "order_count":         int(self.order_count),
            "delivery_count":      int(self.delivery_count),
            "total_events":        int(self.total_events),
            "frame_counter":       int(self.frame_counter),
            "unique_tracks_total": int(len(self.unique_track_ids)),
            "roi_points":          self.convert_to_serializable(self.roi_polygon),
        }


def create_vehicle_processor(**kwargs) -> HummusProcess:
    """Factory de compatibilidad hacia atrás."""
    return HummusProcess(**kwargs)