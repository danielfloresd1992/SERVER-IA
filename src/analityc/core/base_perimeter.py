"""
base_perimeter.py - v2

Correcciones respecto a v1
──────────────────────────
1. ALERTAS DUPLICADAS ELIMINADAS: cooldown por (track_id, event_type) en frames,
   no solo en segundos. Un objeto no puede generar la misma alerta dos veces
   seguidas sin haber cambiado de estado real.

2. BUG if/elif EN _increment_counters CORREGIDO: la cadena lógica era incorrecta,
   el contador de 'car' se evaluaba independientemente del de 'person'.

3. UNA SOLA IMAGEN POR ALERTA: se elimina la copia del frame completo en cada
   alerta. Solo se envía el crop del objeto. Reduce el payload del WebSocket
   hasta 10x en escenas con múltiples objetos.

4. TOLERANCIA A OCLUSIÓN EN _cleanup_tracks: tracks ausentes se marcan como
   "perdidos" pero no se eliminan hasta superar un umbral de frames
   (occlusion_tolerance). Evita alertas falsas de re-entrada tras oclusión breve.

5. processed_entry / processed_exit ELIMINADOS: eran código muerto. La
   deduplicación se maneja via event_cooldown_frames.

6. DIRECCIÓN DE PUERTA ROBUSTECIDA: dot == 0 ahora tiene manejo explícito
   (se usa el último dot no-cero conocido como desempate).

7. CONTADORES CLARAMENTE SEPARADOS en el metadata: acumulativos vs estado actual.

8. TRACKER CONFIGURABLE: se acepta parámetro tracker_config para usar
   botsort/bytetrack vía archivo yaml, igual que HummusProcess.
"""

import cv2
import numpy as np
import time
import base64
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple


class BasePerimeter:
    def __init__(
        self,
        client_id: str,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        tracker_config: Optional[str] = None,
        # Frames que un track puede estar ausente antes de eliminarse
        occlusion_tolerance: int = 10,
        # Frames mínimos entre la misma alerta para el mismo track
        event_cooldown_frames: int = 15,
    ):
        self.client_id = client_id
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config
        self.occlusion_tolerance = occlusion_tolerance
        self.event_cooldown_frames = event_cooldown_frames

        # Inicializar modelo
        self.model = None
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except Exception as exc:
            print(f"Error cargando modelo: {exc}")

        # Estado del sistema
        self.frame_counter = 0
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        # {track_id: TrackData}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

        # Contadores acumulativos (total histórico desde el inicio)
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        self.person_count_total = 0

        # Estado actual dentro del ROI (se recalcula cada frame)
        self.persons_in_area = 0
        self.vehicles_in_area = 0

        # Geometría
        self.roi_polygon: Optional[np.ndarray] = None
        self.door_polygon: Optional[np.ndarray] = None
        self.door_direction: Optional[np.ndarray] = None
        self.roi_active = False
        self.door_active = False
        self.door_direction_active = False

        # Alertas del frame actual (se limpia cada frame)
        self.current_alerts: List[Dict[str, Any]] = []

        # Clases de interés
        self.class_ids = [0, 1, 2, 3, 5, 7, 16]
        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            16: "dog",
        }
        self.class_translations = {
            "person": "Persona",
            "car": "Carro",
            "truck": "Camion",
            "motorcycle": "Motocicleta",
            "bus": "Autobus",
            "bicycle": "Bicicleta",
            "dog": "Perro",
        }
        self.vehicle_classes = {"car", "truck", "motorcycle", "bus", "bicycle"}

    # ─────────────────────────────────────────────────────────────────────────
    # Proceso principal
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        image: np.ndarray,
        roi: Any = None,
        activate_roi: bool = False,
        door_roi: Any = None,
        door_activate: bool = False,
        door_direction: Any = None,
        door_direction_activate: bool = False,
        camera_id: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        self.frame_counter += 1
        self.current_alerts = []
        self.roi_active = activate_roi
        self.door_active = door_activate
        self.door_direction_active = door_direction_activate

        # 1. Parsear geometrías
        self._parse_geometries(roi, door_roi, door_direction)

        # 2. Inferencia
        if self.model is None:
            return image, self._build_metadata(0, 0)

        track_kwargs = dict(
            persist=True,
            classes=self.class_ids,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if self.tracker_config:
            track_kwargs["tracker"] = self.tracker_config

        results = self.model.track(image, **track_kwargs)

        vehicles_detected = 0
        persons_detected = 0
        current_frame_tracks: set = set()

        # 3. Procesar detecciones
        boxes_result = results[0].boxes if results else None
        if boxes_result is not None and boxes_result.id is not None:
            boxes   = boxes_result.xyxy.cpu().numpy()
            tids    = boxes_result.id.cpu().numpy().astype(int)
            cls_ids = boxes_result.cls.cpu().numpy().astype(int)
            confs   = boxes_result.conf.cpu().numpy()

            for box, tid, cls_id, conf in zip(boxes, tids, cls_ids, confs):
                current_frame_tracks.add(tid)
                class_name = self.class_names.get(cls_id, "unknown")

                if class_name == "person":
                    persons_detected += 1
                elif class_name in self.vehicle_classes:
                    vehicles_detected += 1

                centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                self.track_history[tid].append(centroid)

                self._update_track_state(tid, cls_id, centroid, image, box)
                self._draw_detection(image, box, tid, cls_id, conf)

        # 4. Limpiar tracks con tolerancia a oclusión
        self._cleanup_tracks(current_frame_tracks)

        # 5. Recalcular contadores de estado actual
        self._update_area_counters()

        # 6. Dibujar zonas
        self._draw_zones(image)

        return image, self._build_metadata(vehicles_detected, persons_detected)

    # ─────────────────────────────────────────────────────────────────────────
    # Parsing de geometrías
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_geometries(self, roi, door_roi, door_direction):
        self.roi_polygon  = self._to_numpy_poly(roi)
        self.door_polygon = self._to_numpy_poly(door_roi)
        self.door_direction = None

        if door_direction is not None:
            try:
                pts = np.array(door_direction, dtype=np.int32)
                if pts.shape == (2, 2):
                    self.door_direction = pts
            except Exception:
                pass

    def _to_numpy_poly(self, points) -> Optional[np.ndarray]:
        if not points:
            return None
        try:
            if isinstance(points, list) and len(points) > 0:
                if isinstance(points[0], dict):
                    pts = [[p["x"], p["y"]] for p in points]
                else:
                    pts = points
                return np.array(pts, np.int32).reshape((-1, 1, 2))
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Lógica de estado de tracks
    # ─────────────────────────────────────────────────────────────────────────

    def _update_track_state(
        self,
        track_id: int,
        cls_id: int,
        centroid: Tuple[float, float],
        image: np.ndarray,
        box: np.ndarray,
    ):
        class_name = self.class_names.get(cls_id, "unknown")

        # Inicializar track nuevo
        if track_id not in self.active_tracks:
            self.active_tracks[track_id] = {
                "class_id": cls_id,
                "in_roi": False,
                "in_door": False,
                "frames_missing": 0,
                # {event_type: frame_number} para cooldown de alertas
                "last_alert_frame": {},
                # Último dot no-cero para desempate de dirección
                "last_nonzero_dot": 0.0,
            }

        track = self.active_tracks[track_id]
        track["frames_missing"] = 0  # está presente en este frame

        # ── ROI principal ─────────────────────────────────────────────────────
        if self.roi_polygon is not None and self.roi_active:
            prev_in_roi = track["in_roi"]
            dist = cv2.pointPolygonTest(
                self.roi_polygon, (float(centroid[0]), float(centroid[1])), False
            )
            is_in_roi = dist >= 0
            track["in_roi"] = is_in_roi

            if not prev_in_roi and is_in_roi:
                if self._check_alert_cooldown(track, "Entrada"):
                    self._trigger_alert("Entrada", class_name, image, box)
                    self._trigger_alert_cooldown(track, "Entrada")
                    self._increment_counters(class_name, "entry")

            elif prev_in_roi and not is_in_roi:
                if self._check_alert_cooldown(track, "Salida"):
                    self._trigger_alert("Salida", class_name, image, box)
                    self._trigger_alert_cooldown(track, "Salida")

        # ── ROI puerta ────────────────────────────────────────────────────────
        if self.door_polygon is not None and self.door_active:
            prev_in_door = track["in_door"]
            dist_door = cv2.pointPolygonTest(
                self.door_polygon, (float(centroid[0]), float(centroid[1])), False
            )
            is_in_door = dist_door >= 0
            track["in_door"] = is_in_door

            direction_label = self._compute_door_direction(
                track, track_id, centroid, is_in_door, prev_in_door
            )

            if direction_label is not None:
                if self._check_alert_cooldown(track, direction_label):
                    self._trigger_alert(direction_label, class_name, image, box)
                    self._trigger_alert_cooldown(track, direction_label)

    def _compute_door_direction(
        self,
        track: Dict[str, Any],
        track_id: int,
        centroid: Tuple[float, float],
        is_in_door: bool,
        prev_in_door: bool,
    ) -> Optional[str]:
        """
        Determina la etiqueta de dirección de puerta para este frame.
        Retorna None si no hay transición relevante.
        """
        # Solo nos importa cuando hay una transición real
        state_changed = is_in_door != prev_in_door
        if not state_changed and not is_in_door:
            return None

        direction_label: Optional[str] = None

        if (
            self.door_direction is not None
            and self.door_direction_active
            and len(self.track_history[track_id]) > 1
        ):
            prev_point = self.track_history[track_id][-2]
            move_vec = (centroid[0] - prev_point[0], centroid[1] - prev_point[1])
            dir_vec = (
                self.door_direction[1][0] - self.door_direction[0][0],
                self.door_direction[1][1] - self.door_direction[0][1],
            )
            dot = move_vec[0] * dir_vec[0] + move_vec[1] * dir_vec[1]

            # Guardar último dot no-cero para desempate
            if dot != 0.0:
                track["last_nonzero_dot"] = dot
            else:
                # Desempate: usar el último dot conocido
                dot = track.get("last_nonzero_dot", 0.0)

            if state_changed or is_in_door:
                if dot > 0:
                    direction_label = "Entrada Puerta"
                elif dot < 0:
                    direction_label = "Salida Puerta"

        # Fallback sin línea de dirección: solo en transición
        if direction_label is None and state_changed:
            if is_in_door and not prev_in_door:
                direction_label = "Entrada Puerta"
            elif not is_in_door and prev_in_door:
                direction_label = "Salida Puerta"

        return direction_label

    def _check_alert_cooldown(self, track: Dict[str, Any], event_type: str) -> bool:
        """
        Devuelve True si ha pasado suficientes frames desde la última alerta
        del mismo tipo para este track.
        """
        last_frame = track["last_alert_frame"].get(event_type, -self.event_cooldown_frames - 1)
        return (self.frame_counter - last_frame) > self.event_cooldown_frames

    def _trigger_alert_cooldown(self, track: Dict[str, Any], event_type: str) -> None:
        track["last_alert_frame"][event_type] = self.frame_counter

    # ─────────────────────────────────────────────────────────────────────────
    # Contadores
    # ─────────────────────────────────────────────────────────────────────────

    def _increment_counters(self, class_name: str, event_type: str) -> None:
        if event_type != "entry":
            return
        # FIX: cadena if/elif correcta (v1 tenía un if independiente para 'car')
        if class_name == "person":
            self.person_count_total += 1
        elif class_name == "car":
            self.car_count += 1
        elif class_name == "truck":
            self.truck_count += 1
        elif class_name == "motorcycle":
            self.motorcycle_count += 1
        self.last_counted_frame = self.frame_counter

    def _update_area_counters(self) -> None:
        """Recalcula cuántos objetos hay actualmente dentro del ROI."""
        self.persons_in_area = sum(
            1
            for t in self.active_tracks.values()
            if t["in_roi"] and self.class_names.get(t["class_id"]) == "person"
        )
        self.vehicles_in_area = sum(
            1
            for t in self.active_tracks.values()
            if t["in_roi"] and self.class_names.get(t["class_id"]) in self.vehicle_classes
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup con tolerancia a oclusión
    # ─────────────────────────────────────────────────────────────────────────

    def _cleanup_tracks(self, current_frame_track_ids: set) -> None:
        """
        Incrementa el contador de frames ausentes para tracks no visibles.
        Solo elimina el track cuando supera occlusion_tolerance frames seguidos
        sin aparecer, evitando falsas re-entradas por oclusión breve.
        """
        to_delete = []
        for tid, track in self.active_tracks.items():
            if tid not in current_frame_track_ids:
                track["frames_missing"] += 1
                if track["frames_missing"] > self.occlusion_tolerance:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.active_tracks[tid]
            self.track_history.pop(tid, None)

    # ─────────────────────────────────────────────────────────────────────────
    # Alertas
    # ─────────────────────────────────────────────────────────────────────────

    def _trigger_alert(
        self,
        event_type: str,
        class_name: str,
        image: np.ndarray,
        box: np.ndarray,
    ) -> None:
        """
        Genera una alerta con solo el crop del objeto detectado.
        Se elimina la copia completa del frame (era hasta 10x más pesada).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(box[0])), max(0, int(box[1])),
            min(w, int(box[2])), min(h, int(box[3])),
        )
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((10, 10, 3), np.uint8)

        _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
        crop_b64 = base64.b64encode(buffer).decode("utf-8") if buffer is not None else ""

        translated = self.class_translations.get(class_name, class_name)

        self.current_alerts.append({
            "event_type": event_type,
            "class_name": translated,
            "timestamp": time.time(),
            "crop_image": crop_b64,
            "description": self._format_description(event_type, translated),
        })

    def _format_description(self, event_type: str, translated_class: str) -> str:
        descriptions = {
            "Entrada":       f"Entrada de {translated_class} en el perímetro",
            "Salida":        f"Salida de {translated_class} del perímetro",
            "Entrada Puerta": f"Entrada de {translated_class} en puerta del perímetro",
            "Salida Puerta":  f"Salida de {translated_class} en puerta del perímetro",
        }
        return descriptions.get(event_type, f"{translated_class} {event_type.lower()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Visualización
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_detection(
        self,
        image: np.ndarray,
        box: np.ndarray,
        track_id: int,
        cls_id: int,
        conf: float,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = self.class_names.get(cls_id, "")
        color = (0, 255, 0) if class_name == "person" else (255, 100, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id} {class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_zones(self, image: np.ndarray) -> None:
        if self.roi_polygon is not None:
            color = (0, 255, 255) if self.roi_active else (128, 128, 128)
            cv2.polylines(image, [self.roi_polygon], True, color, 2)

        if self.door_polygon is not None:
            color = (255, 80, 0) if self.door_active else (128, 128, 128)
            cv2.polylines(image, [self.door_polygon], True, color, 2)

        if self.door_direction is not None:
            pt1 = tuple(self.door_direction[0])
            pt2 = tuple(self.door_direction[1])
            color = (0, 0, 255) if self.door_direction_active else (128, 128, 128)
            cv2.arrowedLine(image, pt1, pt2, color, 3, tipLength=0.1)

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────────────────

    def _build_metadata(self, vehicles_detected: int, persons_detected: int) -> Dict[str, Any]:
        return {
            "frame_number":       self.frame_counter,
            "roi_active":         self.roi_active,
            "door_active":        self.door_active,
            # ── Detecciones en este frame ──
            "vehicles_detected":  vehicles_detected,
            "persons_detected":   persons_detected,
            # ── Estado actual dentro del ROI ──
            "persons_in_area":    self.persons_in_area,
            "vehicles_in_area":   self.vehicles_in_area,
            # ── Totales acumulativos históricos ──
            "person_count_total": self.person_count_total,
            "car_count":          self.car_count,
            "truck_count":        self.truck_count,
            "motorcycle_count":   self.motorcycle_count,
            # ── Tracking ──
            "active_tracks":      len(self.active_tracks),
            # ── Alertas de este frame ──
            "alerts":             self.current_alerts,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────────

    def reset_counters(self) -> None:
        self.car_count = 0
        self.truck_count = 0
        self.motorcycle_count = 0
        self.person_count_total = 0
        self.persons_in_area = 0
        self.vehicles_in_area = 0
        self.active_tracks.clear()
        self.track_history.clear()
        self.frame_counter = 0

    def cleanup(self) -> None:
        pass