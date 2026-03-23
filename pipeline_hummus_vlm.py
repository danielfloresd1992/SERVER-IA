#!/usr/bin/env python3
"""
Pipeline de detección de eventos en mostrador — YOLO11 + LLaVA VLM
===================================================================
Detecta dos eventos en video de cámara cenital de un local de comida:

  A) "Toma de Orden"   — empleado entrega ticket al cliente  (zona caja)
  B) "Entrega de Comida" — cliente retira bandeja naranja     (zona entrega)

Métrica final:  tiempo_espera = timestamp(B) − timestamp(A)  por track_id.

Arquitectura (4 capas):
  1. YOLO11n  → detección de personas (clase "person" de COCO)
  2. ROI      → dos polígonos configurables (zona_caja, zona_entrega)
  3. ByteTrack→ tracking por ID, detección de "persona detenida"
  4. LLaVA 4b → clasificación visual del crop (solo si detenido en zona)

Hardware objetivo: NVIDIA Quadro RTX 4000 (8 GB VRAM)
  YOLO11n  ≈   10 MB  |  LLaVA-7B 4-bit ≈ 4 GB  |  Total ≈ 4.1 GB ✓

Uso:
    python pipeline_hummus_vlm.py --source video.mp4
    python pipeline_hummus_vlm.py --source rtsp://user:pass@ip:port/stream
    python pipeline_hummus_vlm.py --source video.mp4 --no-vlm   # solo YOLO+zonas
    python pipeline_hummus_vlm.py --source video.mp4 --config config.json

Dependencias:
    pip install ultralytics>=8.3 opencv-python-headless torch torchvision \\
                transformers>=4.36 bitsandbytes>=0.41 accelerate pillow
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — editar coordenadas ROI y parámetros aquí
# ═════════════════════════════════════════════════════════════════════════════
# Las coordenadas ROI por defecto asumen resolución ≈960×540.
# Recalibrar si la cámara tiene otra resolución.

CONFIG: Dict[str, Any] = {
    # ── Polígonos ROI [x, y] en píxeles ──────────────────────────────────
    "zona_caja": [
        [50, 330], [290, 330], [290, 530], [50, 530]
    ],
    "zona_entrega": [
        [300, 150], [720, 150], [720, 390], [300, 390]
    ],
    # ── YOLO ─────────────────────────────────────────────────────────────
    "yolo_model": "yolo11n.pt",
    "yolo_conf": 0.35,
    "yolo_iou": 0.45,
    "yolo_imgsz": 640,
    "person_class_id": 0,           # COCO class 0 = person
    # ── Tracking / detección de parada ───────────────────────────────────
    "stop_threshold_px": 30,        # px máx de desplazamiento
    "stop_duration_s": 3.0,         # segundos detenido para trigger
    "centroid_history_maxlen": 150,  # ~5 s @ 30 fps
    # ── VLM ──────────────────────────────────────────────────────────────
    "vlm_model": "llava-hf/llava-1.5-7b-hf",
    "vlm_cooldown_s": 10.0,         # cooldown tras "Sí"
    "vlm_no_cooldown_s": 2.0,       # cooldown tras "No" (anti-spam)
    "crop_pad_ratio": 0.4,          # expandir bbox un 40 % para el crop
    # ── Prompts binarios (respuesta: Yes / No) ───────────────────────────
    "prompt_caja": (
        "In this overhead image of a fast-food counter, is an employee's hand "
        "extending a small piece of paper (receipt/ticket) toward the customer's "
        "hand? Answer only Yes or No."
    ),
    "prompt_entrega": (
        "In this overhead image of a fast-food counter, is the customer picking "
        "up or holding a large orange tray with food? Answer only Yes or No."
    ),
    # ── Display ──────────────────────────────────────────────────────────
    "show_preview": True,
    "preview_scale": 0.75,
}


# ═════════════════════════════════════════════════════════════════════════════
# CAPA 4 — CLASIFICADOR VLM  (LLaVA-1.5 7B, 4-bit)
# ═════════════════════════════════════════════════════════════════════════════

class VLMClassifier:
    """Carga LLaVA 4-bit una sola vez y corre inferencia en un hilo dedicado."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.device = device
        self._request_q: queue.Queue = queue.Queue(maxsize=4)
        self._results: Dict[int, Optional[bool]] = {}
        self._results_lock = threading.Lock()
        self._next_id = 0
        self._running = True

        logger.info("Cargando VLM %s en 4-bit …", model_name)
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
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info(
            "VLM listo — VRAM ocupada: %.0f MB",
            torch.cuda.memory_allocated() / 1e6,
        )

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── API pública ──────────────────────────────────────────────────────

    def submit(self, crop: np.ndarray, prompt: str) -> int:
        """Encola un crop + prompt. Retorna un request_id."""
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
        """True/False si listo, None si aún pendiente o descartado."""
        with self._results_lock:
            return self._results.pop(req_id, None)

    def stop(self) -> None:
        self._running = False
        self._request_q.put(None)

    # ── Hilo consumidor ──────────────────────────────────────────────────

    def _worker(self) -> None:
        while self._running:
            item = self._request_q.get()
            if item is None:
                break
            req_id, crop, prompt = item
            try:
                result = self._infer(crop, prompt)
            except Exception as exc:
                logger.error("VLM error: %s", exc)
                result = False
            with self._results_lock:
                self._results[req_id] = result

    @torch.no_grad()
    def _infer(self, crop: np.ndarray, prompt: str) -> bool:
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
            out_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip().lower()

        torch.cuda.empty_cache()

        token = generated.split()[0] if generated else ""
        is_yes = token.startswith(("yes", "sí", "si"))
        logger.debug("VLM → '%s' → %s", generated, is_yes)
        return is_yes


# ═════════════════════════════════════════════════════════════════════════════
# CAPA 2 + 3 — ZONAS ROI  +  TRACKER DE ESTADOS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackState:
    """Estado interno por persona rastreada."""
    zone: Optional[str] = None
    t_entry: float = 0.0
    stopped: bool = False
    centroids: deque = field(default_factory=lambda: deque(maxlen=150))
    bbox: Optional[np.ndarray] = None
    vlm_cooldown_until: float = 0.0
    pending_req: Optional[int] = None


class ZoneTracker:
    """Gestiona polígonos ROI y determina cuándo una persona está detenida."""

    def __init__(
        self,
        zones: Dict[str, List[List[int]]],
        stop_px: float = 30.0,
        stop_s: float = 3.0,
        maxlen: int = 150,
    ) -> None:
        self.polys: Dict[str, np.ndarray] = {
            n: np.array(pts, np.int32) for n, pts in zones.items()
        }
        self.stop_px = stop_px
        self.stop_s = stop_s
        self.maxlen = maxlen
        self.tracks: Dict[int, TrackState] = {}

    def update(
        self,
        tid: int,
        centroid: Tuple[float, float],
        bbox: np.ndarray,
        t: float,
    ) -> Optional[str]:
        """Actualiza track. Retorna nombre de zona si detenido, None si no."""
        if tid not in self.tracks:
            self.tracks[tid] = TrackState(
                centroids=deque(maxlen=self.maxlen)
            )
        s = self.tracks[tid]
        s.centroids.append((t, centroid[0], centroid[1]))
        s.bbox = bbox

        zone = self._zone_of(centroid)
        if zone != s.zone:
            s.zone = zone
            s.t_entry = t
            s.stopped = False
            s.centroids.clear()
            s.centroids.append((t, centroid[0], centroid[1]))
            return None

        if zone is None:
            s.stopped = False
            return None

        s.stopped = self._check_stopped(s, t)
        return zone if s.stopped else None

    def crop_bbox(
        self, tid: int, shape: Tuple[int, ...], pad: float = 0.4
    ) -> Optional[Tuple[int, int, int, int]]:
        """Bbox expandido para el crop que se envía al VLM."""
        s = self.tracks.get(tid)
        if s is None or s.bbox is None:
            return None
        x1, y1, x2, y2 = s.bbox
        w, h = x2 - x1, y2 - y1
        px, py = w * pad, h * pad
        H, W = shape[:2]
        return (
            int(max(0, x1 - px)),
            int(max(0, y1 - py)),
            int(min(W, x2 + px)),
            int(min(H, y2 + py)),
        )

    def can_vlm(self, tid: int, t: float) -> bool:
        s = self.tracks.get(tid)
        return s is not None and t >= s.vlm_cooldown_until and s.pending_req is None

    def set_cooldown(self, tid: int, seconds: float, t: float) -> None:
        s = self.tracks.get(tid)
        if s:
            s.vlm_cooldown_until = t + seconds

    def cleanup(self, alive: set, t: float, max_age: float = 30.0) -> None:
        gone = [
            k for k, v in self.tracks.items()
            if k not in alive
            and v.centroids
            and (t - v.centroids[-1][0]) > max_age
        ]
        for k in gone:
            del self.tracks[k]

    # ── privados ─────────────────────────────────────────────────────────

    def _zone_of(self, pt: Tuple[float, float]) -> Optional[str]:
        px, py = int(pt[0]), int(pt[1])
        for name, poly in self.polys.items():
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (px, py), False) >= 0:
                return name
        return None

    def _check_stopped(self, s: TrackState, t: float) -> bool:
        if not s.centroids:
            return False
        cutoff = t - self.stop_s
        recent = [(tt, cx, cy) for tt, cx, cy in s.centroids if tt >= cutoff]
        if not recent or (t - recent[0][0]) < self.stop_s * 0.8:
            return False
        cx0, cy0 = recent[-1][1], recent[-1][2]
        max_d = max(
            ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
            for _, cx, cy in recent
        )
        return max_d < self.stop_px


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════

_ZONE_COLORS = {"caja": (0, 200, 255), "entrega": (0, 255, 128)}


def draw_overlay(
    frame: np.ndarray,
    zt: ZoneTracker,
    events: List[Dict[str, Any]],
) -> np.ndarray:
    out = frame.copy()

    # Zonas semi-transparentes
    for name, poly in zt.polys.items():
        c = _ZONE_COLORS.get(name, (255, 255, 255))
        ov = out.copy()
        cv2.fillPoly(ov, [poly], c)
        cv2.addWeighted(ov, 0.15, out, 0.85, 0, out)
        cv2.polylines(out, [poly], True, c, 2)
        mx = int(poly[:, 0].mean()) - 30
        my = int(poly[:, 1].min()) - 10
        cv2.putText(out, name.upper(), (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    # Tracks
    for tid, s in zt.tracks.items():
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

    # HUD — últimos 5 eventos
    n = min(5, len(events))
    if n > 0:
        hud = out.copy()
        cv2.rectangle(hud, (4, 4), (440, 28 + 22 * n), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.6, out, 0.4, 0, out)
        for i, e in enumerate(events[-5:]):
            txt = f"{e['event']}  ID:{e['track_id']}  {e.get('time_str','')}"
            if "wait_time_s" in e:
                txt += f"  espera:{e['wait_time_s']}s"
            cv2.putText(out, txt, (10, 24 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline YOLO+LLaVA mostrador")
    ap.add_argument("--source", required=True,
                     help="Ruta .mp4, .avi o URL RTSP")
    ap.add_argument("--config", default=None,
                     help="JSON con override de CONFIG (opcional)")
    ap.add_argument("--no-vlm", action="store_true",
                     help="Solo YOLO + zonas, sin LLaVA")
    ap.add_argument("--no-preview", action="store_true",
                     help="Sin ventana OpenCV")
    args = ap.parse_args()

    cfg = CONFIG.copy()
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    if args.no_preview:
        cfg["show_preview"] = False

    # ── CAPA 1: YOLO ─────────────────────────────────────────────────────
    logger.info("Cargando YOLO %s …", cfg["yolo_model"])
    yolo = YOLO(cfg["yolo_model"])
    yolo.to("cuda")
    logger.info("YOLO listo — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    # ── CAPA 4: VLM (opcional) ───────────────────────────────────────────
    vlm: Optional[VLMClassifier] = None
    if not args.no_vlm:
        vlm = VLMClassifier(cfg["vlm_model"])
    else:
        logger.info("VLM desactivado (--no-vlm)")

    # ── CAPAS 2+3: zonas + tracker ───────────────────────────────────────
    zt = ZoneTracker(
        zones={"caja": cfg["zona_caja"], "entrega": cfg["zona_entrega"]},
        stop_px=cfg["stop_threshold_px"],
        stop_s=cfg["stop_duration_s"],
        maxlen=cfg["centroid_history_maxlen"],
    )

    prompts = {"caja": cfg["prompt_caja"], "entrega": cfg["prompt_entrega"]}
    event_names = {"caja": "order_taken", "entrega": "food_delivered"}

    # Estado global
    pending: Dict[int, Tuple[int, str, float]] = {}   # req_id → (tid, zone, t)
    order_ts: Dict[int, float] = {}                    # tid → timestamp de orden
    events_log: List[Dict[str, Any]] = []

    # ── Video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error("No se pudo abrir: %s", args.source)
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frame = 0
    t0 = time.time()
    logger.info("Loop iniciado — source=%s  fps_src=%.1f", args.source, fps_src)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("Fin de video / error de lectura.")
                break

            n_frame += 1
            now = time.time()

            # ── YOLO + ByteTrack ─────────────────────────────────────────
            results = yolo.track(
                frame,
                persist=True,
                conf=cfg["yolo_conf"],
                iou=cfg["yolo_iou"],
                imgsz=cfg["yolo_imgsz"],
                classes=[cfg["person_class_id"]],
                verbose=False,
            )

            alive: set = set()

            if results and results[0].boxes is not None:
                bxs = results[0].boxes
                xyxy = bxs.xyxy.cpu().numpy()
                ids = (
                    bxs.id.cpu().numpy()
                    if bxs.id is not None
                    else [None] * len(xyxy)
                )

                for i in range(len(xyxy)):
                    tid = int(ids[i]) if ids[i] is not None else None
                    if tid is None:
                        continue
                    alive.add(tid)

                    x1, y1, x2, y2 = xyxy[i]
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

                    stopped_zone = zt.update(tid, (cx, cy), xyxy[i], now)

                    # ── Disparar VLM si detenido ─────────────────────────
                    if stopped_zone and vlm and zt.can_vlm(tid, now):
                        cb = zt.crop_bbox(tid, frame.shape, cfg["crop_pad_ratio"])
                        if cb:
                            crop = frame[cb[1]:cb[3], cb[0]:cb[2]]
                            if crop.size > 0:
                                rid = vlm.submit(crop, prompts[stopped_zone])
                                zt.tracks[tid].pending_req = rid
                                pending[rid] = (tid, stopped_zone, now)

            # ── Recoger resultados VLM ───────────────────────────────────
            if vlm:
                done = []
                for rid, (tid, zone, t_sub) in pending.items():
                    res = vlm.poll(rid)
                    if res is None:           # aún pendiente
                        continue
                    done.append(rid)

                    if tid in zt.tracks:
                        zt.tracks[tid].pending_req = None

                    if res is True:
                        ename = event_names[zone]
                        evt: Dict[str, Any] = {
                            "event": ename,
                            "track_id": tid,
                            "timestamp": now,
                            "time_str": time.strftime("%H:%M:%S"),
                            "frame": n_frame,
                        }
                        if ename == "order_taken":
                            order_ts[tid] = now
                        elif ename == "food_delivered" and tid in order_ts:
                            wt = round(now - order_ts.pop(tid), 1)
                            evt["wait_time_s"] = wt
                            logger.info(
                                "Tiempo de espera ID %d: %.1f s", tid, wt
                            )

                        events_log.append(evt)
                        print(json.dumps(evt, ensure_ascii=False), flush=True)
                        zt.set_cooldown(tid, cfg["vlm_cooldown_s"], now)
                    else:
                        zt.set_cooldown(tid, cfg["vlm_no_cooldown_s"], now)

                for rid in done:
                    del pending[rid]

            # ── Limpieza periódica ───────────────────────────────────────
            if n_frame % 150 == 0:
                zt.cleanup(alive, now)

            # ── Preview ──────────────────────────────────────────────────
            if cfg["show_preview"]:
                vis = draw_overlay(frame, zt, events_log)
                sc = cfg["preview_scale"]
                if sc != 1.0:
                    vis = cv2.resize(vis, None, fx=sc, fy=sc)
                cv2.imshow("Pipeline Hummus VLM", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    zt.tracks.clear()
                    order_ts.clear()
                    logger.info("Tracks reiniciados (tecla R)")

            # ── Log FPS cada 300 frames ──────────────────────────────────
            if n_frame % 300 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "f=%d  FPS=%.1f  tracks=%d  events=%d",
                    n_frame,
                    n_frame / elapsed if elapsed > 0 else 0,
                    len(zt.tracks),
                    len(events_log),
                )

    except KeyboardInterrupt:
        logger.info("Interrumpido.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if vlm:
            vlm.stop()

        # ── Resumen final ────────────────────────────────────────────────
        orders = [e for e in events_log if e["event"] == "order_taken"]
        deliv = [e for e in events_log if e["event"] == "food_delivered"]
        waits = [e["wait_time_s"] for e in deliv if "wait_time_s" in e]

        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info("  Frames procesados : %d", n_frame)
        logger.info("  Ordenes tomadas   : %d", len(orders))
        logger.info("  Entregas          : %d", len(deliv))
        if waits:
            logger.info(
                "  Espera — min: %.1fs  max: %.1fs  prom: %.1fs",
                min(waits), max(waits), sum(waits) / len(waits),
            )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
