import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import cv2
import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from ..analityc.core.Perimetrales import MultiObjectProcessor
from ..analityc.core.base_perimeter import BasePerimeter
from ..analityc.core.botsort_wrapper import BoTSORTWrapper
from ..analityc.core.car_washed import VehicleProcessor
from ..analityc.core.hardware_available import device_hardware
from ..analityc.core.Hummus import HummusProcess
try:
    from ..analityc.core.hummus_vlm import HummusVLMProcess
except ImportError:
    HummusVLMProcess = None  # type: ignore
    logging.getLogger(__name__).warning(
        "HummusVLMProcess no disponible (faltan transformers/bitsandbytes). "
        "Instalar con: pip install transformers>=4.36 bitsandbytes>=0.41 accelerate pillow"
    )
from ..analityc.core.Misters import MistersProcess
from ..analityc.core.perimetrales_multicam import PerimetralesMultiCam
from ..analityc.core.person_amazona_inference import PersonAmazonas
from ..analityc.config.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constantes de configuracion
# -----------------------------------------------------------------------------
WEBSOCKET_TIMEOUT: float = 30.0      # segundos sin mensaje antes de ping
MAX_QUEUE_PER_CLIENT: int = 2        # frames maximos en cola por cliente
JPEG_QUALITY: int = 70               # calidad JPEG para imagen de salida
EXECUTOR_WORKERS: int = 8


# -----------------------------------------------------------------------------
# Ciclo de vida: ThreadPoolExecutor gestionado correctamente
# -----------------------------------------------------------------------------
executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Servidor iniciando - ThreadPoolExecutor con %d workers", EXECUTOR_WORKERS)
    yield
    logger.info("Servidor apagandose - cerrando ThreadPoolExecutor")
    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Estado de conexiones
# -----------------------------------------------------------------------------
active_connections: Dict[str, Dict[str, Any]] = {}
connection_lock = asyncio.Lock()


# -----------------------------------------------------------------------------
# Helpers de configuracion / hardware
# -----------------------------------------------------------------------------

def _get_device_default() -> str:
    dev = getattr(device_hardware, "device_default", "cpu")
    if isinstance(dev, dict):
        return dev.get("gpu_use", "cpu")
    return dev if isinstance(dev, str) else "cpu"


def _get_first_gpu() -> str:
    gpus = getattr(device_hardware, "gpu_tuple", [])
    if isinstance(gpus, (list, tuple)) and gpus:
        gpu0 = gpus[0]
        if isinstance(gpu0, dict):
            return gpu0.get("gpu_use", _get_device_default())
    return _get_device_default()


def _get_model_path(config: Dict[str, Any], inference_name: str, default_path: str) -> str:
    model_paths = config.get("model_paths")
    if isinstance(model_paths, dict):
        path = model_paths.get(inference_name)
        if path:
            return path
    return config.get("model_path", default_path)


def _get_person_model_path(config: Dict[str, Any], inference_name: str, default_path: str) -> str:
    person_model_paths = config.get("person_model_paths")
    if isinstance(person_model_paths, dict):
        path = person_model_paths.get(inference_name)
        if path:
            return path
    return default_path


# -----------------------------------------------------------------------------
# Creacion de procesadores
# -----------------------------------------------------------------------------

def _build_processor(type_inference: str, client_id: str, config: Dict[str, Any]) -> Any:
    """Instancia y devuelve el procesador correcto segun el tipo de inferencia."""
    gpu = _get_first_gpu()

    if type_inference == "Misters":
        return MistersProcess(
            client_id=client_id,
            model_path=_get_model_path(config, "Misters", "models/base/misters.pt"),
            person_model_path=_get_person_model_path(config, "Misters", "models/base/yolo12l.pt"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            iou_threshold=config.get("iou_threshold", 0.5),
            device=gpu,
        )

    if type_inference == "Autolavado":
        return VehicleProcessor(
            client_id=client_id,
            model_path=_get_model_path(config, "Lavado", "models/base/best.pt"),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    if type_inference == "Hummus":
        return HummusProcess(
            client_id=client_id,
            model_path=_get_model_path(config, "Hummus", "models/base/1080.pt"),
            person_model_path=_get_person_model_path(config, "Hummus", "models/base/yolo12l.pt"),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    if type_inference == "HummusVLM":
        return HummusVLMProcess(
            client_id=client_id,
            person_model_path=_get_person_model_path(config, "HummusVLM", "models/base/yolo12l.pt"),
            confidence_threshold=config.get("confidence_threshold", 0.35),
            iou_threshold=config.get("iou_threshold", 0.45),
            device=gpu,
        )

    if type_inference == "Perimetrales":
        return BasePerimeter(
            client_id=client_id,
            model_path=_get_model_path(config, "Perimetrales", "yolo12l.pt"),
            device=gpu,
        )

    if type_inference == "PerimetralesMultiCam":
        return PerimetralesMultiCam(
            model_path=_get_model_path(config, "PerimetralesMultiCam", "yolo12l.pt"),
            device=gpu,
            debug=False,
        )

    if type_inference == "PerimetralesBoTSORT":
        return BoTSORTWrapper(
            model_path=_get_model_path(config, "PerimetralesBoTSORT", "yolo12l.pt"),
            device=gpu,
            match_thresh=0.45,
            debug=False,
        )

    if type_inference == "Personal de Amazonas":
        return PersonAmazonas(
            client_id=client_id,
            model_path=_get_model_path(config, "Personal de Amazonas", r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\models\base\amazonas.pt"),
            person_model_path=_get_person_model_path(config, "Personal de Amazonas", r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\models\base\yolo12l.pt"),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    raise ValueError(f"Tipo de inferencia desconocido: '{type_inference}'")


# -----------------------------------------------------------------------------
# Procesamiento sincronico de frames (ejecutado en ThreadPoolExecutor)
# -----------------------------------------------------------------------------

def _decode_image(image_data: Any) -> np.ndarray:
    """Decodifica la imagen desde base64-str, bytes o data-URI."""
    if isinstance(image_data, (bytes, bytearray)):
        image_bytes = bytes(image_data)
    elif isinstance(image_data, str):
        if image_data.startswith("data:") and "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        raise ValueError(f"Tipo de imagen no soportado: {type(image_data)}")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen corrupta o formato invalido")
    return img


def _encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Error en la codificacion JPEG de salida")
    return buf.tobytes()


def _apply_track_classes(processor: Any, track_classes: list) -> None:
    """Aplica dinámicamente las clases a detectar en el procesador."""
    if not track_classes:
        return
    classes = [int(c) for c in track_classes if isinstance(c, (int, float))]
    if not classes:
        return

    # BasePerimeter: usa class_ids
    if isinstance(processor, BasePerimeter):
        processor.class_ids = classes

    # PersonAmazonas: usa all_classes + staff_names
    elif isinstance(processor, PersonAmazonas):
        processor.all_classes = classes
        # Actualizar staff_names para que el filtro post-proceso acepte las clases
        if hasattr(processor, 'model') and processor.model and hasattr(processor.model, 'names'):
            for cid in classes:
                if cid not in processor.staff_names:
                    name = processor.model.names.get(cid, f"Clase_{cid}")
                    processor.staff_names[cid] = name.replace('_', ' ').title()

    # MultiObjectProcessor: usa vehicle_classes + person_classes
    elif isinstance(processor, MultiObjectProcessor):
        processor.vehicle_classes = [c for c in classes if c in {2, 3, 5, 7}]
        processor.person_classes = [c for c in classes if c == 0]
        processor.all_classes = classes

    # BoTSORTWrapper / PerimetralesMultiCam: detectan solo personas por defecto
    elif hasattr(processor, 'detect_classes'):
        processor.detect_classes = classes


def process_image_sync(
    processor: Any,
    img: np.ndarray,
    roi: Any,
    roi_activate: bool,
    door_roi: Any = None,
    door_activate: bool = False,
    door_direction: Any = None,
    door_direction_activate: bool = False,
    pay_roi: Any = None,
    withdraw_roi: Any = None,
    pay_roi_activate: bool = True,
    withdraw_roi_activate: bool = True,
    camera_id: Any = 1,
    track_classes: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Despacha el frame al procesador correcto.
    Se ejecuta en el ThreadPoolExecutor, nunca en el event loop.
    """
    # Aplicar clases dinámicas al procesador si las envía el cliente
    if track_classes is not None and isinstance(track_classes, list):
        _apply_track_classes(processor, track_classes)
    if isinstance(processor, PerimetralesMultiCam):
        tracks = processor.process_frame(
            img, camera_id,
            pay_roi=pay_roi, withdraw_roi=withdraw_roi,
            pay_roi_activate=pay_roi_activate, withdraw_roi_activate=withdraw_roi_activate,
        )
        for t in tracks:
            x1, y1, x2, y2 = t.get("bbox", (0, 0, 0, 0))
            gid = t.get("global_id", 0)
            conf = t.get("conf", 0.0)
            cam = t.get("camera_id", camera_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{gid} CAM:{cam} {conf:.2f}"
            y_text = max(10, y1 - 6)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y_text - th - bl)), (x1 + tw, y_text + bl), (0, 255, 255), -1)
            cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img, {"tracks": tracks}

    if isinstance(processor, PersonAmazonas):
        try:
            cam_proc = processor.get_camera_processor(camera_id)
            return cam_proc.process_frame(img, roi, roi_activate, camera_id)
        except Exception:
            return processor.process_frame(img, roi, roi_activate, camera_id)

    if isinstance(processor, MultiObjectProcessor):
        try:
            cam_proc = processor.get_camera_processor(camera_id)
            return cam_proc.process_frame(
                img, roi, roi_activate,
                door_roi, door_activate, door_direction, door_direction_activate,
            )
        except Exception:
            return processor.process_frame(
                img, roi, roi_activate,
                door_roi, door_activate, door_direction, door_direction_activate,
            )

    # HummusVLMProcess: misma interfaz que HummusProcess
    if isinstance(processor, HummusVLMProcess):
        effective_roi = roi if (roi_activate and isinstance(roi, list) and len(roi) >= 3) else None
        return processor.process_frame(img, effective_roi, send_to_server=False)

    # HummusProcess: solo pasar roi si está activo Y tiene ≥3 puntos válidos
    if isinstance(processor, HummusProcess):
        effective_roi = roi if (roi_activate and isinstance(roi, list) and len(roi) >= 3) else None
        return processor.process_frame(img, effective_roi, send_to_server=False)

    # MistersProcess: misma lógica de ROI
    if isinstance(processor, MistersProcess):
        effective_roi = roi if (roi_activate and isinstance(roi, list) and len(roi) >= 3) else None
        return processor.process_frame(img, effective_roi, send_to_server=False)

    # Procesadores legacy genericos
    return processor.process_frame(
        img, roi, roi_activate,
        door_roi, door_activate, door_direction, door_direction_activate,
    )


def _full_frame_sync(
    processor: Any,
    image_data: Any,
    roi: Any,
    roi_activate: bool,
    door_roi: Any,
    door_activate: bool,
    door_direction: Any,
    door_direction_activate: bool,
    pay_roi: Any,
    withdraw_roi: Any,
    pay_roi_activate: bool,
    withdraw_roi_activate: bool,
    camera_id: Any,
    track_classes: Any = None,
) -> Dict[str, Any]:
    """
    Combina decode + inferencia + encode JPEG en una sola llamada al executor.
    El event loop no se bloquea en ningun paso de CPU.
    """
    t0 = time.time()
    img = _decode_image(image_data)
    processed_img, metadata = process_image_sync(
        processor, img, roi, roi_activate,
        door_roi, door_activate, door_direction, door_direction_activate,
        pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id,
        track_classes=track_classes,
    )
    jpeg_bytes = _encode_jpeg(processed_img)
    processing_time = round(time.time() - t0, 3)

    return {
        "camera_id": camera_id,
        "status": "success",
        "metadata": metadata,
        "processed_image": f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode()}",
        "processing_time": processing_time,
    }


# -----------------------------------------------------------------------------
# Worker por cliente
# -----------------------------------------------------------------------------

class ClientWorker:
    """
    Cola y worker dedicados a UN cliente.

    Ventajas vs cola global original:
      - Un cliente lento NO bloquea a los demas (head-of-line blocking eliminado).
      - Limpieza independiente al desconectar sin afectar al resto.
      - Limite de cola y politica de drop configurables por instancia.
    """

    def __init__(self, client_id: str, processor: Any):
        self.client_id = client_id
        self.processor = processor
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_PER_CLIENT)
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"worker-{self.client_id}")

    async def stop(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        while not self._queue.empty():
            try:
                fut, _ = self._queue.get_nowait()
                if not fut.done():
                    fut.cancel()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def enqueue(self, payload: tuple) -> asyncio.Future:
        """
        Encola un frame.
        Si la cola esta llena aplica drop-head: descarta el frame MAS ANTIGUO
        y acepta el nuevo. Asi siempre se procesa el frame mas reciente.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        if self._queue.full():
            try:
                old_fut, _ = self._queue.get_nowait()
                self._queue.task_done()
                if not old_fut.done():
                    old_fut.set_result({"status": "dropped", "reason": "queue_full"})
                logger.debug("Drop-head frame para client=%s", self.client_id)
            except asyncio.QueueEmpty:
                pass

        await self._queue.put((future, payload))
        return future

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._stopped:
            try:
                future, payload = await self._queue.get()
            except asyncio.CancelledError:
                break

            if future.done():
                self._queue.task_done()
                continue

            try:
                result = await loop.run_in_executor(executor, _full_frame_sync, *payload)
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                logger.error("Error en worker client=%s: %s", self.client_id, exc)
                if not future.done():
                    future.set_result({"status": "error", "message": str(exc)})
            finally:
                self._queue.task_done()


# -----------------------------------------------------------------------------
# Endpoint WebSocket
# -----------------------------------------------------------------------------

@app.websocket("/ws/{type_inference}")
async def websocket_endpoint(websocket: WebSocket, type_inference: str):
    await websocket.accept()
    client_id = f"client_{id(websocket)}"
    worker: Optional[ClientWorker] = None
    processor: Any = None

    try:
        # -- 1. Inicializacion del procesador ----------------------------------
        config = get_config()
        try:
            processor = _build_processor(type_inference, client_id, config)
        except ValueError as exc:
            await websocket.send_text(json.dumps({"status": "error", "message": str(exc)}))
            await websocket.close(code=1008)
            return

        worker = ClientWorker(client_id, processor)
        worker.start()

        async with connection_lock:
            active_connections[client_id] = {
                "websocket": websocket,
                "processor": processor,
                "worker": worker,
                "type_inference": type_inference,
                "connected_at": time.time(),
                "last_active": time.time(),
            }

        logger.info("Cliente conectado: %s tipo=%s", client_id, type_inference)

        await websocket.send_text(json.dumps({
            "id_connection": id(websocket),
            "event": "connection_init",
            "type_inference": type_inference,
            "data": {"roi": False},
        }))

        # -- 2. Bucle principal ------------------------------------------------
        while True:
            try:
                recv = await asyncio.wait_for(websocket.receive(), timeout=WEBSOCKET_TIMEOUT)
            except asyncio.TimeoutError:
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps({"status": "ping"}))
                    except Exception:
                        break
                continue

            # Deserializar (msgpack o JSON)
            incoming_is_binary = recv.get("bytes") is not None
            try:
                if incoming_is_binary:
                    request = msgpack.unpackb(recv["bytes"], raw=False, strict_map_key=False)
                else:
                    request = json.loads(recv.get("text") or "")
            except Exception as exc:
                logger.warning("Error deserializando mensaje client=%s: %s", client_id, exc)
                await _send_error(websocket, incoming_is_binary, str(exc))
                continue

            data = request.get("data", {})
            if not isinstance(data, dict):
                await _send_error(websocket, incoming_is_binary, "Campo 'data' invalido o ausente")
                continue

            if "image" not in data:
                await _send_error(websocket, incoming_is_binary, "Campo 'image' requerido")
                continue

            # Extraer y normalizar campos
            image_data = data["image"]
            roi = data.get("roi_coordinates", "")
            roi_activate = bool(data.get("roi_activate", False))
            door_roi = data.get("door_roi_coordinates")
            door_activate = bool(data.get("door_roi_activate", False))
            door_direction = data.get("door_direction")
            door_direction_activate = data.get("door_direction_activate", False)

            # Compatibilidad: door_direction_activate como lista de puntos
            if isinstance(door_direction_activate, (list, tuple)) and door_direction is None:
                door_direction = door_direction_activate
                door_direction_activate = True

            pay_roi = data.get("pay_roi")
            withdraw_roi = data.get("withdraw_roi")
            pay_roi_activate = bool(data.get("pay_roi_activate", True))
            withdraw_roi_activate = bool(data.get("withdraw_roi_activate", True))

            camera_id_raw = data.get("camera_id", 1)
            try:
                camera_id = int(camera_id_raw)
            except (TypeError, ValueError):
                camera_id = camera_id_raw

            # Clases a trackear (lista de class_ids COCO enviadas por el cliente)
            track_classes = data.get("track_classes", None)

            # Payload para _full_frame_sync (orden fijo de argumentos)
            frame_payload = (
                processor, image_data,
                roi, roi_activate,
                door_roi, door_activate,
                door_direction, door_direction_activate,
                pay_roi, withdraw_roi,
                pay_roi_activate, withdraw_roi_activate,
                camera_id,
                track_classes,
            )

            try:
                future = await worker.enqueue(frame_payload)
                result = await future
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error encolando frame client=%s: %s", client_id, exc)
                result = {"status": "error", "message": str(exc)}

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            request["data"] = result
            try:
                if incoming_is_binary:
                    await websocket.send_bytes(msgpack.packb(request, use_bin_type=True))
                else:
                    await websocket.send_json(request)
            except Exception as exc:
                logger.error("Error enviando respuesta client=%s: %s", client_id, exc)
                break

            async with connection_lock:
                if client_id in active_connections:
                    active_connections[client_id]["last_active"] = time.time()

    except WebSocketDisconnect:
        logger.info("Cliente desconectado: %s", client_id)
    except Exception as exc:
        logger.error("Error critico client=%s: %s", client_id, exc)
    finally:
        # -- 3. Limpieza completa ----------------------------------------------
        if worker:
            await worker.stop()
        async with connection_lock:
            active_connections.pop(client_id, None)
        if processor and hasattr(processor, "cleanup"):
            try:
                processor.cleanup()
            except Exception:
                pass
        logger.info("Limpieza completa para client=%s", client_id)


async def _send_error(websocket: WebSocket, is_binary: bool, message: str) -> None:
    """Envia un error estructurado al cliente."""
    payload = {"data": {"status": "error", "message": message}}
    try:
        if is_binary:
            await websocket.send_bytes(msgpack.packb(payload, use_bin_type=True))
        else:
            await websocket.send_json(payload)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Endpoints HTTP
# -----------------------------------------------------------------------------

@app.get("/")
def init_server():
    return {"status": "active", "connections": len(active_connections)}


@app.get("/health")
def health_check():
    now = time.time()
    clients = [
        {
            "client_id": cid,
            "type_inference": info.get("type_inference", "unknown"),
            "idle_seconds": round(now - info.get("last_active", now), 1),
            "connected_seconds": round(now - info.get("connected_at", now), 1),
        }
        for cid, info in active_connections.items()
    ]
    return {
        "status": "active",
        "total_connections": len(active_connections),
        "executor_max_workers": EXECUTOR_WORKERS,
        "clients": clients,
    }