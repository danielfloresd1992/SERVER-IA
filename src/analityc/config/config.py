"""
config.py - v2

Cambios respecto a v1
---------------------
1. VARIABLES DE ENTORNO: todas las rutas y parametros clave se pueden
   sobreescribir via env vars (compatible con Docker / CI / multi-servidor).
   Los defaults siguen apuntando a las rutas originales de produccion.

2. COPIA PROFUNDA en get_config(): evita corrupcion del dict global cuando
   algun modulo modifica el resultado de get_config() en runtime.

3. VALIDACION AL ARRANCAR: _validate_config() comprueba que los modelos
   existen en disco y lanza errores claros antes de aceptar conexiones WS.

4. DEVICE REAL: la clave "device" se expone para que app.py pueda usarla.

5. RECARGA EN CALIENTE opcional: get_config() puede leer ENV vars en cada
   llamada (util para cambiar umbrales sin reiniciar, via CONFIG_RELOAD=true).
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas base de produccion
# ---------------------------------------------------------------------------
_MODELS_BASE = r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\models\base"
_OUTPUT_BASE  = r"C:\Users\Sistema-1\Desktop\ELDE\SERVER-IA PERIMETRALES\output"


# ---------------------------------------------------------------------------
# Helpers para leer variables de entorno con tipos
# ---------------------------------------------------------------------------

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key, "").strip()
    if val:
        try:
            return float(val)
        except ValueError:
            logger.warning("ENV %s='%s' no es un float valido; usando default=%.3f", key, val, default)
    return default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    if val:
        try:
            return int(val)
        except ValueError:
            logger.warning("ENV %s='%s' no es un int valido; usando default=%d", key, val, default)
    return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------------------
# Construccion del diccionario de configuracion
# ---------------------------------------------------------------------------

def _build_config() -> Dict[str, Any]:
    """
    Construye el dict de config combinando defaults con ENV vars.
    Llamada una vez al arrancar (o en cada get_config() si CONFIG_RELOAD=true).
    """
    models_dir = _env_str("MODELS_DIR", _MODELS_BASE)
    output_dir = _env_str("OUTPUT_DIR", _OUTPUT_BASE)

    def model(filename: str) -> str:
        """Ruta absoluta a un modelo dentro de models_dir."""
        return str(Path(models_dir) / filename)

    def output(filename: str) -> str:
        """Ruta absoluta a un archivo dentro de output_dir."""
        return str(Path(output_dir) / filename)

    return {
        # -- Rutas de modelos ------------------------------------------------
        "model_path": _env_str("MODEL_PATH_DEFAULT", model("yolo12l.pt")),
        "model_paths": {
            "Lavado":               _env_str("MODEL_PATH_LAVADO",                model("yolo12l.pt")),
            "Hummus":               _env_str("MODEL_PATH_HUMMUS",                model("1080.pt")),
            "Perimetrales":         _env_str("MODEL_PATH_PERIMETRALES",          model("yolo12l.pt")),
            "PerimetralesMultiCam": _env_str("MODEL_PATH_PERIMETRALES_MULTICAM", model("yolo12l.pt")),
            "PerimetralesBoTSORT":  _env_str("MODEL_PATH_PERIMETRALES_BOTSORT",  model("yolo12l.pt")),
            "Personal de Amazonas": _env_str("MODEL_PATH_AMAZONAS",              model("amazonas.pt")),
        },
        "person_model_paths": {
            "Hummus": _env_str("PERSON_MODEL_PATH_HUMMUS", model("yolo12l.pt")),
        },

        # -- Directorios de salida -------------------------------------------
        "output_dir":                     output_dir,
        "hummus_order_screenshot_dir":    _env_str("HUMMUS_ORDER_DIR",    output("Toma_de_orden_hummus")),
        "hummus_delivery_screenshot_dir": _env_str("HUMMUS_DELIVERY_DIR", output("Entrega_de_plato_hummus")),
        "hummus_alert_csv":               _env_str("HUMMUS_ALERT_CSV",    output("hummus_alertas.csv")),

        # -- Parametros de inferencia ----------------------------------------
        "confidence_threshold": _env_float("CONFIDENCE_THRESHOLD", 0.70),
        "iou_threshold":        _env_float("IOU_THRESHOLD",        0.45),

        # -- Hardware --------------------------------------------------------
        # "auto"   -> app.py usa _get_first_gpu() / _get_device_default()
        # "cpu"    -> fuerza CPU aunque haya GPU
        # "cuda:0" -> fuerza esa GPU especifica
        "device": _env_str("INFERENCE_DEVICE", "auto"),

        # -- Red -------------------------------------------------------------
        "host":           _env_str("SERVER_HOST", "0.0.0.0"),
        "websocket_port": _env_int("SERVER_PORT", 9000),

        # -- Comportamiento --------------------------------------------------
        # CONFIG_RELOAD=true -> get_config() reconstruye en cada llamada
        # (util para ajustar umbrales sin reiniciar el servidor)
        "config_reload": _env_bool("CONFIG_RELOAD", False),
    }


# ---------------------------------------------------------------------------
# Validacion de modelos al arrancar
# ---------------------------------------------------------------------------

def _validate_config(cfg: Dict[str, Any], *, strict: bool = False) -> List[str]:
    """
    Verifica que los archivos de modelo declarados en el config existan en disco.

    Parametros
    ----------
    strict : bool
        False (default) -> solo warnings, el servidor arranca igual.
        True            -> lanza FileNotFoundError si falta alguno.

    Retorna
    -------
    Lista de paths que no se encontraron.
    """
    paths_to_check: List[str] = []

    default_path = cfg.get("model_path", "")
    if default_path:
        paths_to_check.append(default_path)

    for path in cfg.get("model_paths", {}).values():
        if path:
            paths_to_check.append(path)

    for path in cfg.get("person_model_paths", {}).values():
        if path:
            paths_to_check.append(path)

    missing: List[str] = []
    for path in paths_to_check:
        if not Path(path).exists():
            logger.warning("Modelo no encontrado en disco: %s", path)
            missing.append(path)

    if strict and missing:
        raise FileNotFoundError(
            "Los siguientes modelos no existen (strict=True):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    return missing


# ---------------------------------------------------------------------------
# Cache del config
# ---------------------------------------------------------------------------

_cached_config: Optional[Dict[str, Any]] = None


def get_config(*, validate: bool = True, strict: bool = False) -> Dict[str, Any]:
    """
    Retorna una copia profunda de la configuracion activa.

    Parametros
    ----------
    validate : bool
        Verifica existencia de modelos al primer uso (y cuando CONFIG_RELOAD=true).
    strict : bool
        Si True, lanza FileNotFoundError cuando faltan modelos.
    """
    global _cached_config

    # Reconstruir si no hay cache o si CONFIG_RELOAD esta activo
    if _cached_config is None or _env_bool("CONFIG_RELOAD", False):
        _cached_config = _build_config()
        if validate:
            missing = _validate_config(_cached_config, strict=strict)
            if missing:
                logger.warning(
                    "%d modelo(s) no encontrado(s) en disco. "
                    "El servidor iniciara pero fallara al intentar usarlos.",
                    len(missing),
                )

    # Siempre retornar copia profunda para evitar corrupcion del cache global
    return copy.deepcopy(_cached_config)


def reload_config() -> Dict[str, Any]:
    """Fuerza reconstruccion del config desde ENV vars. Util en tests o hot-reload."""
    global _cached_config
    _cached_config = None
    return get_config()


# ---------------------------------------------------------------------------
# DEFAULT_ROI
#
# Mantenido aqui por retrocompatibilidad con los modulos que lo importan.
# Corresponde al dominio visual (Hummus), no es config de servidor.
# ---------------------------------------------------------------------------

DEFAULT_ROI = [
    [500, 250],   # Esquina superior izquierda
    [900, 250],   # Esquina superior derecha
    [1040, 560],  # Esquina inferior derecha
    [600, 560],   # Esquina inferior izquierda
]