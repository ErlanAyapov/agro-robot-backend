from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from core import DeltaRobot
from models import CalibrateModel
from robot_api import RobotApiRuntime, build_legacy_limitswitch_router, build_legacy_ui_router, build_v1_router
from robot_api.services import DeviceCommandError, ServiceRegistry
import utils_limitswitch_rotate as motion
from utils_limitswitch_rotate import close_limit_switch_client, reset_runtime_state
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import logging
import asyncio
import time
import json
import sys
import os
import importlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from threading import Event, Lock, Thread

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - optional dependency fallback
    rs = None
try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover - optional dependency fallback
    list_ports = None

from visualize_stream import (
    DEFAULT_FOV_HEIGHT_MM,
    DEFAULT_FOV_WIDTH_MM,
    PREDICTION_DELAY_MS,
    REAL_ZONE_SIZE_MM,
    _detect_objects,
    _draw_detection_overlay,
    _get_yolo_model,
    _resolve_model_device,
    build_default_video_path,
    gen_visualization_mjpeg,
)

DEBUG = True

PORT = "COM19"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _v1_error_response(*, status_code: int, code: str, message: str, details: Optional[Dict[str, Any]] = None):
    return JSONResponse(
        status_code=int(status_code),
        content={
            "error": {
                "code": str(code or "device_fault"),
                "message": str(message or "Request failed"),
                "details": dict(details or {}),
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_error_handler(request: Request, exc: RequestValidationError):
    if str(request.url.path).startswith("/api/v1"):
        return _v1_error_response(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            details={"errors": list(exc.errors() or [])},
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    if str(request.url.path).startswith("/api/v1"):
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code") or "command_rejected")
            message = str(detail.get("message") or detail.get("detail") or "Request failed")
            details = {k: v for k, v in detail.items() if k not in {"code", "message"}}
        else:
            code = "command_rejected"
            message = str(detail or "Request failed")
            details = {}
        return _v1_error_response(
            status_code=int(exc.status_code),
            code=code,
            message=message,
            details=details,
        )
    return JSONResponse(status_code=int(exc.status_code), content={"detail": exc.detail})
# Run
# fastapi dev main.py
# python3.8 -m fastapi_cli dev main.py
# python3.8 -m uvicorn main:app --host 0.0.0.0 --port 8000


REALSENSE_CAMERA = True
WEBCAM_INDEX = 0
FRAME_WAIT_TIMEOUT_MS = 5000
FRAME_TIMEOUT_RETRIES = 30
FRAME_TIMEOUT_SLEEP_SEC = 0.1
RESTART_COOLDOWN_SEC = 1.0
FRAME_PROBE_TIMEOUT_MS = 1000
FRAME_PROBE_ATTEMPTS = 5
DEPTH_VISUALIZATION_ALPHA = 0.03
ROBOT_MOVE_DURATION_MS = 12000
ROBOT_Z_UP_MM = -180
ROBOT_Z_DOWN_MM = ROBOT_Z_UP_MM - 300.0
ROBOT_MANUAL_Z_MIN_MM = -520
ROBOT_MANUAL_Z_MAX_MM = -180
ROBOT_COMMAND_COOLDOWN_SEC = 1.5
ROBOT_MIN_DURATION_MS = 500
ROBOT_MAX_DURATION_MS = 60000
ROBOT_HOME_X_MM = 0
ROBOT_HOME_Y_MM = 0
ROBOT_READY_REGISTER_ADDR = 6
ROBOT_READY_IDLE_VALUE = 1
ROBOT_READY_POLL_INTERVAL_SEC = 0.05
ROBOT_BUSY_WAIT_TIMEOUT_SEC = 2.0
ROBOT_READY_WAIT_TIMEOUT_SEC = 30.0
ROBOT_CALIBRATION_WAIT_TIMEOUT_SEC = 180.0
ZONE_CONFIG_PATH = Path(__file__).resolve().parent / "zone_config.json"
ROBOT_SETTINGS_PATH = Path(__file__).resolve().parent / "robot_settings.json"
ZONE_MIN_OFFSET_PX = -4000
ZONE_MAX_OFFSET_PX = 4000
ZONE_MIN_SIZE_PX = 80
ZONE_MAX_SIZE_PX = 2000
ZONE_MIN_SIZE_MM = 10.0
ZONE_MAX_SIZE_MM = 5000.0
ROBOT_XY_ROTATION_MIN_DEG = -360.0
ROBOT_XY_ROTATION_MAX_DEG = 360.0
DETECTION_CONF_MIN = 0.01
DETECTION_CONF_MAX = 1.0
FLOW_SPEED_EMA_ALPHA = 0.35
FLOW_TRACK_MIN_POINTS = 3
FLOW_SPEED_UPDATE_HZ = 3.0
FLOW_SPEED_UPDATE_INTERVAL_SEC = 1.0 / FLOW_SPEED_UPDATE_HZ
FLOW_POINT_MAX_CORNERS = 40
FLOW_POINT_QUALITY_LEVEL = 0.01
FLOW_POINT_MIN_DISTANCE = 14
FLOW_CENTER_ZONE_RATIO = 0.55
TRACK_MATCH_MAX_DISTANCE_PX = 90.0
TRACK_STALE_FRAME_TTL = 12


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


DEFAULT_ROBOT_XY_ROTATION_DEG = _env_float("DELTA_MANUAL_XY_ROTATION_DEG", 95.0)
DEFAULT_ROBOT_INVERT_X = _env_bool("DELTA_MANUAL_INVERT_X", False)
DEFAULT_ROBOT_INVERT_Y = _env_bool("DELTA_MANUAL_INVERT_Y", False)
DEFAULT_ROBOT_INVERT_Z = _env_bool("DELTA_MANUAL_INVERT_Z", False)
DEFAULT_DETECTION_CONF = _env_float("DELTA_DETECTION_CONF", 0.25)
WEBCAM_INDEX = _env_int("DELTA_WEBCAM_INDEX", WEBCAM_INDEX)
DEFAULT_ZONE_SIZE_MM = float(REAL_ZONE_SIZE_MM)
DEFAULT_ZONE_SIZE_PX = int(
    round(
        DEFAULT_ZONE_SIZE_MM
        / (
            ((DEFAULT_FOV_WIDTH_MM / 640.0) + (DEFAULT_FOV_HEIGHT_MM / 480.0))
            / 2.0
        )
    )
)

logging.basicConfig()
log = logging.getLogger(__name__)
pipeline_lock = Lock()
visual_state_lock = Lock()
visual_state = {"seq": 0, "event": "init", "count": 0, "detections": []}
event_log_lock = Lock()
event_log: List[Dict[str, Any]] = []
event_log_seq = 0
MAX_EVENT_LOG_SIZE = 800
robot_command_lock = Lock()
last_robot_command_ts = 0.0
robot_async_state_lock = Lock()
robot_async_inflight = False
robot_async_seq = 0
move_to_coordinate_fn = None
move_to_coordinate_error = None
move_module = None
move_module_port = None
zone_config_lock = Lock()
robot_settings_lock = Lock()
active_port_lock = Lock()
active_port = PORT
stream_stop_event = Event()
manual_robot_state_lock = Lock()
manual_robot_state = {
    "has_last": False,
    "x_mm": float(ROBOT_HOME_X_MM),
    "y_mm": float(ROBOT_HOME_Y_MM),
    "z_mm": float(ROBOT_Z_UP_MM),
}


def _port_sort_key(port_name: str):
    name = (port_name or "").strip().lower()
    if name.startswith("com"):
        suffix = name[3:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)


def _list_serial_ports() -> List[Dict[str, str]]:
    if list_ports is None:
        return []

    discovered: List[Dict[str, str]] = []
    seen = set()
    for item in list_ports.comports():
        port_name = str(getattr(item, "device", "") or "").strip()
        if not port_name or port_name in seen:
            continue
        seen.add(port_name)
        discovered.append(
            {
                "port": port_name,
                "description": str(getattr(item, "description", "") or "").strip(),
            }
        )
    discovered.sort(key=lambda row: _port_sort_key(row["port"]))
    return discovered


def _get_active_port() -> str:
    with active_port_lock:
        return active_port


def _set_active_port(port_name: str) -> str:
    global active_port
    normalized = str(port_name or "").strip()
    if not normalized:
        raise ValueError("Port name is required")
    with active_port_lock:
        active_port = normalized
        return active_port


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _publish_event(event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global event_log_seq, event_log
    payload = dict(payload or {})
    with event_log_lock:
        event_log_seq = int(event_log_seq) + 1
        item = {
            "cursor": int(event_log_seq),
            "event": str(event or "device.state"),
            "timestamp": str(payload.pop("timestamp", _utc_timestamp())),
            **payload,
        }
        event_log.append(item)
        if len(event_log) > int(MAX_EVENT_LOG_SIZE):
            event_log = event_log[-int(MAX_EVENT_LOG_SIZE):]
    return dict(item)


def _pull_events_since(cursor: int = 0, limit: int = 100) -> Dict[str, Any]:
    try:
        cursor_value = int(cursor)
    except Exception:
        cursor_value = 0
    try:
        limit_value = max(1, min(500, int(limit)))
    except Exception:
        limit_value = 100

    with event_log_lock:
        batch = [dict(item) for item in event_log if int(item.get("cursor", 0)) > cursor_value][:limit_value]
        latest_cursor = int(event_log_seq)

    next_cursor = cursor_value if not batch else int(batch[-1].get("cursor", cursor_value))
    return {
        "events": batch,
        "next_cursor": next_cursor,
        "latest_cursor": latest_cursor,
    }


def _update_visual_state(payload: dict):
    global visual_state
    with visual_state_lock:
        next_seq = int(visual_state.get("seq", 0)) + 1
        visual_state = {"seq": next_seq, **payload}
        snapshot = dict(visual_state)

    detection_count = int(snapshot.get("count", 0))
    event_name = str(snapshot.get("event") or "")
    if detection_count > 0 or event_name in {
        "manual_move",
        "robot_command_result",
        "robot_command_accepted",
        "zone_move",
        "zone_size",
        "zone_mm",
        "zone_reset",
        "camera_stream_error",
    }:
        _publish_event(
            "vision.detection" if detection_count > 0 else "device.state",
            {
                "device": "vision",
                "seq": int(snapshot.get("seq", 0)),
                "count": detection_count,
                "state": snapshot,
            },
        )

    if detection_count > 0:
        def _format_detection(detection: Dict[str, Any]) -> str:
            item = f"class={detection['class']} conf={float(detection['confidence']):.2f}"
            if detection.get("track_id") is not None:
                try:
                    item = f"id={int(detection['track_id'])} {item}"
                except Exception:
                    pass
            if detection.get("x_mm") is not None and detection.get("y_mm") is not None:
                try:
                    return f"{item} mm=({int(detection['x_mm'])},{int(detection['y_mm'])})"
                except Exception:
                    return item
            if detection.get("x") is None or detection.get("y") is None:
                return item
            try:
                return f"{item} pos=({int(detection['x'])},{int(detection['y'])})"
            except Exception:
                return item

        details = " | ".join(
            _format_detection(d)
            for d in payload.get("detections", [])[:5]
        )
        log.info(
            "Detection frame=%s count=%s device=%s %s",
            payload.get("frame_id"),
            payload.get("count"),
            payload.get("device"),
            details,
        )


def _get_vision_state_snapshot() -> Dict[str, Any]:
    with visual_state_lock:
        snapshot = dict(visual_state)
    return {
        "streaming": not bool(stream_stop_event.is_set()),
        "detecting": True,
        "active_mode": str(snapshot.get("source") or "manual"),
        "last_event": str(snapshot.get("event") or "init"),
        "last_seq": int(snapshot.get("seq", 0)),
        "detection_count": int(snapshot.get("count", 0)),
    }


def _get_visual_state_snapshot() -> Dict[str, Any]:
    with visual_state_lock:
        return dict(visual_state)


def _vision_start() -> Dict[str, Any]:
    stream_stop_event.clear()
    _publish_event(
        "device.state",
        {
            "device": "vision",
            "action": "start",
            "streaming": True,
        },
    )
    return {
        "streaming": True,
        "detecting": True,
    }


def _vision_stop() -> Dict[str, Any]:
    stream_stop_event.set()
    _publish_event(
        "device.state",
        {
            "device": "vision",
            "action": "stop",
            "streaming": False,
        },
    )
    return {
        "streaming": False,
        "detecting": False,
    }


def _probe_modbus_device(selected_port: str, *, name: str, slave_id: int) -> Dict[str, Any]:
    firmware_controller = motion.firmware_controller_name(name)
    if not selected_port:
        return {
            "name": name,
            "firmware_controller": firmware_controller,
            "slave_id": int(slave_id),
            "status": "transport_unavailable",
            "protocol_version": None,
            "firmware_version": None,
        }

    try:
        probe = motion.run_with_connection(
            selected_port,
            lambda: motion.probe_modbus_device(name=name, slave_id=int(slave_id)),
        )
        if probe.get("status") == "error":
            return {
                "name": name,
                "firmware_controller": firmware_controller,
                "slave_id": int(slave_id),
                "status": "device_unreachable",
                "protocol_version": None,
                "firmware_version": None,
                "error": str(probe.get("message") or "Probe failed"),
            }
        return dict(probe)
    except Exception as exc:
        return {
            "name": name,
            "firmware_controller": firmware_controller,
            "slave_id": int(slave_id),
            "status": "device_unreachable",
            "protocol_version": None,
            "firmware_version": None,
            "error": str(exc),
        }


def _get_system_devices() -> List[Dict[str, Any]]:
    selected_port = str(_get_active_port() or "").strip()
    layout = [
        ("delta", 1),
        ("platform_turn", 2),
        ("platform_width", 3),
    ]
    return [
        _probe_modbus_device(selected_port, name=name, slave_id=slave_id)
        for name, slave_id in layout
    ]


def _sanitize_zone_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}

    def _as_int(name: str, default: int) -> int:
        try:
            return int(raw.get(name, default))
        except Exception:
            return int(default)

    def _as_float(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except Exception:
            return float(default)

    return {
        "offset_x_px": int(np.clip(_as_int("offset_x_px", 0), ZONE_MIN_OFFSET_PX, ZONE_MAX_OFFSET_PX)),
        "offset_y_px": int(np.clip(_as_int("offset_y_px", 0), ZONE_MIN_OFFSET_PX, ZONE_MAX_OFFSET_PX)),
        "size_px": int(np.clip(_as_int("size_px", DEFAULT_ZONE_SIZE_PX), ZONE_MIN_SIZE_PX, ZONE_MAX_SIZE_PX)),
        "size_mm": float(np.clip(_as_float("size_mm", DEFAULT_ZONE_SIZE_MM), ZONE_MIN_SIZE_MM, ZONE_MAX_SIZE_MM)),
    }


def _read_zone_config_from_disk() -> Optional[Dict[str, Any]]:
    if not ZONE_CONFIG_PATH.exists():
        return None

    try:
        return json.loads(ZONE_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read zone config '%s': %s", ZONE_CONFIG_PATH, exc)
        return None


def _write_zone_config_to_disk(cfg: Dict[str, Any]) -> None:
    try:
        ZONE_CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("Failed to write zone config '%s': %s", ZONE_CONFIG_PATH, exc)


zone_config = _sanitize_zone_config(_read_zone_config_from_disk())
_write_zone_config_to_disk(zone_config)


def _get_zone_config() -> Dict[str, Any]:
    with zone_config_lock:
        return dict(zone_config)


def _set_zone_config(
    *,
    offset_x_px: Optional[int] = None,
    offset_y_px: Optional[int] = None,
    size_px: Optional[int] = None,
    size_mm: Optional[float] = None,
) -> Dict[str, Any]:
    with zone_config_lock:
        next_cfg = dict(zone_config)
        if offset_x_px is not None:
            next_cfg["offset_x_px"] = int(offset_x_px)
        if offset_y_px is not None:
            next_cfg["offset_y_px"] = int(offset_y_px)
        if size_px is not None:
            next_cfg["size_px"] = int(size_px)
        if size_mm is not None:
            next_cfg["size_mm"] = float(size_mm)

        sanitized = _sanitize_zone_config(next_cfg)
        zone_config.update(sanitized)
        saved = dict(zone_config)

    _write_zone_config_to_disk(saved)
    return saved


def _move_zone_offset(dx: int, dy: int) -> Dict[str, Any]:
    cfg = _get_zone_config()
    return _set_zone_config(
        offset_x_px=int(cfg["offset_x_px"]) + int(dx),
        offset_y_px=int(cfg["offset_y_px"]) + int(dy),
    )


def _sanitize_robot_settings(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}

    def _as_float(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except Exception:
            return float(default)

    def _as_bool(name: str, default: bool) -> bool:
        value = raw.get(name, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    return {
        "xy_rotation_deg": float(
            np.clip(
                _as_float("xy_rotation_deg", DEFAULT_ROBOT_XY_ROTATION_DEG),
                ROBOT_XY_ROTATION_MIN_DEG,
                ROBOT_XY_ROTATION_MAX_DEG,
            )
        ),
        "invert_x": bool(_as_bool("invert_x", DEFAULT_ROBOT_INVERT_X)),
        "invert_y": bool(_as_bool("invert_y", DEFAULT_ROBOT_INVERT_Y)),
        "invert_z": bool(_as_bool("invert_z", DEFAULT_ROBOT_INVERT_Z)),
        "detection_conf": float(
            np.clip(
                _as_float("detection_conf", DEFAULT_DETECTION_CONF),
                DETECTION_CONF_MIN,
                DETECTION_CONF_MAX,
            )
        ),
    }


def _read_robot_settings_from_disk() -> Optional[Dict[str, Any]]:
    if not ROBOT_SETTINGS_PATH.exists():
        return None
    try:
        return json.loads(ROBOT_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read robot settings '%s': %s", ROBOT_SETTINGS_PATH, exc)
        return None


def _write_robot_settings_to_disk(cfg: Dict[str, Any]) -> None:
    try:
        ROBOT_SETTINGS_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("Failed to write robot settings '%s': %s", ROBOT_SETTINGS_PATH, exc)


def _apply_robot_settings_to_env(cfg: Dict[str, Any]) -> None:
    os.environ["DELTA_MANUAL_XY_ROTATION_DEG"] = f"{float(cfg['xy_rotation_deg']):.8f}"
    os.environ["DELTA_MANUAL_INVERT_X"] = "1" if bool(cfg["invert_x"]) else "0"
    os.environ["DELTA_MANUAL_INVERT_Y"] = "1" if bool(cfg["invert_y"]) else "0"
    os.environ["DELTA_MANUAL_INVERT_Z"] = "1" if bool(cfg["invert_z"]) else "0"
    os.environ["DELTA_DETECTION_CONF"] = f"{float(cfg['detection_conf']):.8f}"


robot_settings = _sanitize_robot_settings(_read_robot_settings_from_disk())
_write_robot_settings_to_disk(robot_settings)
_apply_robot_settings_to_env(robot_settings)


def _get_robot_settings() -> Dict[str, Any]:
    with robot_settings_lock:
        return dict(robot_settings)


def _set_robot_settings(
    *,
    xy_rotation_deg: Optional[float] = None,
    invert_x: Optional[bool] = None,
    invert_y: Optional[bool] = None,
    invert_z: Optional[bool] = None,
    detection_conf: Optional[float] = None,
) -> Dict[str, Any]:
    with robot_settings_lock:
        next_cfg = dict(robot_settings)
        if xy_rotation_deg is not None:
            next_cfg["xy_rotation_deg"] = xy_rotation_deg
        if invert_x is not None:
            next_cfg["invert_x"] = invert_x
        if invert_y is not None:
            next_cfg["invert_y"] = invert_y
        if invert_z is not None:
            next_cfg["invert_z"] = invert_z
        if detection_conf is not None:
            next_cfg["detection_conf"] = detection_conf

        sanitized = _sanitize_robot_settings(next_cfg)
        robot_settings.update(sanitized)
        saved = dict(robot_settings)

    _write_robot_settings_to_disk(saved)
    _apply_robot_settings_to_env(saved)
    return saved


def _resolve_detection_conf(conf: Optional[float]) -> float:
    if conf is None:
        conf = float(_get_robot_settings()["detection_conf"])
    return float(np.clip(float(conf), DETECTION_CONF_MIN, DETECTION_CONF_MAX))


def _resolve_command_duration_ms(duration_ms: Optional[int]) -> int:
    if duration_ms is None:
        duration_ms = ROBOT_MOVE_DURATION_MS
    try:
        value = int(duration_ms)
    except Exception:
        value = int(ROBOT_MOVE_DURATION_MS)
    return int(np.clip(value, ROBOT_MIN_DURATION_MS, ROBOT_MAX_DURATION_MS))


def _predict_position_px(
    center_x: int,
    center_y: int,
    speed_x_mm_per_sec: float,
    speed_y_mm_per_sec: float,
    horizon_ms: int,
    mm_per_pixel: float,
) -> tuple:
    if mm_per_pixel <= 1e-9:
        return int(center_x), int(center_y)

    horizon_sec = max(0.0, float(horizon_ms) / 1000.0)
    delta_px_x = (float(speed_x_mm_per_sec) * horizon_sec) / max(mm_per_pixel, 1e-9)
    delta_px_y = (float(speed_y_mm_per_sec) * horizon_sec) / max(mm_per_pixel, 1e-9)
    return (
        int(round(float(center_x) + float(delta_px_x))),
        int(round(float(center_y) + float(delta_px_y))),
    )


def _build_center_flow_rect(
    frame_w: int,
    frame_h: int,
    zone_x1: int,
    zone_y1: int,
    zone_x2: int,
    zone_y2: int,
    ratio: float = FLOW_CENTER_ZONE_RATIO,
) -> tuple:
    zone_x1 = int(np.clip(zone_x1, 0, max(0, frame_w - 1)))
    zone_y1 = int(np.clip(zone_y1, 0, max(0, frame_h - 1)))
    zone_x2 = int(np.clip(zone_x2, zone_x1 + 1, frame_w))
    zone_y2 = int(np.clip(zone_y2, zone_y1 + 1, frame_h))

    zone_w = max(2, zone_x2 - zone_x1)
    zone_h = max(2, zone_y2 - zone_y1)
    inner_ratio = float(np.clip(float(ratio), 0.2, 1.0))
    inner_w = max(16, int(round(zone_w * inner_ratio)))
    inner_h = max(16, int(round(zone_h * inner_ratio)))

    center_x = int(round((zone_x1 + zone_x2) / 2.0))
    center_y = int(round((zone_y1 + zone_y2) / 2.0))
    x1 = int(np.clip(center_x - inner_w // 2, 0, max(0, frame_w - 1)))
    y1 = int(np.clip(center_y - inner_h // 2, 0, max(0, frame_h - 1)))
    x2 = int(np.clip(x1 + inner_w, x1 + 1, frame_w))
    y2 = int(np.clip(y1 + inner_h, y1 + 1, frame_h))
    return x1, y1, x2, y2


def _pick_optical_flow_points(
    gray: np.ndarray,
    zone_x1: int,
    zone_y1: int,
    zone_x2: int,
    zone_y2: int,
) -> Optional[np.ndarray]:
    if gray is None or gray.size == 0:
        return None

    frame_h, frame_w = gray.shape[:2]
    inner_x1, inner_y1, inner_x2, inner_y2 = _build_center_flow_rect(
        frame_w=frame_w,
        frame_h=frame_h,
        zone_x1=zone_x1,
        zone_y1=zone_y1,
        zone_x2=zone_x2,
        zone_y2=zone_y2,
        ratio=FLOW_CENTER_ZONE_RATIO,
    )

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = 255
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=FLOW_POINT_MAX_CORNERS,
        qualityLevel=FLOW_POINT_QUALITY_LEVEL,
        minDistance=FLOW_POINT_MIN_DISTANCE,
        mask=mask,
        blockSize=5,
    )
    if points is not None and len(points) >= FLOW_TRACK_MIN_POINTS:
        return points

    zone_mask = np.zeros(gray.shape, dtype=np.uint8)
    safe_x1 = int(np.clip(zone_x1, 0, max(0, frame_w - 1)))
    safe_y1 = int(np.clip(zone_y1, 0, max(0, frame_h - 1)))
    safe_x2 = int(np.clip(zone_x2, safe_x1 + 1, frame_w))
    safe_y2 = int(np.clip(zone_y2, safe_y1 + 1, frame_h))
    zone_mask[safe_y1:safe_y2, safe_x1:safe_x2] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=FLOW_POINT_MAX_CORNERS,
        qualityLevel=FLOW_POINT_QUALITY_LEVEL,
        minDistance=FLOW_POINT_MIN_DISTANCE,
        mask=zone_mask,
        blockSize=5,
    )


def _compute_speed_vector_with_optical_flow(
    prev_gray,
    curr_gray,
    prev_points,
    mm_per_pixel: float,
    delta_sec: float,
):
    if prev_points is None or len(prev_points) == 0:
        return 0.0, 0.0, 0.0, None

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)
    if curr_points is None or status is None:
        return 0.0, 0.0, 0.0, None

    status_flat = status.reshape(-1)
    good_mask = status_flat == 1
    if not np.any(good_mask):
        return 0.0, 0.0, 0.0, None

    prev_xy = prev_points.reshape(-1, 2)[good_mask]
    curr_xy = curr_points.reshape(-1, 2)[good_mask]
    if prev_xy.size == 0 or curr_xy.size == 0:
        return 0.0, 0.0, 0.0, None

    displacements = curr_xy - prev_xy
    if displacements.size == 0:
        return 0.0, 0.0, 0.0, curr_xy.reshape(-1, 1, 2)

    median_disp = np.median(displacements, axis=0)
    distances = np.linalg.norm(displacements - median_disp, axis=1)
    mad = float(np.median(distances))
    if mad > 1e-6:
        inlier_mask = distances <= (3.5 * mad)
        if np.any(inlier_mask):
            displacements = displacements[inlier_mask]
            curr_xy = curr_xy[inlier_mask]

    if displacements.size == 0:
        return 0.0, 0.0, 0.0, None

    avg_disp_px = np.mean(displacements, axis=0)
    safe_delta_sec = max(float(delta_sec), 1e-6)
    scale = float(mm_per_pixel) / safe_delta_sec
    speed_x_mm_per_sec = float(avg_disp_px[0]) * scale
    speed_y_mm_per_sec = float(avg_disp_px[1]) * scale
    if not np.isfinite(speed_x_mm_per_sec):
        speed_x_mm_per_sec = 0.0
    if not np.isfinite(speed_y_mm_per_sec):
        speed_y_mm_per_sec = 0.0
    speed_magnitude_mm_per_sec = float(np.hypot(speed_x_mm_per_sec, speed_y_mm_per_sec))
    if not np.isfinite(speed_magnitude_mm_per_sec):
        speed_magnitude_mm_per_sec = 0.0

    next_points = curr_xy.reshape(-1, 1, 2)
    if len(next_points) < FLOW_TRACK_MIN_POINTS:
        next_points = None
    return (
        float(speed_x_mm_per_sec),
        float(speed_y_mm_per_sec),
        float(speed_magnitude_mm_per_sec),
        next_points,
    )


def _match_detection_to_track(
    center_x: int,
    center_y: int,
    class_id: int,
    active_tracks: Dict[int, Dict[str, Any]],
    used_track_ids: set,
    max_distance_px: float,
) -> Optional[int]:
    best_track_id: Optional[int] = None
    best_distance_sq: Optional[float] = None
    max_distance_sq = float(max_distance_px) * float(max_distance_px)

    for track_id, track in active_tracks.items():
        if track_id in used_track_ids:
            continue
        if int(track.get("class_id", -1)) != int(class_id):
            continue
        dx = float(center_x) - float(track.get("center_x", 0.0))
        dy = float(center_y) - float(track.get("center_y", 0.0))
        distance_sq = dx * dx + dy * dy
        if distance_sq > max_distance_sq:
            continue
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = float(distance_sq)
            best_track_id = int(track_id)
    return best_track_id


def _cleanup_stale_tracks(
    active_tracks: Dict[int, Dict[str, Any]],
    frame_id: int,
    stale_frame_ttl: int,
) -> None:
    stale_ids = [
        int(track_id)
        for track_id, track in active_tracks.items()
        if (int(frame_id) - int(track.get("last_seen_frame", frame_id))) > int(stale_frame_ttl)
    ]
    for track_id in stale_ids:
        active_tracks.pop(track_id, None)


def _set_robot_async_inflight(value: bool) -> None:
    global robot_async_inflight
    with robot_async_state_lock:
        robot_async_inflight = bool(value)


def _is_robot_async_inflight() -> bool:
    with robot_async_state_lock:
        return bool(robot_async_inflight)


def _queue_robot_command_async(
    relative_x_mm: int,
    relative_y_mm: int,
    duration_ms: int,
    context: Optional[Dict[str, Any]] = None,
    command_type: str = "process_target",
) -> Dict[str, Any]:
    global robot_async_inflight, robot_async_seq
    with robot_async_state_lock:
        if robot_async_inflight:
            return {"accepted": False, "reason": "busy"}
        robot_async_inflight = True
        robot_async_seq = int(robot_async_seq) + 1
        command_id = int(robot_async_seq)
    command_context = dict(context or {})
    _publish_event(
        "command.accepted",
        {
            "device": "delta",
            "type": str(command_type or "process_target"),
            "command_id": int(command_id),
            "duration_ms": int(duration_ms),
            "target_x_mm": int(relative_x_mm),
            "target_y_mm": int(relative_y_mm),
            "context": dict(command_context),
        },
    )

    def _worker() -> None:
        ok = False
        error_text: Optional[str] = None
        started_at = time.time()
        _publish_event(
            "command.started",
            {
                "device": "delta",
                "type": str(command_type or "process_target"),
                "command_id": int(command_id),
                "duration_ms": int(duration_ms),
                "target_x_mm": int(relative_x_mm),
                "target_y_mm": int(relative_y_mm),
                "context": dict(command_context),
            },
        )
        try:
            ok = bool(_dispatch_robot_command(relative_x_mm, relative_y_mm, duration_ms))
        except Exception as exc:  # pragma: no cover - defensive guard
            error_text = str(exc)
            log.error("Async robot command #%s failed: %s", command_id, exc)
        finally:
            _set_robot_async_inflight(False)
            _update_visual_state(
                {
                    "event": "robot_command_result",
                    "count": 0,
                    "detections": [],
                    "source": "camera",
                    "command_id": int(command_id),
                    "ok": bool(ok),
                    "error": error_text,
                    "duration_ms": int(duration_ms),
                    "elapsed_ms": int(round((time.time() - started_at) * 1000.0)),
                    "target_x_mm": int(relative_x_mm),
                    "target_y_mm": int(relative_y_mm),
                    **command_context,
                }
            )
            if ok:
                _publish_event(
                    "command.completed",
                    {
                        "device": "delta",
                        "type": str(command_type or "process_target"),
                        "command_id": int(command_id),
                        "duration_ms": int(duration_ms),
                        "target_x_mm": int(relative_x_mm),
                        "target_y_mm": int(relative_y_mm),
                        "context": dict(command_context),
                        "result": "done",
                    },
                )
                _publish_event(
                    "fault.cleared",
                    {
                        "device": "delta",
                    },
                )
            else:
                _publish_event(
                    "command.failed",
                    {
                        "device": "delta",
                        "type": str(command_type or "process_target"),
                        "command_id": int(command_id),
                        "duration_ms": int(duration_ms),
                        "target_x_mm": int(relative_x_mm),
                        "target_y_mm": int(relative_y_mm),
                        "context": dict(command_context),
                        "code": "device_fault" if error_text else "execution_timeout",
                        "message": str(error_text or "Command execution failed"),
                    },
                )
                _publish_event(
                    "fault.raised",
                    {
                        "device": "delta",
                        "code": "device_fault" if error_text else "execution_timeout",
                        "message": str(error_text or "Command execution failed"),
                    },
                )

    Thread(target=_worker, name=f"robot-auto-{command_id}", daemon=True).start()
    return {"accepted": True, "command_id": int(command_id)}


def _queue_robot_command_for_runtime(
    relative_x_mm: int,
    relative_y_mm: int,
    duration_ms: int,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = _queue_robot_command_async(
        relative_x_mm=int(relative_x_mm),
        relative_y_mm=int(relative_y_mm),
        duration_ms=int(duration_ms),
        context=context,
        command_type="process_target",
    )
    return result


class _SharedDeltaModbusAdapter:
    """Bridge bres.py calls to the shared backend RS485 owner."""

    def __init__(self, port: str):
        self.port = str(port or "").strip()
        self.connected = bool(self.port)
        self.slave_addr = int(motion.DELTA_SLAVE)

    def _run(self, action):
        if not self.connected or not self.port:
            return {"status": "error", "message": "No active serial port selected"}
        return motion.run_with_connection(self.port, action)

    def start_trajectory(self, segments, duration=0):
        payload = list(segments or [])
        result = self._run(
            lambda: motion.delta_start_trajectory(
                segments=payload,
                duration_ms=int(duration),
            )
        )
        if result.get("status") != "success":
            log.error("Delta trajectory write failed: %s", result.get("message") or result)
            return False
        return True

    def proccess(self, cmd=1, duration=1000):
        result = self._run(lambda: motion.delta_process(cmd=int(cmd), duration_ms=int(duration)))
        if result.get("status") != "success":
            log.error("Delta process command failed: %s", result.get("message") or result)
            return False
        return True

    def rotate(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        return False

    def disconnect(self):
        self.connected = False
        return True


def _delta_stop() -> Dict[str, Any]:
    stream_stop_event.set()
    inflight = bool(_is_robot_async_inflight())
    selected_port = str(_get_active_port() or "").strip()

    hard_stop_sent = False
    ready = None
    busy = None
    reason = None
    if selected_port:
        stop_result = motion.run_with_connection(selected_port, motion.delta_hard_stop)
        if stop_result.get("status") == "success":
            hard_stop_sent = True
            ready = stop_result.get("ready")
            busy = stop_result.get("busy")
        else:
            reason = str(stop_result.get("message") or "hard_stop_failed")
    else:
        reason = "transport_unavailable"

    if not inflight:
        _close_move_module_client()
    payload = {
        "stop_requested": True,
        "inflight": inflight,
        "hard_stop_sent": bool(hard_stop_sent),
        "soft_stop_only": bool(not hard_stop_sent),
        "ready": ready,
        "busy": busy,
        "reason": reason,
    }
    _publish_event(
        "device.state",
        {
            "device": "delta",
            **payload,
        },
    )
    return payload


def _get_manual_robot_state() -> Dict[str, Any]:
    with manual_robot_state_lock:
        return {
            "has_last": bool(manual_robot_state["has_last"]),
            "x_mm": float(manual_robot_state["x_mm"]),
            "y_mm": float(manual_robot_state["y_mm"]),
            "z_mm": float(manual_robot_state["z_mm"]),
            "z_min_mm": float(ROBOT_MANUAL_Z_MIN_MM),
            "z_max_mm": float(ROBOT_MANUAL_Z_MAX_MM),
        }


def _set_manual_robot_last_position(x_mm: float, y_mm: float, z_mm: float) -> Dict[str, Any]:
    with manual_robot_state_lock:
        manual_robot_state["has_last"] = True
        manual_robot_state["x_mm"] = float(x_mm)
        manual_robot_state["y_mm"] = float(y_mm)
        manual_robot_state["z_mm"] = float(z_mm)
    return _get_manual_robot_state()


def _clear_manual_robot_state() -> Dict[str, Any]:
    with manual_robot_state_lock:
        manual_robot_state["has_last"] = False
        manual_robot_state["x_mm"] = float(ROBOT_HOME_X_MM)
        manual_robot_state["y_mm"] = float(ROBOT_HOME_Y_MM)
        manual_robot_state["z_mm"] = float(ROBOT_Z_UP_MM)
    return _get_manual_robot_state()


def _load_move_to_coordinate():
    global move_to_coordinate_fn, move_to_coordinate_error, move_module, move_module_port

    selected_port = _get_active_port()
    if move_to_coordinate_fn is not None and move_module is not None and move_module_port == selected_port:
        return move_to_coordinate_fn

    try:
        api_dir = str(Path(__file__).resolve().parent)
        backup_dir = str(Path(__file__).resolve().parent.parent)
        if api_dir in sys.path:
            sys.path.remove(api_dir)
        sys.path.insert(0, api_dir)
        if backup_dir not in sys.path:
            sys.path.append(backup_dir)

        _apply_robot_settings_to_env(_get_robot_settings())
        os.environ["DELTA_MODBUS_PORT"] = selected_port
        os.environ["DELTA_DISABLE_INTERNAL_MODBUS"] = "1"

        api_bres_path = (Path(__file__).resolve().parent / "bres.py").resolve()
        if "bres" in sys.modules:
            existing_module = sys.modules["bres"]
            existing_file = getattr(existing_module, "__file__", None)
            if not existing_file or Path(existing_file).resolve() != api_bres_path:
                del sys.modules["bres"]
                existing_module = None

        if "bres" in sys.modules:
            existing_module = sys.modules["bres"]
            existing_client = getattr(existing_module, "client", None)
            try:
                existing_disconnect = getattr(existing_client, "disconnect", None)
                if callable(existing_disconnect):
                    existing_disconnect()
                else:
                    existing_serial = getattr(existing_client, "client", None)
                    if existing_serial is not None:
                        existing_serial.close()
            except Exception:
                pass
            module = importlib.reload(existing_module)
        else:
            module = importlib.import_module("bres")

        module.client = _SharedDeltaModbusAdapter(selected_port)
        move_fn = getattr(module, "move_to_coordinate")
        move_to_coordinate_fn = move_fn
        move_module = module
        move_module_port = selected_port
        move_to_coordinate_error = None
        log.info("Loaded robot movement module bres.py (port=%s)", selected_port)
    except Exception as exc:
        move_to_coordinate_error = str(exc)
        move_to_coordinate_fn = None
        move_module = None
        move_module_port = None
        log.error("Failed to load bres.move_to_coordinate: %s", exc)
        return None

    return move_to_coordinate_fn


def _close_move_module_client() -> None:
    global move_to_coordinate_fn, move_module, move_module_port

    module = move_module
    if module is None:
        return

    module_client = getattr(module, "client", None)
    try:
        module_disconnect = getattr(module_client, "disconnect", None)
        if callable(module_disconnect):
            module_disconnect()
        else:
            serial_client = getattr(module_client, "client", None)
            if serial_client is not None:
                serial_client.close()
    except Exception:
        pass

    move_to_coordinate_fn = None
    move_module = None
    move_module_port = None


def _request_stop_and_release_resources() -> None:
    global pipeline, pipeline_mode

    stream_stop_event.set()
    if _is_robot_async_inflight():
        log.warning("Stop requested while robot command is in-flight; deferring robot client close.")
    else:
        _close_move_module_client()
        close_limit_switch_client()
    with pipeline_lock:
        _safe_close_camera_stream(pipeline, pipeline_mode)
        pipeline = None
        pipeline_mode = None


def _read_robot_ready_flag() -> Optional[int]:
    selected_port = str(_get_active_port() or "").strip()
    if not selected_port:
        return None

    try:
        read_result = motion.run_with_connection(
            selected_port,
            lambda: motion.read_registers(
                slave_id=int(motion.DELTA_SLAVE),
                start_addr=int(ROBOT_READY_REGISTER_ADDR),
                count=1,
            ),
        )
        if read_result.get("status") != "success":
            return None
        registers = list(read_result.get("registers") or [])
        if not registers:
            return None
        return int(registers[0])
    except Exception:
        return None


def _wait_until_robot_busy(timeout_sec: float) -> bool:
    deadline = time.time() + max(0.1, float(timeout_sec))
    while time.time() < deadline:
        ready_value = _read_robot_ready_flag()
        if ready_value is not None and ready_value != ROBOT_READY_IDLE_VALUE:
            return True
        time.sleep(ROBOT_READY_POLL_INTERVAL_SEC)
    return False


def _wait_until_robot_ready(timeout_sec: float) -> bool:
    deadline = time.time() + max(0.1, float(timeout_sec))
    while time.time() < deadline:
        ready_value = _read_robot_ready_flag()
        if ready_value is not None and ready_value == ROBOT_READY_IDLE_VALUE:
            return True
        time.sleep(ROBOT_READY_POLL_INTERVAL_SEC)
    return False


def _dispatch_robot_command(relative_x_mm: int, relative_y_mm: int, duration_ms: int = ROBOT_MOVE_DURATION_MS) -> bool:
    global last_robot_command_ts

    if stream_stop_event.is_set():
        return False

    duration_ms = _resolve_command_duration_ms(duration_ms)

    move_fn = _load_move_to_coordinate()
    if move_fn is None:
        return False

    now = time.time()
    with robot_command_lock:
        delta_sec = now - last_robot_command_ts
        if delta_sec < ROBOT_COMMAND_COOLDOWN_SEC:
            log.info(
                "Robot command skipped by cooldown: %.3fs < %.3fs",
                delta_sec,
                ROBOT_COMMAND_COOLDOWN_SEC,
            )
            return False
        last_robot_command_ts = now

        try:
            # Phase 1: move from home (0,0,-180) to detected target and wait until motion is completed.
            move_fn(
                ROBOT_HOME_X_MM,
                ROBOT_HOME_Y_MM,
                ROBOT_Z_UP_MM,
                relative_x_mm,
                relative_y_mm,
                ROBOT_Z_DOWN_MM,
                duration_ms,
                False,
            )
            if not _wait_until_robot_busy(ROBOT_BUSY_WAIT_TIMEOUT_SEC):
                log.warning("Robot did not enter busy state after target command.")
            target_ready = _wait_until_robot_ready(ROBOT_READY_WAIT_TIMEOUT_SEC)
            if not target_ready:
                log.error("Robot target motion timeout (ready flag did not return to idle); forcing return-home command.")
            if stream_stop_event.is_set():
                log.warning("Robot stop requested; skipping return-home command.")
                return False

            # Phase 2: return robot to home pose (0,0,-180) and wait again.
            move_fn(
                relative_x_mm,
                relative_y_mm,
                ROBOT_Z_DOWN_MM,
                ROBOT_HOME_X_MM,
                ROBOT_HOME_Y_MM,
                ROBOT_Z_UP_MM,
                duration_ms,
                False,
            )
            if not _wait_until_robot_busy(ROBOT_BUSY_WAIT_TIMEOUT_SEC):
                log.warning("Robot did not enter busy state after return-home command.")
            if not _wait_until_robot_ready(ROBOT_READY_WAIT_TIMEOUT_SEC):
                log.error("Robot return-home timeout (ready flag did not return to idle).")
                return False

            log.info("Robot cycle completed: target=(%s,%s)mm, home=(0,0,-180)", relative_x_mm, relative_y_mm)
            return bool(target_ready)
        except Exception as exc:
            log.error("Robot command failed: %s", exc)
            return False
        finally:
            # Keep serial resources short-lived: close command ports right after the cycle.
            _close_move_module_client()


def _dispatch_manual_robot_move(
    target_x_mm: float,
    target_y_mm: float,
    target_z_mm: float,
    duration_ms: int = ROBOT_MOVE_DURATION_MS,
) -> Dict[str, Any]:
    # Manual coordinate mode can be used while video stream is stopped.
    stream_stop_event.clear()

    state = _get_manual_robot_state()
    if not bool(state.get("has_last")):
        raise ValueError("calibration_required")

    move_fn = _load_move_to_coordinate()
    if move_fn is None:
        raise RuntimeError(move_to_coordinate_error or "Robot movement module is unavailable.")

    target_x = float(target_x_mm)
    target_y = float(target_y_mm)
    target_z = float(np.clip(float(target_z_mm), ROBOT_MANUAL_Z_MIN_MM, ROBOT_MANUAL_Z_MAX_MM))
    command_duration = _resolve_command_duration_ms(duration_ms)

    with robot_command_lock:
        state = _get_manual_robot_state()
        if not bool(state.get("has_last")):
            raise ValueError("calibration_required")

        start_x = float(state["x_mm"])
        start_y = float(state["y_mm"])
        start_z = float(state["z_mm"])

        try:
            move_fn(
                start_x,
                start_y,
                start_z,
                target_x,
                target_y,
                target_z,
                command_duration,
                False,
            )
            if not _wait_until_robot_busy(ROBOT_BUSY_WAIT_TIMEOUT_SEC):
                log.warning("Robot did not enter busy state after manual move command.")
            if not _wait_until_robot_ready(ROBOT_READY_WAIT_TIMEOUT_SEC):
                raise RuntimeError("Robot manual motion timeout (ready flag did not return to idle).")
        except Exception:
            raise
        finally:
            _close_move_module_client()

    next_state = _set_manual_robot_last_position(target_x, target_y, target_z)
    _update_visual_state(
        {
            "event": "manual_move",
            "count": 0,
            "detections": [],
            "from_x_mm": round(start_x, 3),
            "from_y_mm": round(start_y, 3),
            "from_z_mm": round(start_z, 3),
            "to_x_mm": round(target_x, 3),
            "to_y_mm": round(target_y, 3),
            "to_z_mm": round(target_z, 3),
            "duration_ms": int(command_duration),
            "manual_has_last": bool(next_state["has_last"]),
            "manual_x_mm": float(next_state["x_mm"]),
            "manual_y_mm": float(next_state["y_mm"]),
            "manual_z_mm": float(next_state["z_mm"]),
        }
    )
    return {
        "ok": True,
        "from": {
            "x_mm": round(start_x, 3),
            "y_mm": round(start_y, 3),
            "z_mm": round(start_z, 3),
        },
        "to": {
            "x_mm": round(target_x, 3),
            "y_mm": round(target_y, 3),
            "z_mm": round(target_z, 3),
        },
        "duration_ms": int(command_duration),
        "last": next_state,
    }


def _run_delta_go_home(slave_id: Optional[int] = None) -> Dict[str, Any]:
    slave_id_value = int(motion.DELTA_SLAVE) if slave_id is None else int(slave_id)
    selected_port = str(_get_active_port() or "").strip()
    if not selected_port:
        raise RuntimeError("No active serial port selected")

    with robot_command_lock:
        _close_move_module_client()
        if int(slave_id_value) == int(motion.DELTA_SLAVE):
            if not _wait_until_robot_ready(ROBOT_READY_WAIT_TIMEOUT_SEC):
                raise RuntimeError("Delta controller is busy before home/calibration command.")

            _clear_manual_robot_state()

            calibration = motion.run_with_connection(selected_port, motion.delta_calibrate)
            if calibration.get("status") != "success":
                raise RuntimeError(str(calibration.get("message") or "Delta calibration command failed"))

            calibration_busy_seen = _wait_until_robot_busy(ROBOT_BUSY_WAIT_TIMEOUT_SEC)
            if not calibration_busy_seen:
                raise RuntimeError("Delta calibration did not start (ready flag stayed idle).")

            if not _wait_until_robot_ready(ROBOT_CALIBRATION_WAIT_TIMEOUT_SEC):
                raise RuntimeError("Delta calibration timeout (ready flag did not return to idle).")

            result = motion.run_with_connection(selected_port, motion.delta_go_home)
            if result.get("status") != "success":
                raise RuntimeError(str(result.get("message") or "Delta go-home command failed"))

            home_busy_seen = _wait_until_robot_busy(ROBOT_BUSY_WAIT_TIMEOUT_SEC)
            if home_busy_seen:
                if not _wait_until_robot_ready(ROBOT_READY_WAIT_TIMEOUT_SEC):
                    raise RuntimeError("Delta go-home timeout (ready flag did not return to idle).")
            else:
                ready_value = _read_robot_ready_flag()
                if ready_value is not None and int(ready_value) != int(ROBOT_READY_IDLE_VALUE):
                    if not _wait_until_robot_ready(ROBOT_READY_WAIT_TIMEOUT_SEC):
                        raise RuntimeError("Delta go-home timeout (ready flag did not return to idle).")

            snapshot = motion.run_with_connection(
                selected_port,
                lambda: motion.read_registers(
                    slave_id=int(slave_id_value),
                    start_addr=0,
                    count=31,
                ),
            )
            if snapshot.get("status") != "success":
                raise RuntimeError(str(snapshot.get("message") or "Failed to read Delta registers"))
            registers = list(snapshot.get("registers") or [])

            latest_log = int(registers[11]) if len(registers) > 11 else 0
            if latest_log in {0x2017, 0x3017}:
                raise RuntimeError(
                    f"Delta firmware rejected work-pose command (log=0x{latest_log:04X})."
                )

            result = {
                "status": "success",
                "ok": True,
                "registers": registers,
                "calibration_busy_seen": bool(calibration_busy_seen),
                "home_busy_seen": bool(home_busy_seen),
            }
        else:
            result = motion.run_with_connection(
                selected_port,
                lambda: motion.write_registers(
                    slave_id=int(slave_id_value),
                    start_addr=int(motion.DELTA_WORK_ADDR),
                    values=[1],
                ),
            )
            if result.get("status") == "success":
                snapshot = motion.run_with_connection(
                    selected_port,
                    lambda: motion.read_registers(
                        slave_id=int(slave_id_value),
                        start_addr=0,
                        count=31,
                    ),
                )
                if snapshot.get("status") != "success":
                    result = snapshot
                else:
                    result = {
                        "status": "success",
                        "ok": True,
                        "registers": list(snapshot.get("registers") or []),
                    }
        if result.get("status") != "success":
            raise RuntimeError(str(result.get("message") or "Delta go-home command failed"))

        ok = bool(result.get("ok", result.get("status") == "success"))
        if ok and int(slave_id_value) == int(motion.DELTA_SLAVE):
            _set_manual_robot_last_position(ROBOT_HOME_X_MM, ROBOT_HOME_Y_MM, ROBOT_Z_UP_MM)
        return {
            "ok": bool(ok),
            "registers": list(result.get("registers") or []),
            "manual_state": _get_manual_robot_state(),
        }


def _build_robot_api_runtime() -> RobotApiRuntime:
    return RobotApiRuntime(
        get_active_port=_get_active_port,
        set_active_port=_set_active_port,
        list_serial_ports=_list_serial_ports,
        serial_lock=robot_command_lock,
        reset_motion_runtime_state=reset_runtime_state,
        close_limit_switch_client=close_limit_switch_client,
        close_move_module_client=_close_move_module_client,
        get_manual_robot_state=_get_manual_robot_state,
        clear_manual_robot_state=_clear_manual_robot_state,
        dispatch_manual_robot_move=_dispatch_manual_robot_move,
        get_vision_state=_get_vision_state_snapshot,
        is_robot_busy=_is_robot_async_inflight,
        delta_go_home=_run_delta_go_home,
        queue_robot_command=_queue_robot_command_for_runtime,
        delta_stop=_delta_stop,
        get_visual_state_snapshot=_get_visual_state_snapshot,
        vision_start=_vision_start,
        vision_stop=_vision_stop,
        get_zone_config=_get_zone_config,
        set_zone_config=_set_zone_config,
        get_system_devices=_get_system_devices,
        publish_event=_publish_event,
        pull_events_since=_pull_events_since,
    )


def _build_service_registry() -> ServiceRegistry:
    return ServiceRegistry(_build_robot_api_runtime())


def _start_pipeline(mode: str):
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed")

    stream_pipeline = rs.pipeline()
    stream_config = rs.config()
    if mode == "color":
        stream_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    elif mode == "depth":
        stream_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    else:
        raise ValueError(f"Unsupported RealSense mode: {mode}")

    stream_pipeline.start(stream_config)
    return stream_pipeline, mode


def _safe_stop_pipeline(stream_pipeline):
    if stream_pipeline is None:
        return
    try:
        stream_pipeline.stop()
    except Exception:
        pass


def _safe_release_webcam(stream_capture):
    if stream_capture is None:
        return
    try:
        stream_capture.release()
    except Exception:
        pass


def _safe_close_camera_stream(stream, mode: Optional[str]):
    if mode == "webcam":
        _safe_release_webcam(stream)
        return
    _safe_stop_pipeline(stream)


def _start_webcam(index: int = WEBCAM_INDEX):
    capture = None
    errors: List[str] = []
    backends: List[Optional[int]] = [None]
    if hasattr(cv2, "CAP_DSHOW"):
        backends.insert(0, cv2.CAP_DSHOW)

    for backend in backends:
        try:
            if backend is None:
                capture = cv2.VideoCapture(index)
            else:
                capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                raise RuntimeError("capture is not opened")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("webcam opened but no frames arrived")
            log.info("Webcam stream active (index=%s).", index)
            return capture, "webcam"
        except Exception as exc:
            errors.append(str(exc))
            _safe_release_webcam(capture)
            capture = None

    raise RuntimeError(f"Unable to start webcam stream (index={index}): {'; '.join(errors)}")


def _frames_available(stream_pipeline, mode: str):
    for _ in range(FRAME_PROBE_ATTEMPTS):
        try:
            frames = stream_pipeline.wait_for_frames(timeout_ms=FRAME_PROBE_TIMEOUT_MS)
        except RuntimeError:
            continue

        if mode == "color" and frames.get_color_frame():
            return True
        if mode == "depth" and frames.get_depth_frame():
            return True
    return False


def _start_pipeline_with_fallback(prefer_mode: str = "color"):
    modes = ["color", "depth"] if prefer_mode != "depth" else ["depth", "color"]
    last_exc = None

    for mode in modes:
        stream_pipeline = None
        try:
            stream_pipeline, active_mode = _start_pipeline(mode)
            if not _frames_available(stream_pipeline, active_mode):
                raise RuntimeError(f"RealSense {active_mode} stream started but no frames arrived")
            log.info("RealSense stream active in '%s' mode.", active_mode)
            return stream_pipeline, active_mode
        except Exception as exc:
            last_exc = exc
            _safe_stop_pipeline(stream_pipeline)
            log.warning("RealSense %s stream failed: %s", mode, exc)

    raise RuntimeError(f"Unable to start RealSense stream in modes {modes}: {last_exc}")


def _start_camera_with_fallback(prefer_mode: str = "color"):
    if REALSENSE_CAMERA:
        try:
            return _start_pipeline_with_fallback(prefer_mode=prefer_mode)
        except Exception as exc:
            log.warning(
                "RealSense is unavailable (%s). Falling back to webcam index %s.",
                exc,
                WEBCAM_INDEX,
            )
    return _start_webcam(index=WEBCAM_INDEX)


def _restart_pipeline(prefer_mode: str = "color"):
    global pipeline, pipeline_mode
    with pipeline_lock:
        _safe_close_camera_stream(pipeline, pipeline_mode)
        pipeline = None
        pipeline_mode = None
        pipeline, pipeline_mode = _start_pipeline_with_fallback(prefer_mode)


pipeline = None
pipeline_mode = None


def _frame_to_image(frames, mode: str):
    if mode == "color":
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth = np.asanyarray(depth_frame.get_data())
    depth_u8 = cv2.convertScaleAbs(depth, alpha=DEPTH_VISUALIZATION_ALPHA)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def _draw_locked_command_point(
    frame: np.ndarray,
    command_point: Optional[Dict[str, Any]],
) -> np.ndarray:
    if frame is None or command_point is None:
        return frame

    try:
        px = int(command_point.get("x"))
        py = int(command_point.get("y"))
    except Exception:
        return frame

    frame_h, frame_w = frame.shape[:2]
    if px < 0 or py < 0 or px >= frame_w or py >= frame_h:
        return frame

    cv2.circle(frame, (px, py), 16, (0, 255, 255), 2)
    cv2.drawMarker(
        frame,
        (px, py),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

    label = "Delta target (locked)"
    target_x_mm = command_point.get("target_x_mm")
    target_y_mm = command_point.get("target_y_mm")
    if target_x_mm is not None and target_y_mm is not None:
        try:
            label = f"Delta target (locked): {int(target_x_mm)},{int(target_y_mm)} mm"
        except Exception:
            pass
    cv2.putText(
        frame,
        label,
        (px + 14, max(24, py - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    return frame


@app.on_event("startup")
def startup_event():
    global pipeline, pipeline_mode
    try:
        with pipeline_lock:
            pipeline, pipeline_mode = _start_camera_with_fallback(prefer_mode="color")
    except Exception as exc:
        pipeline = None
        pipeline_mode = None
        log.error("Camera start failed on startup: %s", exc)


def gen_mjpeg():
    global pipeline, pipeline_mode
    timeout_streak = 0
    stream_stop_event.clear()

    try:
        while True:
            if stream_stop_event.is_set():
                return
            if pipeline is None:
                try:
                    with pipeline_lock:
                        pipeline, pipeline_mode = _start_camera_with_fallback(prefer_mode="color")
                    timeout_streak = 0
                except Exception as exc:
                    log.warning("Camera stream is unavailable: %s", exc)
                    time.sleep(1.0)
                    continue

            if pipeline_mode == "webcam":
                ok, img = pipeline.read()
                if not ok or img is None:
                    timeout_streak += 1
                    if timeout_streak % 5 == 0:
                        log.warning(
                            "No webcam frames for %s tries (index=%s).",
                            timeout_streak,
                            WEBCAM_INDEX,
                        )
                    if timeout_streak >= FRAME_TIMEOUT_RETRIES:
                        log.warning(
                            "No webcam frames for %s tries. Reopening webcam (index=%s).",
                            timeout_streak,
                            WEBCAM_INDEX,
                        )
                        with pipeline_lock:
                            _safe_release_webcam(pipeline)
                            pipeline = None
                            pipeline_mode = None
                        timeout_streak = 0
                        time.sleep(RESTART_COOLDOWN_SEC)
                    time.sleep(FRAME_TIMEOUT_SLEEP_SEC)
                    continue
                timeout_streak = 0
            else:
                try:
                    with pipeline_lock:
                        frames = pipeline.wait_for_frames(timeout_ms=FRAME_WAIT_TIMEOUT_MS)
                    timeout_streak = 0
                except RuntimeError as exc:
                    if "Frame didn't arrive" in str(exc):
                        timeout_streak += 1
                        if timeout_streak % 5 == 0:
                            log.warning(
                                "No RealSense frames for %s tries (mode=%s).",
                                timeout_streak,
                                pipeline_mode or "unknown",
                            )
                        if timeout_streak >= FRAME_TIMEOUT_RETRIES:
                            log.warning(
                                "No RealSense frames for %s tries. Restarting pipeline (mode=%s).",
                                timeout_streak,
                                pipeline_mode or "unknown",
                            )
                            try:
                                _restart_pipeline(prefer_mode=pipeline_mode or "color")
                                time.sleep(RESTART_COOLDOWN_SEC)
                                timeout_streak = 0
                            except Exception as restart_exc:
                                log.error("RealSense pipeline restart failed: %s", restart_exc)
                                with pipeline_lock:
                                    pipeline = None
                                    pipeline_mode = None
                            timeout_streak = 0
                        time.sleep(FRAME_TIMEOUT_SLEEP_SEC)
                        continue
                    raise

                img = _frame_to_image(frames, pipeline_mode or "color")
                if img is None:
                    continue

            if img is None:
                with pipeline_lock:
                    _safe_close_camera_stream(pipeline, pipeline_mode)
                    pipeline = None
                    pipeline_mode = None
                time.sleep(FRAME_TIMEOUT_SLEEP_SEC)
                continue

            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            frame = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
    except GeneratorExit:
        return
    except Exception as exc:
        log.error("MJPEG stream failed: %s", exc)


def gen_processed_mjpeg(
    send_command: bool = True,
    enable_detection: bool = True,
    detection_interval: int = 1,
    confidence_threshold: float = 0.3,
    command_duration_ms: int = ROBOT_MOVE_DURATION_MS,
    jpeg_quality: int = 75,
    max_width: int = 1280,
    detection_imgsz: int = 640,
    model_device: Optional[str] = None,
    use_half: bool = True,
):
    global pipeline, pipeline_mode
    timeout_streak = 0
    frame_id = 0
    flow_prev_gray = None
    flow_prev_points = None
    last_speed_update_ts = None
    speed_x_mm_per_sec = 0.0
    speed_y_mm_per_sec = 0.0
    speed_magnitude_mm_per_sec = 0.0
    last_detections = []
    tracked_detections: List[Dict[str, Any]] = []
    detections_payload: List[Dict[str, Any]] = []
    active_tracks: Dict[int, Dict[str, Any]] = {}
    locked_command_point: Optional[Dict[str, Any]] = None
    simulated_command_id = 0
    next_track_id = 1
    stream_stop_event.clear()
    command_duration_ms = _resolve_command_duration_ms(command_duration_ms)
    predict_horizon_ms = int(command_duration_ms)

    detection_interval = max(1, int(detection_interval))
    resolved_device = _resolve_model_device(model_device)
    use_half = bool(use_half and "cuda" in resolved_device.lower())
    model = None
    model_error = None

    def _locked_command_point_payload() -> Dict[str, Any]:
        if locked_command_point is None:
            return {
                "locked_command_point_x": None,
                "locked_command_point_y": None,
                "locked_target_x_mm": None,
                "locked_target_y_mm": None,
                "locked_track_id": None,
                "locked_command_id": None,
            }
        return {
            "locked_command_point_x": int(locked_command_point.get("x", 0)),
            "locked_command_point_y": int(locked_command_point.get("y", 0)),
            "locked_target_x_mm": int(locked_command_point.get("target_x_mm", 0)),
            "locked_target_y_mm": int(locked_command_point.get("target_y_mm", 0)),
            "locked_track_id": int(locked_command_point.get("track_id", 0)),
            "locked_command_id": int(locked_command_point.get("command_id", 0)),
        }

    def _clear_locked_command_point_if_expired(now_ts: float) -> None:
        nonlocal locked_command_point
        if locked_command_point is None:
            return
        try:
            expires_at_ts = float(locked_command_point.get("expires_at_ts", 0.0))
        except Exception:
            expires_at_ts = 0.0
        if expires_at_ts <= 0.0 or float(now_ts) < expires_at_ts:
            return

        command_id = locked_command_point.get("command_id")
        track_id = locked_command_point.get("track_id")
        locked_command_point = None
        _update_visual_state(
            {
                "event": "locked_target_cleared",
                "count": 0,
                "detections": [],
                "source": "camera",
                "reason": "ttl_expired",
                "ttl_ms": int(command_duration_ms * 2),
                "command_id": int(command_id) if command_id is not None else None,
                "track_id": int(track_id) if track_id is not None else None,
                **_locked_command_point_payload(),
            }
        )

    if enable_detection:
        try:
            model = _get_yolo_model()
            warmup = np.zeros((320, 320, 3), dtype=np.uint8)
            model.predict(
                source=warmup,
                conf=confidence_threshold,
                verbose=False,
                device=resolved_device,
                imgsz=320,
                half=use_half,
            )
        except Exception as exc:
            model_error = str(exc)
            log.error("Camera detection model is unavailable: %s", exc)

    try:
        while True:
            if stream_stop_event.is_set():
                return
            frame_started = time.perf_counter()
            _clear_locked_command_point_if_expired(frame_started)
            if pipeline is None:
                try:
                    with pipeline_lock:
                        pipeline, pipeline_mode = _start_camera_with_fallback(prefer_mode="color")
                    timeout_streak = 0
                except Exception as exc:
                    log.warning("Camera stream is unavailable: %s", exc)
                    time.sleep(1.0)
                    continue

            if pipeline_mode == "webcam":
                ok, frame = pipeline.read()
                if not ok or frame is None:
                    timeout_streak += 1
                    if timeout_streak % 5 == 0:
                        log.warning(
                            "No webcam frames for %s tries (index=%s).",
                            timeout_streak,
                            WEBCAM_INDEX,
                        )
                    if timeout_streak >= FRAME_TIMEOUT_RETRIES:
                        log.warning(
                            "No webcam frames for %s tries. Reopening webcam (index=%s).",
                            timeout_streak,
                            WEBCAM_INDEX,
                        )
                        with pipeline_lock:
                            _safe_release_webcam(pipeline)
                            pipeline = None
                            pipeline_mode = None
                        timeout_streak = 0
                        time.sleep(RESTART_COOLDOWN_SEC)
                    time.sleep(FRAME_TIMEOUT_SLEEP_SEC)
                    continue
                timeout_streak = 0
            else:
                try:
                    with pipeline_lock:
                        frames = pipeline.wait_for_frames(timeout_ms=FRAME_WAIT_TIMEOUT_MS)
                    timeout_streak = 0
                except RuntimeError as exc:
                    if "Frame didn't arrive" in str(exc):
                        timeout_streak += 1
                        if timeout_streak % 5 == 0:
                            log.warning(
                                "No RealSense frames for %s tries (mode=%s).",
                                timeout_streak,
                                pipeline_mode or "unknown",
                            )
                        if timeout_streak >= FRAME_TIMEOUT_RETRIES:
                            log.warning(
                                "No RealSense frames for %s tries. Restarting pipeline (mode=%s).",
                                timeout_streak,
                                pipeline_mode or "unknown",
                            )
                            try:
                                _restart_pipeline(prefer_mode=pipeline_mode or "color")
                                time.sleep(RESTART_COOLDOWN_SEC)
                                timeout_streak = 0
                            except Exception as restart_exc:
                                log.error("RealSense pipeline restart failed: %s", restart_exc)
                                with pipeline_lock:
                                    pipeline = None
                                    pipeline_mode = None
                            timeout_streak = 0
                        time.sleep(FRAME_TIMEOUT_SLEEP_SEC)
                        continue
                    raise

                frame = _frame_to_image(frames, pipeline_mode or "color")
                if frame is None:
                    continue

            if frame is None:
                with pipeline_lock:
                    _safe_close_camera_stream(pipeline, pipeline_mode)
                    pipeline = None
                    pipeline_mode = None
                time.sleep(FRAME_TIMEOUT_SLEEP_SEC)
                continue

            frame_id += 1
            frame_h, frame_w = frame.shape[:2]
            frame_now_ts = time.perf_counter()
            fallback_mm_per_pixel = (
                (DEFAULT_FOV_WIDTH_MM / max(frame_w, 1))
                + (DEFAULT_FOV_HEIGHT_MM / max(frame_h, 1))
            ) / 2.0

            zone_cfg = _get_zone_config()
            zone_offset_x = int(zone_cfg["offset_x_px"])
            zone_offset_y = int(zone_cfg["offset_y_px"])
            zone_size_px = int(np.clip(int(zone_cfg["size_px"]), ZONE_MIN_SIZE_PX, min(ZONE_MAX_SIZE_PX, max(80, min(frame_w, frame_h) - 4))))
            zone_size_mm = float(zone_cfg["size_mm"])
            mm_per_pixel = zone_size_mm / max(zone_size_px, 1)
            if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 1e-9:
                mm_per_pixel = fallback_mm_per_pixel

            zone_center_x = int(frame_w // 2 + zone_offset_x)
            zone_center_y = int(frame_h // 2 + zone_offset_y)
            half_zone = max(1, zone_size_px // 2)
            zone_center_x = int(np.clip(zone_center_x, half_zone, max(half_zone, frame_w - half_zone)))
            zone_center_y = int(np.clip(zone_center_y, half_zone, max(half_zone, frame_h - half_zone)))
            zone_x1 = zone_center_x - zone_size_px // 2
            zone_y1 = zone_center_y - zone_size_px // 2
            zone_x2 = zone_center_x + zone_size_px // 2
            zone_y2 = zone_center_y + zone_size_px // 2

            projected_speed_x_mm_per_sec = 0.0
            projected_speed_y_mm_per_sec = 0.0
            projected_speed_magnitude_mm_per_sec = 0.0
            direction_clamped = False
            if model is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if flow_prev_gray is None:
                    flow_prev_gray = gray
                if flow_prev_points is None or len(flow_prev_points) < FLOW_TRACK_MIN_POINTS:
                    flow_prev_points = _pick_optical_flow_points(
                        gray,
                        zone_x1=zone_x1,
                        zone_y1=zone_y1,
                        zone_x2=zone_x2,
                        zone_y2=zone_y2,
                    )
                    flow_prev_gray = gray
                    if last_speed_update_ts is None:
                        last_speed_update_ts = frame_now_ts

                if last_speed_update_ts is None:
                    last_speed_update_ts = frame_now_ts

                if flow_prev_gray is not None and flow_prev_points is not None:
                    elapsed_since_update = float(frame_now_ts - float(last_speed_update_ts))
                    if elapsed_since_update >= FLOW_SPEED_UPDATE_INTERVAL_SEC:
                        raw_vx_mm_per_sec, raw_vy_mm_per_sec, raw_speed_mm_per_sec, next_points = _compute_speed_vector_with_optical_flow(
                            flow_prev_gray,
                            gray,
                            flow_prev_points,
                            mm_per_pixel,
                            elapsed_since_update,
                        )
                        speed_x_mm_per_sec = (
                            (1.0 - FLOW_SPEED_EMA_ALPHA) * float(speed_x_mm_per_sec)
                            + FLOW_SPEED_EMA_ALPHA * float(raw_vx_mm_per_sec)
                        )
                        speed_y_mm_per_sec = (
                            (1.0 - FLOW_SPEED_EMA_ALPHA) * float(speed_y_mm_per_sec)
                            + FLOW_SPEED_EMA_ALPHA * float(raw_vy_mm_per_sec)
                        )
                        speed_magnitude_mm_per_sec = float(np.hypot(speed_x_mm_per_sec, speed_y_mm_per_sec))
                        if not np.isfinite(speed_magnitude_mm_per_sec):
                            speed_magnitude_mm_per_sec = float(raw_speed_mm_per_sec)
                        flow_prev_points = next_points
                        if flow_prev_points is None or len(flow_prev_points) < FLOW_TRACK_MIN_POINTS:
                            flow_prev_points = _pick_optical_flow_points(
                                gray,
                                zone_x1=zone_x1,
                                zone_y1=zone_y1,
                                zone_x2=zone_x2,
                                zone_y2=zone_y2,
                            )
                        flow_prev_gray = gray
                        last_speed_update_ts = frame_now_ts

                projected_speed_x_mm_per_sec = float(speed_x_mm_per_sec)
                projected_speed_y_mm_per_sec = float(speed_y_mm_per_sec)
                projected_speed_magnitude_mm_per_sec = float(speed_magnitude_mm_per_sec)
                direction_clamped = False

                if frame_id % detection_interval == 0:
                    detect_started = time.perf_counter()
                    last_detections = _detect_objects(
                        model,
                        frame,
                        confidence_threshold=confidence_threshold,
                        device=resolved_device,
                        imgsz=max(320, int(detection_imgsz)),
                        use_half=use_half,
                    )
                    detect_ms = (time.perf_counter() - detect_started) * 1000.0

                    detections_payload = []
                    tracked_detections = []
                    half_zone_mm = float(zone_size_mm) / 2.0
                    match_distance_px = float(np.clip(float(zone_size_px) * 0.18, 28.0, TRACK_MATCH_MAX_DISTANCE_PX))
                    _cleanup_stale_tracks(
                        active_tracks=active_tracks,
                        frame_id=frame_id,
                        stale_frame_ttl=TRACK_STALE_FRAME_TTL,
                    )
                    used_track_ids = set()

                    for det in last_detections[:20]:
                        x1 = float(det[0])
                        y1 = float(det[1])
                        x2 = float(det[2])
                        y2 = float(det[3])
                        class_id = int(det[5])
                        confidence = round(float(det[4]), 4)
                        center_x = int(round((x1 + x2) / 2.0))
                        center_y = int(round((y1 + y2) / 2.0))

                        track_id = _match_detection_to_track(
                            center_x=center_x,
                            center_y=center_y,
                            class_id=class_id,
                            active_tracks=active_tracks,
                            used_track_ids=used_track_ids,
                            max_distance_px=match_distance_px,
                        )
                        if track_id is None:
                            track_id = int(next_track_id)
                            next_track_id += 1
                            active_tracks[track_id] = {
                                "class_id": int(class_id),
                                "center_x": int(center_x),
                                "center_y": int(center_y),
                                "confidence": float(confidence),
                                "last_seen_frame": int(frame_id),
                                "processed": False,
                            }
                        else:
                            current = dict(active_tracks.get(track_id, {}))
                            active_tracks[track_id] = {
                                **current,
                                "class_id": int(class_id),
                                "center_x": int(center_x),
                                "center_y": int(center_y),
                                "confidence": float(confidence),
                                "last_seen_frame": int(frame_id),
                                "processed": bool(current.get("processed", False)),
                            }
                        used_track_ids.add(int(track_id))
                        is_processed = bool(active_tracks.get(int(track_id), {}).get("processed", False))

                        predicted_px, predicted_py = _predict_position_px(
                            center_x=center_x,
                            center_y=center_y,
                            speed_x_mm_per_sec=projected_speed_x_mm_per_sec,
                            speed_y_mm_per_sec=projected_speed_y_mm_per_sec,
                            horizon_ms=predict_horizon_ms,
                            mm_per_pixel=mm_per_pixel,
                        )

                        tracked_detections.append(
                            {
                                "track_id": int(track_id),
                                "class_id": int(class_id),
                                "confidence": float(confidence),
                                "center_x": int(center_x),
                                "center_y": int(center_y),
                                "predicted_px": int(predicted_px),
                                "predicted_py": int(predicted_py),
                                "processed": bool(is_processed),
                            }
                        )

                        detection_payload = {
                            "track_id": int(track_id),
                            "processed": bool(is_processed),
                            "class": int(class_id),
                            "confidence": float(confidence),
                            "x": center_x,
                            "y": center_y,
                            "predicted_x": int(predicted_px),
                            "predicted_y": int(predicted_py),
                            "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                            "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                            "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                            "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                            "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                            "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                            "predict_horizon_ms": int(predict_horizon_ms),
                            "speed_direction_clamped": bool(direction_clamped),
                        }
                        if zone_x1 < predicted_px < zone_x2 and zone_y1 < predicted_py < zone_y2:
                            # Delta-robot coordinates are centered at the zone center.
                            rel_x = (predicted_py - zone_center_y) * mm_per_pixel
                            rel_y = -(predicted_px - zone_center_x) * mm_per_pixel
                            detection_payload["x_mm"] = int(round(np.clip(rel_x, -half_zone_mm, half_zone_mm)))
                            detection_payload["y_mm"] = int(round(np.clip(rel_y, -half_zone_mm, half_zone_mm)))
                            detection_payload["command_point_x"] = int(predicted_px)
                            detection_payload["command_point_y"] = int(predicted_py)
                        detections_payload.append(detection_payload)
                    _update_visual_state(
                        {
                            "event": "camera_detection",
                            "frame_id": frame_id,
                            "count": len(last_detections),
                            "detections": detections_payload,
                            "device": resolved_device,
                            "detect_ms": round(detect_ms, 2),
                            "source": "camera",
                            "send_command": bool(send_command),
                            "simulation_mode": not bool(send_command),
                            "zone_offset_x": zone_offset_x,
                            "zone_offset_y": zone_offset_y,
                            "zone_size_px": zone_size_px,
                            "zone_size_mm": round(zone_size_mm, 3),
                            "mm_per_pixel": round(float(mm_per_pixel), 5),
                            "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                            "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                            "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                            "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                            "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                            "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                            "predict_horizon_ms": int(predict_horizon_ms),
                            "command_duration_ms": int(command_duration_ms),
                            "direction_clamped": bool(direction_clamped),
                            "command_inflight": bool(_is_robot_async_inflight()),
                            **_locked_command_point_payload(),
                        }
                    )

                    if tracked_detections:
                        command_candidates = [item for item in tracked_detections if not bool(item.get("processed"))]
                        if command_candidates:
                            target = max(command_candidates, key=lambda item: float(item["confidence"]))
                            target_track_id = int(target["track_id"])
                            center_x = int(target["center_x"])
                            center_y = int(target["center_y"])
                            predicted_px = int(target["predicted_px"])
                            predicted_py = int(target["predicted_py"])
                            relative_x_mm = int(round((predicted_py - zone_center_y) * mm_per_pixel))
                            relative_y_mm = int(round(-(predicted_px - zone_center_x) * mm_per_pixel))
                            inside_zone = bool(zone_x1 < predicted_px < zone_x2 and zone_y1 < predicted_py < zone_y2)
                            if target_track_id in active_tracks:
                                active_tracks[target_track_id]["processed"] = True

                            if inside_zone:
                                if send_command:
                                    queue_result = _queue_robot_command_async(
                                        relative_x_mm=relative_x_mm,
                                        relative_y_mm=relative_y_mm,
                                        duration_ms=command_duration_ms,
                                        context={
                                            "frame_id": int(frame_id),
                                            "track_id": int(target_track_id),
                                            "detection_center_x": int(center_x),
                                            "detection_center_y": int(center_y),
                                            "command_point_x": int(predicted_px),
                                            "command_point_y": int(predicted_py),
                                            "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                                            "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                                            "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                                            "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                                            "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                                            "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                                            "predict_horizon_ms": int(predict_horizon_ms),
                                            "zone_x1": int(zone_x1),
                                            "zone_y1": int(zone_y1),
                                            "zone_x2": int(zone_x2),
                                            "zone_y2": int(zone_y2),
                                        },
                                    )
                                    if queue_result.get("accepted"):
                                        locked_command_point = {
                                            "x": int(predicted_px),
                                            "y": int(predicted_py),
                                            "target_x_mm": int(relative_x_mm),
                                            "target_y_mm": int(relative_y_mm),
                                            "track_id": int(target_track_id),
                                            "command_id": int(queue_result["command_id"]),
                                            "frame_id": int(frame_id),
                                            "expires_at_ts": float(frame_started) + (float(command_duration_ms) * 2.0 / 1000.0),
                                        }
                                        _update_visual_state(
                                            {
                                                "event": "robot_command",
                                                "frame_id": frame_id,
                                                "count": len(last_detections),
                                                "detections": detections_payload,
                                                "device": resolved_device,
                                                "source": "camera",
                                                "command_status": "queued",
                                                "simulated": False,
                                                "command_id": int(queue_result["command_id"]),
                                                "track_id": int(target_track_id),
                                                "target_x_mm": int(relative_x_mm),
                                                "target_y_mm": int(relative_y_mm),
                                                "detection_center_x": int(center_x),
                                                "detection_center_y": int(center_y),
                                                "command_point_x": int(predicted_px),
                                                "command_point_y": int(predicted_py),
                                                "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                                                "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                                                "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                                                "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                                                "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                                                "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                                                "predict_horizon_ms": int(predict_horizon_ms),
                                                "command_duration_ms": int(command_duration_ms),
                                                **_locked_command_point_payload(),
                                            }
                                        )
                                    else:
                                        reject_reason = str(queue_result.get("reason") or "unknown")
                                        _update_visual_state(
                                            {
                                                "event": "robot_command_rejected",
                                                "frame_id": frame_id,
                                                "count": len(last_detections),
                                                "detections": detections_payload,
                                                "device": resolved_device,
                                                "source": "camera",
                                                "reason": reject_reason,
                                                "simulated": False,
                                                "track_id": int(target_track_id),
                                                "target_x_mm": int(relative_x_mm),
                                                "target_y_mm": int(relative_y_mm),
                                                "detection_center_x": int(center_x),
                                                "detection_center_y": int(center_y),
                                                "command_point_x": int(predicted_px),
                                                "command_point_y": int(predicted_py),
                                                "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                                                "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                                                "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                                                "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                                                "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                                                "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                                                "predict_horizon_ms": int(predict_horizon_ms),
                                                "command_duration_ms": int(command_duration_ms),
                                                **_locked_command_point_payload(),
                                            }
                                        )
                                        log.info(
                                            "Robot command rejected (reason=%s): frame=%s track=%s det=(%s,%s) pred=(%s,%s) target=(%s,%s)mm",
                                            reject_reason,
                                            frame_id,
                                            target_track_id,
                                            center_x,
                                            center_y,
                                            predicted_px,
                                            predicted_py,
                                            relative_x_mm,
                                            relative_y_mm,
                                        )
                                else:
                                    simulated_command_id += 1
                                    locked_command_point = {
                                        "x": int(predicted_px),
                                        "y": int(predicted_py),
                                        "target_x_mm": int(relative_x_mm),
                                        "target_y_mm": int(relative_y_mm),
                                        "track_id": int(target_track_id),
                                        "command_id": int(simulated_command_id),
                                        "frame_id": int(frame_id),
                                        "expires_at_ts": float(frame_started) + (float(command_duration_ms) * 2.0 / 1000.0),
                                    }
                                    _update_visual_state(
                                        {
                                            "event": "robot_command",
                                            "frame_id": frame_id,
                                            "count": len(last_detections),
                                            "detections": detections_payload,
                                            "device": resolved_device,
                                            "source": "camera",
                                            "command_status": "simulated",
                                            "simulated": True,
                                            "command_id": int(simulated_command_id),
                                            "track_id": int(target_track_id),
                                            "target_x_mm": int(relative_x_mm),
                                            "target_y_mm": int(relative_y_mm),
                                            "detection_center_x": int(center_x),
                                            "detection_center_y": int(center_y),
                                            "command_point_x": int(predicted_px),
                                            "command_point_y": int(predicted_py),
                                            "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                                            "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                                            "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                                            "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                                            "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                                            "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                                            "predict_horizon_ms": int(predict_horizon_ms),
                                            "command_duration_ms": int(command_duration_ms),
                                            **_locked_command_point_payload(),
                                        }
                                    )
                                    log.info(
                                        "Robot command simulated: frame=%s track=%s det=(%s,%s) pred=(%s,%s) target=(%s,%s)mm",
                                        frame_id,
                                        target_track_id,
                                        center_x,
                                        center_y,
                                        predicted_px,
                                        predicted_py,
                                        relative_x_mm,
                                        relative_y_mm,
                                    )
                            else:
                                _update_visual_state(
                                    {
                                        "event": "robot_command_rejected",
                                        "frame_id": frame_id,
                                        "count": len(last_detections),
                                        "detections": detections_payload,
                                        "device": resolved_device,
                                        "source": "camera",
                                        "reason": "outside_zone",
                                        "simulated": not bool(send_command),
                                        "track_id": int(target_track_id),
                                        "target_x_mm": int(relative_x_mm),
                                        "target_y_mm": int(relative_y_mm),
                                        "detection_center_x": int(center_x),
                                        "detection_center_y": int(center_y),
                                        "command_point_x": int(predicted_px),
                                        "command_point_y": int(predicted_py),
                                        "zone_x1": int(zone_x1),
                                        "zone_y1": int(zone_y1),
                                        "zone_x2": int(zone_x2),
                                        "zone_y2": int(zone_y2),
                                        "zone_center_x": int(zone_center_x),
                                        "zone_center_y": int(zone_center_y),
                                        "mm_per_pixel": round(float(mm_per_pixel), 5),
                                        "platform_speed_x_mm_per_sec": round(float(projected_speed_x_mm_per_sec), 3),
                                        "platform_speed_y_mm_per_sec": round(float(projected_speed_y_mm_per_sec), 3),
                                        "platform_speed_magnitude_mm_per_sec": round(float(projected_speed_magnitude_mm_per_sec), 3),
                                        "platform_speed_x_m_per_sec": round(float(projected_speed_x_mm_per_sec) / 1000.0, 4),
                                        "platform_speed_y_m_per_sec": round(float(projected_speed_y_mm_per_sec) / 1000.0, 4),
                                        "platform_speed_m_per_sec": round(float(projected_speed_magnitude_mm_per_sec) / 1000.0, 4),
                                        "predict_horizon_ms": int(predict_horizon_ms),
                                        "command_duration_ms": int(command_duration_ms),
                                        "speed_direction_clamped": bool(direction_clamped),
                                        **_locked_command_point_payload(),
                                    }
                                )
                                log.info(
                                    "Robot command rejected (outside zone): frame=%s track=%s det=(%s,%s) pred=(%s,%s) "
                                    "zone=[(%s,%s)-(%s,%s)] speed=(%.4f,%.4f)|%.4f m/s horizon=%sms mm_per_pixel=%.5f target=(%s,%s)mm",
                                    frame_id,
                                    target_track_id,
                                    center_x,
                                    center_y,
                                    predicted_px,
                                    predicted_py,
                                    zone_x1,
                                    zone_y1,
                                    zone_x2,
                                    zone_y2,
                                    projected_speed_x_mm_per_sec / 1000.0,
                                    projected_speed_y_mm_per_sec / 1000.0,
                                    projected_speed_magnitude_mm_per_sec / 1000.0,
                                    predict_horizon_ms,
                                    mm_per_pixel,
                                    relative_x_mm,
                                    relative_y_mm,
                                )

                frame = _draw_detection_overlay(
                    frame,
                    last_detections,
                    float(projected_speed_magnitude_mm_per_sec),
                    mm_per_pixel,
                    zone_x1,
                    zone_y1,
                    zone_x2,
                    zone_y2,
                    speed_x_mm_per_sec=float(projected_speed_x_mm_per_sec),
                    speed_y_mm_per_sec=float(projected_speed_y_mm_per_sec),
                    prediction_horizon_ms=float(predict_horizon_ms),
                )
            elif model_error:
                cv2.putText(
                    frame,
                    f"Detection disabled: {model_error}",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            frame = _draw_locked_command_point(frame, locked_command_point)

            frame_ms = (time.perf_counter() - frame_started) * 1000.0
            frame_fps = 1000.0 / max(frame_ms, 1e-6)
            mode_text = "Command mode: ON" if send_command else "Command mode: OFF"
            queue_text = "BUSY" if _is_robot_async_inflight() else "READY"
            speed_text = (
                f"V: {projected_speed_magnitude_mm_per_sec / 1000.0:.3f} m/s "
                f"(Vx:{projected_speed_x_mm_per_sec / 1000.0:.3f}, Vy:{projected_speed_y_mm_per_sec / 1000.0:.3f}) | "
                f"horizon: {predict_horizon_ms} ms | queue: {queue_text}"
            )
            cv2.putText(frame, f"YOLO device: {resolved_device}", (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, mode_text, (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {frame_fps:.1f}", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, speed_text, (24, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 220, 120), 2)
            cv2.putText(
                frame,
                f"Zone offset px: ({zone_offset_x:+d}, {zone_offset_y:+d})",
                (24, 142),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Zone size: {zone_size_px}px ({zone_size_mm:.1f}mm)",
                (24, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if max_width > 0 and frame.shape[1] > max_width:
                resized_h = int(frame.shape[0] * max_width / frame.shape[1])
                frame = cv2.resize(frame, (max_width, resized_h), interpolation=cv2.INTER_AREA)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, jpeg_quality))])
            if not ok:
                continue

            payload = jpg.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
    except GeneratorExit:
        return
    except Exception as exc:
        log.error("Processed camera stream failed: %s", exc)


@app.on_event("shutdown")
def shutdown_event():
    _request_stop_and_release_resources()


app.include_router(
    build_legacy_ui_router(
        request_stop_and_release_resources=_request_stop_and_release_resources,
        gen_mjpeg=gen_mjpeg,
        get_zone_config=_get_zone_config,
        move_zone_offset=_move_zone_offset,
        set_zone_config=_set_zone_config,
        update_visual_state=_update_visual_state,
        get_robot_settings=_get_robot_settings,
        set_robot_settings=_set_robot_settings,
        close_move_module_client=_close_move_module_client,
        resolve_detection_conf=_resolve_detection_conf,
        resolve_command_duration_ms=_resolve_command_duration_ms,
        gen_processed_mjpeg=gen_processed_mjpeg,
        get_visual_state_snapshot=lambda: dict(visual_state),
        gen_visualization_mjpeg=gen_visualization_mjpeg,
        build_default_video_path=build_default_video_path,
    )
)

app.include_router(
    build_legacy_limitswitch_router(_build_robot_api_runtime())
)

app.include_router(
    build_v1_router(_build_robot_api_runtime())
)


@app.get("/api/v1/system/settings/robot")
def get_v1_robot_settings():
    return _get_robot_settings()


@app.post("/api/v1/system/settings/robot")
def set_v1_robot_settings(payload: Dict[str, Any] = Body(default={})):
    raw = payload or {}
    cfg = _set_robot_settings(
        xy_rotation_deg=raw.get("xy_rotation_deg"),
        invert_x=raw.get("invert_x"),
        invert_y=raw.get("invert_y"),
        invert_z=raw.get("invert_z"),
        detection_conf=raw.get("detection_conf"),
    )
    _close_move_module_client()
    return cfg


@app.post("/api/v1/platform-turn/target-angle")
def v1_platform_turn_target_angle(payload: Dict[str, Any] = Body(default={})):
    raw = payload or {}

    raw_target_angle = raw.get("target_angle_deg")
    if raw_target_angle is None:
        raw_target_angle = raw.get("angle_deg")
    if raw_target_angle is None:
        raw_target_angle = raw.get("angle")
    if raw_target_angle is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": "target_angle_deg is required."},
        )

    try:
        target_angle = float(raw_target_angle)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": "target_angle_deg must be numeric."},
        ) from exc

    crab_mode_raw = raw.get("crab_mode")
    if isinstance(crab_mode_raw, bool):
        crab_mode = crab_mode_raw
    elif crab_mode_raw is None:
        crab_mode = False
    elif isinstance(crab_mode_raw, (int, float)):
        crab_mode = bool(crab_mode_raw)
    else:
        crab_mode = str(crab_mode_raw).strip().lower() in {"1", "true", "yes", "on"}

    if crab_mode:
        wheel_1_deg = float(target_angle)
        wheel_2_deg = float(target_angle)
        wheel_3_deg = float(target_angle)
        wheel_4_deg = float(target_angle)
    else:
        wheel_1_deg = float(target_angle)
        wheel_2_deg = float(target_angle)
        wheel_3_deg = float(-target_angle)
        wheel_4_deg = float(-target_angle)

    speed_profile = str(raw.get("speed_profile") or "normal")

    try:
        result = _build_service_registry().platform_turn.move_wheels_absolute(
            wheel_1_deg=wheel_1_deg,
            wheel_2_deg=wheel_2_deg,
            wheel_3_deg=wheel_3_deg,
            wheel_4_deg=wheel_4_deg,
            speed_profile=speed_profile,
        )
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    data = dict(result.get("data") or {})
    data["requested_target_angle_deg"] = float(target_angle)
    data["crab_mode"] = bool(crab_mode)
    data["requested_wheels"] = [
        {"index": 1, "target_angle_deg": float(wheel_1_deg)},
        {"index": 2, "target_angle_deg": float(wheel_2_deg)},
        {"index": 3, "target_angle_deg": float(wheel_3_deg)},
        {"index": 4, "target_angle_deg": float(wheel_4_deg)},
    ]
    return {
        **dict(result or {}),
        "data": data,
    }


@app.get("/api/v1/vision/video")
def v1_video_stream():
    return StreamingResponse(gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/v1/vision/video/stop")
def v1_stop_video_stream():
    _request_stop_and_release_resources()
    return {"stopped": True}


@app.get("/api/v1/vision/processed-video")
def v1_processed_video(
    send_command: bool = True,
    detect: bool = True,
    detection_interval: int = 1,
    conf: Optional[float] = None,
    command_duration_ms: int = 12000,
    jpeg_quality: int = 72,
    max_width: int = 960,
    detection_imgsz: int = 640,
    device: Optional[str] = None,
    half: bool = True,
):
    resolved_conf = _resolve_detection_conf(conf)
    resolved_duration = _resolve_command_duration_ms(command_duration_ms)
    return StreamingResponse(
        gen_processed_mjpeg(
            send_command=send_command,
            enable_detection=detect,
            detection_interval=max(1, detection_interval),
            confidence_threshold=resolved_conf,
            command_duration_ms=resolved_duration,
            jpeg_quality=max(40, min(95, jpeg_quality)),
            max_width=max(480, max_width),
            detection_imgsz=max(320, min(1280, detection_imgsz)),
            model_device=device,
            use_half=half,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/vision/visualize-video")
def v1_visualize_video(
    source: Optional[str] = None,
    detect: bool = True,
    detection_interval: int = 4,
    conf: Optional[float] = None,
    jpeg_quality: int = 75,
    max_width: int = 1280,
    detection_imgsz: int = 512,
    device: Optional[str] = None,
    half: bool = True,
):
    resolved_conf = _resolve_detection_conf(conf)
    video_source = source or build_default_video_path()
    return StreamingResponse(
        gen_visualization_mjpeg(
            video_path=video_source,
            loop=True,
            enable_detection=detect,
            detection_interval=max(1, detection_interval),
            confidence_threshold=resolved_conf,
            jpeg_quality=max(40, min(95, jpeg_quality)),
            max_width=max(480, max_width),
            detection_imgsz=max(320, min(1280, detection_imgsz)),
            model_device=device,
            use_half=half,
            weights_path="yolov11large.pt",
            on_detections=_update_visual_state,
            zone_config_provider=_get_zone_config,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/connection/ports")
def get_connection_ports():
    try:
        return _build_service_registry().transport.get_ports()
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@app.post("/connection/port")
def set_connection_port(port: str):
    try:
        return _build_service_registry().transport.set_port(port)
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@app.get("/read/{slave_id}")
def read_status(slave_id: int):
    try:
        return _build_service_registry().legacy_slave.read_registers(slave_id=slave_id, start_addr=0, count=31)
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@app.post("/calibrate/{slave_id}")
def calibrate(slave_id: int):
    try:
        return _build_service_registry().legacy_slave.calibrate(slave_id=slave_id)
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@app.post("/go-work/{slave_id}")
def go_work(slave_id: int):
    try:
        return _build_service_registry().legacy_slave.go_work(slave_id=slave_id)
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@app.get("/robot/manual/state")
def robot_manual_state():
    try:
        return _build_service_registry().legacy_slave.manual_state()
    except DeviceCommandError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


@app.post("/robot/manual/move")
def robot_manual_move(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    duration_ms: int = ROBOT_MOVE_DURATION_MS,
):
    try:
        return _build_service_registry().legacy_slave.manual_move(
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            duration_ms=duration_ms,
        )
    except ValueError as exc:
        if str(exc) == "calibration_required":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "calibration_required",
                    "message": "Go work calibration is required before manual move.",
                },
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeviceCommandError as exc:
        if exc.code == "calibration_required":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "calibration_required",
                    "message": exc.message,
                },
            ) from exc
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}
