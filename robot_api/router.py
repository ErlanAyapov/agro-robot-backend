import asyncio
import json

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .runtime import RobotApiRuntime
from .services import DeviceCommandError, ServiceRegistry


class PortSelectionRequest(BaseModel):
    port: str = Field(..., min_length=1)


class WidthTargetRequest(BaseModel):
    target_position_cm: float
    rpm: int = 60


class TurnTargetRequest(BaseModel):
    wheel_1_deg: float
    wheel_2_deg: float
    wheel_3_deg: float
    wheel_4_deg: float
    speed_profile: str = "normal"


class DeltaMoveRequest(BaseModel):
    x_mm: float
    y_mm: float
    z_mm: float
    duration_ms: int = 12000


class DeltaProcessTargetRequest(BaseModel):
    target_x_mm: float
    target_y_mm: float
    source: str = "vision"
    duration_ms: int = 900


def build_v1_router(runtime: RobotApiRuntime) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["api-v1"])
    services = ServiceRegistry(runtime)

    def _error_response(
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict = None,
    ) -> JSONResponse:
        payload = {
            "error": {
                "code": str(code or "device_fault"),
                "message": str(message or "Request failed"),
                "details": details or {},
            }
        }
        return JSONResponse(status_code=int(status_code), content=payload)

    def _handle_device_error(exc: DeviceCommandError) -> JSONResponse:
        return _error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details={},
        )

    def _parse_float(value, field_name: str):
        try:
            return float(value)
        except Exception:
            return _error_response(
                status_code=400,
                code="validation_error",
                message=f"{field_name} must be numeric.",
                details={"field": field_name},
            )

    def _parse_int(value, field_name: str):
        try:
            return int(value)
        except Exception:
            return _error_response(
                status_code=400,
                code="validation_error",
                message=f"{field_name} must be integer.",
                details={"field": field_name},
            )

    def _rpm_from_speed_profile(speed_profile: str, default_rpm: int = 60) -> int:
        profile = str(speed_profile or "normal").strip().lower()
        mapping = {
            "slow": 35,
            "normal": int(default_rpm),
            "fast": 80,
        }
        return int(mapping.get(profile, default_rpm))

    @router.get("/system/ports")
    def get_ports():
        try:
            return services.transport.get_ports()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/system/port")
    def set_port(payload: PortSelectionRequest):
        try:
            return services.transport.set_port(payload.port)
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/system/devices")
    def get_devices():
        try:
            return services.transport.get_devices()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/robot/state")
    def get_robot_state():
        return services.robot.get_robot_state()

    @router.get("/platform-width/state")
    def get_platform_width_state():
        try:
            return services.platform_width.get_state()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-width/home")
    def home_platform_width(payload: dict = Body(default={})):
        data = payload or {}
        rpm_raw = data.get("rpm", _rpm_from_speed_profile(str(data.get("speed_profile") or "slow"), default_rpm=40))
        rpm = _parse_int(rpm_raw, "rpm")
        if isinstance(rpm, JSONResponse):
            return rpm
        try:
            return services.platform_width.home(rpm=rpm)
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-width/target")
    def set_platform_width_target(payload: dict = Body(default={})):
        data = payload or {}
        target_raw = data.get("target_width_cm", data.get("target_position_cm"))
        if target_raw is None:
            return _error_response(
                status_code=400,
                code="validation_error",
                message="target_width_cm is required.",
                details={"field": "target_width_cm"},
            )
        target_position = _parse_float(target_raw, "target_width_cm")
        if isinstance(target_position, JSONResponse):
            return target_position

        speed_profile = str(data.get("speed_profile") or "normal")
        rpm_raw = data.get("rpm", _rpm_from_speed_profile(speed_profile))
        rpm = _parse_int(rpm_raw, "rpm")
        if isinstance(rpm, JSONResponse):
            return rpm

        try:
            return services.platform_width.move_to_position(
                target_position_cm=target_position,
                rpm=rpm,
                speed_profile=speed_profile,
            )
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-width/stop")
    def stop_platform_width():
        try:
            return services.platform_width.stop()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/platform-turn/state")
    def get_platform_turn_state():
        try:
            return services.platform_turn.get_state()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-turn/calibrate")
    def calibrate_platform_turn():
        try:
            return services.platform_turn.calibrate()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-turn/align-zero")
    def align_platform_turn_zero():
        try:
            return services.platform_turn.align_zero()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-turn/target")
    def set_platform_turn_target(payload: TurnTargetRequest):
        try:
            return services.platform_turn.move_wheels_absolute(
                wheel_1_deg=payload.wheel_1_deg,
                wheel_2_deg=payload.wheel_2_deg,
                wheel_3_deg=payload.wheel_3_deg,
                wheel_4_deg=payload.wheel_4_deg,
                speed_profile=payload.speed_profile,
            )
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/platform-turn/stop")
    def stop_platform_turn():
        try:
            return services.platform_turn.stop()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/delta/state")
    def get_delta_state():
        try:
            return services.delta.get_state()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/delta/home")
    def home_delta():
        try:
            return services.delta.home()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/delta/move")
    def move_delta(payload: DeltaMoveRequest):
        try:
            return services.delta.move(
                x_mm=payload.x_mm,
                y_mm=payload.y_mm,
                z_mm=payload.z_mm,
                duration_ms=payload.duration_ms,
            )
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/delta/process-target")
    def process_delta_target(payload: DeltaProcessTargetRequest):
        try:
            return services.delta.process_target(
                target_x_mm=payload.target_x_mm,
                target_y_mm=payload.target_y_mm,
                duration_ms=payload.duration_ms,
                source=payload.source,
            )
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/delta/stop")
    def stop_delta():
        try:
            return services.delta.stop()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/vision/start")
    def start_vision():
        try:
            return services.vision.start()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/vision/stop")
    def stop_vision():
        try:
            return services.vision.stop()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/vision/state")
    def get_vision_state():
        try:
            return services.vision.get_state()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/vision/detections")
    def get_vision_detections():
        try:
            return services.vision.get_detections()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.get("/vision/zone")
    def get_vision_zone():
        try:
            return services.vision.get_zone()
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.post("/vision/zone")
    def set_vision_zone(payload: dict = Body(default={})):
        data = payload or {}
        offset_x = data.get("offset_x_px", data.get("x"))
        offset_y = data.get("offset_y_px", data.get("y"))
        size_px = data.get("size_px")
        size_mm = data.get("size_mm")

        if offset_x is not None:
            offset_x = _parse_int(offset_x, "offset_x_px")
            if isinstance(offset_x, JSONResponse):
                return offset_x
        if offset_y is not None:
            offset_y = _parse_int(offset_y, "offset_y_px")
            if isinstance(offset_y, JSONResponse):
                return offset_y
        if size_px is not None:
            size_px = _parse_int(size_px, "size_px")
            if isinstance(size_px, JSONResponse):
                return size_px
        if size_mm is not None:
            size_mm = _parse_float(size_mm, "size_mm")
            if isinstance(size_mm, JSONResponse):
                return size_mm

        try:
            return services.vision.set_zone(
                offset_x_px=offset_x,
                offset_y_px=offset_y,
                size_px=size_px,
                size_mm=size_mm,
            )
        except DeviceCommandError as exc:
            return _handle_device_error(exc)

    @router.websocket("/ws")
    async def api_v1_ws(websocket: WebSocket):
        await websocket.accept()
        poll_interval_sec = 1.0
        cursor = 0
        last_state_signature = ""
        try:
            while True:
                events = []
                pull_events = getattr(runtime, "pull_events_since", None)
                if callable(pull_events):
                    try:
                        packet = dict(pull_events(cursor=cursor, limit=100) or {})
                    except TypeError:
                        packet = dict(pull_events(cursor, 100) or {})
                    events = list(packet.get("events") or [])

                for item in events:
                    await websocket.send_json(item)
                if events:
                    cursor = int(events[-1].get("cursor", cursor))

                robot_state = services.robot.get_robot_state()
                robot_event = {
                    "event": "robot.state",
                    "timestamp": str(robot_state.get("timestamp") or ""),
                    "state": robot_state,
                }
                signature = json.dumps(robot_event, sort_keys=True, ensure_ascii=False)
                if signature != last_state_signature:
                    await websocket.send_json(robot_event)
                    last_state_signature = signature

                await asyncio.sleep(poll_interval_sec)
        except WebSocketDisconnect:
            return

    return router


def build_legacy_limitswitch_router(runtime: RobotApiRuntime) -> APIRouter:
    router = APIRouter(prefix="/limitswitch", tags=["limitswitch"])
    services = ServiceRegistry(runtime)

    def _bool_value(value, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _selected_port() -> str:
        return str(runtime.get_active_port() or "").strip()

    def _with_selected_port(payload: dict) -> dict:
        return {"selected_port": _selected_port(), **payload}

    def _handle_device_error(exc: DeviceCommandError) -> None:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": exc.code,
                "message": exc.message,
            },
        ) from exc

    @router.get("/position")
    def api_current_position():
        try:
            state = services.platform_width.get_state()
            return _with_selected_port(
                {
                    "status": "success",
                    "position": float(state.get("current_position_cm", 0.0)),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.get("/linear/state")
    def api_linear_state():
        try:
            state = services.platform_width.get_state()
            return _with_selected_port(
                {
                    "status": "success",
                    "current_position": float(state.get("current_position_cm", 0.0)),
                    "target_position": float(state.get("target_position_cm", 0.0)),
                    "min_position": float(state.get("min_position_cm", 0.0)),
                    "max_position": float(state.get("max_position_cm", 0.0)),
                    "width_change_allowed": bool(state.get("width_change_allowed", False)),
                    "blocked_reason": state.get("blocked_reason"),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.post("/linear/target")
    def api_linear_target(payload: dict = Body(default={})):
        data = payload or {}
        try:
            target_position = float(data.get("target_position", 0))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="target_position must be numeric") from exc

        try:
            rpm = int(data.get("rpm", 60))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="rpm must be numeric") from exc

        try:
            result = services.platform_width.move_to_position(target_position_cm=target_position, rpm=rpm)
            info = dict(result.get("data") or {})
            return _with_selected_port(
                {
                    "status": "success",
                    "message": "Linear move started",
                    "current_position": float(info.get("current_position_cm", target_position)),
                    "target_position": float(info.get("target_position_cm", target_position)),
                    "delta_position": float(info.get("delta_position_cm", 0.0)),
                    "rpm": int(info.get("rpm", rpm)),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.post("/move")
    def api_move(payload: dict = Body(default={})):
        data = payload or {}
        direction = data.get("direction")
        if direction in (0, 1):
            parsed_direction = int(direction)
        else:
            normalized = str(direction or "").strip().lower()
            if normalized in {"1", "up", "forward", "open"}:
                parsed_direction = 1
            elif normalized in {"0", "down", "back", "close"}:
                parsed_direction = 0
            else:
                raise HTTPException(status_code=400, detail="direction must be 0/1 or up/down")

        motors = {
            "motor1": 1 if _bool_value(data.get("motor1"), True) else 0,
            "motor2": 1 if _bool_value(data.get("motor2"), True) else 0,
            "motor3": 1 if _bool_value(data.get("motor3"), True) else 0,
            "motor4": 1 if _bool_value(data.get("motor4"), True) else 0,
        }
        if sum(motors.values()) == 0:
            raise HTTPException(status_code=400, detail="At least one motor must be enabled")

        try:
            result = services.platform_width.jog(
                direction=parsed_direction,
                motor1=motors["motor1"],
                motor2=motors["motor2"],
                motor3=motors["motor3"],
                motor4=motors["motor4"],
            )
            info = dict(result.get("data") or {})
            return _with_selected_port(
                {
                    "status": "success",
                    "message": str(info.get("message") or "Individual move started"),
                    "steps": int(info.get("steps", 0)),
                    "direction": int(info.get("direction", parsed_direction)),
                    "motors": list(info.get("motors") or []),
                    "position": float(info.get("position_cm", 0.0)),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.post("/stop")
    def api_stop():
        try:
            result = services.platform_width.stop()
            info = dict(result.get("data") or {})
            return _with_selected_port(
                {
                    "status": "success",
                    "message": str(info.get("message") or "All motors stopped"),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.post("/turn/calibrate")
    def api_turn_calibrate():
        try:
            services.platform_turn.calibrate()
            state = services.platform_turn.get_state()
            return _with_selected_port(
                {
                    "status": "success",
                    "message": "Turn calibration started",
                    "current_angle": float(state.get("current_angle_deg", 0.0)),
                    "target_angle": float(state.get("target_angle_deg", 0.0)),
                    "delta_angle": float(state.get("target_angle_deg", 0.0) - state.get("current_angle_deg", 0.0)),
                    "calibrated": bool(state.get("calibrated", False)),
                    "wheel_state_known": bool(state.get("wheel_state_known", False)),
                    "all_wheels_zero": bool(state.get("all_wheels_zero", False)),
                    "all_wheels_width_aligned": bool(state.get("all_wheels_width_aligned", False)),
                    "width_alignment_deg": float(state.get("width_alignment_deg", 90.0)),
                    "wheels": list(state.get("wheels") or []),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.get("/turn/state")
    def api_turn_state():
        try:
            state = services.platform_turn.get_state()
            return _with_selected_port(
                {
                    "status": "success",
                    "current_angle": float(state.get("current_angle_deg", 0.0)),
                    "target_angle": float(state.get("target_angle_deg", 0.0)),
                    "delta_angle": float(state.get("target_angle_deg", 0.0) - state.get("current_angle_deg", 0.0)),
                    "min_angle": float(state.get("min_angle_deg", 0.0)),
                    "max_angle": float(state.get("max_angle_deg", 0.0)),
                    "calibrated": bool(state.get("calibrated", False)),
                    "zero_reference_valid": bool(state.get("zero_reference_valid", False)),
                    "wheel_state_known": bool(state.get("wheel_state_known", False)),
                    "all_wheels_zero": bool(state.get("all_wheels_zero", False)),
                    "all_wheels_width_aligned": bool(state.get("all_wheels_width_aligned", False)),
                    "width_alignment_deg": float(state.get("width_alignment_deg", 90.0)),
                    "wheels": list(state.get("wheels") or []),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    @router.post("/turn/angle")
    def api_turn_angle(payload: dict = Body(default={})):
        data = payload or {}
        try:
            angle = float(data.get("angle", 0))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="angle must be numeric") from exc

        try:
            services.platform_turn.move_to_angle(angle_deg=angle)
            state = services.platform_turn.get_state()
            return _with_selected_port(
                {
                    "status": "success",
                    "message": "Turn command sent",
                    "current_angle": float(state.get("current_angle_deg", 0.0)),
                    "target_angle": float(state.get("target_angle_deg", 0.0)),
                    "delta_angle": float(state.get("target_angle_deg", 0.0) - state.get("current_angle_deg", 0.0)),
                    "min_angle": float(state.get("min_angle_deg", 0.0)),
                    "max_angle": float(state.get("max_angle_deg", 0.0)),
                    "calibrated": bool(state.get("calibrated", False)),
                    "wheel_state_known": bool(state.get("wheel_state_known", False)),
                    "all_wheels_zero": bool(state.get("all_wheels_zero", False)),
                    "all_wheels_width_aligned": bool(state.get("all_wheels_width_aligned", False)),
                    "width_alignment_deg": float(state.get("width_alignment_deg", 90.0)),
                    "wheels": list(state.get("wheels") or []),
                }
            )
        except DeviceCommandError as exc:
            _handle_device_error(exc)

    return router
