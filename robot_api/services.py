from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import count
from typing import Any, Callable, Dict, Optional

import utils_limitswitch_rotate as motion

from .runtime import RobotApiRuntime


COMMAND_CODES_WITH_CONFLICT = {
    "turn_calibration_required",
    "wheel_state_unknown",
    "wheel_zero_required",
    "wheel_alignment_required",
    "command_rejected",
    "calibration_required",
    "safety_interlock",
}
_command_seq = count(1)


class DeviceCommandError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503):
        super().__init__(message)
        self.code = str(code or "device_error")
        self.message = str(message or "Device request failed")
        self.status_code = int(status_code)


def _next_command_id() -> int:
    return int(next(_command_seq))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _publish_runtime_event(runtime: RobotApiRuntime, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    publisher = getattr(runtime, "publish_event", None)
    if not callable(publisher):
        return
    event_payload = dict(payload or {})
    try:
        publisher(event=event, payload=event_payload)
    except TypeError:
        try:
            publisher(event, event_payload)
        except Exception:
            pass
    except Exception:
        pass


def _status_code_for_result(result: Dict[str, Any]) -> int:
    code = str(result.get("code") or "").strip().lower()
    if code in COMMAND_CODES_WITH_CONFLICT:
        return 409
    return 503


def _ensure_result_ok(result: Dict[str, Any]) -> Dict[str, Any]:
    if str(result.get("status") or "").lower() == "error":
        raise DeviceCommandError(
            code=str(result.get("code") or "device_error"),
            message=str(result.get("message") or "Device request failed"),
            status_code=_status_code_for_result(result),
        )
    return result


def _publish_command_event(
    runtime: RobotApiRuntime,
    *,
    event: str,
    command_id: int,
    device: str,
    command_type: str,
    **extra: Any,
) -> None:
    payload = {
        "timestamp": _utc_timestamp(),
        "command_id": int(command_id),
        "device": str(device),
        "type": str(command_type),
        **extra,
    }
    _publish_runtime_event(runtime, event, payload)


def _run_command_with_events(
    runtime: RobotApiRuntime,
    *,
    device: str,
    command_type: str,
    action: Callable[[], Dict[str, Any]],
    command_id: Optional[int] = None,
) -> tuple[int, Dict[str, Any]]:
    cmd_id = int(command_id) if command_id is not None else _next_command_id()
    _publish_command_event(
        runtime,
        event="command.accepted",
        command_id=cmd_id,
        device=device,
        command_type=command_type,
    )
    _publish_command_event(
        runtime,
        event="command.started",
        command_id=cmd_id,
        device=device,
        command_type=command_type,
    )

    try:
        result = dict(action() or {})
    except DeviceCommandError as exc:
        _publish_command_event(
            runtime,
            event="command.failed",
            command_id=cmd_id,
            device=device,
            command_type=command_type,
            code=exc.code,
            message=exc.message,
        )
        _publish_runtime_event(
            runtime,
            "fault.raised",
            {
                "timestamp": _utc_timestamp(),
                "device": str(device),
                "code": str(exc.code or "device_fault"),
                "message": str(exc.message or "Device request failed"),
            },
        )
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        _publish_command_event(
            runtime,
            event="command.failed",
            command_id=cmd_id,
            device=device,
            command_type=command_type,
            code="device_fault",
            message=str(exc),
        )
        _publish_runtime_event(
            runtime,
            "fault.raised",
            {
                "timestamp": _utc_timestamp(),
                "device": str(device),
                "code": "device_fault",
                "message": str(exc),
            },
        )
        raise

    _publish_command_event(
        runtime,
        event="command.completed",
        command_id=cmd_id,
        device=device,
        command_type=command_type,
        result="done",
    )
    _publish_runtime_event(
        runtime,
        "fault.cleared",
        {
            "timestamp": _utc_timestamp(),
            "device": str(device),
        },
    )
    return cmd_id, result


class BaseSerialDeviceService:
    def __init__(self, runtime: RobotApiRuntime):
        self.runtime = runtime

    def _selected_port(self) -> str:
        port = str(self.runtime.get_active_port() or "").strip()
        if not port:
            raise DeviceCommandError("transport_unavailable", "No active serial port selected.", 400)
        return port

    def _run_serial(self, action: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        port = self._selected_port()
        lock_ctx = self.runtime.serial_lock if self.runtime.serial_lock is not None else nullcontext()
        with lock_ctx:
            if not bool(self.runtime.is_robot_busy()):
                self.runtime.close_move_module_client()
            result = motion.run_with_connection(port, action)
        return _ensure_result_ok(result)

    def _publish_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        _publish_runtime_event(self.runtime, event, payload)


class TransportService:
    def __init__(self, runtime: RobotApiRuntime):
        self.runtime = runtime

    def get_ports(self) -> Dict[str, Any]:
        return {
            "ports": self.runtime.list_serial_ports(),
            "selected_port": self.runtime.get_active_port(),
        }

    def get_devices(self) -> Dict[str, Any]:
        bus_layout = motion.get_bus_layout()
        controller_by_name = {
            "delta": motion.firmware_controller_name("delta"),
            "platform_turn": motion.firmware_controller_name("platform_turn"),
            "platform_width": motion.firmware_controller_name("platform_width"),
        }
        provider = getattr(self.runtime, "get_system_devices", None)
        if callable(provider):
            if not bool(self.runtime.is_robot_busy()):
                self.runtime.close_move_module_client()
            devices = list(provider() or [])
            normalized = []
            for item in devices:
                row = dict(item or {})
                name = str(row.get("name") or "")
                if name:
                    row.setdefault("firmware_controller", controller_by_name.get(name, motion.firmware_controller_name(name)))
                normalized.append(row)
            return {"devices": normalized, "bus_layout": bus_layout}

        selected_port = str(self.runtime.get_active_port() or "").strip()
        status = "unknown" if selected_port else "transport_unavailable"
        return {
            "devices": [
                {
                    "name": "delta",
                    "firmware_controller": controller_by_name["delta"],
                    "slave_id": int(bus_layout.get("delta", 1)),
                    "status": status,
                    "protocol_version": None,
                    "firmware_version": None,
                },
                {
                    "name": "platform_turn",
                    "firmware_controller": controller_by_name["platform_turn"],
                    "slave_id": int(bus_layout.get("platform_turn", 2)),
                    "status": status,
                    "protocol_version": None,
                    "firmware_version": None,
                },
                {
                    "name": "platform_width",
                    "firmware_controller": controller_by_name["platform_width"],
                    "slave_id": int(bus_layout.get("platform_width", 3)),
                    "status": status,
                    "protocol_version": None,
                    "firmware_version": None,
                },
            ],
            "bus_layout": bus_layout,
        }

    def set_port(self, port: str) -> Dict[str, Any]:
        normalized = str(port or "").strip()
        if not normalized:
            raise DeviceCommandError("validation_error", "Port name is required.", 400)

        lock_ctx = self.runtime.serial_lock if self.runtime.serial_lock is not None else nullcontext()
        with lock_ctx:
            self.runtime.close_move_module_client()
            self.runtime.close_limit_switch_client()
            selected = self.runtime.set_active_port(normalized)
            self.runtime.reset_motion_runtime_state()
            self.runtime.clear_manual_robot_state()
        _publish_runtime_event(
            self.runtime,
            "device.state",
            {
                "device": "transport",
                "selected_port": selected,
                "timestamp": _utc_timestamp(),
            },
        )
        return {"selected_port": selected}


class PlatformWidthService(BaseSerialDeviceService):
    def get_state(self) -> Dict[str, Any]:
        def _action() -> Dict[str, Any]:
            position_result = motion.current_position()
            if position_result.get("status") != "success":
                return position_result
            return motion.get_linear_state()

        raw = self._run_serial(_action)
        current_position = float(raw.get("current_position", 0.0))
        target_position = float(raw.get("target_position", 0.0))
        min_position = float(raw.get("min_position", 0.0))
        max_position = float(raw.get("max_position", 0.0))
        width_change_allowed = bool(raw.get("width_change_allowed", False))
        blocked_reason = str(raw.get("blocked_reason") or "") or None
        limit_switch_bitmap = int(raw.get("limit_switch_bitmap", 0) or 0)
        busy = bool(raw.get("busy", False))
        return {
            "device": "platform_width",
            "firmware_controller": motion.firmware_controller_name("platform_width"),
            "state": "moving" if busy else "idle",
            "busy": busy,
            "current_position_cm": current_position,
            "target_position_cm": target_position,
            "min_position_cm": min_position,
            "max_position_cm": max_position,
            "width_change_allowed": width_change_allowed,
            "blocked_reason": blocked_reason,
            "current_width_cm": current_position,
            "target_width_cm": target_position,
            "left_extension_cm": float(round(current_position / 2.0, 3)),
            "right_extension_cm": float(round(current_position / 2.0, 3)),
            "limit_switch_bitmap": limit_switch_bitmap,
            "blocked_by_turn": bool(not width_change_allowed),
        }

    def home(self, rpm: int = 40) -> Dict[str, Any]:
        state = self.get_state()
        target = float(state.get("min_position_cm", 0.0))
        command_id, raw = _run_command_with_events(
            self.runtime,
            device="platform_width",
            command_type="home",
            action=lambda: self._run_serial(
                lambda: motion.move_linear_to_position(
                    target_position=target,
                    rpm=int(rpm),
                )
            ),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_width",
                "type": "home",
                "state": "accepted",
            },
            "data": {
                "current_position_cm": float(raw.get("current_position", target)),
                "target_position_cm": float(raw.get("target_position", target)),
                "current_width_cm": float(raw.get("current_position", target)),
                "target_width_cm": float(raw.get("target_position", target)),
                "delta_position_cm": float(raw.get("delta_position", 0.0)),
                "rpm": int(raw.get("rpm", rpm)),
            },
        }

    def move_to_position(
        self,
        target_position_cm: float,
        rpm: int,
        speed_profile: str = "normal",
    ) -> Dict[str, Any]:
        command_id, raw = _run_command_with_events(
            self.runtime,
            device="platform_width",
            command_type="move_to_position",
            action=lambda: self._run_serial(
                lambda: motion.move_linear_to_position(
                    target_position=float(target_position_cm),
                    rpm=int(rpm),
                )
            ),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_width",
                "type": "move_to_position",
                "state": "accepted",
            },
            "data": {
                "current_position_cm": float(raw.get("current_position", 0.0)),
                "target_position_cm": float(raw.get("target_position", 0.0)),
                "current_width_cm": float(raw.get("current_position", 0.0)),
                "target_width_cm": float(raw.get("target_position", 0.0)),
                "delta_position_cm": float(raw.get("delta_position", 0.0)),
                "rpm": int(raw.get("rpm", rpm)),
                "speed_profile": str(speed_profile or "normal"),
            },
        }

    def jog(self, direction: int, motor1: int, motor2: int, motor3: int, motor4: int) -> Dict[str, Any]:
        command_id, raw = _run_command_with_events(
            self.runtime,
            device="platform_width",
            command_type="jog",
            action=lambda: self._run_serial(
                lambda: motion.start_individual_moving(
                    motor1=motor1,
                    motor2=motor2,
                    motor3=motor3,
                    motor4=motor4,
                    direction=direction,
                )
            ),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_width",
                "type": "jog",
                "state": "accepted",
            },
            "data": {
                "steps": int(raw.get("steps", 0)),
                "direction": int(raw.get("direction", direction)),
                "motors": list(raw.get("motors") or [motor1, motor2, motor3, motor4]),
                "position_cm": float(raw.get("position", 0.0)),
                "message": str(raw.get("message") or "Individual move started"),
            },
        }

    def stop(self) -> Dict[str, Any]:
        command_id, raw = _run_command_with_events(
            self.runtime,
            device="platform_width",
            command_type="stop",
            action=lambda: self._run_serial(motion.stop_all_motors),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_width",
                "type": "stop",
                "state": "accepted",
            },
            "data": {"message": str(raw.get("message") or "All width motors stopped")},
        }


class PlatformTurnService(BaseSerialDeviceService):
    def get_state(self) -> Dict[str, Any]:
        raw = motion.get_turn_runtime_state()
        live_busy = False
        live_sensor_bitmap = 0
        selected_port = str(self.runtime.get_active_port() or "").strip()
        if selected_port:
            try:
                live = self._run_serial(motion.get_turn_live_state)
                live_busy = bool(live.get("busy", False))
                live_sensor_bitmap = int(live.get("sensor_bitmap", 0) or 0)
            except DeviceCommandError:
                pass
        wheels = list(raw.get("wheels") or [])
        wheel_payload = [
            {
                "index": int(wheel.get("index", index + 1)),
                "current_angle_deg": float(wheel.get("current_angle", 0.0)),
                "target_angle_deg": float(wheel.get("target_angle", 0.0)),
            }
            for index, wheel in enumerate(wheels)
        ]
        while len(wheel_payload) < 4:
            wheel_payload.append(
                {
                    "index": int(len(wheel_payload) + 1),
                    "current_angle_deg": 0.0,
                    "target_angle_deg": 0.0,
                }
            )
        return {
            "device": "platform_turn",
            "firmware_controller": motion.firmware_controller_name("platform_turn"),
            "state": "moving" if live_busy else "idle",
            "busy": live_busy,
            "current_angle_deg": float(raw.get("current_angle", 0.0)),
            "target_angle_deg": float(raw.get("target_angle", 0.0)),
            "min_angle_deg": float(raw.get("min_angle", 0.0)),
            "max_angle_deg": float(raw.get("max_angle", 0.0)),
            "calibrated": bool(raw.get("calibrated", False)),
            "zero_reference_valid": bool(raw.get("zero_reference_valid", False)),
            "wheel_state_known": bool(raw.get("wheel_state_known", False)),
            "all_wheels_zero": bool(raw.get("all_wheels_zero", False)),
            "all_wheels_width_aligned": bool(raw.get("all_wheels_width_aligned", False)),
            "width_alignment_deg": float(raw.get("width_alignment_deg", 90.0)),
            "sensor_bitmap": int(live_sensor_bitmap),
            "last_command_id": int(raw.get("last_command_id", 0)),
            "wheels": wheel_payload,
            "wheel_1_angle_deg": float(wheel_payload[0]["current_angle_deg"]),
            "wheel_2_angle_deg": float(wheel_payload[1]["current_angle_deg"]),
            "wheel_3_angle_deg": float(wheel_payload[2]["current_angle_deg"]),
            "wheel_4_angle_deg": float(wheel_payload[3]["current_angle_deg"]),
            "wheel_1_target_deg": float(wheel_payload[0]["target_angle_deg"]),
            "wheel_2_target_deg": float(wheel_payload[1]["target_angle_deg"]),
            "wheel_3_target_deg": float(wheel_payload[2]["target_angle_deg"]),
            "wheel_4_target_deg": float(wheel_payload[3]["target_angle_deg"]),
        }

    def calibrate(self) -> Dict[str, Any]:
        command_id, _ = _run_command_with_events(
            self.runtime,
            device="platform_turn",
            command_type="calibrate_zero",
            action=lambda: self._run_serial(motion.calibrate_turn_motors),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_turn",
                "type": "calibrate_zero",
                "state": "accepted",
            },
            "data": self.get_state(),
        }

    def align_zero(self) -> Dict[str, Any]:
        def _action() -> Dict[str, Any]:
            state = self.get_state()
            if bool(state.get("all_wheels_zero")):
                return state
            self._run_serial(
                lambda: motion.move_turn_wheels_to_angles(
                    wheel_1_deg=0.0,
                    wheel_2_deg=0.0,
                    wheel_3_deg=0.0,
                    wheel_4_deg=0.0,
                )
            )
            return self.get_state()

        command_id, state = _run_command_with_events(
            self.runtime,
            device="platform_turn",
            command_type="align_zero",
            action=_action,
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_turn",
                "type": "align_zero",
                "state": "accepted",
            },
            "data": state,
        }

    def move_wheels_absolute(
        self,
        wheel_1_deg: float,
        wheel_2_deg: float,
        wheel_3_deg: float,
        wheel_4_deg: float,
        speed_profile: str = "normal",
    ) -> Dict[str, Any]:
        command_id, _ = _run_command_with_events(
            self.runtime,
            device="platform_turn",
            command_type="move_wheels_absolute",
            action=lambda: self._run_serial(
                lambda: motion.move_turn_wheels_to_angles(
                    wheel_1_deg=wheel_1_deg,
                    wheel_2_deg=wheel_2_deg,
                    wheel_3_deg=wheel_3_deg,
                    wheel_4_deg=wheel_4_deg,
                )
            ),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_turn",
                "type": "move_wheels_absolute",
                "state": "accepted",
            },
            "data": {
                **self.get_state(),
                "speed_profile": str(speed_profile or "normal"),
            },
        }

    def stop(self) -> Dict[str, Any]:
        command_id, raw = _run_command_with_events(
            self.runtime,
            device="platform_turn",
            command_type="stop",
            action=lambda: self._run_serial(motion.stop_turn_motors),
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "platform_turn",
                "type": "stop",
                "state": "accepted",
            },
            "data": {
                **self.get_state(),
                "message": str(raw.get("message") or "Turn motors stopped"),
            },
        }

    def move_to_angle(self, angle_deg: float, speed_profile: str = "normal") -> Dict[str, Any]:
        return self.move_wheels_absolute(
            wheel_1_deg=float(angle_deg),
            wheel_2_deg=float(angle_deg),
            wheel_3_deg=float(angle_deg),
            wheel_4_deg=float(angle_deg),
            speed_profile=speed_profile,
        )


class DeltaService:
    def __init__(self, runtime: RobotApiRuntime):
        self.runtime = runtime

    def get_state(self) -> Dict[str, Any]:
        manual = self.runtime.get_manual_robot_state()
        busy = bool(self.runtime.is_robot_busy())
        has_pose = bool(manual.get("has_last"))
        return {
            "device": "delta",
            "firmware_controller": motion.firmware_controller_name("delta"),
            "state": "moving" if busy else ("idle" if has_pose else "not_ready"),
            "busy": busy,
            "homed": has_pose,
            "pose_mm": {
                "x": float(manual.get("x_mm", 0.0)),
                "y": float(manual.get("y_mm", 0.0)),
                "z": float(manual.get("z_mm", 0.0)),
            },
            "z_min_mm": float(manual.get("z_min_mm", 0.0)),
            "z_max_mm": float(manual.get("z_max_mm", 0.0)),
            "trajectory_state": "running" if busy else "idle",
            "queue_depth": 1 if busy else 0,
        }

    def home(self) -> Dict[str, Any]:
        def _action() -> Dict[str, Any]:
            try:
                result = dict(self.runtime.delta_go_home() or {})
            except RuntimeError as exc:
                raise DeviceCommandError("device_unreachable", str(exc), 503) from exc
            except DeviceCommandError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                raise DeviceCommandError("device_fault", str(exc), 503) from exc
            if not bool(result.get("ok")):
                raise DeviceCommandError("command_rejected", "Delta move to work pose was rejected.", 409)
            return result

        command_id, _ = _run_command_with_events(
            self.runtime,
            device="delta",
            command_type="home",
            action=_action,
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "delta",
                "type": "home",
                "state": "accepted",
            },
            "data": self.get_state(),
        }

    def move(self, x_mm: float, y_mm: float, z_mm: float, duration_ms: int) -> Dict[str, Any]:
        def _action() -> Dict[str, Any]:
            try:
                return dict(
                    self.runtime.dispatch_manual_robot_move(
                        target_x_mm=float(x_mm),
                        target_y_mm=float(y_mm),
                        target_z_mm=float(z_mm),
                        duration_ms=int(duration_ms),
                    )
                    or {}
                )
            except ValueError as exc:
                if str(exc) == "calibration_required":
                    raise DeviceCommandError(
                        "calibration_required",
                        "Go work calibration is required before manual move.",
                        409,
                    ) from exc
                raise DeviceCommandError("validation_error", str(exc), 400) from exc
            except RuntimeError as exc:
                raise DeviceCommandError("device_unreachable", str(exc), 503) from exc
            except DeviceCommandError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                raise DeviceCommandError("device_fault", str(exc), 503) from exc

        command_id, result = _run_command_with_events(
            self.runtime,
            device="delta",
            command_type="move",
            action=_action,
        )

        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "delta",
                "type": "move",
                "state": "accepted",
            },
            "data": result,
        }

    def process_target(
        self,
        target_x_mm: float,
        target_y_mm: float,
        duration_ms: int,
        source: str = "vision",
    ) -> Dict[str, Any]:
        queue_fn = getattr(self.runtime, "queue_robot_command", None)
        if not callable(queue_fn):
            raise DeviceCommandError(
                "command_rejected",
                "Delta target processing is not available in current runtime.",
                409,
            )

        try:
            queue_result = queue_fn(
                relative_x_mm=int(round(float(target_x_mm))),
                relative_y_mm=int(round(float(target_y_mm))),
                duration_ms=int(duration_ms),
                context={"source": str(source or "vision")},
            )
        except Exception as exc:
            raise DeviceCommandError("device_fault", str(exc), 503) from exc

        if not bool(queue_result.get("accepted")):
            reason = str(queue_result.get("reason") or "busy")
            if reason == "busy":
                raise DeviceCommandError("device_busy", "Delta robot is busy.", 409)
            raise DeviceCommandError("command_rejected", f"Delta target rejected: {reason}", 409)

        command_id = int(queue_result.get("command_id") or _next_command_id())
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "delta",
                "type": "process_target",
                "state": "accepted",
            },
            "data": {
                "target_x_mm": float(target_x_mm),
                "target_y_mm": float(target_y_mm),
                "source": str(source or "vision"),
                "duration_ms": int(duration_ms),
                "queue_depth": 1 if bool(self.runtime.is_robot_busy()) else 0,
            },
        }

    def stop(self) -> Dict[str, Any]:
        stop_fn = getattr(self.runtime, "delta_stop", None)
        if not callable(stop_fn):
            raise DeviceCommandError(
                "command_rejected",
                "Current delta controller does not expose a dedicated stop command yet.",
                409,
            )

        def _action() -> Dict[str, Any]:
            try:
                result = dict(stop_fn() or {})
                if not bool(result.get("hard_stop_sent", False)):
                    reason = str(result.get("reason") or "hard_stop_failed")
                    raise DeviceCommandError(
                        "command_rejected",
                        f"Delta hard stop failed: {reason}",
                        409,
                    )
                return result
            except DeviceCommandError:
                raise
            except Exception as exc:
                raise DeviceCommandError("device_fault", str(exc), 503) from exc

        command_id, stop_result = _run_command_with_events(
            self.runtime,
            device="delta",
            command_type="stop",
            action=_action,
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "delta",
                "type": "stop",
                "state": "accepted",
            },
            "data": stop_result,
        }


class VisionService:
    def __init__(self, runtime: RobotApiRuntime):
        self.runtime = runtime

    def get_state(self) -> Dict[str, Any]:
        payload = dict(self.runtime.get_vision_state() or {})
        payload["timestamp"] = _utc_timestamp()
        return payload

    def get_detections(self) -> Dict[str, Any]:
        snapshot_fn = getattr(self.runtime, "get_visual_state_snapshot", None)
        snapshot = dict(snapshot_fn() or {}) if callable(snapshot_fn) else {}
        detections = list(snapshot.get("detections") or [])
        return {
            "last_seq": int(snapshot.get("seq", 0)),
            "event": str(snapshot.get("event") or ""),
            "count": int(snapshot.get("count", len(detections))),
            "detections": detections,
        }

    def get_zone(self) -> Dict[str, Any]:
        provider = getattr(self.runtime, "get_zone_config", None)
        if not callable(provider):
            return {
                "offset_x_px": 0,
                "offset_y_px": 0,
                "size_px": 0,
                "size_mm": 0.0,
            }
        return dict(provider() or {})

    def set_zone(
        self,
        *,
        offset_x_px: Optional[int] = None,
        offset_y_px: Optional[int] = None,
        size_px: Optional[int] = None,
        size_mm: Optional[float] = None,
    ) -> Dict[str, Any]:
        setter = getattr(self.runtime, "set_zone_config", None)
        if not callable(setter):
            raise DeviceCommandError("command_rejected", "Vision zone configuration is unavailable.", 409)

        kwargs: Dict[str, Any] = {}
        if offset_x_px is not None:
            kwargs["offset_x_px"] = int(offset_x_px)
        if offset_y_px is not None:
            kwargs["offset_y_px"] = int(offset_y_px)
        if size_px is not None:
            kwargs["size_px"] = int(size_px)
        if size_mm is not None:
            kwargs["size_mm"] = float(size_mm)

        def _action() -> Dict[str, Any]:
            try:
                return dict(setter(**kwargs) or {})
            except ValueError as exc:
                raise DeviceCommandError("validation_error", str(exc), 400) from exc
            except DeviceCommandError:
                raise
            except Exception as exc:
                raise DeviceCommandError("device_fault", str(exc), 503) from exc

        command_id, cfg = _run_command_with_events(
            self.runtime,
            device="vision",
            command_type="set_zone",
            action=_action,
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "vision",
                "type": "set_zone",
                "state": "accepted",
            },
            "data": cfg,
        }

    def start(self) -> Dict[str, Any]:
        starter = getattr(self.runtime, "vision_start", None)

        def _action() -> Dict[str, Any]:
            if callable(starter):
                try:
                    return dict(starter() or {})
                except DeviceCommandError:
                    raise
                except Exception as exc:
                    raise DeviceCommandError("device_fault", str(exc), 503) from exc
            return {"streaming": True}

        command_id, data = _run_command_with_events(
            self.runtime,
            device="vision",
            command_type="start",
            action=_action,
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "vision",
                "type": "start",
                "state": "accepted",
            },
            "data": data,
        }

    def stop(self) -> Dict[str, Any]:
        stopper = getattr(self.runtime, "vision_stop", None)

        def _action() -> Dict[str, Any]:
            if callable(stopper):
                try:
                    return dict(stopper() or {})
                except DeviceCommandError:
                    raise
                except Exception as exc:
                    raise DeviceCommandError("device_fault", str(exc), 503) from exc
            return {"streaming": False}

        command_id, data = _run_command_with_events(
            self.runtime,
            device="vision",
            command_type="stop",
            action=_action,
        )
        return {
            "ok": True,
            "command": {
                "command_id": command_id,
                "device": "vision",
                "type": "stop",
                "state": "accepted",
            },
            "data": data,
        }


class LegacySlaveService:
    def __init__(self, runtime: RobotApiRuntime):
        self.runtime = runtime

    def _selected_port(self) -> str:
        port = str(self.runtime.get_active_port() or "").strip()
        if not port:
            raise DeviceCommandError("transport_unavailable", "No active serial port selected.", 400)
        return port

    def read_registers(self, slave_id: int, start_addr: int = 0, count: int = 31) -> Dict[str, Any]:
        try:
            lock_ctx = self.runtime.serial_lock if self.runtime.serial_lock is not None else nullcontext()
            with lock_ctx:
                result = motion.run_with_connection(
                    self._selected_port(),
                    lambda: motion.read_registers(
                        slave_id=int(slave_id),
                        start_addr=int(start_addr),
                        count=int(count),
                    ),
                )
            if result.get("status") != "success":
                raise DeviceCommandError(
                    "device_unreachable",
                    str(result.get("message") or "Read registers failed"),
                    503,
                )
            return {"registers": list(result.get("registers") or [])}
        except DeviceCommandError:
            raise
        except Exception as exc:
            raise DeviceCommandError("device_unreachable", str(exc), 503) from exc

    def calibrate(self, slave_id: int) -> Dict[str, Any]:
        lock_ctx = self.runtime.serial_lock if self.runtime.serial_lock is not None else nullcontext()
        try:
            with lock_ctx:
                self.runtime.close_move_module_client()
                if int(slave_id) == int(motion.DELTA_SLAVE):
                    calibration = motion.run_with_connection(self._selected_port(), motion.delta_calibrate)
                else:
                    calibration = motion.run_with_connection(
                        self._selected_port(),
                        lambda: motion.write_registers(
                            slave_id=int(slave_id),
                            start_addr=406,
                            values=[1],
                        ),
                    )
                if calibration.get("status") != "success":
                    raise DeviceCommandError(
                        "device_unreachable",
                        str(calibration.get("message") or "Calibration command failed"),
                        503,
                    )
                ok = bool(calibration.get("ok", True))
                snapshot = motion.run_with_connection(
                    self._selected_port(),
                    lambda: motion.read_registers(
                        slave_id=int(slave_id),
                        start_addr=0,
                        count=31,
                    ),
                )
                if snapshot.get("status") != "success":
                    raise DeviceCommandError(
                        "device_unreachable",
                        str(snapshot.get("message") or "Read registers failed"),
                        503,
                    )
                if ok and int(slave_id) == int(motion.DELTA_SLAVE):
                    self.runtime.clear_manual_robot_state()
                return {
                    "ok": bool(ok),
                    "registers": list(snapshot.get("registers") or []),
                    "manual_state": self.runtime.get_manual_robot_state(),
                }
        except Exception as exc:
            raise DeviceCommandError("device_unreachable", str(exc), 503) from exc

    def go_work(self, slave_id: int) -> Dict[str, Any]:
        try:
            return dict(self.runtime.delta_go_home(int(slave_id)))
        except Exception as exc:
            raise DeviceCommandError("device_unreachable", str(exc), 503) from exc

    def manual_state(self) -> Dict[str, Any]:
        return self.runtime.get_manual_robot_state()

    def manual_move(self, x_mm: float, y_mm: float, z_mm: float, duration_ms: int) -> Dict[str, Any]:
        return self.runtime.dispatch_manual_robot_move(
            target_x_mm=float(x_mm),
            target_y_mm=float(y_mm),
            target_z_mm=float(z_mm),
            duration_ms=int(duration_ms),
        )


class RobotStateService:
    def __init__(
        self,
        runtime: RobotApiRuntime,
        transport: TransportService,
        width: PlatformWidthService,
        turn: PlatformTurnService,
        delta: DeltaService,
    ):
        self.runtime = runtime
        self.transport = transport
        self.width = width
        self.turn = turn
        self.delta = delta

    def _safe_state(self, reader: Callable[[], Dict[str, Any]], device: str) -> Dict[str, Any]:
        try:
            state = reader()
            return {**state, "fault": None}
        except DeviceCommandError as exc:
            return {
                "device": device,
                "state": "error",
                "busy": False,
                "fault": {
                    "code": exc.code,
                    "message": exc.message,
                    "status_code": exc.status_code,
                },
            }

    def get_robot_state(self) -> Dict[str, Any]:
        width_state = self._safe_state(self.width.get_state, "platform_width")
        turn_state = self._safe_state(self.turn.get_state, "platform_turn")
        delta_state = self._safe_state(self.delta.get_state, "delta")
        vision_state = dict(self.runtime.get_vision_state() or {})

        faults = [
            item.get("fault")
            for item in (width_state, turn_state, delta_state)
            if isinstance(item.get("fault"), dict)
        ]
        fault = faults[0] if faults else None

        selected_port = str(self.runtime.get_active_port() or "").strip()
        turn_ready = bool(turn_state.get("calibrated")) and bool(turn_state.get("wheel_state_known"))
        width_ready = bool(width_state.get("width_change_allowed", False))
        delta_ready = bool(delta_state.get("homed", False))
        busy = bool(width_state.get("busy")) or bool(turn_state.get("busy")) or bool(delta_state.get("busy"))
        emergency_stop = False

        robot_state = "ready"
        if fault is not None:
            robot_state = "fault"
        elif emergency_stop:
            robot_state = "emergency_stop"
        elif not selected_port:
            robot_state = "boot"
        elif busy:
            robot_state = "executing"
        elif not (turn_ready and width_ready and delta_ready):
            robot_state = "homing"

        return {
            "timestamp": _utc_timestamp(),
            "selected_port": selected_port,
            "robot_state": robot_state,
            "active_mode": str(vision_state.get("active_mode") or "manual"),
            "safety": {
                "emergency_stop": emergency_stop,
                "faulted": bool(fault is not None),
                "fault_code": None if fault is None else fault.get("code"),
                "fault_message": None if fault is None else fault.get("message"),
            },
            "platform_width": width_state,
            "platform_turn": turn_state,
            "delta": delta_state,
            "vision": vision_state,
        }


class ServiceRegistry:
    def __init__(self, runtime: RobotApiRuntime):
        self.runtime = runtime
        self.transport = TransportService(runtime)
        self.platform_width = PlatformWidthService(runtime)
        self.platform_turn = PlatformTurnService(runtime)
        self.delta = DeltaService(runtime)
        self.vision = VisionService(runtime)
        self.legacy_slave = LegacySlaveService(runtime)
        self.robot = RobotStateService(
            runtime=runtime,
            transport=self.transport,
            width=self.platform_width,
            turn=self.platform_turn,
            delta=self.delta,
        )

    def get_devices(self) -> Dict[str, Dict[str, Any]]:
        return {
            "platform_width": self.platform_width.get_state(),
            "platform_turn": self.platform_turn.get_state(),
            "delta": self.delta.get_state(),
            "vision": self.vision.get_state(),
        }
