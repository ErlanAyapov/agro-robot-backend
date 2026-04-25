import logging
import struct
import time
from contextlib import nullcontext
from threading import RLock, Thread
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from pymodbus.client import ModbusSerialClient
try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover - optional dependency fallback
    list_ports = None


CURRENT_POSITION = 0.0
TURN_POSITION = 0.0
TURN_WHEEL_COUNT = 4
TURN_CALIBRATED = False
TURN_ZERO_REFERENCE_VALID = False
TURN_WHEEL_STATE_KNOWN = False
TURN_LAST_COMMAND_ID = 0
TURN_WHEEL_ANGLES = [0.0] * TURN_WHEEL_COUNT
TURN_WHEEL_TARGETS = [0.0] * TURN_WHEEL_COUNT
REDUCTOR_CONF = 1
ROTATION_PER_SM = 1
LEGS_LENGTH = 20

MAX_POSITION = ROTATION_PER_SM * REDUCTOR_CONF * LEGS_LENGTH * 10
LINEAR_POSITION_MIN = 0.0
LINEAR_POSITION_MAX = float(LEGS_LENGTH)

TURN_PROFILE_SEGMENTS = 12
TURN_PROFILE_MIN_RPM = 25
TURN_PROFILE_MAX_RPM = 60
TURN_PROFILE_SETTLE_SEC = 0.01
TURN_TIME_GUARD = 1.12
TURN_BASE_RPM = 60.0
TURN_BASE_STEP_FREQ = 6400.0
TURN_STEPS_PER_DEGREE = (401.0 * 32.0 * 10.0) / 360.0
TURN_PROFILE_ENABLE = False
TURN_MIN_ANGLE = -90.0
TURN_MAX_ANGLE = 90.0
TURN_WIDTH_ALIGNMENT_DEG = 90.0
TURN_WIDTH_ALIGNMENT_TOLERANCE_DEG = 0.5

DEVICE_FIRMWARE_CONTROLLERS = {
    "platform_width": "WheelBaseWidthController",
    "platform_turn": "WheelSteeringController",
    "delta": "DeltaWeedingController",
}

# Unified bus mapping:
#   1 -> delta
#   2 -> platform_turn
#   3 -> platform_width
LIMIT_SWITCH_SLAVE = 3
TURN_SLAVE = 2
DELTA_SLAVE = 1
DELTA_SEGMENT_BASE_ADDR = 400
DELTA_TRAJECTORY_ADDR = 300
DELTA_PROCESS_ADDR = 302
DELTA_STOP_ADDR = 303
DELTA_CALIBRATE_ADDR = 406
DELTA_WORK_ADDR = 407
MODBUS_MAX_BATCH_REGS = 120


client = None
client_port = None
client_lock = RLock()
logger = logging.getLogger(__name__)
TURN_RESTORE_WAIT_TIMEOUT_SEC = 180.0
TURN_RESTORE_POLL_SEC = 0.2
TURN_RESTORE_TOKEN = 0
TURN_RESTORE_TOKEN_LOCK = RLock()


def get_bus_layout() -> Dict[str, int]:
    return {
        "delta": 1,
        "platform_turn": 2,
        "platform_width": 3,
    }


def firmware_controller_name(device_name: str) -> str:
    return str(DEVICE_FIRMWARE_CONTROLLERS.get(str(device_name), str(device_name)))


def _int16_to_u16(value: Any) -> int:
    return int(int(value)) & 0xFFFF


def _read_holding_registers_locked(slave: int, address: int, count: int):
    if not _is_client_connected():
        return None
    try:
        response = client.read_holding_registers(address=int(address), count=int(count), slave=int(slave))
    except Exception as exc:
        logger.debug("Register read failed (slave=%s addr=%s count=%s): %s", slave, address, count, exc)
        return None
    if response is None or response.isError():
        return None
    return list(response.registers or [])


def _read_switch_bitmap_locked(slave: int) -> int:
    registers = _read_holding_registers_locked(slave=slave, address=25, count=8)
    if not registers:
        return 0
    bitmap = 0
    for idx, value in enumerate(registers[:8]):
        if int(value) == 1:
            bitmap |= (1 << idx)
    return int(bitmap)


def combine_total_rotation(regs) -> int:
    if len(regs) != 4:
        return 0

    value = (
        ((regs[0] & 0xFF) << 24)
        | ((regs[1] & 0xFF) << 16)
        | ((regs[2] & 0xFF) << 8)
        | (regs[3] & 0xFF)
    )
    if value & (1 << 31):
        value -= 1 << 32
    return value


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _smoothstep01(x):
    x = _clamp(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _round_angle(value: Any) -> float:
    return round(float(value), 3)


def _next_turn_command_id() -> int:
    global TURN_LAST_COMMAND_ID
    TURN_LAST_COMMAND_ID = int(TURN_LAST_COMMAND_ID) + 1
    return int(TURN_LAST_COMMAND_ID)


def _set_turn_angles(
    *,
    current_angles: Optional[List[float]] = None,
    target_angles: Optional[List[float]] = None,
) -> None:
    global TURN_POSITION, TURN_WHEEL_ANGLES, TURN_WHEEL_TARGETS

    if current_angles is not None:
        TURN_WHEEL_ANGLES = [_round_angle(value) for value in list(current_angles)[:TURN_WHEEL_COUNT]]
        if len(TURN_WHEEL_ANGLES) < TURN_WHEEL_COUNT:
            TURN_WHEEL_ANGLES.extend([0.0] * (TURN_WHEEL_COUNT - len(TURN_WHEEL_ANGLES)))
    if target_angles is not None:
        TURN_WHEEL_TARGETS = [_round_angle(value) for value in list(target_angles)[:TURN_WHEEL_COUNT]]
        if len(TURN_WHEEL_TARGETS) < TURN_WHEEL_COUNT:
            TURN_WHEEL_TARGETS.extend([0.0] * (TURN_WHEEL_COUNT - len(TURN_WHEEL_TARGETS)))

    if TURN_WHEEL_ANGLES:
        TURN_POSITION = float(sum(TURN_WHEEL_ANGLES) / len(TURN_WHEEL_ANGLES))
    else:
        TURN_POSITION = 0.0


def _turn_all_wheels_aligned_to(target_angle_deg: float, tolerance_deg: float) -> bool:
    target = float(target_angle_deg)
    tolerance = max(0.0, float(tolerance_deg))
    return all(abs(float(angle) - target) <= tolerance for angle in TURN_WHEEL_ANGLES)


def _turn_all_wheels_zero() -> bool:
    return _turn_all_wheels_aligned_to(0.0, 1e-6)


def _turn_all_wheels_width_pose() -> bool:
    return _turn_all_wheels_aligned_to(
        TURN_WIDTH_ALIGNMENT_DEG,
        TURN_WIDTH_ALIGNMENT_TOLERANCE_DEG,
    )


def _normalize_turn_angles(values: Any) -> List[float]:
    normalized = [_round_angle(float(value)) for value in list(values or [])[:TURN_WHEEL_COUNT]]
    if len(normalized) < TURN_WHEEL_COUNT:
        normalized.extend([0.0] * (TURN_WHEEL_COUNT - len(normalized)))
    return normalized


def _angles_close(lhs: List[float], rhs: List[float], tolerance: float = 1e-3) -> bool:
    if len(lhs) != len(rhs):
        return False
    tol = max(0.0, float(tolerance))
    return all(abs(float(lhs[idx]) - float(rhs[idx])) <= tol for idx in range(len(lhs)))


def _turn_restore_needed(previous_angles: List[float]) -> bool:
    prev = _normalize_turn_angles(previous_angles)
    width_pose = [float(TURN_WIDTH_ALIGNMENT_DEG)] * TURN_WHEEL_COUNT
    return not _angles_close(prev, width_pose, tolerance=TURN_WIDTH_ALIGNMENT_TOLERANCE_DEG)


def _reserve_turn_restore_token() -> int:
    global TURN_RESTORE_TOKEN
    with TURN_RESTORE_TOKEN_LOCK:
        TURN_RESTORE_TOKEN = int(TURN_RESTORE_TOKEN) + 1
        return int(TURN_RESTORE_TOKEN)


def _is_turn_restore_token_current(token: int) -> bool:
    with TURN_RESTORE_TOKEN_LOCK:
        return int(token) == int(TURN_RESTORE_TOKEN)


def reset_runtime_state() -> None:
    global CURRENT_POSITION, TURN_POSITION
    global TURN_CALIBRATED, TURN_ZERO_REFERENCE_VALID, TURN_WHEEL_STATE_KNOWN, TURN_LAST_COMMAND_ID

    CURRENT_POSITION = 0.0
    TURN_POSITION = 0.0
    TURN_CALIBRATED = False
    TURN_ZERO_REFERENCE_VALID = False
    TURN_WHEEL_STATE_KNOWN = False
    TURN_LAST_COMMAND_ID = 0
    _set_turn_angles(
        current_angles=[0.0] * TURN_WHEEL_COUNT,
        target_angles=[0.0] * TURN_WHEEL_COUNT,
    )


def _turn_state_payload() -> Dict[str, Any]:
    current_angles = list(TURN_WHEEL_ANGLES)
    target_angles = list(TURN_WHEEL_TARGETS)
    current_angle = float(current_angles[0]) if current_angles else float(TURN_POSITION)
    target_angle = float(target_angles[0]) if target_angles else float(current_angle)
    return {
        "status": "success",
        "current_angle": float(current_angle),
        "target_angle": float(target_angle),
        "delta_angle": float(target_angle - current_angle),
        "min_angle": float(TURN_MIN_ANGLE),
        "max_angle": float(TURN_MAX_ANGLE),
        "calibrated": bool(TURN_CALIBRATED),
        "zero_reference_valid": bool(TURN_ZERO_REFERENCE_VALID),
        "wheel_state_known": bool(TURN_WHEEL_STATE_KNOWN),
        "all_wheels_zero": bool(_turn_all_wheels_zero()),
        "width_alignment_deg": float(TURN_WIDTH_ALIGNMENT_DEG),
        "all_wheels_width_aligned": bool(_turn_all_wheels_width_pose()),
        "last_command_id": int(TURN_LAST_COMMAND_ID),
        "wheels": [
            {
                "index": int(index + 1),
                "current_angle": float(current_angles[index]),
                "target_angle": float(target_angles[index]),
            }
            for index in range(TURN_WHEEL_COUNT)
        ],
    }


def _turn_calibration_required_result() -> Dict[str, Any]:
    return {
        "status": "error",
        "code": "turn_calibration_required",
        "message": "Turn calibration is required before absolute steering control.",
    }


def _wheel_state_unknown_result() -> Dict[str, Any]:
    return {
        "status": "error",
        "code": "wheel_state_unknown",
        "message": "Wheel state is unknown. Run turn calibration first.",
    }


def _wheel_alignment_required_result() -> Dict[str, Any]:
    return {
        "status": "error",
        "code": "wheel_alignment_required",
        "message": f"All steering wheels must be aligned to {int(TURN_WIDTH_ALIGNMENT_DEG)} degrees before width change.",
        "required_angle_deg": float(TURN_WIDTH_ALIGNMENT_DEG),
    }


def ensure_turn_calibrated() -> Dict[str, Any]:
    if not bool(TURN_CALIBRATED) or not bool(TURN_ZERO_REFERENCE_VALID):
        return _turn_calibration_required_result()
    if not bool(TURN_WHEEL_STATE_KNOWN):
        return _wheel_state_unknown_result()
    return {"status": "success"}


def ensure_width_change_allowed(*, auto_align: bool = False) -> Dict[str, Any]:
    calibration_check = ensure_turn_calibrated()
    if calibration_check.get("status") != "success":
        return calibration_check
    if _turn_all_wheels_width_pose():
        return {"status": "success"}
    if bool(auto_align):
        align_result = move_turn_wheels_to_angles(
            wheel_1_deg=TURN_WIDTH_ALIGNMENT_DEG,
            wheel_2_deg=TURN_WIDTH_ALIGNMENT_DEG,
            wheel_3_deg=TURN_WIDTH_ALIGNMENT_DEG,
            wheel_4_deg=TURN_WIDTH_ALIGNMENT_DEG,
        )
        if align_result.get("status") == "success" and _turn_all_wheels_width_pose():
            return {"status": "success"}
        align_message = str(align_result.get("message") or "auto alignment failed")
        return {
            **_wheel_alignment_required_result(),
            "message": (
                f"Width change requires all steering wheels at {int(TURN_WIDTH_ALIGNMENT_DEG)} degrees. "
                f"Auto-alignment failed: {align_message}"
            ),
        }
    return _wheel_alignment_required_result()


def get_turn_runtime_state() -> Dict[str, Any]:
    return _turn_state_payload()


def _build_turn_profile(total_angle_deg):
    total_angle_deg = int(max(0, total_angle_deg))
    if total_angle_deg == 0:
        return []

    segment_count = int(max(1, min(TURN_PROFILE_SEGMENTS, total_angle_deg)))
    profile = []
    prev_target = 0

    for i in range(1, segment_count + 1):
        target = int(round((total_angle_deg * i) / segment_count))
        segment_angle = target - prev_target
        prev_target = target
        if segment_angle <= 0:
            continue

        phase = (i - 0.5) / segment_count
        if phase <= 0.5:
            ramp = _smoothstep01(phase / 0.5)
        else:
            ramp = _smoothstep01((1.0 - phase) / 0.5)

        rpm = int(round(TURN_PROFILE_MIN_RPM + (TURN_PROFILE_MAX_RPM - TURN_PROFILE_MIN_RPM) * ramp))
        rpm = int(_clamp(rpm, TURN_PROFILE_MIN_RPM, TURN_PROFILE_MAX_RPM))
        profile.append((segment_angle, rpm))

    if not profile:
        profile.append((total_angle_deg, TURN_PROFILE_MAX_RPM))
    return profile


def _estimate_turn_segment_time(segment_angle_deg, rpm):
    if segment_angle_deg <= 0:
        return 0.0

    rpm = max(1, int(rpm))
    total_steps = TURN_STEPS_PER_DEGREE * float(segment_angle_deg)
    step_freq = TURN_BASE_STEP_FREQ * (float(rpm) / TURN_BASE_RPM)
    if step_freq <= 0:
        return TURN_PROFILE_SETTLE_SEC

    return (total_steps / step_freq) * TURN_TIME_GUARD + TURN_PROFILE_SETTLE_SEC


def _is_client_connected():
    if client is None:
        return False

    socket_checker = getattr(client, "is_socket_open", None)
    if callable(socket_checker):
        try:
            return bool(socket_checker())
        except Exception:
            return False

    return True


def _no_connection_response():
    return {
        "status": "error",
        "code": "transport_unavailable",
        "message": "No active connection",
    }


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_direction(value: Any) -> Optional[int]:
    if value in (0, 1):
        return int(value)

    normalized = str(value or "").strip().lower()
    if normalized in {"1", "up", "forward", "open"}:
        return 1
    if normalized in {"0", "down", "back", "close"}:
        return 0
    return None


def get_com_ports():
    if list_ports is None:
        return []
    ports = list_ports.comports()
    return [port.device for port in ports]


def _parse_version_pair(major_value: Any, minor_value: Any) -> Optional[str]:
    try:
        major = int(major_value)
        minor = int(minor_value)
    except Exception:
        return None
    if major < 0 or minor < 0:
        return None
    if major == 0xFFFF or minor == 0xFFFF:
        return None
    return f"{major}.{minor}"


def read_registers(slave_id: int, start_addr: int = 0, count: int = 31) -> Dict[str, Any]:
    if not _is_client_connected():
        return _no_connection_response()

    registers = _read_holding_registers_locked(
        slave=int(slave_id),
        address=int(start_addr),
        count=int(count),
    )
    if registers is None:
        return {"status": "error", "message": "Read registers failed"}
    return {
        "status": "success",
        "slave_id": int(slave_id),
        "start_addr": int(start_addr),
        "count": int(count),
        "registers": list(registers),
    }


def write_registers(slave_id: int, start_addr: int, values: List[int]) -> Dict[str, Any]:
    if not _is_client_connected():
        return _no_connection_response()

    payload = [int(v) & 0xFFFF for v in list(values or [])]
    if not payload:
        return {"status": "error", "message": "No register values provided"}

    try:
        result = client.write_registers(
            address=int(start_addr),
            values=payload,
            slave=int(slave_id),
        )
        if result.isError():
            logger.error("Register write error (slave=%s addr=%s): %s", slave_id, start_addr, result)
            return {"status": "error", "message": str(result)}
        return {
            "status": "success",
            "slave_id": int(slave_id),
            "start_addr": int(start_addr),
            "count": len(payload),
        }
    except Exception as exc:
        logger.error("Register write failed (slave=%s addr=%s): %s", slave_id, start_addr, exc)
        return {"status": "error", "message": str(exc)}


def probe_modbus_device(name: str, slave_id: int) -> Dict[str, Any]:
    controller_name = firmware_controller_name(str(name))
    if not _is_client_connected():
        return {
            "name": str(name),
            "firmware_controller": controller_name,
            "slave_id": int(slave_id),
            "status": "device_unreachable",
            "protocol_version": None,
            "firmware_version": None,
        }

    read_result = read_registers(slave_id=int(slave_id), start_addr=0, count=37)
    if read_result.get("status") != "success":
        return {
            "name": str(name),
            "firmware_controller": controller_name,
            "slave_id": int(slave_id),
            "status": "device_unreachable",
            "protocol_version": None,
            "firmware_version": None,
            "error": str(read_result.get("message") or "Read failed"),
        }

    registers = list(read_result.get("registers") or [])
    protocol_version = None
    firmware_version = None
    if len(registers) >= 37:
        protocol_version = _parse_version_pair(registers[33], registers[34])
        firmware_version = _parse_version_pair(registers[35], registers[36])

    if str(name) == "platform_width" and firmware_version is None and len(registers) >= 25:
        packed_fw = (
            ((int(registers[21]) & 0xFF) << 24)
            | ((int(registers[22]) & 0xFF) << 16)
            | ((int(registers[23]) & 0xFF) << 8)
            | (int(registers[24]) & 0xFF)
        )
        if packed_fw not in (0, 0xFFFFFFFF):
            firmware_version = str(int(packed_fw))

    return {
        "name": str(name),
        "firmware_controller": controller_name,
        "slave_id": int(slave_id),
        "status": "online" if registers else "device_unreachable",
        "protocol_version": protocol_version,
        "firmware_version": firmware_version,
        "register_count": len(registers),
        "ready": bool(int(registers[6])) if len(registers) > 6 and int(registers[6]) in (0, 1) else None,
        "busy": bool(int(registers[7])) if len(registers) > 7 and int(registers[7]) in (0, 1) else None,
    }


def delta_calibrate() -> Dict[str, Any]:
    result = write_registers(slave_id=DELTA_SLAVE, start_addr=DELTA_CALIBRATE_ADDR, values=[1])
    if result.get("status") != "success":
        return result
    return {
        "status": "success",
        "ok": True,
        "message": "Delta calibration command sent",
    }


def delta_go_home() -> Dict[str, Any]:
    write_result = write_registers(slave_id=DELTA_SLAVE, start_addr=DELTA_WORK_ADDR, values=[1])
    if write_result.get("status") != "success":
        return write_result

    state = read_registers(slave_id=DELTA_SLAVE, start_addr=0, count=31)
    return {
        "status": "success",
        "ok": True,
        "registers": list(state.get("registers") or []),
    }


def delta_hard_stop() -> Dict[str, Any]:
    write_result = write_registers(slave_id=DELTA_SLAVE, start_addr=DELTA_STOP_ADDR, values=[1])
    if write_result.get("status") != "success":
        return write_result

    state = read_registers(slave_id=DELTA_SLAVE, start_addr=6, count=2)
    registers = list(state.get("registers") or [])
    return {
        "status": "success",
        "ok": True,
        "message": "Delta hard stop command sent",
        "ready": bool(int(registers[0]) == 1) if len(registers) > 0 else None,
        "busy": bool(int(registers[1]) == 1) if len(registers) > 1 else None,
    }


def delta_process(cmd: int = 1, duration_ms: int = 1000) -> Dict[str, Any]:
    return write_registers(
        slave_id=DELTA_SLAVE,
        start_addr=DELTA_PROCESS_ADDR,
        values=[int(cmd), int(duration_ms)],
    )


def delta_write_segments(segments: List[List[int]]) -> Dict[str, Any]:
    segment_rows = list(segments or [])
    registers: List[int] = []
    for segment in segment_rows:
        row = list(segment or [])
        if len(row) != 3:
            return {"status": "error", "message": "Each segment must contain exactly 3 values"}
        for value in row:
            int_value = int(value)
            if int_value < -32768 or int_value > 32767:
                return {"status": "error", "message": f"Segment value out of int16 range: {int_value}"}
            packed = struct.pack(">h", int_value)
            registers.append((packed[0] << 8) | packed[1])

    if not registers:
        return {"status": "error", "message": "No segments to write"}

    for offset in range(0, len(registers), MODBUS_MAX_BATCH_REGS):
        batch = registers[offset : offset + MODBUS_MAX_BATCH_REGS]
        write_result = write_registers(
            slave_id=DELTA_SLAVE,
            start_addr=DELTA_SEGMENT_BASE_ADDR + int(offset),
            values=batch,
        )
        if write_result.get("status") != "success":
            return write_result
    return {"status": "success", "segments": int(len(segment_rows))}


def delta_start_trajectory(segments: List[List[int]], duration_ms: int) -> Dict[str, Any]:
    segment_rows = list(segments or [])
    write_result = delta_write_segments(segment_rows)
    if write_result.get("status") != "success":
        return write_result
    launch_result = write_registers(
        slave_id=DELTA_SLAVE,
        start_addr=DELTA_TRAJECTORY_ADDR,
        values=[int(len(segment_rows)), int(duration_ms)],
    )
    if launch_result.get("status") != "success":
        return launch_result
    return {
        "status": "success",
        "ok": True,
        "segments": int(len(segment_rows)),
        "duration_ms": int(duration_ms),
    }


def connect(port):
    global client, client_port
    with client_lock:
        try:
            normalized_port = str(port or "").strip()
            if not normalized_port:
                return {"status": "error", "message": "Port is required"}

            if client is not None and _is_client_connected() and str(client_port or "") == normalized_port:
                return {"status": "success", "message": f"Already connected to {normalized_port}"}

            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
                finally:
                    client = None
                    client_port = None
                logger.info("Closed previous limit-switch connection")

            client = ModbusSerialClient(
                port=normalized_port,
                baudrate=115200,
                stopbits=1,
                bytesize=8,
                parity="N",
                timeout=0.5,
            )
            if client.connect():
                client_port = normalized_port
                logger.info("Limit-switch controller connected to %s", normalized_port)
                return {"status": "success", "message": f"Connected to {normalized_port}"}

            client.close()
            client = None
            client_port = None
            logger.error("Failed to connect to %s", normalized_port)
            return {"status": "error", "message": f"Failed to connect to {normalized_port}"}
        except Exception as exc:
            logger.error("Connection error: %s", exc)
            client = None
            client_port = None
            return {"status": "error", "message": str(exc)}


def disconnect():
    global client, client_port
    with client_lock:
        if client:
            try:
                client.close()
            finally:
                client = None
                client_port = None
            logger.info("Limit-switch controller disconnected")
            return {"status": "success", "message": "Disconnected"}
        return _no_connection_response()


def close_limit_switch_client():
    try:
        disconnect()
    except Exception:
        pass


def is_connected():
    return {"connected": _is_client_connected()}


def _read_current_position_locked():
    global CURRENT_POSITION
    if not _is_client_connected():
        return _no_connection_response()

    try:
        def _read_position_registers(slave: int) -> Optional[List[int]]:
            for _ in range(3):
                registers = _read_holding_registers_locked(slave=int(slave), address=21, count=4)
                if registers is not None and len(registers) >= 4:
                    return list(registers[:4])
                time.sleep(0.05)
            return None

        regs = _read_position_registers(int(LIMIT_SWITCH_SLAVE))
        if regs is None:
            return {
                "status": "error",
                "code": "device_unreachable",
                "slave_id": int(LIMIT_SWITCH_SLAVE),
                "message": f"Position read failed on slave {int(LIMIT_SWITCH_SLAVE)}: no valid response",
            }

        CURRENT_POSITION = combine_total_rotation(list(regs)) / 20.0
        logger.info("Current limit-switch position: %s", CURRENT_POSITION)
        return {"status": "success", "position": CURRENT_POSITION}
    except Exception as exc:
        logger.error("Position error: %s", exc)
        return {
            "status": "error",
            "code": "device_unreachable",
            "slave_id": int(LIMIT_SWITCH_SLAVE),
            "message": f"Position read failed on slave {int(LIMIT_SWITCH_SLAVE)}: {exc}",
        }


def current_position():
    return _read_current_position_locked()


def get_linear_state():
    interlock = ensure_width_change_allowed(auto_align=False)
    status_regs = _read_holding_registers_locked(slave=LIMIT_SWITCH_SLAVE, address=6, count=2) or []
    ready_flag = int(status_regs[0]) if len(status_regs) > 0 else None
    busy_flag = int(status_regs[1]) if len(status_regs) > 1 else None
    limit_switch_bitmap = _read_switch_bitmap_locked(LIMIT_SWITCH_SLAVE)
    return {
        "status": "success",
        "slave_id": int(LIMIT_SWITCH_SLAVE),
        "current_position": float(CURRENT_POSITION),
        "target_position": float(CURRENT_POSITION),
        "min_position": float(LINEAR_POSITION_MIN),
        "max_position": float(LINEAR_POSITION_MAX),
        "width_change_allowed": bool(interlock.get("status") == "success"),
        "blocked_reason": None if interlock.get("status") == "success" else str(interlock.get("code") or ""),
        "ready": bool(ready_flag == 1),
        "busy": bool(busy_flag == 1),
        "limit_switch_bitmap": int(limit_switch_bitmap),
    }


def start_individual_moving(motor1, motor2, motor3, motor4, direction):
    if not _is_client_connected():
        return _no_connection_response()

    pre_width_angles = _normalize_turn_angles(TURN_WHEEL_ANGLES)
    interlock = ensure_width_change_allowed(auto_align=True)
    if interlock.get("status") != "success":
        return interlock
    post_width_angles = _normalize_turn_angles(TURN_WHEEL_ANGLES)
    auto_aligned_for_width = not _angles_close(
        pre_width_angles,
        post_width_angles,
        tolerance=TURN_WIDTH_ALIGNMENT_TOLERANCE_DEG,
    )

    direction = _parse_direction(direction)
    if direction not in (0, 1):
        return {"status": "error", "message": "Invalid direction value"}
    position_result = _read_current_position_locked()
    if position_result.get("status") != "success":
        return position_result

    try:
        if direction == 1:
            remaining_distance = max(0.0, float(LINEAR_POSITION_MAX) - float(CURRENT_POSITION))
        else:
            remaining_distance = max(0.0, float(CURRENT_POSITION) - float(LINEAR_POSITION_MIN))

        steps = round(abs(remaining_distance) * 10)
        values = [
            int(motor1 if motor1 is not None else 0),
            int(motor2 if motor2 is not None else 0),
            int(motor3 if motor3 is not None else 0),
            int(motor4 if motor4 is not None else 0),
            0,
            steps,
            int(direction),
            0,
        ]
        result = client.write_registers(address=0, values=values, slave=LIMIT_SWITCH_SLAVE)
        if result.isError():
            logger.error("Individual move error: %s", result)
            if auto_aligned_for_width:
                _restore_turn_angles_now(pre_width_angles, reason="width-jog-write-error")
            return {"status": "error", "message": str(result)}

        restore_scheduled = False
        if auto_aligned_for_width and _turn_restore_needed(pre_width_angles):
            if int(steps) > 0:
                restore_scheduled = bool(_schedule_turn_restore_after_width(pre_width_angles))
            else:
                _restore_turn_angles_now(pre_width_angles, reason="width-jog-zero-steps")

        logger.info("Individual move started: motors=%s dir=%s steps=%s", values[:4], direction, steps)
        return {
            "status": "success",
            "message": "Individual move started",
            "steps": int(steps),
            "direction": int(direction),
            "motors": values[:4],
            "position": float(CURRENT_POSITION),
            "turn_restore_scheduled": bool(restore_scheduled),
        }
    except Exception as exc:
        logger.error("Individual move failed: %s", exc)
        if auto_aligned_for_width:
            _restore_turn_angles_now(pre_width_angles, reason="width-jog-exception")
        return {"status": "error", "message": str(exc)}


def stop_all_motors():
    if not _is_client_connected():
        return _no_connection_response()

    try:
        values = [0, 0, 0, 0, 0, 0, 2, 1]
        result = client.write_registers(address=0, values=values, slave=LIMIT_SWITCH_SLAVE)
        if result.isError():
            logger.error("Stop error: %s", result)
            return {"status": "error", "message": str(result)}
        logger.info("All limit-switch motors stopped")
        return {"status": "success", "message": "All motors stopped"}
    except Exception as exc:
        logger.error("Stop failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def calibrate_turn_motors():
    global TURN_CALIBRATED, TURN_ZERO_REFERENCE_VALID, TURN_WHEEL_STATE_KNOWN
    if not _is_client_connected():
        return _no_connection_response()

    try:
        result = client.write_registers(address=0, values=[1, 5], slave=TURN_SLAVE)
        if result.isError():
            logger.error("Turn calibration error: %s", result)
            return {"status": "error", "message": str(result)}
        TURN_CALIBRATED = True
        TURN_ZERO_REFERENCE_VALID = True
        TURN_WHEEL_STATE_KNOWN = True
        _next_turn_command_id()
        _set_turn_angles(
            current_angles=[0.0] * TURN_WHEEL_COUNT,
            target_angles=[0.0] * TURN_WHEEL_COUNT,
        )
        logger.info("Turn calibration started")
        return {
            **_turn_state_payload(),
            "message": "Turn calibration started",
        }
    except Exception as exc:
        logger.error("Turn calibration failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def get_turn_live_state():
    if not _is_client_connected():
        return _no_connection_response()

    status_regs = _read_holding_registers_locked(slave=TURN_SLAVE, address=6, count=2) or []
    ready_flag = int(status_regs[0]) if len(status_regs) > 0 else None
    busy_flag = int(status_regs[1]) if len(status_regs) > 1 else None
    return {
        "status": "success",
        "slave_id": int(TURN_SLAVE),
        "ready": bool(ready_flag == 1),
        "busy": bool(busy_flag == 1),
        "sensor_bitmap": int(_read_switch_bitmap_locked(TURN_SLAVE)),
    }


def stop_turn_motors():
    if not _is_client_connected():
        return _no_connection_response()

    try:
        result = client.write_registers(address=0, values=[0, 0], slave=TURN_SLAVE)
        if result.isError():
            logger.error("Turn stop error: %s", result)
            return {"status": "error", "message": str(result)}
        _next_turn_command_id()
        _set_turn_angles(
            current_angles=list(TURN_WHEEL_ANGLES),
            target_angles=list(TURN_WHEEL_ANGLES),
        )
        return {
            **_turn_state_payload(),
            "message": "Turn motors stopped",
        }
    except Exception as exc:
        logger.error("Turn stop failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def send_angles(angle):
    if not _is_client_connected():
        return _no_connection_response()

    try:
        angle = float(angle)
        direction = 1 if angle >= 0 else 0
        total_angle = int(round(abs(angle)))
        if total_angle == 0:
            return {"status": "success", "message": "Zero angle, no movement required", "angle_deg": 0}

        if not TURN_PROFILE_ENABLE:
            values = [0, int(total_angle), int(direction), int(TURN_PROFILE_MAX_RPM)]
            logger.debug("Turn command: %s", values)
            result = client.write_registers(address=0, values=values, slave=TURN_SLAVE)
            if result.isError():
                logger.error("Turn write error: %s", result)
                return {"status": "error", "message": str(result)}
            return {
                "status": "success",
                "message": "Turn command sent",
                "segments": 1,
                "angle_deg": total_angle,
                "direction": int(direction),
            }

        profile = _build_turn_profile(total_angle)
        logger.debug("S-curve profile: %s", profile)

        for segment_angle, segment_rpm in profile:
            values = [0, int(segment_angle), int(direction), int(segment_rpm)]
            result = client.write_registers(address=0, values=values, slave=TURN_SLAVE)
            if result.isError():
                logger.error("Turn segment error: %s", result)
                return {"status": "error", "message": str(result)}
            time.sleep(_estimate_turn_segment_time(segment_angle, segment_rpm))

        return {
            "status": "success",
            "message": "Turn profile completed",
            "segments": len(profile),
            "angle_deg": total_angle,
            "direction": int(direction),
        }
    except Exception as exc:
        logger.error("Turn failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def get_turn_state():
    return _turn_state_payload()


def move_turn_to_angle(target_angle):
    if not _is_client_connected():
        return _no_connection_response()

    calibration_check = ensure_turn_calibrated()
    if calibration_check.get("status") != "success":
        return calibration_check

    try:
        target_angle = float(target_angle)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    target_angle = float(_clamp(target_angle, TURN_MIN_ANGLE, TURN_MAX_ANGLE))

    current_angle = float(TURN_POSITION)
    delta_angle = float(target_angle - current_angle)

    if abs(delta_angle) < 1e-9:
        return {
            "status": "success",
            "message": "Turn position already reached",
            "current_angle": current_angle,
            "target_angle": target_angle,
            "delta_angle": 0.0,
        }

    result = send_angles(delta_angle)
    if result.get("status") != "success":
        return result

    _next_turn_command_id()
    _set_turn_angles(
        current_angles=[float(target_angle)] * TURN_WHEEL_COUNT,
        target_angles=[float(target_angle)] * TURN_WHEEL_COUNT,
    )
    return {
        **result,
        **_turn_state_payload(),
    }


def _send_turn_wheel_delta(wheel_index: int, delta_angle_deg: float, rpm: int) -> Dict[str, Any]:
    if not _is_client_connected():
        return _no_connection_response()

    step_degrees = int(round(abs(float(delta_angle_deg))))
    if step_degrees <= 0:
        return {"status": "success", "message": "No wheel movement required"}

    direction = 1 if float(delta_angle_deg) >= 0.0 else 0
    motors = [0, 0, 0, 0]
    if wheel_index < 0 or wheel_index >= TURN_WHEEL_COUNT:
        return {"status": "error", "message": f"Invalid wheel index: {wheel_index}"}
    motors[wheel_index] = 1
    values = [
        int(motors[0]),
        int(motors[1]),
        int(motors[2]),
        int(motors[3]),
        int(max(1, rpm)),
        int(step_degrees),
        int(direction),
        0,
    ]
    try:
        result = client.write_registers(address=0, values=values, slave=TURN_SLAVE)
        if result.isError():
            logger.error("Wheel %s move error: %s", wheel_index + 1, result)
            return {"status": "error", "message": str(result)}
        time.sleep(_estimate_turn_segment_time(step_degrees, rpm))
        return {
            "status": "success",
            "wheel_index": int(wheel_index + 1),
            "step_degrees": int(step_degrees),
            "direction": int(direction),
        }
    except Exception as exc:
        logger.error("Wheel %s move failed: %s", wheel_index + 1, exc)
        return {"status": "error", "message": str(exc)}


def _send_turn_wheel_deltas_sync(deltas_angle_deg: List[float], rpm: int) -> Dict[str, Any]:
    if not _is_client_connected():
        return _no_connection_response()

    if len(list(deltas_angle_deg or [])) != TURN_WHEEL_COUNT:
        return {"status": "error", "message": "Exactly 4 wheel deltas are required"}

    quantized_deltas = [int(round(float(value))) for value in list(deltas_angle_deg)]
    motors = [1 if abs(delta) > 0 else 0 for delta in quantized_deltas]
    if sum(motors) == 0:
        return {"status": "success", "message": "No wheel movement required"}

    safe_rpm = int(max(1, rpm))
    values = [
        int(motors[0]),
        int(motors[1]),
        int(motors[2]),
        int(motors[3]),
        safe_rpm,
        _int16_to_u16(quantized_deltas[0]),
        _int16_to_u16(quantized_deltas[1]),
        _int16_to_u16(quantized_deltas[2]),
        _int16_to_u16(quantized_deltas[3]),
        0,  # ena=0 -> run
    ]
    max_segment = max(abs(delta) for delta in quantized_deltas)
    try:
        result = client.write_registers(address=0, values=values, slave=TURN_SLAVE)
        if result.isError():
            logger.error("Synchronized wheel move error: %s", result)
            return {"status": "error", "message": str(result)}
        time.sleep(_estimate_turn_segment_time(max_segment, safe_rpm))
        return {
            "status": "success",
            "mode": "sync",
            "wheel_deltas_deg": [int(value) for value in quantized_deltas],
            "rpm": int(safe_rpm),
        }
    except Exception as exc:
        logger.error("Synchronized wheel move failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def move_turn_wheels_to_angles(
    wheel_1_deg: float,
    wheel_2_deg: float,
    wheel_3_deg: float,
    wheel_4_deg: float,
):
    targets = [
        float(_clamp(float(wheel_1_deg), TURN_MIN_ANGLE, TURN_MAX_ANGLE)),
        float(_clamp(float(wheel_2_deg), TURN_MIN_ANGLE, TURN_MAX_ANGLE)),
        float(_clamp(float(wheel_3_deg), TURN_MIN_ANGLE, TURN_MAX_ANGLE)),
        float(_clamp(float(wheel_4_deg), TURN_MIN_ANGLE, TURN_MAX_ANGLE)),
    ]
    calibration_check = ensure_turn_calibrated()
    if calibration_check.get("status") != "success":
        return calibration_check

    current_angles = list(TURN_WHEEL_ANGLES)
    while len(current_angles) < TURN_WHEEL_COUNT:
        current_angles.append(0.0)

    deltas = [
        float(targets[index] - current_angles[index])
        for index in range(TURN_WHEEL_COUNT)
    ]

    result = _send_turn_wheel_deltas_sync(
        deltas_angle_deg=deltas,
        rpm=TURN_PROFILE_MAX_RPM,
    )
    if result.get("status") != "success":
        return result

    for idx, target_angle in enumerate(targets):
        current_angles[idx] = float(target_angle)

    _next_turn_command_id()
    _set_turn_angles(
        current_angles=current_angles,
        target_angles=targets,
    )
    return {
        **_turn_state_payload(),
        "requested_wheels": [
            {"index": 1, "target_angle": float(targets[0])},
            {"index": 2, "target_angle": float(targets[1])},
            {"index": 3, "target_angle": float(targets[2])},
            {"index": 4, "target_angle": float(targets[3])},
        ],
    }


def _restore_turn_angles_now(previous_angles: List[float], *, reason: str) -> Dict[str, Any]:
    if not _turn_restore_needed(previous_angles):
        return {"status": "success", "message": "Turn restore not required"}

    target_angles = _normalize_turn_angles(previous_angles)
    result = move_turn_wheels_to_angles(
        wheel_1_deg=target_angles[0],
        wheel_2_deg=target_angles[1],
        wheel_3_deg=target_angles[2],
        wheel_4_deg=target_angles[3],
    )
    if result.get("status") == "success":
        logger.info("Turn angles restored (%s): %s", reason, target_angles)
    else:
        logger.warning("Turn restore failed (%s): %s", reason, result)
    return result


def _wait_linear_motion_complete(timeout_sec: float = TURN_RESTORE_WAIT_TIMEOUT_SEC) -> Dict[str, Any]:
    timeout = max(0.5, float(timeout_sec))
    deadline = time.time() + timeout
    started_at = time.time()
    seen_busy = False

    while time.time() < deadline:
        with client_lock:
            if not _is_client_connected():
                return _no_connection_response()
            status_regs = _read_holding_registers_locked(slave=LIMIT_SWITCH_SLAVE, address=6, count=2) or []

        if len(status_regs) >= 2:
            busy = bool(int(status_regs[1]) == 1)
            if busy:
                seen_busy = True
            elif seen_busy or (time.time() - started_at) >= 0.8:
                return {"status": "success", "busy": False}

        time.sleep(TURN_RESTORE_POLL_SEC)

    return {
        "status": "error",
        "code": "execution_timeout",
        "message": "Timed out while waiting for width motion completion before turn restore.",
    }


def _turn_restore_worker(token: int, previous_angles: List[float]) -> None:
    try:
        wait_result = _wait_linear_motion_complete()
        if wait_result.get("status") != "success":
            logger.warning("Post-width turn restore skipped: %s", wait_result)
            return
        if not _is_turn_restore_token_current(token):
            logger.info("Post-width turn restore skipped: superseded token=%s", token)
            return
        with client_lock:
            if not _is_client_connected():
                logger.warning("Post-width turn restore skipped: no active connection")
                return
            _restore_turn_angles_now(previous_angles, reason="post-width-auto-align")
    except Exception as exc:
        logger.error("Post-width turn restore worker failed: %s", exc)


def _schedule_turn_restore_after_width(previous_angles: List[float]) -> bool:
    if not _turn_restore_needed(previous_angles):
        return False
    token = _reserve_turn_restore_token()
    Thread(
        target=_turn_restore_worker,
        args=(token, _normalize_turn_angles(previous_angles)),
        daemon=True,
        name="turn-restore-worker",
    ).start()
    return True


def send(rpm, target_position):
    global CURRENT_POSITION
    if not _is_client_connected():
        return _no_connection_response()

    try:
        target_position = float(target_position)
        rpm = int(rpm)
        direction = 1 if target_position > CURRENT_POSITION else 0
        step = round(abs(target_position - CURRENT_POSITION) * 10)
        CURRENT_POSITION = target_position

        values = [1, 1, 1, 1, rpm, step, direction, 0]
        logger.debug("Linear move payload: %s", values)

        request = bytearray([1, 0x10, 0x00, 0x00, 0x00, 0x04, 0x08])
        for val in values:
            request.extend(struct.pack(">H", val))
        crc = 0xFFFF
        for byte in request[:-2]:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        request.extend([crc & 0xFF, (crc >> 8) & 0xFF])
        logger.debug("Linear move request bytes: %s", [hex(b) for b in request])

        result = client.write_registers(address=0, values=values, slave=LIMIT_SWITCH_SLAVE)
        if result.isError():
            logger.error("Linear move write error: %s", result)
            return {"status": "error", "message": str(result)}

        time.sleep(0.2)
        response = client.read_holding_registers(address=0, count=4, slave=LIMIT_SWITCH_SLAVE)
        if response.isError():
            logger.error("Linear move readback error: %s", response)
            return {"status": "error", "message": str(response)}

        data = {
            "dir": response.registers[0],
            "rpm": response.registers[1],
            "steps": response.registers[2],
            "motor": response.registers[3],
        }
        logger.info("Linear move started: %s", data)
        return {"status": "success", "message": "Linear move started", "data": data}
    except Exception as exc:
        logger.error("Linear move failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def move_linear_to_position(target_position, rpm):
    if not _is_client_connected():
        return _no_connection_response()

    pre_width_angles = _normalize_turn_angles(TURN_WHEEL_ANGLES)
    interlock = ensure_width_change_allowed(auto_align=True)
    if interlock.get("status") != "success":
        return interlock
    post_width_angles = _normalize_turn_angles(TURN_WHEEL_ANGLES)
    auto_aligned_for_width = not _angles_close(
        pre_width_angles,
        post_width_angles,
        tolerance=TURN_WIDTH_ALIGNMENT_TOLERANCE_DEG,
    )

    position_result = _read_current_position_locked()
    if position_result.get("status") != "success":
        if auto_aligned_for_width:
            _restore_turn_angles_now(pre_width_angles, reason="width-position-read-failed")
        return position_result

    try:
        rpm = int(rpm)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    try:
        target_position = float(target_position)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    target_position = float(_clamp(target_position, LINEAR_POSITION_MIN, LINEAR_POSITION_MAX))
    current_position_value = float(CURRENT_POSITION)
    delta_position = float(target_position - current_position_value)

    if abs(delta_position) < 1e-9:
        if auto_aligned_for_width:
            _restore_turn_angles_now(pre_width_angles, reason="width-no-move-required")
        return {
            "status": "success",
            "message": "Linear position already reached",
            "current_position": current_position_value,
            "target_position": target_position,
            "delta_position": 0.0,
            "rpm": int(rpm),
        }

    result = send(rpm, target_position)
    if result.get("status") != "success":
        if auto_aligned_for_width:
            _restore_turn_angles_now(pre_width_angles, reason="width-send-failed")
        return result

    restore_scheduled = False
    if auto_aligned_for_width and _turn_restore_needed(pre_width_angles):
        restore_scheduled = bool(_schedule_turn_restore_after_width(pre_width_angles))

    return {
        **result,
        "current_position": float(CURRENT_POSITION),
        "target_position": float(target_position),
        "delta_position": float(delta_position),
        "rpm": int(rpm),
        "turn_restore_scheduled": bool(restore_scheduled),
    }


def _call_with_connection(port: str, action: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    with client_lock:
        connect_result = connect(port)
        if connect_result.get("status") != "success":
            return connect_result
        return action()


def run_with_connection(port: str, action: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    return _call_with_connection(port, action)


def build_limitswitch_router(
    port_provider: Callable[[], str],
    serial_lock: Optional[Any] = None,
) -> APIRouter:
    router = APIRouter(prefix="/limitswitch", tags=["limitswitch"])

    def _status_code_from_result(result: Dict[str, Any]) -> int:
        code = str(result.get("code") or "").strip().lower()
        if code in {
            "turn_calibration_required",
            "wheel_state_unknown",
            "wheel_zero_required",
            "wheel_alignment_required",
            "command_rejected",
        }:
            return 409
        return 503

    def _selected_port() -> str:
        selected_port = str(port_provider() or "").strip()
        if not selected_port:
            raise HTTPException(status_code=400, detail="No active serial port selected")
        return selected_port

    def _run(action: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        port = _selected_port()
        lock_ctx = serial_lock if serial_lock is not None else nullcontext()
        with lock_ctx:
            result = _call_with_connection(port, action)
        if result.get("status") == "error":
            raise HTTPException(
                status_code=_status_code_from_result(result),
                detail={
                    "code": result.get("code") or "device_error",
                    "message": result.get("message", "Limit-switch request failed"),
                },
            )
        return {"selected_port": port, **result}

    @router.get("/position")
    def api_current_position():
        return _run(current_position)

    @router.get("/linear/state")
    def api_linear_state():
        def _action():
            position_result = current_position()
            if position_result.get("status") != "success":
                return position_result
            return get_linear_state()

        return _run(_action)

    @router.post("/linear/target")
    def api_linear_target(payload: Dict[str, Any] = Body(default={})):
        data = payload or {}
        try:
            target_position = float(data.get("target_position", 0))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="target_position must be numeric") from exc

        try:
            rpm = int(data.get("rpm", 60))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="rpm must be numeric") from exc

        return _run(lambda: move_linear_to_position(target_position=target_position, rpm=rpm))

    @router.post("/move")
    def api_move(payload: Dict[str, Any] = Body(default={})):
        data = payload or {}

        direction = _parse_direction(data.get("direction"))
        if direction is None:
            raise HTTPException(status_code=400, detail="direction must be 0/1 or up/down")

        motors = {
            "motor1": 1 if _bool_value(data.get("motor1"), True) else 0,
            "motor2": 1 if _bool_value(data.get("motor2"), True) else 0,
            "motor3": 1 if _bool_value(data.get("motor3"), True) else 0,
            "motor4": 1 if _bool_value(data.get("motor4"), True) else 0,
        }
        if sum(motors.values()) == 0:
            raise HTTPException(status_code=400, detail="At least one motor must be enabled")

        return _run(
            lambda: start_individual_moving(
                motors["motor1"],
                motors["motor2"],
                motors["motor3"],
                motors["motor4"],
                direction,
            )
        )

    @router.post("/stop")
    def api_stop():
        return _run(stop_all_motors)

    @router.post("/turn/calibrate")
    def api_turn_calibrate():
        return _run(calibrate_turn_motors)

    @router.get("/turn/state")
    def api_turn_state():
        return {"selected_port": _selected_port(), **get_turn_state()}

    @router.post("/turn/angle")
    def api_turn_angle(payload: Dict[str, Any] = Body(default={})):
        data = payload or {}
        try:
            angle = float(data.get("angle", 0))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="angle must be numeric") from exc
        return _run(lambda: move_turn_to_angle(angle))

    return router
