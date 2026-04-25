import logging
from DeltaRobot import DeltaRobot, ModbusClient
from time import sleep
import math
import os
import sys


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _configure_utf8_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass


_configure_utf8_console()


# Firmware command resolution is 1 arcmin (1/60 deg). Using a coarser grid here
# introduces visible Cartesian jitter, especially on constant-Z moves.
ANGLE_QUANTUM_DEG = 1.0 / 60.0
FW_SEGMENTS_MAX = 200
FW_MAX_FREQ_STEPS_PER_SEC = 64000
FW_STEP_TOGGLE_FACTOR = 2
FW_MICRO_STEP = 32
FW_REDUCTOR_CONF = 10


def quantize_deg_to_grid(angle_deg, quantum_deg=ANGLE_QUANTUM_DEG):
    """Snap angle to the nearest motor quantum in degrees."""
    return round(angle_deg / quantum_deg) * quantum_deg


def to_arcmin_quantized(angle_deg):
    """Convert degrees to firmware arcmin command with minimal quantization loss."""
    return int(round(float(angle_deg) * 60.0))


# Logging setup
logger = logging.getLogger("bres")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s| %(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logging.getLogger("pymodbus").setLevel(logging.CRITICAL)
logging.getLogger("pymodbus.logging").setLevel(logging.CRITICAL)


R = 170   # Р Р°РґРёСѓСЃ Р±Р°Р·РѕРІРѕР№ РїР»Р°С‚С„РѕСЂРјС‹
r = 85   # Р Р°РґРёСѓСЃ РїРѕРґРІРёР¶РЅРѕР№ РїР»Р°С‚С„РѕСЂРјС‹
l = 620  # Р”Р»РёРЅР° РїР°СЂР°Р»Р»РµР»РѕРіСЂР°РјРјРЅС‹С… СЂС‹С‡Р°РіРѕРІ
L = 420  # Р”Р»РёРЅР° РІРµСЂС…РЅРёС… СЂС‹С‡Р°РіРѕРІ

MANUAL_XY_ROTATION_DEG = _env_float("DELTA_MANUAL_XY_ROTATION_DEG", 95.0)
MANUAL_INVERT_X = _env_bool("DELTA_MANUAL_INVERT_X", False)
MANUAL_INVERT_Y = _env_bool("DELTA_MANUAL_INVERT_Y", False)
MANUAL_INVERT_Z = _env_bool("DELTA_MANUAL_INVERT_Z", False)
USER_ZERO_XYZ_AT_JOINT_ZERO = True
PRINT_USER_CUBE_RANGES = True
CUBE_THETA_MIN = -5.0
CUBE_THETA_MAX = 90.0
CUBE_APPLY_COMMAND_LIMITS = False

robot = DeltaRobot(
    R, r, l, L,
    xy_rotation_deg=MANUAL_XY_ROTATION_DEG,
    invert_x=MANUAL_INVERT_X,
    invert_y=MANUAL_INVERT_Y,
    invert_z=MANUAL_INVERT_Z
)
if USER_ZERO_XYZ_AT_JOINT_ZERO:
    x0_before, y0_before, z0_before, _ = robot.fkinem(0, 0, 0)
    robot.set_user_origin_from_angles(0, 0, 0, zero_x=True, zero_y=True, zero_z=True)
    x0_after, y0_after, z0_after, _ = robot.fkinem(0, 0, 0)
    logger.info(
        f"User XYZ zero anchored at joints(0,0,0): "
        f"before=({x0_before:.3f}, {y0_before:.3f}, {z0_before:.3f}) mm, "
        f"after=({x0_after:.3f}, {y0_after:.3f}, {z0_after:.3f}) mm, "
        f"offsets=({robot.offset_x:.3f}, {robot.offset_y:.3f}, {robot.offset_z:.3f}) mm"
    )
# Р’С‹С‡РёСЃР»СЏРµРј РїР°СЂР°РјРµС‚СЂС‹ РІРїРёСЃР°РЅРЅРѕРіРѕ РєСѓР±Р°
# robot.cuboid()

# Р’С‹С‡РёСЃР»СЏРµРј СЂР°Р±РѕС‡СѓСЋ РѕР±Р»Р°СЃС‚СЊ (РѕРїС†РёРѕРЅР°Р»СЊРЅРѕ, РјРѕР¶РЅРѕ Р·Р°РєРѕРјРјРµРЅС‚РёСЂРѕРІР°С‚СЊ РґР»СЏ СѓСЃРєРѕСЂРµРЅРёСЏ)
# A = robot.workspace()

class NoOpModbusClient:
    """Fallback client used when serial port is unavailable."""

    def __init__(self, reason):
        self.connected = False
        self._warned = False

    def _warn(self, action):
        if not self._warned:
            logger.warning("Modbus отключен: координаты на контроллер не отправляются.")
            self._warned = True
        return False

    def start_trajectory(self, segments, duration=0):
        return self._warn("start_trajectory")

    def proccess(self, cmd=1, duration=1000):
        return self._warn("proccess")

    def rotate(self, *args, **kwargs):
        return self._warn("rotate")


def _create_modbus_client():
    if _env_bool("DELTA_DISABLE_INTERNAL_MODBUS", False):
        logger.info("Internal Modbus client disabled by DELTA_DISABLE_INTERNAL_MODBUS.")
        return NoOpModbusClient("disabled")
    port = os.getenv("DELTA_MODBUS_PORT", "COM20")
    try:
        modbus_client = ModbusClient(port, move_to_start=False)
        setattr(modbus_client, "connected", True)
        logger.info("Modbus подключен: %s", port)
        return modbus_client
    except Exception as exc:
        logger.warning("Не удалось открыть Modbus-порт %s. Отправка координат отключена.", port)
        logger.debug("Ошибка подключения Modbus: %r", exc)
        return NoOpModbusClient(exc)


client = _create_modbus_client()
    


def create_delta_robot(R, r, l, L):
    """
    РЎРѕР·РґР°РµС‚ СЌРєР·РµРјРїР»СЏСЂ РєР»Р°СЃСЃР° DeltaRobot СЃ Р·Р°РґР°РЅРЅС‹РјРё РїР°СЂР°РјРµС‚СЂР°РјРё.
    """
    inst = DeltaRobot(
        R, r, l, L,
        xy_rotation_deg=MANUAL_XY_ROTATION_DEG,
        invert_x=MANUAL_INVERT_X,
        invert_y=MANUAL_INVERT_Y,
        invert_z=MANUAL_INVERT_Z
    )
    if USER_ZERO_XYZ_AT_JOINT_ZERO:
        inst.set_user_origin_from_angles(0, 0, 0, zero_x=True, zero_y=True, zero_z=True)
    return inst


def print_user_cube_ranges(
    robot_inst,
    theta_min=CUBE_THETA_MIN,
    theta_max=CUBE_THETA_MAX,
    apply_command_limits=CUBE_APPLY_COMMAND_LIMITS
):
    cube_size, cube_origin, middle_z, _ = robot_inst.cuboid(
        theta_min=theta_min,
        theta_max=theta_max,
        apply_command_limits=apply_command_limits,
        force_center_xy=True,
        center_x=0.0,
        center_y=0.0
    )

    if cube_size <= 0:
        logger.warning("User cube range is empty (cube_size <= 0).")
        return None

    x_min = float(cube_origin[0])
    x_max = x_min + float(cube_size)
    y_min = float(cube_origin[1])
    y_max = y_min + float(cube_size)
    z_min = float(cube_origin[2])
    z_max = z_min + float(cube_size)

    logger.info(
        "User cube ranges: "
        f"X[{x_min:.2f}, {x_max:.2f}] mm, "
        f"Y[{y_min:.2f}, {y_max:.2f}] mm, "
        f"Z[{z_min:.2f}, {z_max:.2f}] mm"
    )
    logger.info(
        "User cube sizes: "
        f"dX={x_max - x_min:.2f} mm, dY={y_max - y_min:.2f} mm, dZ={z_max - z_min:.2f} mm, "
        f"center=(0.00, 0.00, {middle_z:.2f}) mm"
    )
    return cube_size, cube_origin, middle_z


def _smootherstep(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def s_curve_interpolation_3d(x0, y0, z0, x1, y1, z1, num_points):
    if num_points < 2:
        return [(float(x0), float(y0), float(z0))]

    points = []
    steps = num_points - 1
    for i in range(num_points):
        t = i / steps
        s = _smootherstep(t)
        x = x0 + (x1 - x0) * s
        y = y0 + (y1 - y0) * s
        z = z0 + (z1 - z0) * s
        points.append((float(x), float(y), float(z)))
    return points


def move_to_coordinate(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    duration=5000,
    reverse=False,
    use_s_curve=True,
    segment_budget=FW_SEGMENTS_MAX,
):
    """
    Build one smooth trajectory and distribute it into <= 200 firmware segments.
    Start/end increments are smaller, middle increments are larger (S-curve).
    """
    if not getattr(client, "connected", True):
        client.start_trajectory([], 0)
        return

    _ = robot.ikinem(x1, y1, z1)

    budget = int(segment_budget)
    if budget < 20:
        budget = 20
    if budget > FW_SEGMENTS_MAX:
        budget = FW_SEGMENTS_MAX

    num_points = budget + 1
    logger.info(
        f"Path sampling by segment budget: segments={budget}, points={num_points}, "
        f"profile={'s-curve' if use_s_curve else 'linear'}"
    )

    if use_s_curve:
        line_points = s_curve_interpolation_3d(x1, y1, z1, x2, y2, z2, num_points=num_points)
    else:
        line_points = robot.linear_interpolation_3d(x1, y1, z1, x2, y2, z2, num_points=num_points)

    logger.info("Path points generated: %s", len(line_points))

    failed_points = []
    steps = []
    quantized_angles = []

    for point in line_points:
        x, y, z = point
        theta1, theta2, theta3, fl = robot.ikinem(x, y, z)

        if fl:
            quantized_angles.append([
                to_arcmin_quantized(theta1),
                to_arcmin_quantized(theta2),
                to_arcmin_quantized(theta3),
            ])
            logger.debug("Path point: (%.3f, %.3f, %.3f)", x, y, z)
        else:
            failed_points.append((x, y, z))

    if failed_points:
        sample = ", ".join(
            f"({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})"
            for pt in failed_points[:3]
        )
        logger.warning(
            "Skipped %s points outside workspace. First points: %s",
            len(failed_points),
            sample,
        )

    if len(quantized_angles) > (FW_SEGMENTS_MAX + 1):
        before = len(quantized_angles)
        quantized_angles = _downsample_joint_points(quantized_angles, FW_SEGMENTS_MAX + 1)
        logger.warning(
            f"Trajectory downsampled for firmware limit: {before} -> {len(quantized_angles)} points "
            f"(max segments: {FW_SEGMENTS_MAX})."
        )

    for i in range(1, len(quantized_angles)):
        prev = quantized_angles[i - 1]
        curr = quantized_angles[i]
        seg = [prev[0] - curr[0], prev[1] - curr[1], prev[2] - curr[2]]
        if seg[0] != 0 or seg[1] != 0 or seg[2] != 0:
            steps.append(seg)

    if len(steps) == 0:
        logger.warning("No valid non-zero trajectory segments to send.")
        return

    if len(steps) > FW_SEGMENTS_MAX:
        logger.warning(
            f"Segment count {len(steps)} exceeds firmware limit {FW_SEGMENTS_MAX}. Applying final downsample."
        )
        quantized_angles = _downsample_joint_points(quantized_angles, FW_SEGMENTS_MAX + 1)
        steps = []
        for i in range(1, len(quantized_angles)):
            prev = quantized_angles[i - 1]
            curr = quantized_angles[i]
            seg = [prev[0] - curr[0], prev[1] - curr[1], prev[2] - curr[2]]
            if seg[0] != 0 or seg[1] != 0 or seg[2] != 0:
                steps.append(seg)

    logger.info("Trajectory segments: %s", len(steps))
    logger.info("Points outside workspace: %s", len(failed_points))

    total_steps = [0, 0, 0]
    for step in steps:
        total_steps[0] += abs(step[0])
        total_steps[1] += abs(step[1])
        total_steps[2] += abs(step[2])

    logger.info("Total motor steps by axis: %s", total_steps)
    logger.info("Total steps sum: %s", sum(total_steps))

    safe_duration = duration
    required_duration = _required_duration_ms_for_profile(steps)
    if required_duration > safe_duration:
        logger.warning(
            f"Duration increased from {safe_duration}ms to {required_duration}ms "
            f"to satisfy firmware max frequency ({FW_MAX_FREQ_STEPS_PER_SEC} steps/sec)."
        )
        safe_duration = required_duration

    client.start_trajectory(steps, safe_duration)
    if reverse:
        sleep(safe_duration / 1000.0)
        client.proccess()
        sleep(2)
        move_to_coordinate(
            x2, y2, z2, x1, y1, z1,
            duration=safe_duration,
            segment_budget=budget,
            use_s_curve=use_s_curve,
        )
 

def _interpolate_joint_space(start_angles, end_angles, num_points, use_s_curve=True):
    if num_points < 2:
        return [tuple(float(v) for v in start_angles)]

    points = []
    steps = num_points - 1
    for i in range(num_points):
        t = i / steps
        s = _smootherstep(t) if use_s_curve else t
        p = (
            start_angles[0] + (end_angles[0] - start_angles[0]) * s,
            start_angles[1] + (end_angles[1] - start_angles[1]) * s,
            start_angles[2] + (end_angles[2] - start_angles[2]) * s,
        )
        points.append((float(p[0]), float(p[1]), float(p[2])))
    return points


def _downsample_joint_points(points, max_points):
    """Downsample points preserving first/last to fit firmware segment limit."""
    if len(points) <= max_points:
        return points
    if max_points < 2:
        return [points[0]]

    n = len(points)
    out = []
    for i in range(max_points):
        if i == max_points - 1:
            idx = n - 1
        else:
            idx = int(i * (n - 1) / (max_points - 1))
        out.append(points[idx])
    return out


def _arcmin_to_driver_steps(angle_min):
    """Mirror firmware conversion from arcmin to pulse toggles (motor.c)."""
    driver_steps = int(round(abs(angle_min) * FW_MICRO_STEP * 200.0 * FW_REDUCTOR_CONF / 21600.0))
    return driver_steps * FW_STEP_TOGGLE_FACTOR


def _required_duration_ms_for_profile(steps):
    """
    Compute minimal duration so each segment stays <= FW_MAX_FREQ_STEPS_PER_SEC.
    Firmware uses same time_per_segment for all segments.
    """
    if not steps:
        return 0

    seg_count = len(steps)
    max_steps_in_segment = 0
    for seg in steps:
        for motor_arcmin in seg:
            seg_steps = _arcmin_to_driver_steps(motor_arcmin)
            if seg_steps > max_steps_in_segment:
                max_steps_in_segment = seg_steps

    if max_steps_in_segment == 0:
        return 0

    required = (max_steps_in_segment * seg_count * 1000.0) / FW_MAX_FREQ_STEPS_PER_SEC
    return int(math.ceil(required))


def move_to_angles(
    target_theta1,
    target_theta2,
    target_theta3,
    duration=5000,
    start_theta1=0.0,
    start_theta2=0.0,
    start_theta3=0.0,
    num_points=None,
    use_s_curve=True,
):
    """
    РџРµСЂРµРјРµС‰Р°РµС‚ СЂРѕР±РѕС‚Р° РІ РїСЂРѕСЃС‚СЂР°РЅСЃС‚РІРµ СѓРіР»РѕРІ (РіСЂР°РґСѓСЃС‹), РѕС‚РїСЂР°РІР»СЏСЏ СЃРµРіРјРµРЅС‚С‹ РІ arcmin.
    """
    if not getattr(client, "connected", True):
        client.start_trajectory([], 0)
        return

    start_angles = (float(start_theta1), float(start_theta2), float(start_theta3))
    end_angles = (float(target_theta1), float(target_theta2), float(target_theta3))

    if num_points is None:
        max_delta = max(
            abs(end_angles[0] - start_angles[0]),
            abs(end_angles[1] - start_angles[1]),
            abs(end_angles[2] - start_angles[2]),
        )
        num_points = max(20, int(max_delta) + 1)

    joint_points = _interpolate_joint_space(
        start_angles=start_angles,
        end_angles=end_angles,
        num_points=num_points,
        use_s_curve=use_s_curve,
    )
    joint_points = _downsample_joint_points(joint_points, FW_SEGMENTS_MAX + 1)

    quantized_angles = []
    for p in joint_points:
        q = [
            to_arcmin_quantized(p[0]),
            to_arcmin_quantized(p[1]),
            to_arcmin_quantized(p[2]),
        ]
        quantized_angles.append(q)
        logger.info(f"Joint point (deg): {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}")
        logger.info(f"Command (arcmin): {q[0]}, {q[1]}, {q[2]}")

    steps = []
    for i in range(1, len(quantized_angles)):
        prev = quantized_angles[i - 1]
        curr = quantized_angles[i]
        seg = [prev[0] - curr[0], prev[1] - curr[1], prev[2] - curr[2]]
        if seg[0] != 0 or seg[1] != 0 or seg[2] != 0:
            steps.append(seg)

    if len(steps) == 0:
        logger.warning("No valid non-zero trajectory segments to send (move_to_angles).")
        return

    total_steps = [0, 0, 0]
    for step in steps:
        total_steps[0] += abs(step[0])
        total_steps[1] += abs(step[1])
        total_steps[2] += abs(step[2])

    logger.info("Joint trajectory points: %s", len(joint_points))
    logger.info("Trajectory segments: %s", len(steps))
    logger.info("Total motor steps by axis: %s", total_steps)
    logger.info("Total steps sum: %s", sum(total_steps))

    safe_duration = duration
    required_duration = _required_duration_ms_for_profile(steps)
    if required_duration > safe_duration:
        logger.warning(
            f"Duration increased from {safe_duration}ms to {required_duration}ms "
            f"to satisfy firmware max frequency ({FW_MAX_FREQ_STEPS_PER_SEC} steps/sec)."
        )
        safe_duration = required_duration

    client.start_trajectory(steps, safe_duration)


def move_to_one_segment(x1, y1, z1, x2, y2, z2):
    theta11, theta12, theta13, fl1 = robot.ikinem(x1, y1, z1)
    theta21, theta22, theta23, fl2 = robot.ikinem(x2, y2, z2)
    if fl1 and fl2:
        theta = [
            to_arcmin_quantized(theta11 - theta21),
            to_arcmin_quantized(theta12 - theta22),
            to_arcmin_quantized(theta13 - theta23),
        ]
        # client.start_trajectory([theta], 5000)
    else:
        logger.warning("Point is outside workspace")
        return False, None


# is_within_cube ===============
# -x, -y, -z = -187.04, -187.04, -550.07
# x, y, z = 187.04, 187.04, -177
def main(): 
    if PRINT_USER_CUBE_RANGES:
        print_user_cube_ranges(robot)

    if USER_ZERO_XYZ_AT_JOINT_ZERO:
        # User origin is anchored at joints(0,0,0): home point becomes (0,0,0).
        x1, y1, z1 = 0, 0, -180
        x2, y2, z2 = 0, 0, -500
    else:
        x1, y1, z1 = 0, 0, -359
        x2, y2, z2 = 150, 0, -459

    # revers = 1
    duration = 12000

    move_to_coordinate(x1, y1, z1, x2, y2, z2, duration)
    # move_to_angles(
    #     360, 360, 360,
    #     duration=3000,
    #     start_theta1=0, start_theta2=0, start_theta3=0,
    #     use_s_curve=True
    # )


    # sleep(duration/1000)

    # move_to_coordinate(x2, y2, z2, x1, y1, z1, duration)

if __name__ == "__main__":
    main()
