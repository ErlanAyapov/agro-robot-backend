import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


logging.basicConfig()
log = logging.getLogger()
# log.setLevel(logging.DEBUG)
 
def bresenham3d_indices(i0, j0, k0, i1, j1, k1):
    """
    Алгоритм Брезенхэма для 3D по целочисленным индексам (i,j,k).
    Возвращает список (i,j,k) от стартовой точки до конечной.
    """
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    dk = abs(k1 - k0)

    si = 1 if i1 >= i0 else -1
    sj = 1 if j1 >= j0 else -1
    sk = 1 if k1 >= k0 else -1

    coords = []

    # Определяем, какая из компонент самая "длинная"
    if di >= dj and di >= dk:
        # i - главный цикл
        p1 = 2*dj - di
        p2 = 2*dk - di
        j = j0
        k = k0
        for i in range(i0, i1 + si, si):
            coords.append((i, j, k))
            if p1 >= 0:
                j += sj
                p1 -= 2*di
            p1 += 2*dj
            if p2 >= 0:
                k += sk
                p2 -= 2*di
            p2 += 2*dk

    elif dj >= di and dj >= dk:
        # j - главный цикл
        p1 = 2*di - dj
        p2 = 2*dk - dj
        i = i0
        k = k0
        for j in range(j0, j1 + sj, sj):
            coords.append((i, j, k))
            if p1 >= 0:
                i += si
                p1 -= 2*dj
            p1 += 2*di
            if p2 >= 0:
                k += sk
                p2 -= 2*dj
            p2 += 2*dk

    else:
        # k - главный цикл
        p1 = 2*di - dk
        p2 = 2*dj - dk
        i = i0
        j = j0
        for k in range(k0, k1 + sk, sk):
            coords.append((i, j, k))
            if p1 >= 0:
                i += si
                p1 -= 2*dk
            p1 += 2*di
            if p2 >= 0:
                j += sj
                p2 -= 2*dk
            p2 += 2*dj

    return coords


class DeltaRobot:
    def __init__(
        self,
        R,
        r,
        l,
        L,
        angle_per_step=0.05625,
        invert_x=False,
        invert_y=False,
        invert_z=False,
        xy_rotation_deg=0.0,
        offset_x=0.0,
        offset_y=0.0,
        offset_z=0.0
    ):
        # Инициализация параметров робота
        self.R = R  # Радиус базовой платформы 68.63 
        self.r = r  # Радиус подвижной платформы (рабочий орган)
        self.l = l  # Длина параллелограммных рычагов (предплечье)
        self.L = L  # Длина верхних рычагов (плечо)

        # Параметры куба и рабочей области будут вычислены позже
        self.cube_size = None
        self.cube_origin = None
        self.middleZ = None
        self.total = None

        # Отключаем смещения по X и Y
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.offset_z = float(offset_z)
        self.axis_sign = np.array([
            -1.0 if invert_x else 1.0,
            -1.0 if invert_y else 1.0,
            -1.0 if invert_z else 1.0,
        ], dtype=float)
        self.xy_rotation_deg = float(xy_rotation_deg)
        self._update_xy_rotation_cache()
        self.angle_per_step = angle_per_step  # Угол на один шаг в градусах

    def _update_xy_rotation_cache(self):
        angle_rad = np.deg2rad(self.xy_rotation_deg)
        self._xy_cos = float(np.cos(angle_rad))
        self._xy_sin = float(np.sin(angle_rad))

    def set_axis_transform(
        self,
        invert_x=None,
        invert_y=None,
        invert_z=None,
        xy_rotation_deg=None,
        offset_x=None,
        offset_y=None,
        offset_z=None
    ):
        """
        Configure user->robot coordinate transform.
        Useful when physical axes are mirrored/rotated relative to software axes.
        """
        if invert_x is not None:
            self.axis_sign[0] = -1.0 if invert_x else 1.0
        if invert_y is not None:
            self.axis_sign[1] = -1.0 if invert_y else 1.0
        if invert_z is not None:
            self.axis_sign[2] = -1.0 if invert_z else 1.0
        if xy_rotation_deg is not None:
            self.xy_rotation_deg = float(xy_rotation_deg)
            self._update_xy_rotation_cache()
        if offset_x is not None:
            self.offset_x = float(offset_x)
        if offset_y is not None:
            self.offset_y = float(offset_y)
        if offset_z is not None:
            self.offset_z = float(offset_z)

    def set_user_origin_from_angles(
        self,
        theta1=0.0,
        theta2=0.0,
        theta3=0.0,
        zero_x=True,
        zero_y=True,
        zero_z=True
    ):
        """
        Shift user coordinates so FK(theta1, theta2, theta3) becomes zero on selected axes.
        Returns tuple (offset_x, offset_y, offset_z).
        """
        x_robot, y_robot, z_robot, fl = self._fkinem_raw(theta1, theta2, theta3)
        if fl != 0 or np.isnan(x_robot) or np.isnan(y_robot) or np.isnan(z_robot):
            raise ValueError(
                f"Cannot set origin: FK is invalid for angles ({theta1}, {theta2}, {theta3})."
            )

        xu, yu, zu = self._from_robot_frame(x_robot, y_robot, z_robot)
        if zero_x:
            self.offset_x -= float(xu)
        if zero_y:
            self.offset_y -= float(yu)
        if zero_z:
            self.offset_z -= float(zu)
        return self.offset_x, self.offset_y, self.offset_z

    def set_user_z_zero_from_angles(self, theta1=0.0, theta2=0.0, theta3=0.0):
        """
        Define user Z=0 at the FK position for given joint angles.
        Example: set_user_z_zero_from_angles(0, 0, 0) makes fkinem(0,0,0).Z == 0.
        """
        _, _, offset_z = self.set_user_origin_from_angles(
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            zero_x=False,
            zero_y=False,
            zero_z=True
        )
        return offset_z

    def _to_robot_frame(self, X, Y, Z):
        """
        Convert point from user frame to robot frame.
        Order: translation -> axis sign -> XY rotation.
        """
        x = (float(X) - self.offset_x) * self.axis_sign[0]
        y = (float(Y) - self.offset_y) * self.axis_sign[1]
        z = (float(Z) - self.offset_z) * self.axis_sign[2]

        xr = self._xy_cos * x - self._xy_sin * y
        yr = self._xy_sin * x + self._xy_cos * y
        return xr, yr, z

    def _from_robot_frame(self, X, Y, Z):
        """
        Convert point from robot frame to user frame (inverse transform).
        """
        x = self._xy_cos * float(X) + self._xy_sin * float(Y)
        y = -self._xy_sin * float(X) + self._xy_cos * float(Y)
        z = float(Z)

        xu = x * self.axis_sign[0] + self.offset_x
        yu = y * self.axis_sign[1] + self.offset_y
        zu = z * self.axis_sign[2] + self.offset_z
        return xu, yu, zu

    def ikinem_th(self, x0, y0, z0):
        if abs(z0) < 1e-9:
            return np.nan

        y1 = -self.R
        y0 = y0 - self.r  # Смещение центра к краю

        a = (x0**2 + y0**2 + z0**2 + self.L**2 - self.l**2 - y1**2) / (2 * z0)
        b = (y1 - y0) / z0

        # Дискриминант
        D = -(a + b * y1)**2 + self.L**2 * (b**2 + 1)

        if D < 0:
            theta = np.nan  # Нет реального решения
        else:
            yj = (y1 - a * b - np.sqrt(D)) / (b**2 + 1)
            zj = a + b * yj
            theta = np.arctan2(-zj, y1 - yj)
            if yj > y1:
                theta += np.pi
            theta = np.degrees(theta)
        return theta

    def ikinem(self, X, Y, Z):
        # Respect current cube boundaries (if cube is configured).
        within_cube = self.is_within_cube(X, Y, Z)

        Xr, Yr, Zr = self._to_robot_frame(X, Y, Z)

        # First drive
        x0 = Xr
        y0 = Yr
        z0 = Zr
        theta1 = self.ikinem_th(x0, y0, z0)

        # Second drive (+120 deg)
        x0 = Xr * np.cos(2 * np.pi / 3) + Yr * np.sin(2 * np.pi / 3)
        y0 = Yr * np.cos(2 * np.pi / 3) - Xr * np.sin(2 * np.pi / 3)
        theta2 = self.ikinem_th(x0, y0, z0)

        # Third drive (-120 deg)
        x0 = Xr * np.cos(2 * np.pi / 3) - Yr * np.sin(2 * np.pi / 3)
        y0 = Yr * np.cos(2 * np.pi / 3) + Xr * np.sin(2 * np.pi / 3)
        theta3 = self.ikinem_th(x0, y0, z0)

        if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
            available = within_cube
        else:
            available = False

        return theta1, theta2, theta3, available

    def _fkinem_raw(self, theta1, theta2, theta3):
        t = self.R - self.r
        theta1_rad = np.radians(theta1)
        theta2_rad = np.radians(theta2)
        theta3_rad = np.radians(theta3)

        y1 = -(t + self.L * np.cos(theta1_rad))
        z1 = -self.L * np.sin(theta1_rad)

        y2 = (t + self.L * np.cos(theta2_rad)) * np.sin(np.pi / 6)
        x2 = y2 * np.tan(np.pi / 3)
        z2 = -self.L * np.sin(theta2_rad)

        y3 = (t + self.L * np.cos(theta3_rad)) * np.sin(np.pi / 6)
        x3 = -y3 * np.tan(np.pi / 3)
        z3 = -self.L * np.sin(theta3_rad)

        w1 = y1**2 + z1**2
        w2 = x2**2 + y2**2 + z2**2
        w3 = x3**2 + y3**2 + z3**2

        dnm = (y2 - y1) * x3 - (y3 - y1) * x2

        a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
        b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2
        a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
        b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2

        a = a1**2 + a2**2 + dnm**2
        b = 2 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm**2)
        c = (b2 - y1 * dnm)**2 + b1**2 + dnm**2 * (z1**2 - self.l**2)

        d = b**2 - 4 * a * c
        if d >= 0:
            Z = -0.5 * (b + np.sqrt(d)) / a
            X = (a1 * Z + b1) / dnm
            Y = (a2 * Z + b2) / dnm
            return X, Y, Z, 0

        return np.nan, np.nan, np.nan, -1

    def fkinem(self, theta1, theta2, theta3):
        Xr, Yr, Zr, fl = self._fkinem_raw(theta1, theta2, theta3)
        if fl != 0:
            return Xr, Yr, Zr, fl

        X, Y, Z = self._from_robot_frame(Xr, Yr, Zr)

        if not self.is_within_cube(X, Y, Z):
            return X, Y, Z, -1

        return X, Y, Z, 0

    def is_within_cube(self, X, Y, Z, show=False):
        if self.cube_origin is None or self.cube_size is None:
            # Параметры куба не вычислены, считаем, что точка внутри рабочей области
            return True

        x_min = self.cube_origin[0]
        x_max = x_min + self.cube_size
        y_min = self.cube_origin[1]
        y_max = y_min + self.cube_size
        z_min = self.cube_origin[2]
        z_max = z_min + self.cube_size

        if show:
            print("is_within_cube ===============")
            print(f"{x_min} <= {X} <= {x_max} | {(x_min <= X <= x_max)}")
            print(f"{y_min} <= {Y} <= {y_max} | {(y_min <= Y <= y_max)}")
            print(f"{z_min} <= {Z} <= {z_max} | {(z_min <= Z <= z_max)}")
            print("=============== ===============")

        return (x_min <= X <= x_max) and (y_min <= Y <= y_max) and (z_min <= Z <= z_max)

    def workspace(
        self,
        cache_file='workspace_new_delta.npy',
        h=1,
        force_rebuild=False,
        theta_min=-5,
        theta_max=90,
        apply_command_limits=False,
        x_abs_max=500,
        y_abs_max=500,
        z_min=-1200,
        z_max=0
    ):
        if (not force_rebuild) and os.path.exists(cache_file):
            print(f"[INFO] Loading workspace from '{cache_file}'...")
            return np.load(cache_file)

        if force_rebuild:
            print(f"[INFO] Force rebuild workspace (cache: '{cache_file}')...")
        print(
            f"[INFO] Calculating workspace with step {h} deg, "
            f"theta range [{theta_min}, {theta_max}], "
            f"apply_command_limits={apply_command_limits}..."
        )

        # Never clip workspace by an already-configured cube.
        prev_cube_size = self.cube_size
        prev_cube_origin = self.cube_origin
        prev_middle_z = self.middleZ
        prev_total = self.total
        self.cube_size = None
        self.cube_origin = None
        self.middleZ = None
        self.total = None

        try:
            th_range = np.arange(theta_min, theta_max + h, h)
            A_list = []

            total = len(th_range) ** 3
            processed = 0
            print_every = max(1, total // 100)

            for th1 in th_range:
                for th2 in th_range:
                    for th3 in th_range:
                        processed += 1

                        X, Y, Z, fl = self._fkinem_raw(th1, th2, th3)
                        if fl != 0:
                            continue

                        X, Y, Z = self._from_robot_frame(X, Y, Z)

                        if apply_command_limits:
                            if abs(X) > x_abs_max or abs(Y) > y_abs_max:
                                continue
                            if z_max is not None and Z > z_max:
                                continue
                            if z_min is not None and Z < z_min:
                                continue

                        A_list.append([X, Y, Z])

                        if processed % print_every == 0:
                            percent = processed / total * 100
                            print(f"\r[INFO] Progress: {percent:.1f}%...", end='')

            A = np.array(A_list)
            np.save(cache_file, A)
            print(f"\n[INFO] Saved {len(A)} points into '{cache_file}'")
            return A
        finally:
            self.cube_size = prev_cube_size
            self.cube_origin = prev_cube_origin
            self.middleZ = prev_middle_z
            self.total = prev_total

    def _point_reachable_strict(
        self,
        x,
        y,
        z,
        angle_min=-35.0,
        angle_max=105.0,
        pos_tol=1e-2,
        apply_command_limits=False,
        x_abs_max=300.0,
        y_abs_max=300.0,
        z_min=-500.0,
        z_max=0.0
    ):
        x_robot, y_robot, z_robot = self._to_robot_frame(x, y, z)

        # IK per arm directly (independent from current cube limits).
        th1 = self.ikinem_th(x_robot, y_robot, z_robot)

        x2 = x_robot * np.cos(2 * np.pi / 3) + y_robot * np.sin(2 * np.pi / 3)
        y2 = y_robot * np.cos(2 * np.pi / 3) - x_robot * np.sin(2 * np.pi / 3)
        th2 = self.ikinem_th(x2, y2, z_robot)

        x3 = x_robot * np.cos(2 * np.pi / 3) - y_robot * np.sin(2 * np.pi / 3)
        y3 = y_robot * np.cos(2 * np.pi / 3) + x_robot * np.sin(2 * np.pi / 3)
        th3 = self.ikinem_th(x3, y3, z_robot)

        if np.isnan(th1) or np.isnan(th2) or np.isnan(th3):
            return False

        if not (angle_min <= th1 <= angle_max):
            return False
        if not (angle_min <= th2 <= angle_max):
            return False
        if not (angle_min <= th3 <= angle_max):
            return False

        x_fk, y_fk, z_fk, fl = self._fkinem_raw(th1, th2, th3)
        if fl != 0:
            return False

        if np.isnan(x_fk) or np.isnan(y_fk) or np.isnan(z_fk):
            return False

        if abs(x_fk - x_robot) > pos_tol:
            return False
        if abs(y_fk - y_robot) > pos_tol:
            return False
        if abs(z_fk - z_robot) > pos_tol:
            return False

        if apply_command_limits:
            if abs(x) > x_abs_max or abs(y) > y_abs_max:
                return False
            if z_max is not None and z > z_max:
                return False
            if z_min is not None and z < z_min:
                return False

        return True

    def _cube_points_reachable_at(
        self,
        center_x,
        center_y,
        center_z,
        side,
        samples_per_axis=3,
        **reach_kwargs
    ):
        if side <= 0:
            return True

        half = side / 2.0
        xs = np.linspace(center_x - half, center_x + half, samples_per_axis)
        ys = np.linspace(center_y - half, center_y + half, samples_per_axis)
        zs = np.linspace(center_z - half, center_z + half, samples_per_axis)

        for x in xs:
            for y in ys:
                for z in zs:
                    if not self._point_reachable_strict(x, y, z, **reach_kwargs):
                        return False
        return True

    def cuboid(
        self,
        side_tol=1.0,
        coarse_samples=3,
        final_samples=5,
        max_centers=300,
        theta_min=-5.0,
        theta_max=90.0,
        apply_command_limits=False,
        x_abs_max=300.0,
        y_abs_max=300.0,
        z_min=-500.0,
        z_max=0.0,
        force_center_xy=False,
        center_x=0.0,
        center_y=0.0
    ):
        """
        Find an approximately maximal axis-aligned cube in reachable space.
        By default, cube center is searched in X/Y/Z.
        If force_center_xy=True, X/Y center is fixed (e.g. 0,0) and only Z is searched.
        """
        prev_cube_size = self.cube_size
        prev_cube_origin = self.cube_origin
        prev_middle_z = self.middleZ
        prev_total = self.total

        # Disable cube limits during cube search.
        self.cube_size = None
        self.cube_origin = None
        self.middleZ = None
        self.total = None

        try:
            # Build coarse reachable cloud for center candidates.
            th_range = np.arange(theta_min, theta_max + 5.0, 5.0)
            pts = []
            for th1 in th_range:
                for th2 in th_range:
                    for th3 in th_range:
                        x, y, z, fl = self._fkinem_raw(th1, th2, th3)
                        if fl != 0:
                            continue
                        x, y, z = self._from_robot_frame(x, y, z)
                        if apply_command_limits:
                            if abs(x) > x_abs_max or abs(y) > y_abs_max:
                                continue
                            if z_max is not None and z > z_max:
                                continue
                            if z_min is not None and z < z_min:
                                continue
                        pts.append((float(x), float(y), float(z)))

            if not pts:
                self.cube_size = prev_cube_size
                self.cube_origin = prev_cube_origin
                self.middleZ = prev_middle_z
                self.total = prev_total
                return 0.0, [0.0, 0.0, 0.0], 0.0, 0.0

            pts_np = np.array(pts, dtype=float)
            min_x, min_y, min_z_cloud = np.min(pts_np[:, 0]), np.min(pts_np[:, 1]), np.min(pts_np[:, 2])
            max_x, max_y, max_z_cloud = np.max(pts_np[:, 0]), np.max(pts_np[:, 1]), np.max(pts_np[:, 2])
            mid_z_cloud = (min_z_cloud + max_z_cloud) / 2.0

            if force_center_xy:
                # Search from robot center in XY: keep X/Y fixed, vary only Z.
                cx_fixed = float(center_x)
                cy_fixed = float(center_y)

                # Use reachable Z values ordered from the cloud middle outward.
                z_values = np.unique(np.round(pts_np[:, 2], 3))
                order = np.argsort(np.abs(z_values - mid_z_cloud))
                z_values = z_values[order]
                z_values = z_values[:max(1, int(max_centers))]

                candidate_centers = [(cx_fixed, cy_fixed, float(mid_z_cloud))]
                for zc in z_values:
                    zc = float(zc)
                    if abs(zc - mid_z_cloud) <= 1e-9:
                        continue
                    candidate_centers.append((cx_fixed, cy_fixed, zc))
            else:
                # Candidate centers: decimated reachable points + bounding-box center.
                stride = max(1, len(pts) // max_centers)
                candidate_centers = list(pts[::stride][:max_centers])
                candidate_centers.append(
                    (
                        (min_x + max_x) / 2.0,
                        (min_y + max_y) / 2.0,
                        mid_z_cloud
                    )
                )

            reach_kwargs = dict(
                angle_min=theta_min,
                angle_max=theta_max,
                apply_command_limits=apply_command_limits,
                x_abs_max=x_abs_max,
                y_abs_max=y_abs_max,
                z_min=z_min,
                z_max=z_max
            )

            best_side = 0.0
            if force_center_xy:
                best_center = (float(center_x), float(center_y), mid_z_cloud)
            else:
                best_center = (0.0, 0.0, mid_z_cloud)

            for cx, cy, cz in candidate_centers:
                if not self._point_reachable_strict(cx, cy, cz, **reach_kwargs):
                    continue

                # Upper bound from margins to the reachable cloud bounding box.
                hi = 2.0 * min(
                    cx - min_x, max_x - cx,
                    cy - min_y, max_y - cy,
                    cz - min_z_cloud, max_z_cloud - cz
                )
                if hi <= 0:
                    continue
                if hi <= best_side:
                    continue

                lo = 0.0
                for _ in range(14):
                    if hi - lo <= side_tol:
                        break
                    mid = (lo + hi) / 2.0
                    if self._cube_points_reachable_at(
                        cx, cy, cz, mid, samples_per_axis=coarse_samples, **reach_kwargs
                    ):
                        lo = mid
                    else:
                        hi = mid

                candidate_side = lo
                while candidate_side > 0 and not self._cube_points_reachable_at(
                    cx, cy, cz, candidate_side, samples_per_axis=final_samples, **reach_kwargs
                ):
                    candidate_side -= side_tol

                if candidate_side > best_side:
                    best_side = candidate_side
                    best_center = (cx, cy, cz)

            cx, cy, cz = best_center
            half = best_side / 2.0
            cube_size = float(best_side)
            cube_origin = [float(cx - half), float(cy - half), float(cz - half)]
            middle_z = float(cz)
            total = float(half)

            self.cube_size = cube_size
            self.cube_origin = cube_origin
            self.middleZ = middle_z
            self.total = total

            return cube_size, cube_origin, middle_z, total
        finally:
            # Keep the found cube in self.* when successful.
            if self.cube_size is None or self.cube_origin is None:
                self.cube_size = prev_cube_size
                self.cube_origin = prev_cube_origin
                self.middleZ = prev_middle_z
                self.total = prev_total

    def plot_cube(self, ax, edges, origin, alpha=1, color=(1, 0, 0)):
        """
        Рисует куб по длинам ребер edges = (x_size, y_size, z_size)
        с началом в точке origin = (x0, y0, z0).
        """
        X = [0, edges[0]]
        Y = [0, edges[1]]
        Z = [0, edges[2]]
        vertices = np.array([[x, y, z] for x in X for y in Y for z in Z])
        vertices += origin

        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],  # Нижняя грань
            [vertices[4], vertices[5], vertices[7], vertices[6]],  # Верхняя грань
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Передняя грань
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Задняя грань
            [vertices[1], vertices[3], vertices[7], vertices[5]],  # Правая грань
            [vertices[0], vertices[2], vertices[6], vertices[4]],  # Левая грань
        ]

        cube = Poly3DCollection(faces, linewidths=1, edgecolors='k')
        cube.set_facecolor((*color, alpha))
        ax.add_collection3d(cube)

    def set_axes_equal(self, ax):
        """Устанавливает равные масштабы по осям."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def graf_space(self, A, cube_size=None, cube_origin=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Окружность базовой платформы
        t = np.linspace(0, 2 * np.pi, 360)
        x_base = self.R * np.cos(t)
        y_base = self.R * np.sin(t)
        z_base = np.zeros_like(x_base)
        ax.plot(x_base, y_base, z_base, linewidth=2)

        # Синие точки (workspace)
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='blue', s=3, alpha=0.5)

        # Куб, если параметры указаны
        if cube_size is not None and cube_origin is not None:
            self.plot_cube(ax, [cube_size, cube_size, cube_size], cube_origin, alpha=0.3, color=(0, 1, 0))

        ax.grid(True)
        self.set_axes_equal(ax)
        plt.show()

    def bresenham_line_3d(self, x0, y0, z0, x1, y1, z1, resolution=1.0):
        """
        Возвращает список точек (X, Y, Z),
        лежащих "по линии" между (x0,y0,z0) и (x1,y1,z1),
        используя 3D-Брезенхэм по дискретной сетке с шагом resolution.
        """
        i0 = int(round(x0 / resolution))
        j0 = int(round(y0 / resolution))
        k0 = int(round(z0 / resolution))

        i1 = int(round(x1 / resolution))
        j1 = int(round(y1 / resolution))
        k1 = int(round(z1 / resolution))

        ijk_coords = bresenham3d_indices(i0, j0, k0, i1, j1, k1)

        line_points = []
        for (ii, jj, kk) in ijk_coords:
            X = ii * resolution
            Y = jj * resolution
            Z = kk * resolution
            line_points.append((X, Y, Z))

        return line_points

    def plot_trjaectory_line_3d(self, line_points):
        """
        Рисует 3D-график с "идеальной" (Брезенхэм) линией (красной) + базовая окружность (серой).
        """
        # line_points = self.bresenham_line_3d(x0, y0, z0, x1, y1, z1, resolution)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Окружность
        t = np.linspace(0, 2*np.pi, 360)
        x_base = self.R*np.cos(t)
        y_base = self.R*np.sin(t)
        z_base = np.zeros_like(x_base)
        ax.plot(x_base, y_base, z_base, linewidth=2, color='gray', label='Base Circle')

        line_points_arr = np.array(line_points)
        ax.scatter(line_points_arr[:,0], line_points_arr[:,1], line_points_arr[:,2],
                   color='red', s=20, label='Bresenham 3D')

        self.set_axes_equal(ax)
        ax.grid(True)
        ax.legend()
        plt.show()

    # -------------------------------------------------------------------------
    # НОВЫЙ МЕТОД: "приближение" линии точками из workspace
    # -------------------------------------------------------------------------
    def plot_approx_line_in_workspace(self, x0, y0, z0, x1, y1, z1, resolution=1.0):
        """
        1) Генерирует "синие" точки (A) рабочей области.
        2) Строит "идеальную" 3D-линию (line_points) по Брезенхэму.
        3) Для каждой точки линии ищет ближайшую точку из A.
        4) Рисует:
           - все A (синим),
           - приближённую "линию" (красным),
           - окружность базовой платформы (серым).
        """

        # 1) Рабочая область
        A = self.workspace()  # shape (N, 3)

        # Если рабочая область пустая, смысла нет
        if len(A) == 0:
            print("Рабочая область пуста!")
            return

        # 2) Идеальная линия
        line_points = self.bresenham_line_3d(x0, y0, z0, x1, y1, z1, resolution)

        # 3) Для каждой точки линии находим ближайшую в A
        A_np = np.array(A)  # (N,3)
        approx_line = []
        for (lx, ly, lz) in line_points:
            # Наивный поиск ближайшей точки из A
            # (O(N) на каждую точку линии)
            best_dist = float('inf')
            best_idx = -1
            for i in range(A_np.shape[0]):
                dx = A_np[i,0] - lx
                dy = A_np[i,1] - ly
                dz = A_np[i,2] - lz
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < best_dist:
                    best_dist = dist_sq
                    best_idx = i
            if best_idx >= 0:
                approx_line.append(A_np[best_idx,:])

        approx_line = np.array(approx_line)  # (M,3) - возможно, с дубликатами

        # Чтобы исключить повторения при "застревании" на одной точке,
        # можно оставить только уникальные, если хочется "ломаную":
        if approx_line.shape[0] > 0:
            approx_line_unique = np.unique(approx_line, axis=0)
        else:
            approx_line_unique = approx_line
        distances = np.linalg.norm(approx_line_unique - np.array([x0, y0, z0]), axis=1)
        sorted_indices = np.argsort(distances)
        approx_line_sorted = approx_line_unique[sorted_indices]
        # 4) Визуализация
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Окружность (серым)
        t = np.linspace(0, 2*np.pi, 360)
        x_base = self.R*np.cos(t)
        y_base = self.R*np.sin(t)
        z_base = np.zeros_like(x_base)
        ax.plot(x_base, y_base, z_base, linewidth=2, color='gray', label='Base Circle')

        # Синие точки workspace
        minX, maxX = min(approx_line_unique[:,0]), max(approx_line_unique[:,0])
        minY, maxY = min(approx_line_unique[:,1]), max(approx_line_unique[:,1])
        minZ, maxZ = min(approx_line_unique[:,2]), max(approx_line_unique[:,2])

        # # Фильтруем точки A (N×3):
        # # Берём только те, что по каждой координате X, Y, Z попадают в [min, max].
        mask = (
            (A[:,0] >= minX) & (A[:,0] <= maxX) &
            (A[:,1] >= minY) & (A[:,1] <= maxY) &
            (A[:,2] >= minZ) & (A[:,2] <= maxZ)
        )

        A_filtered = A[mask]  # это подмассив, где все условия выполнены

        print("Число точек в отфильтрованном A:", A_filtered.shape[0])
        ax.scatter(A_filtered[:,0], A_filtered[:,1], A_filtered[:,2],
                color='blue', s=5, alpha=0.4, label='Workspace filtered')
        # Рисуем их
        # ax.scatter(A[:,0], A[:,1], A[:,2],
        #         color='blue', s=5, alpha=0.4, label='Workspace filtered')


        # Красные точки = приближённая линия
        if approx_line_unique.shape[0] > 0:
            # ax.scatter(approx_line_unique[:,0],
            #            approx_line_unique[:,1],
            #            approx_line_unique[:,2],
            #            color='red', s=30, label='Approx line in workspace')
            # Вместо scatter рисуем текст (цифры) в каждой точке
            for idx, (x, y, z) in enumerate(approx_line_sorted):
                # Например, нумеруем их с 1
                ax.text(x, y, z, str(idx+1), color='red')
        else:
            print("Не удалось найти приближённых точек в workspace!")


        self.set_axes_equal(ax)
        ax.grid(True)
        ax.legend()
        ax.set_title("Синие: workspace, Красные: приближённая линия")
        plt.show()

    def linear_interpolation_3d(self, x0, y0, z0, x1, y1, z1, num_points=None, step_size=None):
        """
        Линейная интерполяция в 3D между двумя точками.

        Аргументы:
            x0, y0, z0 — координаты начальной точки
            x1, y1, z1 — координаты конечной точки
            num_points — количество интерполированных точек (включая начальную и конечную)
            step_size — шаг между точками (если не задан num_points)

        Возвращает:
            Список точек [(x, y, z), ...]
        """
        # Вектор разницы
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if num_points is not None:
            if num_points < 2:
                raise ValueError("num_points должно быть >= 2")
            steps = num_points - 1
        elif step_size is not None:
            if step_size <= 0:
                raise ValueError("step_size должно быть > 0")
            steps = int(np.ceil(distance / step_size))
        else:
            raise ValueError("Укажите либо num_points, либо step_size")

        points = []
        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * dx
            y = y0 + t * dy
            z = z0 + t * dz
            points.append((x, y, z))
        return points


    def plot_coordinates_over_time(self, trajectory_points, total_time_ms=5000):
        """
        trajectory_points: список кортежей [(x, y, z), ...]
        total_time_ms: общее время на всю траекторию, в миллисекундах
        """
        if len(trajectory_points) < 2:
            print("Недостаточно точек для построения графика.")
            return

        times = np.linspace(0, total_time_ms / 1000.0, len(trajectory_points))  # в секундах
        xs = [p[0] for p in trajectory_points]
        ys = [p[1] for p in trajectory_points]
        zs = [p[2] for p in trajectory_points]

        plt.figure(figsize=(10, 6))
        plt.plot(times, xs, label='X', linewidth=2)
        plt.plot(times, ys, label='Y', linewidth=2)
        plt.plot(times, zs, label='Z', linewidth=2)
        plt.xlabel("Время (сек)")
        plt.ylabel("Координата (мм)")
        plt.title("Координаты X, Y, Z во времени")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_trajectory_projections(self, trajectory_points):
        """
        Строит 2D-проекции траектории на плоскостях XY, XZ и YZ.
        
        Аргументы:
            trajectory_points: список кортежей [(x, y, z), ...]
        """
        if len(trajectory_points) < 2:
            print("Недостаточно точек для построения проекций.")
            return

        xs = [p[0] for p in trajectory_points]
        ys = [p[1] for p in trajectory_points]
        zs = [p[2] for p in trajectory_points]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # XY-проекция
        axs[0].plot(xs, ys, 'b-', linewidth=2)
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].set_title("Проекция XY")
        axs[0].grid(True)
        axs[0].axis('equal')

        # XZ-проекция
        axs[1].plot(xs, zs, 'g-', linewidth=2)
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Z")
        axs[1].set_title("Проекция XZ")
        axs[1].grid(True)
        axs[1].axis('equal')

        # YZ-проекция
        axs[2].plot(ys, zs, 'r-', linewidth=2)
        axs[2].set_xlabel("Y")
        axs[2].set_ylabel("Z")
        axs[2].set_title("Проекция YZ")
        axs[2].grid(True)
        axs[2].axis('equal')

        plt.tight_layout()
        plt.show()

    def plot_trajectory_in_workspace(self, workspace_points, trajectory_points, show_labels=True):
        """
        Рисует 3D-график:
        - workspace_points — синие точки рабочего пространства (Nx3)
        - trajectory_points — траектория (Mx3), рисуется красными точками
        - show_labels — показывать ли индекс каждой точки траектории
        """
        A = np.array(workspace_points)
        approx_line_sorted = np.array(trajectory_points)

        if A.shape[1] != 3 or approx_line_sorted.shape[1] != 3:
            raise ValueError("Ожидаются 3D-точки вида (x, y, z)")

        # Определяем ограничивающий бокс для workspace
        minX, maxX = np.min(approx_line_sorted[:,0]), np.max(approx_line_sorted[:,0])
        minY, maxY = np.min(approx_line_sorted[:,1]), np.max(approx_line_sorted[:,1])
        minZ, maxZ = np.min(approx_line_sorted[:,2]), np.max(approx_line_sorted[:,2])

        mask = (
            (A[:,0] >= minX) & (A[:,0] <= maxX) &
            (A[:,1] >= minY) & (A[:,1] <= maxY) &
            (A[:,2] >= minZ) & (A[:,2] <= maxZ)
        )
        A_filtered = A[mask]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Синие точки — рабочее пространство
        # ax.scatter(A_filtered[:,0], A_filtered[:,1], A_filtered[:,2],
        #         color='blue', s=5, alpha=0.4, label='Workspace')

        # Красные точки — траектория
        for idx, (x, y, z) in enumerate(approx_line_sorted):
            if show_labels:
                ax.text(x, y, z, str(idx+1), color='red', fontsize=8)
            else:
                ax.scatter(x, y, z, color='red', s=30)

        ax.set_title("Рабочее пространство и траектория")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1,1,1])
        plt.show() 
