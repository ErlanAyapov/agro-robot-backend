#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


Point = Tuple[int, float, float]

DEFAULT_FOV_WIDTH_MM = 180.0
DEFAULT_FOV_HEIGHT_MM = 180.0
REAL_ZONE_SIZE_MM = 138.0
PREDICTION_DELAY_MS = 500.0

_MODEL_LOCK = Lock()
_MODEL = None
_MODEL_PATH = None
_VIDEO_CACHE_LOCK = Lock()
VISUALIZATION_CACHE_WIDTH = 640
VISUALIZATION_CACHE_HEIGHT = 480
_VIDEO_CACHE_DIRNAME = "cache"
log = logging.getLogger(__name__)


def _resolve_model_device(model_device: Optional[str] = None) -> str:
    env_device = os.getenv("VISUALIZE_MODEL_DEVICE")
    requested = (model_device or env_device or "").strip()

    cuda_available = bool(torch is not None and torch.cuda.is_available())

    if requested:
        req = requested.lower()
        if req.startswith("cuda") or req.isdigit():
            if cuda_available:
                return requested
            return "cpu"
        return requested

    if cuda_available:
        return "cuda:0"
    return "cpu"


def configure_utf8_console() -> None:
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


def load_points(path: str) -> List[Point]:
    points: List[Point] = []
    if not os.path.exists(path):
        return points

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            try:
                weed_id = int(parts[0])
                global_x_mm = float(parts[1])
                global_y_mm = float(parts[2])
            except ValueError:
                continue

            points.append((weed_id, global_x_mm, global_y_mm))

    return points


def build_default_input_path() -> str:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "data" / "weed_report" / "weed_cor_to_send.txt",
        base_dir.parent / "data" / "weed_report" / "weed_cor_to_send.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def build_default_video_path() -> str:
    base_dir = Path(__file__).resolve().parent
    env_path = os.getenv("VISUALIZE_VIDEO_PATH")

    candidates = [
        str(Path(env_path).expanduser().resolve()) if env_path else None,
        str(base_dir / "video.mp4"),
        str(base_dir / "video_compressed.mp4"),
        str(base_dir.parent / "old_scripts" / "video.mp4"),
        str(base_dir.parent / "old_scripts" / "video_compressed.mp4"),
        str(base_dir.parent / "data" / "tracked_video.mp4"),
        str(base_dir.parent / "data" / "weed_report" / "tracked_video.mp4"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return str(base_dir / "video.mp4")


def _build_cached_video_paths(source_path: Path, width: int, height: int) -> Tuple[Path, Path]:
    cache_dir = Path(__file__).resolve().parent / _VIDEO_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(f"{source_path}|{width}x{height}".encode("utf-8")).hexdigest()[:16]
    base_name = f"visualization_{cache_key}_{width}x{height}"
    return cache_dir / f"{base_name}.mp4", cache_dir / f"{base_name}.json"


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _is_cached_video_valid(
    cached_video_path: Path,
    cached_meta_path: Path,
    source_path: Path,
    width: int,
    height: int,
) -> bool:
    if not cached_video_path.exists() or not cached_meta_path.exists():
        return False
    try:
        if cached_video_path.stat().st_size <= 0:
            return False
    except Exception:
        return False

    metadata = _read_json_dict(cached_meta_path)
    if metadata is None:
        return False

    try:
        source_stat = source_path.stat()
    except Exception:
        return False

    if str(metadata.get("source_path", "")) != str(source_path):
        return False
    if int(metadata.get("source_size", -1)) != int(source_stat.st_size):
        return False
    if int(metadata.get("source_mtime_ns", -1)) != int(source_stat.st_mtime_ns):
        return False
    if int(metadata.get("width", -1)) != int(width):
        return False
    if int(metadata.get("height", -1)) != int(height):
        return False
    return True


def _prepare_cached_visualization_video(
    video_path: str,
    target_width: int = VISUALIZATION_CACHE_WIDTH,
    target_height: int = VISUALIZATION_CACHE_HEIGHT,
) -> str:
    source_path = Path(video_path).expanduser().resolve()
    if not source_path.exists():
        return str(source_path)

    width = max(160, int(target_width))
    height = max(120, int(target_height))
    cached_video_path, cached_meta_path = _build_cached_video_paths(source_path, width, height)

    with _VIDEO_CACHE_LOCK:
        if _is_cached_video_valid(cached_video_path, cached_meta_path, source_path, width, height):
            return str(cached_video_path)

        temp_video_path = cached_video_path.with_suffix(".tmp.mp4")
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return str(source_path)

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(source_fps) or source_fps <= 1e-3:
            source_fps = 25.0

        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            return str(source_path)

        frames_written = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(resized)
                frames_written += 1
        finally:
            capture.release()
            writer.release()

        if frames_written <= 0 or not temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except Exception:
                pass
            return str(source_path)

        try:
            if temp_video_path.stat().st_size <= 0:
                temp_video_path.unlink()
                return str(source_path)
        except Exception:
            return str(source_path)

        try:
            temp_video_path.replace(cached_video_path)
        except Exception:
            try:
                temp_video_path.unlink()
            except Exception:
                pass
            return str(source_path)

        metadata = {
            "source_path": str(source_path),
            "source_size": int(source_path.stat().st_size),
            "source_mtime_ns": int(source_path.stat().st_mtime_ns),
            "width": int(width),
            "height": int(height),
            "fps": float(source_fps),
            "frames": int(frames_written),
            "prepared_at": int(time.time()),
        }
        try:
            cached_meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        log.info(
            "Prepared visualization cache '%s' -> '%s' (%dx%d, frames=%d).",
            source_path.name,
            cached_video_path.name,
            width,
            height,
            frames_written,
        )
        return str(cached_video_path)


def build_default_weights_path() -> str:
    base_dir = Path(__file__).resolve().parent
    env_path = os.getenv("VISUALIZE_WEIGHTS_PATH")

    candidates = [
        str(Path(env_path).expanduser().resolve()) if env_path else None,
        str(base_dir / "models" / "best_l.pt"),
        str(base_dir.parent / "models" / "best_l.pt"),
        str(base_dir.parent / "models" / "best.pt"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return str(base_dir.parent / "models" / "best_l.pt")


def _get_yolo_model(weights_path: Optional[str] = None):
    global _MODEL, _MODEL_PATH

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not available in this environment.")

    model_path = weights_path or build_default_weights_path()
    if not os.path.exists(model_path):
        raise RuntimeError(f"Weights file not found: {model_path}")

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PATH != model_path:
            _MODEL = YOLO(model_path)
            _MODEL_PATH = model_path
    return _MODEL


def _detect_objects(
    model,
    frame,
    confidence_threshold: float,
    device: Optional[str] = None,
    imgsz: int = 512,
    use_half: bool = False,
):
    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        verbose=False,
        device=device,
        imgsz=imgsz,
        half=use_half,
    )
    if not results:
        return []

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    det = np.concatenate([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)], axis=1)
    return [d for d in det if d[4] >= confidence_threshold]


def _render_error_frame(text: str, width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "Visualization stream error", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    cv2.putText(frame, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame


def _overlay_points_info(frame: np.ndarray, points: List[Point], show_ids: bool) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (620, 150), (20, 20, 20), thickness=-1)
    alpha = 0.45
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, f"Weed points loaded: {len(points)}", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if show_ids and points:
        sample_ids = ", ".join(str(p[0]) for p in points[:8])
        cv2.putText(frame, f"IDs: {sample_ids}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
    cv2.putText(frame, "Mode: detection + prediction overlay", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    return frame


def _compute_speed_with_optical_flow(prev_gray, curr_gray, prev_points, mm_per_pixel: float, fps: float):
    if prev_points is None or len(prev_points) == 0:
        return 0.0, None

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)
    if curr_points is None or status is None:
        return 0.0, None

    displacements = []
    for i, st in enumerate(status):
        if st == 1:
            dx = curr_points[i][0][0] - prev_points[i][0][0]
            dy = curr_points[i][0][1] - prev_points[i][0][1]
            displacements.append((dx * dx + dy * dy) ** 0.5)

    if not displacements:
        return 0.0, curr_points

    avg_disp_px = float(np.mean(displacements))
    speed_mm_per_sec = avg_disp_px * mm_per_pixel * max(fps, 1.0)
    return speed_mm_per_sec, curr_points


def _draw_detection_overlay(
    frame: np.ndarray,
    detections,
    cart_speed_mm_per_sec: float,
    mm_per_pixel: float,
    zone_x1: int,
    zone_y1: int,
    zone_x2: int,
    zone_y2: int,
    speed_x_mm_per_sec: Optional[float] = None,
    speed_y_mm_per_sec: Optional[float] = None,
    prediction_horizon_ms: Optional[float] = None,
):
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
    speed_m_per_sec = float(cart_speed_mm_per_sec) / 1000.0
    horizon_ms = float(prediction_horizon_ms) if prediction_horizon_ms is not None else float(PREDICTION_DELAY_MS)
    if speed_x_mm_per_sec is not None and speed_y_mm_per_sec is not None:
        speed_text = (
            f"Platform Speed: {speed_m_per_sec:.3f} m/s "
            f"(vx={float(speed_x_mm_per_sec) / 1000.0:.3f}, vy={float(speed_y_mm_per_sec) / 1000.0:.3f})"
        )
    else:
        speed_text = f"Platform Speed: {speed_m_per_sec:.3f} m/s"
    cv2.putText(
        frame,
        speed_text,
        (zone_x1 + 12, zone_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 0),
        2,
    )

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"id:{int(cls)} conf:{float(conf):.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        if mm_per_pixel <= 1e-9:
            continue

        if speed_x_mm_per_sec is not None and speed_y_mm_per_sec is not None:
            delta_px_x = (float(speed_x_mm_per_sec) * (horizon_ms / 1000.0)) / max(mm_per_pixel, 1e-9)
            delta_px_y = (float(speed_y_mm_per_sec) * (horizon_ms / 1000.0)) / max(mm_per_pixel, 1e-9)
            predicted_px = int(round(float(center_x) + float(delta_px_x)))
            predicted_py = int(round(float(center_y) + float(delta_px_y)))
        else:
            delta_mm = float(cart_speed_mm_per_sec) * (float(PREDICTION_DELAY_MS) / 1000.0)
            predicted_px = int(center_x - delta_mm / mm_per_pixel)
            predicted_py = int(center_y)

        if zone_x1 < predicted_px < zone_x2 and zone_y1 < predicted_py < zone_y2:
            cv2.circle(frame, (predicted_px, predicted_py), 12, (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Predicted",
                (predicted_px + 10, max(20, predicted_py - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    return frame


def gen_visualization_mjpeg(
    video_path: Optional[str] = None,
    loop: bool = True,
    input_path: Optional[str] = None,
    show_ids: bool = False,
    jpeg_quality: int = 75,
    enable_detection: bool = True,
    confidence_threshold: float = 0.3,
    detection_interval: int = 4,
    detection_imgsz: int = 512,
    model_device: Optional[str] = None,
    use_half: bool = True,
    weights_path: Optional[str] = None,
    max_width: int = 1280,
    on_detections: Optional[Callable[[Dict[str, Any]], None]] = None,
    zone_config_provider: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Generator[bytes, None, None]:
    points = load_points(input_path or build_default_input_path())
    original_source = video_path or build_default_video_path()
    source = _prepare_cached_visualization_video(
        original_source,
        target_width=VISUALIZATION_CACHE_WIDTH,
        target_height=VISUALIZATION_CACHE_HEIGHT,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and source != original_source:
        cap = cv2.VideoCapture(original_source)
        source = original_source
    if not cap.isOpened():
        error_frame = _render_error_frame(f"Cannot open video source: {source}")
        ok, jpg = cv2.imencode(".jpg", error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            return
        payload = jpg.tobytes()
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(1.0)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    resolved_device = _resolve_model_device(model_device)
    use_half = bool(use_half and "cuda" in resolved_device.lower())
    model = None
    model_error = None
    if enable_detection:
        try:
            model = _get_yolo_model(weights_path=weights_path)
            # Warm-up avoids first-frame lag spikes when using GPU.
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

    frame_id = 0
    prev_gray = None
    prev_points = None
    cart_speed_mm_per_sec = 0.0
    last_detections = []

    try:
        while True:
            start_ts = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if not loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_id += 1
            frame_h, frame_w = frame.shape[:2]
            fallback_mm_per_pixel = (
                (DEFAULT_FOV_WIDTH_MM / max(frame_w, 1))
                + (DEFAULT_FOV_HEIGHT_MM / max(frame_h, 1))
            ) / 2.0

            zone_cfg: Dict[str, Any] = {}
            if zone_config_provider is not None:
                try:
                    zone_cfg = zone_config_provider() or {}
                except Exception:
                    zone_cfg = {}

            zone_offset_x = int(zone_cfg.get("offset_x_px", 0))
            zone_offset_y = int(zone_cfg.get("offset_y_px", 0))
            default_zone_px = int(REAL_ZONE_SIZE_MM / max(fallback_mm_per_pixel, 1e-9))
            zone_size_px = int(zone_cfg.get("size_px", default_zone_px))
            zone_size_px = int(np.clip(zone_size_px, 80, max(80, min(frame_w, frame_h) - 4)))
            zone_size_mm = float(zone_cfg.get("size_mm", REAL_ZONE_SIZE_MM))
            if not np.isfinite(zone_size_mm) or zone_size_mm <= 1e-9:
                zone_size_mm = float(REAL_ZONE_SIZE_MM)

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

            if model is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and prev_points is not None:
                    cart_speed_mm_per_sec, prev_points = _compute_speed_with_optical_flow(
                        prev_gray, gray, prev_points, mm_per_pixel, fps
                    )
                else:
                    prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=5, qualityLevel=0.01, minDistance=40)
                prev_gray = gray

                if frame_id % max(1, detection_interval) == 0:
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

                    if on_detections is not None:
                        det_payload = []
                        delta_mm = cart_speed_mm_per_sec * (PREDICTION_DELAY_MS / 1000.0)
                        half_zone_mm = float(zone_size_mm) / 2.0
                        for d in last_detections[:20]:
                            x1 = float(d[0])
                            y1 = float(d[1])
                            x2 = float(d[2])
                            y2 = float(d[3])
                            center_x = int(round((x1 + x2) / 2.0))
                            center_y = int(round((y1 + y2) / 2.0))
                            predicted_px = int(center_x - delta_mm / max(mm_per_pixel, 1e-9))
                            predicted_py = int(center_y)
                            entry = {
                                "class": int(d[5]),
                                "confidence": round(float(d[4]), 4),
                                "x": center_x,
                                "y": center_y,
                            }
                            if zone_x1 < predicted_px < zone_x2 and zone_y1 < predicted_py < zone_y2:
                                # Delta-robot coordinates are centered at the zone center.
                                rel_x = (predicted_py - zone_center_y) * mm_per_pixel
                                rel_y = -(predicted_px - zone_center_x) * mm_per_pixel
                                entry["x_mm"] = int(round(np.clip(rel_x, -half_zone_mm, half_zone_mm)))
                                entry["y_mm"] = int(round(np.clip(rel_y, -half_zone_mm, half_zone_mm)))
                                entry["command_point_x"] = int(predicted_px)
                                entry["command_point_y"] = int(predicted_py)
                            det_payload.append(entry)
                        on_detections(
                            {
                                "event": "detection",
                                "frame_id": frame_id,
                                "count": len(last_detections),
                                "detections": det_payload,
                                "device": resolved_device,
                                "detect_ms": round(detect_ms, 2),
                                "source": "visualization",
                                "zone_offset_x": int(zone_offset_x),
                                "zone_offset_y": int(zone_offset_y),
                                "zone_size_px": int(zone_size_px),
                                "zone_size_mm": float(zone_size_mm),
                                "mm_per_pixel": round(float(mm_per_pixel), 5),
                            }
                        )

                frame = _draw_detection_overlay(
                    frame,
                    last_detections,
                    cart_speed_mm_per_sec,
                    mm_per_pixel,
                    zone_x1,
                    zone_y1,
                    zone_x2,
                    zone_y2,
                )
            elif model_error:
                cv2.putText(
                    frame,
                    f"Detection disabled: {model_error}",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            frame = _overlay_points_info(frame, points, show_ids=show_ids)
            cv2.putText(
                frame,
                f"YOLO device: {resolved_device}",
                (24, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if "cuda" in resolved_device.lower() else (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Zone offset px: ({zone_offset_x:+d}, {zone_offset_y:+d})",
                (24, 198),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Zone size: {zone_size_px}px ({zone_size_mm:.1f}mm)",
                (24, 226),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            cv2.putText(
                frame,
                f"Frame time: {elapsed_ms:.1f} ms",
                (24, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if max_width > 0 and frame.shape[1] > max_width:
                resized_h = int(frame.shape[0] * max_width / frame.shape[1])
                frame = cv2.resize(frame, (max_width, resized_h), interpolation=cv2.INTER_AREA)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                continue

            payload = jpg.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
    finally:
        cap.release()


def plot_points(points: List[Point], show_ids: bool) -> None:
    import matplotlib.pyplot as plt

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, c="red", s=35)

    if show_ids:
        for weed_id, x_mm, y_mm in points:
            plt.text(x_mm, y_mm, str(weed_id), fontsize=8)

    plt.title("Weed Global Coordinates")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detected weed coordinates.")
    parser.add_argument(
        "--input",
        default=build_default_input_path(),
        help="Path to weed_cor_to_send.txt (default: data/weed_report/weed_cor_to_send.txt).",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Draw weed IDs near points.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    points = load_points(args.input)

    if not points:
        print(f"Нет валидных точек для отображения: {args.input}")
        return

    print(f"Загружено точек: {len(points)}")
    plot_points(points, show_ids=args.show_ids)


if __name__ == "__main__":
    main()
