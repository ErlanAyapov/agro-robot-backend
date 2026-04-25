import asyncio
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse


def build_legacy_ui_router(
    *,
    request_stop_and_release_resources: Callable[[], None],
    gen_mjpeg: Callable[[], Any],
    get_zone_config: Callable[[], Dict[str, Any]],
    move_zone_offset: Callable[..., Dict[str, Any]],
    set_zone_config: Callable[..., Dict[str, Any]],
    update_visual_state: Callable[[Dict[str, Any]], None],
    get_robot_settings: Callable[[], Dict[str, Any]],
    set_robot_settings: Callable[..., Dict[str, Any]],
    close_move_module_client: Callable[[], None],
    resolve_detection_conf: Callable[[Optional[float]], float],
    resolve_command_duration_ms: Callable[[Optional[int]], int],
    gen_processed_mjpeg: Callable[..., Any],
    get_visual_state_snapshot: Callable[[], Dict[str, Any]],
    gen_visualization_mjpeg: Callable[..., Any],
    build_default_video_path: Callable[[], str],
    websocket_poll_interval_sec: float = 0.1,
) -> APIRouter:
    router = APIRouter(tags=["legacy-ui"])

    @router.get("/video")
    def video():
        return StreamingResponse(gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @router.post("/video/stop")
    def stop_video():
        request_stop_and_release_resources()
        return {"stopped": True}

    @router.get("/video/zone")
    def get_video_zone_offset():
        cfg = get_zone_config()
        return {
            "x": int(cfg["offset_x_px"]),
            "y": int(cfg["offset_y_px"]),
            "size_px": int(cfg["size_px"]),
            "size_mm": float(cfg["size_mm"]),
        }

    @router.post("/video/zone/move")
    def move_video_zone_offset(dx: int = 0, dy: int = 0):
        cfg = move_zone_offset(dx=dx, dy=dy)
        update_visual_state(
            {
                "event": "zone_move",
                "count": 0,
                "detections": [],
                "zone_offset_x": int(cfg["offset_x_px"]),
                "zone_offset_y": int(cfg["offset_y_px"]),
                "zone_size_px": int(cfg["size_px"]),
                "zone_size_mm": float(cfg["size_mm"]),
            }
        )
        return {
            "x": int(cfg["offset_x_px"]),
            "y": int(cfg["offset_y_px"]),
            "size_px": int(cfg["size_px"]),
            "size_mm": float(cfg["size_mm"]),
        }

    @router.post("/video/zone/size")
    def resize_video_zone(delta_px: int = 0, size_px: Optional[int] = None):
        cfg = get_zone_config()
        target_size = int(cfg["size_px"]) + int(delta_px)
        if size_px is not None:
            target_size = int(size_px)

        cfg = set_zone_config(size_px=target_size)
        update_visual_state(
            {
                "event": "zone_size",
                "count": 0,
                "detections": [],
                "zone_offset_x": int(cfg["offset_x_px"]),
                "zone_offset_y": int(cfg["offset_y_px"]),
                "zone_size_px": int(cfg["size_px"]),
                "zone_size_mm": float(cfg["size_mm"]),
            }
        )
        return {
            "x": int(cfg["offset_x_px"]),
            "y": int(cfg["offset_y_px"]),
            "size_px": int(cfg["size_px"]),
            "size_mm": float(cfg["size_mm"]),
        }

    @router.post("/video/zone/mm")
    def update_video_zone_mm(size_mm: float):
        cfg = set_zone_config(size_mm=size_mm)
        update_visual_state(
            {
                "event": "zone_mm",
                "count": 0,
                "detections": [],
                "zone_offset_x": int(cfg["offset_x_px"]),
                "zone_offset_y": int(cfg["offset_y_px"]),
                "zone_size_px": int(cfg["size_px"]),
                "zone_size_mm": float(cfg["size_mm"]),
            }
        )
        return {
            "x": int(cfg["offset_x_px"]),
            "y": int(cfg["offset_y_px"]),
            "size_px": int(cfg["size_px"]),
            "size_mm": float(cfg["size_mm"]),
        }

    @router.post("/video/zone/reset")
    def reset_video_zone_offset():
        cfg = set_zone_config(offset_x_px=0, offset_y_px=0)
        update_visual_state(
            {
                "event": "zone_reset",
                "count": 0,
                "detections": [],
                "zone_offset_x": int(cfg["offset_x_px"]),
                "zone_offset_y": int(cfg["offset_y_px"]),
                "zone_size_px": int(cfg["size_px"]),
                "zone_size_mm": float(cfg["size_mm"]),
            }
        )
        return {
            "x": int(cfg["offset_x_px"]),
            "y": int(cfg["offset_y_px"]),
            "size_px": int(cfg["size_px"]),
            "size_mm": float(cfg["size_mm"]),
        }

    @router.get("/settings/robot")
    def legacy_get_robot_settings():
        return get_robot_settings()

    @router.post("/settings/robot")
    def legacy_set_robot_settings(payload: Dict[str, Any] = Body(default={})):
        raw = payload or {}
        cfg = set_robot_settings(
            xy_rotation_deg=raw.get("xy_rotation_deg"),
            invert_x=raw.get("invert_x"),
            invert_y=raw.get("invert_y"),
            invert_z=raw.get("invert_z"),
            detection_conf=raw.get("detection_conf"),
        )
        close_move_module_client()
        return cfg

    @router.get("/video/processed")
    def processed_video(
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
        resolved_conf = resolve_detection_conf(conf)
        resolved_duration = resolve_command_duration_ms(command_duration_ms)
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

    @router.websocket("/ws")
    async def detections_ws(websocket: WebSocket):
        await websocket.accept()
        last_seq = 0
        try:
            while True:
                state = dict(get_visual_state_snapshot() or {})
                if int(state.get("seq", 0)) > last_seq:
                    await websocket.send_json(state)
                    last_seq = int(state["seq"])
                await asyncio.sleep(float(websocket_poll_interval_sec))
        except WebSocketDisconnect:
            return

    @router.get("/visualize/video")
    def visualize_video(
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
        resolved_conf = resolve_detection_conf(conf)
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
                on_detections=update_visual_state,
                zone_config_provider=get_zone_config,
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return router
