from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RobotApiRuntime:
    get_active_port: Callable[[], str]
    set_active_port: Callable[[str], str]
    list_serial_ports: Callable[[], List[Dict[str, str]]]
    serial_lock: Any
    reset_motion_runtime_state: Callable[[], None]
    close_limit_switch_client: Callable[[], None]
    close_move_module_client: Callable[[], None]
    get_manual_robot_state: Callable[[], Dict[str, Any]]
    clear_manual_robot_state: Callable[[], Dict[str, Any]]
    dispatch_manual_robot_move: Callable[..., Dict[str, Any]]
    get_vision_state: Callable[[], Dict[str, Any]]
    is_robot_busy: Callable[[], bool]
    delta_go_home: Callable[[], Dict[str, Any]]
    queue_robot_command: Optional[Callable[..., Dict[str, Any]]] = None
    delta_stop: Optional[Callable[[], Dict[str, Any]]] = None
    get_visual_state_snapshot: Optional[Callable[[], Dict[str, Any]]] = None
    vision_start: Optional[Callable[[], Dict[str, Any]]] = None
    vision_stop: Optional[Callable[[], Dict[str, Any]]] = None
    get_zone_config: Optional[Callable[[], Dict[str, Any]]] = None
    set_zone_config: Optional[Callable[..., Dict[str, Any]]] = None
    get_system_devices: Optional[Callable[[], List[Dict[str, Any]]]] = None
    publish_event: Optional[Callable[..., Dict[str, Any]]] = None
    pull_events_since: Optional[Callable[..., Dict[str, Any]]] = None
