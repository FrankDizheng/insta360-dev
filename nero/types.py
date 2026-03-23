from dataclasses import dataclass, field
from typing import Any


NUM_JOINTS = 7

JOINT_LIMITS_DEG: list[tuple[float, float]] = [
    (-175.0, 175.0),   # J1
    (-175.0, 175.0),   # J2
    (-175.0, 175.0),   # J3
    (-175.0, 175.0),   # J4
    (-175.0, 175.0),   # J5
    (-175.0, 175.0),   # J6
    (-175.0, 175.0),   # J7
]

HOME_ANGLES_DEG: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def clamp_joints(angles_deg: list[float]) -> list[float]:
    """Clamp each joint angle to its safe operating range."""
    clamped = []
    for i, a in enumerate(angles_deg):
        lo, hi = JOINT_LIMITS_DEG[i] if i < len(JOINT_LIMITS_DEG) else (-175.0, 175.0)
        clamped.append(max(lo, min(hi, a)))
    return clamped


@dataclass
class Position3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, coords: list[float]) -> "Position3D":
        return cls(x=coords[0], y=coords[1], z=coords[2] if len(coords) > 2 else 0.0)


@dataclass
class FlangePose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

    @classmethod
    def from_list(cls, values: list[float]) -> "FlangePose":
        return cls(
            x=values[0] if len(values) > 0 else 0.0,
            y=values[1] if len(values) > 1 else 0.0,
            z=values[2] if len(values) > 2 else 0.0,
            roll=values[3] if len(values) > 3 else 0.0,
            pitch=values[4] if len(values) > 4 else 0.0,
            yaw=values[5] if len(values) > 5 else 0.0,
        )


@dataclass
class GripperState:
    width: float | None = None
    force: float | None = None
    enabled: bool = False
    error: bool = False
    homed: bool = False
    available: bool = False


@dataclass
class ArmState:
    connected: bool = False
    enabled: bool = False
    mode: str = "unknown"
    joint_angles_deg: list[float] = field(default_factory=list)
    flange_pose: FlangePose | None = None
    gripper: GripperState = field(default_factory=GripperState)
    error_msg: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "connected": self.connected,
            "enabled": self.enabled,
            "mode": self.mode,
            "joint_angles_deg": self.joint_angles_deg,
        }
        if self.flange_pose:
            d["flange_pose"] = self.flange_pose.as_list()
        d["gripper"] = {
            "width": self.gripper.width,
            "force": self.gripper.force,
            "enabled": self.gripper.enabled,
            "available": self.gripper.available,
        }
        if self.error_msg:
            d["error"] = self.error_msg
        return d


@dataclass
class ActionDecision:
    action: str
    target: str = ""
    reason: str = ""
    source: str = "vlm"
    latency_ms: float = 0.0
    step: int = 0
    case_id: str = ""
    task_description: str = ""
    status: str = "ok"
    target_pos: Position3D | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionDecision":
        return cls(
            action=payload.get("action", "done"),
            target=payload.get("target", ""),
            reason=payload.get("reason", ""),
            source=payload.get("source", "vlm"),
            latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
            step=int(payload.get("step", 0) or 0),
            case_id=payload.get("case_id", ""),
            task_description=payload.get("task_description", ""),
            status=payload.get("status", "ok"),
            target_pos=Position3D.from_list(payload["target_pos"]) if isinstance(payload.get("target_pos"), list) else None,
        )
