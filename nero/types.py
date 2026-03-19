from dataclasses import dataclass
from typing import Any


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