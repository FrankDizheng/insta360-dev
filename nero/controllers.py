import os
import time
from abc import ABC, abstractmethod

from nero.types import Position3D


class BaseRobotController(ABC):
    def __init__(self):
        self.connected = False

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def get_status(self) -> dict: ...

    @abstractmethod
    def move_above(self, target_name: str, target_pos: Position3D | None = None) -> bool: ...

    @abstractmethod
    def lower(self, target_name: str, target_pos: Position3D | None = None) -> bool: ...

    @abstractmethod
    def grasp(self) -> bool: ...

    @abstractmethod
    def lift(self) -> bool: ...

    @abstractmethod
    def release(self) -> bool: ...

    @abstractmethod
    def stop(self) -> bool: ...


class MockRobotController(BaseRobotController):
    def __init__(self, sleep_s: float = 0.1):
        super().__init__()
        self.sleep_s = sleep_s
        self.last_action = ""

    def connect(self):
        print("[Robot] Connecting to robot arm...")
        self.connected = True
        print("[Robot] Connected (mock mode)")

    def get_status(self) -> dict:
        return {
            "connected": self.connected,
            "mode": "mock",
            "last_action": self.last_action,
        }

    def move_above(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        self.last_action = f"move_above:{target_name}"
        print(f"[Robot] Moving above {target_name}")
        time.sleep(self.sleep_s)
        return True

    def lower(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        self.last_action = f"lower:{target_name}"
        print(f"[Robot] Lowering to {target_name}")
        time.sleep(self.sleep_s)
        return True

    def grasp(self) -> bool:
        self.last_action = "grasp"
        print("[Robot] Closing gripper")
        time.sleep(self.sleep_s)
        return True

    def lift(self) -> bool:
        self.last_action = "lift"
        print("[Robot] Lifting")
        time.sleep(self.sleep_s)
        return True

    def release(self) -> bool:
        self.last_action = "release"
        print("[Robot] Opening gripper")
        time.sleep(self.sleep_s)
        return True

    def stop(self) -> bool:
        self.last_action = "stop"
        print("[Robot] Stop requested")
        return True


class NeroRobotController(BaseRobotController):
    def connect(self):
        raise RuntimeError(
            "NeroRobotController is scaffolded but not connected to the official SDK yet. "
            "Use mock mode until hardware and SDK integration are ready."
        )

    def get_status(self) -> dict:
        return {"connected": False, "mode": "nero", "ready": False}

    def move_above(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        raise NotImplementedError("NERO SDK integration pending")

    def lower(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        raise NotImplementedError("NERO SDK integration pending")

    def grasp(self) -> bool:
        raise NotImplementedError("NERO SDK integration pending")

    def lift(self) -> bool:
        raise NotImplementedError("NERO SDK integration pending")

    def release(self) -> bool:
        raise NotImplementedError("NERO SDK integration pending")

    def stop(self) -> bool:
        raise NotImplementedError("NERO SDK integration pending")


def get_robot_controller(mode: str | None = None) -> BaseRobotController:
    selected_mode = (mode or os.getenv("ROBOT_MODE", "mock")).lower()
    if selected_mode == "nero":
        return NeroRobotController()
    return MockRobotController()


def dispatch_action(robot: BaseRobotController, action: str, target: str = "", target_pos: Position3D | None = None) -> bool:
    if action == "move_above":
        return robot.move_above(target, target_pos)
    if action == "lower":
        return robot.lower(target, target_pos)
    if action == "grasp":
        return robot.grasp()
    if action == "lift":
        return robot.lift()
    if action == "release":
        return robot.release()
    if action == "stop":
        return robot.stop()
    if action in {"done", "wait"}:
        return True
    raise ValueError(f"Unsupported robot action: {action}")