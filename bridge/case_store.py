import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


DEFAULT_CASES_ROOT = Path(__file__).resolve().parent.parent / "data" / "cases"


class CaseStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else DEFAULT_CASES_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def create_case(
        self,
        session_id: str,
        step: int,
        task_description: str,
        context: str,
        image_bytes: bytes,
        image_name: str = "input.jpg",
    ) -> tuple[str, Path]:
        case_id = f"case_{step:03d}_{uuid4().hex[:8]}"
        case_dir = self._session_dir(session_id) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        image_path = case_dir / image_name
        image_path.write_bytes(image_bytes)

        log_path = case_dir / "log.json"
        log_path.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "session_id": session_id,
                    "step": step,
                    "timestamp": self._utc_now(),
                    "task_description": task_description,
                    "context": context,
                    "image_path": str(image_path.relative_to(self.root)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return case_id, case_dir

    def _merge_log(self, case_dir: Path, patch: dict):
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / "log.json"
        data = {}
        if log_path.exists():
            data = json.loads(log_path.read_text(encoding="utf-8"))
        data.update(patch)
        log_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_request(
        self,
        case_dir: Path,
        task_description: str,
        context: str,
        robot_state: dict,
        spatial_context: dict,
        metadata: dict,
        use_cache: bool,
    ):
        self._merge_log(
            case_dir,
            {
                "request": {
                    "task_description": task_description,
                    "context": context,
                    "robot_state": robot_state,
                    "spatial_context": spatial_context,
                    "metadata": metadata,
                    "use_cache": use_cache,
                }
            },
        )

    def log_decision(
        self,
        session_id: str,
        case_id: str,
        prompt: str,
        raw_response: str,
        parsed_action: dict,
        source: str,
        latency_ms: float,
    ):
        case_dir = self._session_dir(session_id) / case_id
        self._merge_log(
            case_dir,
            {
                "decision": {
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_action": parsed_action,
                    "source": source,
                    "latency_ms": latency_ms,
                }
            },
        )

    def log_feedback(
        self,
        session_id: str,
        case_id: str,
        action: str,
        target: str,
        success: bool,
        detail: str = "",
    ):
        case_dir = self._session_dir(session_id) / case_id
        self._merge_log(
            case_dir,
            {
                "feedback": {
                    "timestamp": self._utc_now(),
                    "action": action,
                    "target": target,
                    "success": success,
                    "detail": detail,
                }
            },
        )

    def log_correction(
        self,
        session_id: str,
        case_id: str,
        corrected_action: dict,
        reviewer: str,
        notes: str = "",
    ):
        case_dir = self._session_dir(session_id) / case_id
        correction_path = case_dir / "correction.json"
        correction_path.write_text(
            json.dumps(
                {
                    "timestamp": self._utc_now(),
                    "reviewer": reviewer,
                    "notes": notes,
                    "corrected_action": corrected_action,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )