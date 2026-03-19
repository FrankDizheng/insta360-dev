import json
import tempfile
import unittest
from pathlib import Path

from bridge.case_store import CaseStore


class CaseStoreTest(unittest.TestCase):
    def test_case_lifecycle_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CaseStore(tmpdir)
            case_id, case_dir = store.create_case(
                session_id="session_a",
                step=1,
                task_description="pick red block",
                context="workspace_a",
                image_bytes=b"fake-image-bytes",
            )

            store.log_decision(
                "session_a",
                case_id,
                prompt="test prompt",
                raw_response='{"action":"grasp"}',
                parsed_action={"action": "grasp", "target": "red block"},
                source="vlm",
                latency_ms=123,
            )
            store.log_feedback("session_a", case_id, "grasp", "red block", True, "ok")
            store.log_correction(
                "session_a",
                case_id,
                corrected_action={"action": "move_above", "target": "red block"},
                reviewer="human",
                notes="grasp was too early",
            )

            log = json.loads((Path(case_dir) / "log.json").read_text(encoding="utf-8"))
            correction = json.loads((Path(case_dir) / "correction.json").read_text(encoding="utf-8"))

            self.assertEqual(log["case_id"], case_id)
            self.assertEqual(log["decision"]["parsed_action"]["action"], "grasp")
            self.assertTrue(log["feedback"]["success"])
            self.assertEqual(correction["corrected_action"]["action"], "move_above")


if __name__ == "__main__":
    unittest.main()