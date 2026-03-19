import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api_server
from bridge.case_store import CaseStore


class ApiServerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        api_server.case_store = CaseStore(self.tmpdir.name)
        api_server.plan_cache.clear()
        api_server.sessions.clear()
        self.client = TestClient(api_server.app)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_decide_feedback_correct_status_flow(self):
        image_bytes = b"fake-jpeg-bytes"
        with patch.object(
            api_server,
            "analyze_image",
            return_value='{"action":"move_above","target":"red block","reason":"target visible"}',
        ):
            response = self.client.post(
                "/decide",
                files={"image": ("frame.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                data={
                    "task_description": "Pick the red block",
                    "context": "workspace_a",
                    "robot_state": json.dumps({"mode": "mock", "connected": True}),
                    "spatial_context": json.dumps({"camera": "top_down"}),
                    "metadata": json.dumps({"source": "test"}),
                    "session_id": "session_test",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["action"], "move_above")
        self.assertEqual(payload["target"], "red block")
        self.assertTrue(payload["case_id"])

        case_id = payload["case_id"]
        feedback = self.client.post(
            "/feedback",
            json={
                "action": "move_above",
                "target": "red block",
                "success": True,
                "case_id": case_id,
                "session_id": "session_test",
            },
        )
        self.assertEqual(feedback.status_code, 200)

        correction = self.client.post(
            f"/correct/{case_id}",
            json={
                "session_id": "session_test",
                "corrected_action": "lower",
                "corrected_target": "red block",
                "reason": "Need to descend before grasping",
                "reviewer": "human",
            },
        )
        self.assertEqual(correction.status_code, 200)

        status = self.client.get("/status?session_id=session_test")
        self.assertEqual(status.status_code, 200)
        self.assertGreaterEqual(status.json()["total_cases"], 1)

        case_root = Path(self.tmpdir.name)
        case_dir = None
        for log_path in sorted(case_root.rglob("log.json")):
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if data.get("request", {}).get("task_description") == "Pick the red block":
                case_dir = log_path.parent
                break

        self.assertIsNotNone(case_dir)
        log = json.loads((case_dir / "log.json").read_text(encoding="utf-8"))
        correction_path = None
        for candidate in sorted(case_root.rglob("correction.json")):
            correction_path = candidate
            break

        self.assertIsNotNone(correction_path)
        correction_file = json.loads(correction_path.read_text(encoding="utf-8"))

        self.assertEqual(log["request"]["task_description"], "Pick the red block")
        self.assertEqual(log["request"]["robot_state"]["mode"], "mock")
        self.assertEqual(log["decision"]["parsed_action"]["action"], "move_above")
        self.assertEqual(correction_file["corrected_action"]["action"], "lower")


if __name__ == "__main__":
    unittest.main()