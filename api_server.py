"""Generic vision-decision service for robot MVP integration.

This service accepts an image plus task context, calls a multimodal backend,
returns a structured action, and logs cases for replay/human correction.
"""
import os, sys, json, re, time
from typing import Any, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bridge"))
from model_feed import analyze_image
from case_store import CaseStore

app = FastAPI(title="Vision Decision Service", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

case_store = CaseStore()

DECISION_PROMPT = """You are a robot controller.
You see the current workspace from a camera and must decide the next single robot action.

Output EXACTLY ONE JSON:
{"action": "<action>", "target": "<target>", "reason": "<brief reason>"}

Valid actions: move_above, lower, grasp, lift, release, wait, done
Valid targets: an object name, a zone name, or an empty string.

Follow the task description strictly.
Return only the best next action, not the whole plan.

Rules:
- Output ONLY the JSON.
- Keep actions simple and executable.
"""


def _default_session_state() -> dict:
    return {
        "action_history": [],
        "held_object": None,
        "completed_targets": [],
        "template_learned": False,
        "last_task_description": "",
        "vlm_calls": 0,
        "cache_hits": 0,
        "case_count": 0,
    }


sessions: dict[str, dict] = {}
plan_cache: dict[tuple, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = _default_session_state()
    return sessions[session_id]


class ActionResponse(BaseModel):
    action: str
    target: str
    reason: str = ""
    source: str = "vlm"
    latency_ms: float = 0
    step: int = 0
    case_id: str = ""
    task_description: str = ""
    status: str = "ok"


class CorrectionRequest(BaseModel):
    session_id: str = "default"
    corrected_action: str
    corrected_target: str = ""
    reason: str = ""
    reviewer: str = "human"


class FeedbackRequest(BaseModel):
    action: str
    target: str = ""
    success: bool
    detail: str = ""
    case_id: str = ""
    session_id: str = "default"


class SessionStatus(BaseModel):
    session_id: str
    action_history: list
    held_object: Optional[str]
    completed_targets: list
    template_learned: bool
    total_vlm_calls: int
    total_cache_hits: int
    total_cases: int


def parse_action(text):
    m = re.search(r'\{[^}]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"action": "done", "target": "", "reason": "parse_error"}


def parse_json_dict(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        return {}
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"raw": raw_text}
    return value if isinstance(value, dict) else {"value": value}


@app.post("/decide", response_model=ActionResponse)
async def decide_action(
    image: UploadFile = File(...),
    task_description: str = Form(""),
    context: str = Form(""),
    robot_state: str = Form(""),
    spatial_context: str = Form(""),
    metadata: str = Form(""),
    use_cache: bool = Form(True),
    session_id: str = Form("default"),
):
    """
    Send a camera image, receive the next robot action.

    - image: current camera frame (JPEG/PNG)
    - task_description: high-level task to execute
    - context: optional extra context
    - use_cache: whether to use plan template caching
    """
    ss = _get_session(session_id)
    step = len(ss["action_history"])
    ss["last_task_description"] = task_description
    parsed_robot_state = parse_json_dict(robot_state)
    parsed_spatial_context = parse_json_dict(spatial_context)
    parsed_metadata = parse_json_dict(metadata)

    cache_key = (
        session_id,
        task_description,
        json.dumps(parsed_robot_state, ensure_ascii=False, sort_keys=True),
        json.dumps(parsed_spatial_context, ensure_ascii=False, sort_keys=True),
        tuple(ss["completed_targets"]),
        ss["held_object"],
        tuple(ss["action_history"][-3:]) if ss["action_history"] else (),
    )

    img_bytes = await image.read()
    case_id, case_dir = case_store.create_case(
        session_id=session_id,
        step=step,
        task_description=task_description,
        context=context,
        image_bytes=img_bytes,
    )
    case_store.log_request(
        case_dir,
        task_description=task_description,
        context=context,
        robot_state=parsed_robot_state,
        spatial_context=parsed_spatial_context,
        metadata=parsed_metadata,
        use_cache=use_cache,
    )
    ss["case_count"] += 1

    if use_cache and ss["template_learned"] and cache_key in plan_cache:
        cached = plan_cache[cache_key]
        ss["cache_hits"] += 1
        case_store.log_decision(
            session_id,
            case_id,
            prompt="cached_template",
            raw_response=json.dumps(cached, ensure_ascii=False),
            parsed_action=cached,
            source="cache",
            latency_ms=0,
        )
        return ActionResponse(
            action=cached["action"],
            target=cached.get("target", ""),
            reason="cached plan template",
            source="cache",
            latency_ms=0,
            step=step,
            case_id=case_id,
            task_description=task_description,
        )

    image_path = str(case_dir / "input.jpg")

    hist = ""
    if ss["action_history"]:
        hist = "\n\nRecent actions:\n"
        for a in ss["action_history"][-10:]:
            hist += f"  {a}\n"

    status_parts = [f"\nCompleted targets: {ss['completed_targets']}"]
    if ss["held_object"]:
        status_parts.append(f"Currently holding: {ss['held_object']}")
    if context:
        status_parts.append(f"Context: {context}")
    if task_description:
        status_parts.append(f"Task: {task_description}")
    if parsed_robot_state:
        status_parts.append(f"Robot state: {json.dumps(parsed_robot_state, ensure_ascii=False)}")
    if parsed_spatial_context:
        status_parts.append(f"Spatial context: {json.dumps(parsed_spatial_context, ensure_ascii=False)}")
    if parsed_metadata:
        status_parts.append(f"Metadata: {json.dumps(parsed_metadata, ensure_ascii=False)}")

    prompt = DECISION_PROMPT + "\n".join(status_parts) + hist + f"\nStep {step}. Next action?"

    t0 = time.time()
    resp = analyze_image(image_path, prompt, stream=False)
    latency = (time.time() - t0) * 1000
    ss["vlm_calls"] += 1

    act = parse_action(resp)

    if use_cache:
        plan_cache[cache_key] = act

    case_store.log_decision(
        session_id,
        case_id,
        prompt=prompt,
        raw_response=resp,
        parsed_action=act,
        source="vlm",
        latency_ms=round(latency, 1),
    )

    ss["action_history"].append(
        f"{act.get('action', '?')} {act.get('target', '')}".strip()
    )

    return ActionResponse(
        action=act.get("action", "done"),
        target=act.get("target", ""),
        reason=act.get("reason", ""),
        source="vlm",
        latency_ms=round(latency, 1),
        step=step,
        case_id=case_id,
        task_description=task_description,
    )


@app.post("/feedback")
async def action_feedback(payload: FeedbackRequest):
    """
    Hardware agent reports action execution result.
    Used to update session state and log case outcomes.
    """
    ss = _get_session(payload.session_id)
    if payload.action == "grasp" and payload.success:
        ss["held_object"] = payload.target or payload.detail
    elif payload.action == "release" and payload.success:
        if ss["held_object"]:
            ss["completed_targets"].append(ss["held_object"])
        ss["held_object"] = None
        if not ss["template_learned"]:
            ss["template_learned"] = True
    if payload.case_id:
        case_store.log_feedback(
            payload.session_id,
            payload.case_id,
            payload.action,
            payload.target,
            payload.success,
            payload.detail,
        )
    return {"status": "ok", "session_id": payload.session_id, "state": ss}


@app.post("/correct/{case_id}")
async def correct_case(case_id: str, payload: CorrectionRequest):
    corrected_action = {
        "action": payload.corrected_action,
        "target": payload.corrected_target,
        "reason": payload.reason,
    }
    case_store.log_correction(
        session_id=payload.session_id,
        case_id=case_id,
        corrected_action=corrected_action,
        reviewer=payload.reviewer,
        notes=payload.reason,
    )
    return {"status": "ok", "case_id": case_id, "corrected_action": corrected_action}


@app.get("/status")
async def get_status(session_id: str = "default"):
    ss = _get_session(session_id)
    return SessionStatus(
        session_id=session_id,
        action_history=ss["action_history"],
        held_object=ss["held_object"],
        completed_targets=ss["completed_targets"],
        template_learned=ss["template_learned"],
        total_vlm_calls=ss["vlm_calls"],
        total_cache_hits=ss["cache_hits"],
        total_cases=ss["case_count"],
    )


@app.post("/reset")
async def reset_session(session_id: str = "default"):
    sessions[session_id] = _default_session_state()
    keys_to_remove = [k for k in plan_cache if k[0] == session_id]
    for k in keys_to_remove:
        del plan_cache[k]
    return {"status": "reset", "session_id": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(sessions),
        "cache_entries": len(plan_cache),
        "cases_root": str(case_store.root),
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Vision Decision Service...")
    print("  Docs:   http://0.0.0.0:8100/docs")
    print("  Health: http://0.0.0.0:8100/health")
    uvicorn.run(app, host="0.0.0.0", port=8100)
