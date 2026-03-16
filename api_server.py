"""
VLM Sorting API Server
Receives camera images, returns action commands.
Deploy on: local GPU machine / cloud GPU server / edge device.

Usage:
    pip install fastapi uvicorn python-multipart
    python api_server.py
"""
import os, sys, json, re, time, hashlib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bridge"))
from model_feed import analyze_image

app = FastAPI(title="VLM Sorting Controller", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Plan template cache ──
PLAN_TEMPLATE = [
    {"action": "move_above", "target": "{object}"},
    {"action": "lower", "target": "{object}"},
    {"action": "grasp", "target": ""},
    {"action": "lift", "target": ""},
    {"action": "move_above", "target": "{bin}"},
    {"action": "lower", "target": "{bin}"},
    {"action": "release", "target": ""},
]

plan_cache = {}
session_state = {
    "action_history": [],
    "held_object": None,
    "sorted_objects": [],
    "template_learned": False,
}

SORTING_PROMPT = """You are a robot controller for a sorting task.
You see the scene from a camera. Identify objects and decide the next action.

Output EXACTLY ONE JSON:
{"action": "<action>", "target": "<target>", "reason": "<brief reason>"}

Valid actions: move_above, lower, grasp, lift, release, done
Valid targets: any object name or bin name visible in the scene.

For each object: move_above → lower → grasp → lift → move_above <bin> → lower <bin> → release
When all objects sorted: {"action": "done"}

Rules:
- Output ONLY the JSON.
- Sort one object at a time.
"""


class ActionResponse(BaseModel):
    action: str
    target: str
    reason: str = ""
    source: str = "vlm"
    latency_ms: float = 0
    step: int = 0


class SessionStatus(BaseModel):
    action_history: list
    held_object: Optional[str]
    sorted_objects: list
    template_learned: bool
    total_vlm_calls: int
    total_cache_hits: int


stats = {"vlm_calls": 0, "cache_hits": 0}


def parse_action(text):
    m = re.search(r'\{[^}]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"action": "done", "target": "", "reason": "parse_error"}


@app.post("/decide", response_model=ActionResponse)
async def decide_action(
    image: UploadFile = File(...),
    context: str = "",
    use_cache: bool = True,
):
    """
    Send a camera image, receive the next robot action.
    
    - image: current camera frame (JPEG/PNG)
    - context: optional extra context (e.g. "objects: bottle_a, bottle_b")
    - use_cache: whether to use plan template caching
    """
    step = len(session_state["action_history"])

    cache_key = (
        tuple(session_state["sorted_objects"]),
        session_state["held_object"],
        tuple(session_state["action_history"][-3:]) if session_state["action_history"] else (),
    )

    if use_cache and session_state["template_learned"] and cache_key in plan_cache:
        cached = plan_cache[cache_key]
        stats["cache_hits"] += 1
        return ActionResponse(
            action=cached["action"],
            target=cached.get("target", ""),
            reason="cached plan template",
            source="cache",
            latency_ms=0,
            step=step,
        )

    img_bytes = await image.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_bytes)
        tmp_path = f.name

    try:
        hist = ""
        if session_state["action_history"]:
            hist = "\n\nRecent actions:\n"
            for i, a in enumerate(session_state["action_history"][-10:]):
                hist += f"  {a}\n"

        status = f"\nSorted: {session_state['sorted_objects']}"
        if session_state["held_object"]:
            status += f"\nCurrently holding: {session_state['held_object']}"
        if context:
            status += f"\nContext: {context}"

        prompt = SORTING_PROMPT + status + hist + f"\nStep {step}. Next action?"

        t0 = time.time()
        resp = analyze_image(tmp_path, prompt, stream=False)
        latency = (time.time() - t0) * 1000
        stats["vlm_calls"] += 1

        act = parse_action(resp)

        if use_cache:
            plan_cache[cache_key] = act

        session_state["action_history"].append(
            f"{act.get('action', '?')} {act.get('target', '')}".strip()
        )

        return ActionResponse(
            action=act.get("action", "done"),
            target=act.get("target", ""),
            reason=act.get("reason", ""),
            source="vlm",
            latency_ms=round(latency, 1),
            step=step,
        )
    finally:
        os.unlink(tmp_path)


@app.post("/feedback")
async def action_feedback(action: str, target: str, success: bool, detail: str = ""):
    """
    Hardware agent reports action execution result.
    Used to update session state (e.g. grasp succeeded, object released).
    """
    if action == "grasp" and success:
        session_state["held_object"] = target or detail
    elif action == "release" and success:
        if session_state["held_object"]:
            session_state["sorted_objects"].append(session_state["held_object"])
        session_state["held_object"] = None
        if not session_state["template_learned"]:
            session_state["template_learned"] = True
    return {"status": "ok", "state": session_state}


@app.get("/status", response_model=SessionStatus)
async def get_status():
    return SessionStatus(
        **session_state,
        total_vlm_calls=stats["vlm_calls"],
        total_cache_hits=stats["cache_hits"],
    )


@app.post("/reset")
async def reset_session():
    session_state["action_history"] = []
    session_state["held_object"] = None
    session_state["sorted_objects"] = []
    session_state["template_learned"] = False
    plan_cache.clear()
    stats["vlm_calls"] = 0
    stats["cache_hits"] = 0
    return {"status": "reset"}


@app.get("/health")
async def health():
    return {"status": "ok", "vlm": "qwen3.5", "cache_entries": len(plan_cache)}


if __name__ == "__main__":
    import uvicorn
    print("Starting VLM Sorting API Server...")
    print("  Docs:   http://0.0.0.0:8100/docs")
    print("  Health: http://0.0.0.0:8100/health")
    uvicorn.run(app, host="0.0.0.0", port=8100)
