"""
Search-enabled chat proxy for Qwen3.5-27B.
Sits between the frontend and vLLM, handles tool calling loop.

Endpoints:
  GET  /           → serves chat.html
  POST /api/chat   → chat with web search capability
  GET  /v1/models  → proxy to vLLM

Usage:
  pip install fastapi uvicorn httpx duckduckgo-search
  python search_proxy.py
"""

import json
import re
import asyncio
import datetime
from pathlib import Path
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

VLLM_URL = "http://localhost:8000/v1"
MODEL = "qwen3.5-27b-opus"
MAX_TOOL_ROUNDS = 3
PROXY_PORT = 8094

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for real-time information. Use when user asks about current events, recent news, or anything requiring up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query keywords"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def do_web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return json.dumps({"query": query, "results": "No results found"}, ensure_ascii=False)
        formatted = []
        for r in results:
            formatted.append({"title": r.get("title", ""), "body": r.get("body", ""), "url": r.get("href", "")})
        return json.dumps({"query": query, "results": formatted}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"query": query, "error": str(e)}, ensure_ascii=False)


def do_get_time() -> str:
    import datetime
    now = datetime.datetime.now()
    return json.dumps({
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][now.weekday()],
    })


TOOL_FUNCTIONS = {
    "web_search": lambda args: do_web_search(args.get("query", "")),
    "get_current_time": lambda args: do_get_time(),
}


async def call_vllm(messages, tools=None, stream=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.6,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            return client.stream("POST", f"{VLLM_URL}/chat/completions", json=payload)
        else:
            resp = await client.post(f"{VLLM_URL}/chat/completions", json=payload)
            return resp.json()


async def handle_tool_calls(messages):
    """Run tool calling loop, return final messages list and whether tools were used."""
    tools_used = []

    for _ in range(MAX_TOOL_ROUNDS):
        result = await call_vllm(messages, tools=TOOLS, stream=False)
        msg = result["choices"][0]["message"]

        if not msg.get("tool_calls"):
            return messages, msg, tools_used

        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            tools_used.append({"tool": fn_name, "args": fn_args})

            if fn_name in TOOL_FUNCTIONS:
                fn = TOOL_FUNCTIONS[fn_name]
                if asyncio.iscoroutinefunction(fn):
                    tool_result = await fn(fn_args)
                else:
                    result_or_coro = fn(fn_args)
                    if asyncio.iscoroutine(result_or_coro):
                        tool_result = await result_or_coro
                    else:
                        tool_result = result_or_coro
            else:
                tool_result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

    result = await call_vllm(messages, tools=None, stream=False)
    return messages, result["choices"][0]["message"], tools_used


async def stream_final_response(messages) -> AsyncGenerator[str, None]:
    """Stream the final response after tool calls are done."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{VLLM_URL}/chat/completions", json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.6,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"


@app.get("/")
async def serve_ui():
    html_path = Path(__file__).parent / "chat.html"
    return FileResponse(html_path, media_type="text/html")


@app.get("/v1/models")
async def proxy_models():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{VLLM_URL}/models")
        return JSONResponse(content=resp.json())


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    """Forward chat completions requests directly to vLLM."""
    body = await request.json()
    stream = body.get("stream", False)
    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{VLLM_URL}/chat/completions", json=body) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{VLLM_URL}/chat/completions", json=body)
            return JSONResponse(content=resp.json())


@app.post("/api/chat")
async def chat_with_tools(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    use_search = body.get("enable_search", True)

    if use_search:
        today = datetime.datetime.now().strftime("%Y-%m-%d %A")
        sys_msg = {"role": "system", "content": f"Today is {today}. You have access to web_search and get_current_time tools. Use them when the user asks about current events, news, or time. After getting search results, summarize them clearly for the user. Do NOT output raw tool_call tags in your final answer."}
        messages_copy = [sys_msg] + [m for m in messages if isinstance(m, dict)]
        _, final_msg, tools_used = await handle_tool_calls(messages_copy)

        if tools_used:
            content = final_msg.get("content", "") or ""
            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
            content = re.sub(r'<tool_call>[\s\S]*$', '', content)
            content = content.strip()
            if not content or content.endswith('</think>'):
                messages_copy.append({"role": "user", "content": "Please summarize what you found from the search results above. Answer in the same language as the original question. Do not attempt to call any tools."})
                followup = await call_vllm(messages_copy, tools=None, stream=False)
                content = followup["choices"][0]["message"].get("content", "")
                content = re.sub(r'<tool_call>[\s\S]*$', '', content).strip()
            return JSONResponse({
                "content": content,
                "tools_used": tools_used,
            })

    async def generate():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{VLLM_URL}/chat/completions", json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.6,
                "stream": True,
            }) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
