"""
Tool Calling Demo with Qwen3.5-27B via vLLM.

Flow:
  1. User sends message
  2. Model decides: reply directly OR call a tool
  3. If tool call → execute tool → feed result back → model generates final answer
  4. Loop until model replies directly

Usage: python tool_call_demo.py
"""

import json
import time
import datetime
import requests
from openai import OpenAI

client = OpenAI(base_url="http://8.211.22.21:8094/v1", api_key="none")
MODEL = "qwen3.5-27b-opus"

# ============================================================
# Define your tools here - each tool is a Python function
# ============================================================

def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.datetime.now()
    return json.dumps({
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()],
        "timezone": "local",
    })


def web_search(query: str) -> str:
    """搜索互联网获取实时信息"""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=5,
        )
        data = r.json()
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        return json.dumps({"query": query, "results": results or ["No results found"]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})


def control_robot_arm(action: str, target: str = "", speed: str = "normal") -> str:
    """控制机械臂执行动作（模拟）"""
    valid_actions = ["grab", "release", "move_to", "home", "wave"]
    if action not in valid_actions:
        return json.dumps({"error": f"Unknown action. Valid: {valid_actions}"})
    return json.dumps({
        "status": "success",
        "action": action,
        "target": target,
        "speed": speed,
        "message": f"Robot arm: {action} {target} at {speed} speed",
    })


# Registry: name → (function, schema)
TOOLS = {
    "get_current_time": {
        "fn": get_current_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前日期、时间和星期",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    },
    "web_search": {
        "fn": web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "搜索互联网获取实时信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "calculate": {
        "fn": calculate,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "计算数学表达式，支持加减乘除和幂运算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式，如 '2**10' 或 '3.14*5*5'"},
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "control_robot_arm": {
        "fn": control_robot_arm,
        "schema": {
            "type": "function",
            "function": {
                "name": "control_robot_arm",
                "description": "控制机械臂执行动作：grab(抓取), release(释放), move_to(移动到目标), home(回到原点), wave(招手)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["grab", "release", "move_to", "home", "wave"], "description": "动作类型"},
                        "target": {"type": "string", "description": "目标物体或位置（可选）"},
                        "speed": {"type": "string", "enum": ["slow", "normal", "fast"], "description": "速度"},
                    },
                    "required": ["action"],
                },
            },
        },
    },
}


def chat_with_tools(user_message: str, history: list) -> str:
    """Send message, handle tool calls, return final response."""
    history.append({"role": "user", "content": user_message})
    tool_schemas = [t["schema"] for t in TOOLS.values()]

    max_rounds = 5
    for round_i in range(max_rounds):
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=tool_schemas,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.6,
        )
        elapsed = time.time() - t0
        msg = response.choices[0].message

        if msg.tool_calls:
            history.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"  🔧 Tool call: {fn_name}({fn_args})")

                if fn_name in TOOLS:
                    result = TOOLS[fn_name]["fn"](**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                print(f"  📋 Result: {result[:200]}")
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        # No tool call → final text response
        content = msg.content or ""
        history.append({"role": "assistant", "content": content})
        tokens = response.usage.completion_tokens if response.usage else 0
        print(f"  ⏱  {elapsed:.1f}s | {tokens} tokens")
        return content

    return "[Max tool call rounds reached]"


def main():
    print("=" * 50)
    print("Qwen3.5-27B Tool Calling Demo")
    print("=" * 50)
    print("Available tools: get_current_time, web_search, calculate, control_robot_arm")
    print("Type 'quit' to exit, 'clear' to reset history\n")

    history = [{"role": "system", "content": "你是一个智能机器人助手，可以使用工具来帮助用户。如果用户的问题需要实时信息、计算或机械臂控制，请调用相应的工具。"}]

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = [history[0]]
            print("History cleared.")
            continue

        print("🤖 Assistant:")
        reply = chat_with_tools(user_input, history)
        print(f"\n{reply}")


if __name__ == "__main__":
    main()
