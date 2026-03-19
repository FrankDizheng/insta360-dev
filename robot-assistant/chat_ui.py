"""
Chat UI for Qwen3.5-27B-Opus via vLLM.
Usage: uv run --with gradio --with openai chat_ui.py
"""

import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "qwen3.5-27b-opus"

def chat(message, history):
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["content"]}) if h["role"] == "user" else messages.append({"role": "assistant", "content": h["content"]})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.6,
        stream=True,
    )

    partial = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial

demo = gr.ChatInterface(
    fn=chat,
    title="Qwen3.5-27B · Claude Opus 4.6 Distill",
    description="Running on H20 GPU via vLLM (126 tok/s)",
    type="messages",
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
