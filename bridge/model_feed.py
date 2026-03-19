"""Send images to a configurable multimodal backend for analysis.

Supported backends:
  - ollama
  - openai_compatible
"""

import base64
import io
import json
import os
import requests
import numpy as np
from pathlib import Path

try:
    import cv2  # type: ignore[import-not-found]
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

from PIL import Image

DEFAULT_BACKEND = os.getenv("VLM_BACKEND", "ollama").strip().lower()
API_BASE = os.getenv("VLM_API_BASE", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("VLM_MODEL", "qwen3.5:9b")
MAX_IMAGE_DIM = 640
JPEG_QUALITY = 80


def check_ollama() -> bool:
    """Compatibility wrapper retained for existing callers."""
    return check_backend()


def check_backend() -> bool:
    """Check if the configured backend is reachable and the model is available."""
    try:
        if DEFAULT_BACKEND == "ollama":
            resp = requests.get(f"{API_BASE}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(DEFAULT_MODEL.split(":")[0] in m for m in models)

        if DEFAULT_BACKEND == "openai_compatible":
            resp = requests.get(f"{API_BASE}/models", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["id"] for m in resp.json().get("data", [])]
            return not models or DEFAULT_MODEL in models

        raise ValueError(f"Unsupported VLM backend: {DEFAULT_BACKEND}")
    except requests.ConnectionError:
        return False


def _prepare_image(image_path: Path) -> str:
    """Load, downscale, and base64-encode an image for fast inference."""
    if HAS_CV2:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buf).decode("utf-8")

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        image = image.resize((int(w * scale), int(h * scale)))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_image(image_path: str | Path, prompt: str, model: str = DEFAULT_MODEL, stream: bool = True) -> str:
    """Send an image to the configured backend and get analysis text back.

    Args:
        image_path: Path to a JPEG/PNG image file.
        prompt: Text prompt to send alongside the image.
        model: Ollama model name.
        stream: If True, print tokens as they arrive.

    Returns:
        The complete response text from the model.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_b64 = _prepare_image(image_path)

    if DEFAULT_BACKEND == "ollama":
        return _analyze_with_ollama(image_b64, prompt, model, stream)
    if DEFAULT_BACKEND == "openai_compatible":
        return _analyze_with_openai_compatible(image_b64, prompt, model, stream)
    raise ValueError(f"Unsupported VLM backend: {DEFAULT_BACKEND}")


def _analyze_with_ollama(image_b64: str, prompt: str, model: str, stream: bool) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": stream,
        "think": False,
    }

    if stream:
        return _stream_ollama_response(payload)

    resp = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _extract_openai_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def _analyze_with_openai_compatible(image_b64: str, prompt: str, model: str, stream: bool) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "stream": stream,
    }

    if stream:
        return _stream_openai_compatible_response(payload)

    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    message = resp.json()["choices"][0]["message"]["content"]
    return _extract_openai_text(message)


def _stream_ollama_response(payload: dict) -> str:
    """Stream Ollama response tokens and print them as they arrive."""
    resp = requests.post(f"{API_BASE}/api/chat", json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    full_response = []
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                full_response.append(token)
            if data.get("done"):
                break

    print()
    return "".join(full_response)


def _stream_openai_compatible_response(payload: dict) -> str:
    """Stream SSE tokens from an OpenAI-compatible backend."""
    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    full_response = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        data = json.loads(data_str)
        delta = data.get("choices", [{}])[0].get("delta", {})
        token = _extract_openai_text(delta.get("content", ""))
        if token:
            print(token, end="", flush=True)
            full_response.append(token)

    print()
    return "".join(full_response)


if __name__ == "__main__":
    print("Checking VLM backend connection...")
    if check_backend():
        print(f"Backend '{DEFAULT_BACKEND}' is reachable with model {DEFAULT_MODEL}")
    else:
        print(f"WARNING: backend '{DEFAULT_BACKEND}' not reachable or model missing at {API_BASE}")
        if DEFAULT_BACKEND == "ollama":
            print("Make sure Ollama is running in WSL: ollama serve")
            print(f"And the model is pulled: ollama pull {DEFAULT_MODEL}")
