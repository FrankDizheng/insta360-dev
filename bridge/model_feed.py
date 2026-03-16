"""Send images to Qwen3.5 via Ollama HTTP API for multimodal analysis."""

import base64
import json
import requests
import cv2
import numpy as np
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
MAX_IMAGE_DIM = 640
JPEG_QUALITY = 80


def check_ollama() -> bool:
    """Check if Ollama is reachable and the model is available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(DEFAULT_MODEL.split(":")[0] in m for m in models)
    except requests.ConnectionError:
        return False


def _prepare_image(image_path: Path) -> str:
    """Load, downscale, and base64-encode an image for fast inference."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def analyze_image(image_path: str | Path, prompt: str, model: str = DEFAULT_MODEL, stream: bool = True) -> str:
    """Send an image to Qwen3.5 via Ollama and get analysis text back.

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
        return _stream_response(payload)
    else:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _stream_response(payload: dict) -> str:
    """Stream response tokens and print them as they arrive."""
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120)
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


if __name__ == "__main__":
    print("Checking Ollama connection...")
    if check_ollama():
        print(f"Ollama is running with {DEFAULT_MODEL}")
    else:
        print(f"WARNING: Ollama not reachable or {DEFAULT_MODEL} not found at {OLLAMA_URL}")
        print("Make sure Ollama is running in WSL: ollama serve")
        print(f"And the model is pulled: ollama pull {DEFAULT_MODEL}")
