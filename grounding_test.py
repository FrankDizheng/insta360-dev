"""Test VLM visual grounding -- can Qwen3.5 output bounding box coordinates?"""

import cv2
import base64
import json
import requests
import numpy as np
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:9b"
MAX_DIM = 640
CAPTURES = Path("d:/DevProjects/insta360-dev/captures")

GROUNDING_PROMPT = """Look at this image carefully. Detect ALL distinct objects you can see.
For each object, output its bounding box as pixel coordinates.

Use this EXACT format for each object (one per line):
OBJECT_NAME: [x1, y1, x2, y2]

Where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner.
The image dimensions are {width}x{height} pixels.

List every visible object: furniture, electronics, appliances, bottles, containers, people, etc."""


def prepare_and_encode(image_path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    new_h, new_w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return b64, img, new_w, new_h


def query_grounding(image_path):
    b64, img, w, h = prepare_and_encode(image_path)
    prompt = GROUNDING_PROMPT.format(width=w, height=h)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1},
    }

    print(f"  Sending {w}x{h} image to {MODEL}...")
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()["message"]["content"]
    return result, img, w, h


def parse_boxes(text):
    """Parse OBJECT_NAME: [x1, y1, x2, y2] lines from model output."""
    boxes = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or "[" not in line:
            continue
        try:
            label_part = line.split("[")[0].strip().rstrip(":")
            coord_str = line[line.index("["):line.index("]") + 1]
            coords = json.loads(coord_str)
            if len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords):
                boxes.append({"label": label_part, "bbox": [int(c) for c in coords]})
        except (ValueError, json.JSONDecodeError):
            continue
    return boxes


def draw_boxes(img, boxes, output_path):
    vis = img.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = box["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = box["label"][:30]
        cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(output_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def test_view(view_name):
    image_path = CAPTURES / f"grounding_{view_name}.jpg"
    if not image_path.exists():
        print(f"  {view_name}: image not found")
        return

    print(f"\n{'='*50}")
    print(f"Testing: {view_name}")
    print(f"{'='*50}")

    result, img, w, h = query_grounding(image_path)
    print(f"\n  Raw model output:")
    print(f"  {'-'*40}")
    for line in result.strip().split("\n"):
        print(f"  {line}")
    print(f"  {'-'*40}")

    boxes = parse_boxes(result)
    print(f"\n  Parsed {len(boxes)} bounding boxes:")
    for b in boxes:
        print(f"    {b['label']}: {b['bbox']}")

    if boxes:
        out_path = CAPTURES / f"grounding_{view_name}_result.jpg"
        draw_boxes(img, boxes, out_path)
        print(f"\n  Result saved: {out_path.name}")
    else:
        print("\n  No boxes parsed -- model may not support coordinate output well")


if __name__ == "__main__":
    print("VLM Visual Grounding Test")
    print(f"Model: {MODEL}")
    for view in ["front", "right", "back", "left"]:
        test_view(view)
    print("\n\nDone. Check captures/ folder for *_result.jpg files.")
