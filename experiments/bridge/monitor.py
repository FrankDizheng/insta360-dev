"""Insta360 X5 continuous monitoring mode.

Captures frames at regular intervals, sends perspective views to Qwen3.5,
and outputs activity summaries -- who is doing what.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from capture import Insta360Capture, find_x5_device
from perspective import equirect_to_perspective, PRESET_VIEWS
from model_feed import analyze_image, check_ollama, DEFAULT_MODEL, OLLAMA_URL
import cv2
import numpy as np

CAPTURES_DIR = Path(__file__).parent.parent / "captures" / "monitor"
INTERVAL_SECONDS = 5

MONITOR_PROMPT = (
    "You are a security/activity monitor analyzing a room camera feed. "
    "Describe ONLY what you observe RIGHT NOW in this image:\n"
    "1. How many people are visible and where are they?\n"
    "2. What is each person doing?\n"
    "3. Any notable activity or changes in the environment?\n"
    "Be concise -- 2-3 sentences max. If no people are visible, say so."
)

SUMMARY_PROMPT = (
    "You are summarizing a monitoring session. Below are timestamped observations "
    "from a room camera taken every few seconds. Write a brief activity summary "
    "(3-5 sentences) covering: who was present, what they did, and any notable events.\n\n"
    "{observations}"
)


def save_perspective_views(equirect: np.ndarray, timestamp: str, out_dir: Path) -> list[Path]:
    """Extract and save key perspective views, return paths."""
    views = ["front", "right", "back", "left"]
    paths = []
    for name in views:
        params = PRESET_VIEWS[name]
        persp = equirect_to_perspective(equirect, **params)
        p = out_dir / f"{timestamp}_{name}.jpg"
        cv2.imwrite(str(p), persp, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def analyze_frame(equirect: np.ndarray, timestamp: str, out_dir: Path) -> str:
    """Capture perspective views and get AI analysis."""
    wide = equirect_to_perspective(equirect, yaw=0, pitch=0, hfov=120)
    wide_path = out_dir / f"{timestamp}_wide.jpg"
    cv2.imwrite(str(wide_path), wide, [cv2.IMWRITE_JPEG_QUALITY, 85])

    try:
        result = analyze_image(wide_path, MONITOR_PROMPT, stream=False)
        return result.strip()
    except Exception as e:
        return f"[Analysis error: {e}]"


def main():
    interval = INTERVAL_SECONDS
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 60)
    print("  Insta360 X5 -- Continuous Room Monitor")
    print(f"  Interval: {interval}s | Model: {DEFAULT_MODEL}")
    print("  Press Ctrl+C to stop and get a session summary")
    print("=" * 60)

    if not check_ollama():
        print(f"\nERROR: Ollama not reachable at {OLLAMA_URL}")
        print("Start it in WSL: ollama serve")
        return

    print("\nLooking for Insta360 X5...")
    device_idx = find_x5_device()
    if device_idx is None:
        print("ERROR: X5 not found. Connect it in USB Camera mode.")
        return

    cam = Insta360Capture(device_idx)
    cam.open()

    session_start = datetime.now()
    session_id = session_start.strftime("%Y%m%d_%H%M%S")
    out_dir = CAPTURES_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSession: {session_id}")
    print(f"Saving to: {out_dir}")
    print("-" * 60)

    observations = []
    frame_count = 0

    try:
        while True:
            timestamp = datetime.now().strftime("%H%M%S")
            time_display = datetime.now().strftime("%H:%M:%S")

            try:
                equirect = cam.capture_frame()
            except RuntimeError as e:
                print(f"[{time_display}] Capture error: {e}")
                time.sleep(interval)
                continue

            frame_count += 1
            print(f"\n[{time_display}] Frame #{frame_count} captured, analyzing...", end=" ")

            start = time.time()
            observation = analyze_frame(equirect, timestamp, out_dir)
            elapsed = time.time() - start

            entry = f"[{time_display}] {observation}"
            observations.append(entry)

            print(f"({elapsed:.1f}s)")
            print(f"  >> {observation}")

            time.sleep(max(0, interval - elapsed))

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  MONITORING STOPPED")
        print("=" * 60)

        duration = datetime.now() - session_start
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)

        print(f"\nSession duration: {minutes}m {seconds}s")
        print(f"Frames analyzed: {frame_count}")
        print(f"Images saved to: {out_dir}\n")

        if len(observations) >= 2:
            print("Generating session summary...\n")
            obs_text = "\n".join(observations)

            log_path = out_dir / "observations.txt"
            log_path.write_text(obs_text, encoding="utf-8")

            summary_path = out_dir / "summary.txt"
            try:
                from model_feed import _prepare_image
                import base64, json, requests

                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": SUMMARY_PROMPT.format(observations=obs_text)}],
                    "stream": False,
                    "think": False,
                }
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
                resp.raise_for_status()
                summary = resp.json()["message"]["content"]
                print("-" * 60)
                print("SESSION SUMMARY:")
                print("-" * 60)
                print(summary)
                print("-" * 60)

                summary_path.write_text(
                    f"Session: {session_id}\nDuration: {minutes}m {seconds}s\nFrames: {frame_count}\n\n"
                    f"OBSERVATIONS:\n{obs_text}\n\nSUMMARY:\n{summary}",
                    encoding="utf-8",
                )
                print(f"\nFull report saved: {summary_path}")
            except Exception as e:
                print(f"Summary generation failed: {e}")
                summary_path.write_text(
                    f"Session: {session_id}\nDuration: {minutes}m {seconds}s\n\nOBSERVATIONS:\n{obs_text}",
                    encoding="utf-8",
                )
        else:
            print("Too few frames for a summary.")

    finally:
        cam.close()
        print("\nCamera closed.")


if __name__ == "__main__":
    main()
