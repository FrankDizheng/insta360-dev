"""Insta360 X5 + Qwen3.5 Bridge -- Interactive CLI.

Captures 360 equirectangular frames from the X5 webcam and sends them
to Qwen3.5 running on Ollama (WSL) for multimodal AI analysis.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from capture import Insta360Capture, find_x5_device
from model_feed import analyze_image, check_ollama, OLLAMA_URL, DEFAULT_MODEL

CAPTURES_DIR = Path(__file__).parent.parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)


def print_banner():
    print("=" * 60)
    print("  Insta360 X5 + Qwen3.5 Vision Bridge")
    print("  Camera: Insta360 X5 (USB webcam mode)")
    print(f"  Model:  {DEFAULT_MODEL} via Ollama")
    print("=" * 60)


def main():
    print_banner()

    print("\n[1/3] Checking Ollama connection...")
    if check_ollama():
        print(f"  OK -- {DEFAULT_MODEL} available at {OLLAMA_URL}")
    else:
        print(f"  WARNING: Ollama not reachable or model not ready.")
        print(f"  In WSL, run: ollama serve && ollama pull {DEFAULT_MODEL}")
        print(f"  Continuing anyway -- you can still capture frames.\n")

    print("\n[2/3] Looking for Insta360 X5...")
    device_idx = find_x5_device()
    if device_idx is None:
        print("  ERROR: X5 not found. Make sure it's connected in USB Camera mode.")
        return
    print(f"  OK -- X5 found at device {device_idx}")

    print("\n[3/3] Opening camera stream...")
    cam = Insta360Capture(device_idx)
    cam.open()
    print("  OK -- stream ready\n")

    print("-" * 60)
    print("Commands:")
    print("  [Enter]     Capture + analyze with default prompt")
    print("  <text>      Capture + analyze with custom prompt")
    print("  save        Capture and save without AI analysis")
    print("  quit        Exit")
    print("-" * 60)

    default_prompt = (
        "This is a 360-degree equirectangular panoramic image. "
        "Describe the scene in detail: what objects, people, and "
        "environment do you see? Note any interesting details."
    )

    try:
        while True:
            print()
            user_input = input(">>> ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = CAPTURES_DIR / f"capture_{timestamp}.jpg"

            print("Capturing 360 frame...")
            try:
                cam.capture_to_file(image_path)
                print(f"  Saved: {image_path.name} ({image_path.stat().st_size // 1024} KB)")
            except RuntimeError as e:
                print(f"  Capture error: {e}")
                continue

            if user_input.lower() == "save":
                continue

            prompt = user_input if user_input else default_prompt
            print(f"\nAnalyzing with Qwen3.5...\n")

            try:
                response = analyze_image(image_path, prompt)
                print(f"\n[Analysis saved alongside image]")
                result_path = image_path.with_suffix(".txt")
                result_path.write_text(
                    f"Prompt: {prompt}\n\nResponse:\n{response}",
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"\n  AI analysis error: {e}")
                print("  Make sure Ollama is running in WSL with the model loaded.")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        cam.close()
        print("Camera closed. Goodbye.")


if __name__ == "__main__":
    main()
