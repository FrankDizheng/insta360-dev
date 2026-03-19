import cv2
import time
import os
import numpy as np

print("Scanning for Insta360 X5...")
x5_idx = None
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Device {i}: {w}x{h}")
        cap.release()
    else:
        print(f"  Device {i}: not available")

print()
print("Trying each device with 2880x1440 (X5 360 mode)...")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 2880 and h == 1440:
        x5_idx = i
        print(f"  Device {i}: accepted 2880x1440 -- this is the X5!")
        cap.release()
        break
    cap.release()

if x5_idx is None:
    print("X5 not found at 2880x1440, trying any 2:1 ratio device...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w >= 2880 and abs(w / h - 2.0) < 0.1:
            x5_idx = i
            print(f"  Device {i}: {w}x{h} -- likely X5")
            cap.release()
            break
        cap.release()

if x5_idx is None:
    print("ERROR: Could not find X5")
    exit(1)

print(f"\nOpening X5 at device {x5_idx} with 2880x1440...")
cap = cv2.VideoCapture(x5_idx, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 30)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual resolution: {w}x{h}")

print("Warming up stream (8 seconds)...")
time.sleep(8)

print("Reading frames...")
good_frame = None
for attempt in range(60):
    ret, frame = cap.read()
    if ret and frame is not None:
        mean_val = np.mean(frame)
        max_val = int(np.max(frame))
        if attempt % 10 == 0:
            print(f"  Frame {attempt}: mean={mean_val:.2f} max={max_val}")
        if mean_val > 1.0:
            good_frame = frame
            print(f"  Frame {attempt}: GOT IMAGE! mean={mean_val:.2f} max={max_val}")
            break

if good_frame is not None:
    path = "d:/DevProjects/insta360-dev/captures/x5_360_capture.jpg"
    cv2.imwrite(path, good_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved: {path}")
    print(f"Shape: {good_frame.shape}")
    print(f"Size: {os.path.getsize(path)} bytes")
elif ret:
    path = "d:/DevProjects/insta360-dev/captures/x5_360_capture.jpg"
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    mean_val = np.mean(frame)
    print(f"\nFrame still dark (mean={mean_val:.2f}), saved anyway: {path}")
    print(f"Shape: {frame.shape}")
else:
    print("\nERROR: failed to read any frame")

cap.release()
print("Done.")
