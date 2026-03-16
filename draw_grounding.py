"""Re-draw grounding boxes with correct 0-1000 -> pixel coordinate scaling."""

import cv2
import json
import numpy as np
from pathlib import Path

CAPTURES = Path("d:/DevProjects/insta360-dev/captures")

RESULTS = {
    "front": [
        {"bbox": [0, 471, 253, 998], "label": "vacuum cleaner"},
        {"bbox": [460, 618, 563, 794], "label": "desk"},
        {"bbox": [565, 622, 626, 786], "label": "chair"},
        {"bbox": [490, 712, 536, 815], "label": "bin"},
        {"bbox": [706, 632, 756, 776], "label": "suitcase"},
        {"bbox": [479, 509, 531, 588], "label": "monitor"},
        {"bbox": [468, 607, 481, 628], "label": "bottle"},
        {"bbox": [481, 609, 493, 635], "label": "bottle"},
        {"bbox": [497, 612, 509, 635], "label": "bottle"},
        {"bbox": [511, 612, 523, 638], "label": "bottle"},
        {"bbox": [740, 588, 759, 646], "label": "bottle-r1"},
        {"bbox": [719, 592, 739, 654], "label": "bottle-r2"},
        {"bbox": [708, 596, 722, 657], "label": "bottle-r3"},
    ],
    "right": [
        {"bbox": [518, 0, 999, 997], "label": "person"},
        {"bbox": [146, 593, 698, 998], "label": "sofa"},
        {"bbox": [0, 58, 205, 674], "label": "shelf"},
        {"bbox": [1, 588, 203, 998], "label": "chair"},
        {"bbox": [649, 773, 849, 998], "label": "desk"},
        {"bbox": [395, 501, 485, 613], "label": "pillow"},
        {"bbox": [474, 560, 592, 613], "label": "pillow"},
    ],
    "back": [
        {"bbox": [0, 0, 479, 1000], "label": "person"},
        {"bbox": [385, 0, 762, 1000], "label": "curtain"},
        {"bbox": [765, 397, 1000, 1000], "label": "monitor"},
        {"bbox": [580, 774, 740, 1000], "label": "speaker"},
        {"bbox": [494, 927, 580, 1000], "label": "remote"},
    ],
    "left": [
        {"bbox": [0, 375, 498, 1000], "label": "monitor"},
        {"bbox": [636, 650, 683, 1000], "label": "antenna"},
        {"bbox": [827, 475, 999, 1000], "label": "printer"},
    ],
}

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255),
]


def draw(view_name, boxes):
    img_path = CAPTURES / f"grounding_{view_name}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        x1 = int(box["bbox"][0] / 1000.0 * w)
        y1 = int(box["bbox"][1] / 1000.0 * h)
        x2 = int(box["bbox"][2] / 1000.0 * w)
        y2 = int(box["bbox"][3] / 1000.0 * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = box["label"]
        cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out_path = CAPTURES / f"grounding_{view_name}_result.jpg"
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{view_name}: {len(boxes)} boxes -> {out_path.name}")


for view, boxes in RESULTS.items():
    draw(view, boxes)
print("Done. Open the *_result.jpg files to check accuracy.")
