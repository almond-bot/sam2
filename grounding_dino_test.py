import cv2
import numpy as np

from bin_picking import grounding_dino_inference

rgb = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2RGB)

bboxes = grounding_dino_inference(rgb, "circular pieces of metal")

for bbox in bboxes:
    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("rgb_with_bboxes.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
