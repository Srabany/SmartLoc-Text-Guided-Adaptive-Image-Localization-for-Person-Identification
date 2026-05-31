import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

IMAGE_FILES = [
    "../../data/images/Group.JPG",
    "../../data/images/group1.jpg",
]

yolo = YOLO("yolov8n.pt")

image_labels = []
person_counts = []

for path in IMAGE_FILES:
    img = cv2.imread(path)
    results = yolo(path, conf=0.5)   # higher conf threshold

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    valid = 0

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
        # only class 0 (person)
        if int(cls_id) != 0:
            continue

        w = x2 - x1
        h = y2 - y1

        # ignore tiny / partial detections
        if w < 80 or h < 120:
            continue

        valid += 1

    filename = os.path.basename(path)
    image_labels.append(filename)
    person_counts.append(valid)
    print(f"{filename}: {valid} filtered persons")

plt.figure(figsize=(8, 5))
plt.bar(image_labels, person_counts, color="orange", edgecolor="black")

plt.title("Filtered Person Count per Image")
plt.xlabel("Image")
plt.ylabel("Person Count")
plt.grid(axis='y')

save_path = "../../graphs/person_count_filtered.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)

plt.show()
