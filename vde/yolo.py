from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
import json
from config.config import Config

class YOLODetector:
    def __init__(self, config: Config):
        self.config = config
        self.model_path = self.config.yolo_weights_path
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLOv8 model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"Error loading YOLOv8 model from {self.model_path}: {e}")
            self.model = None

    def detect_and_crop_vehicles(self, image_path, output_folder_cropped, output_folder_visualized, log_data):
        if self.model is None:
            print(f"YOLO model not loaded. Skipping detection for {image_path}")
            return []

        os.makedirs(output_folder_cropped, exist_ok=True)
        os.makedirs(output_folder_visualized, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return []

        results = self.model(img, verbose=False)
        detections_data_for_image = []

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        img_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        original_img_filename = os.path.basename(image_path)

        for i, r in enumerate(results):
            for j, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x1 + 5, y1 - 15), f"{class_name}: {conf:.2f}", fill="green")

                y1_safe = max(0, y1)
                y2_safe = min(img.shape[0], y2)
                x1_safe = max(0, x1)
                x2_safe = min(img.shape[1], x2)

                cropped_img_cv2 = img[y1_safe:y2_safe, x1_safe:x2_safe]

                if cropped_img_cv2.size == 0:
                    print(f"Warning: Empty crop for {original_img_filename} (box {j}). Skipping this crop.")
                    continue

                cropped_filename = f"{img_name_without_ext}_vehicle_crop_{j}_{class_name}.jpg"
                cropped_filepath = os.path.join(output_folder_cropped, cropped_filename)
                cv2.imwrite(cropped_filepath, cropped_img_cv2)

                detection_info = {
                    "original_image": original_img_filename,
                    "cropped_image_path": cropped_filepath,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": class_name
                }
                detections_data_for_image.append(detection_info)
                log_data.append(detection_info)

        if detections_data_for_image:
            visualized_filepath = os.path.join(output_folder_visualized, f"{img_name_without_ext}_detected.jpg")
            pil_img.save(visualized_filepath)

        return detections_data_for_image