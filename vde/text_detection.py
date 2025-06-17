import os
import json
import numpy as np
import requests
import time
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO
from tqdm import tqdm
from config.config import Config
import cv2
import base64
class TextDetector:
    def __init__(self, config: Config):
        self.config = config
        self.detection_api_url = self.config.detection_api_url
        self.detection_headers = self.config.detection_headers
        self.request_delay_seconds = self.config.request_delay_seconds

    def _apply_api_delay(self):
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)

    def encode_image_to_base64(self, image_path):
        img = Image.open(image_path).convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _convert_numpy_to_python_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python_types(elem) for elem in obj]
        else:
            return obj

    def shrink_bbox(self, pil_image, bbox):
        image = pil_image.copy()
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        x_min_crop = max(0, int(bbox[0]))
        x_max_crop = min(image.shape[1], int(bbox[1]))
        y_min_crop = max(0, int(bbox[2]))
        y_max_crop = min(image.shape[0], int(bbox[3]))

        if x_max_crop <= x_min_crop or y_max_crop <= y_min_crop:
            return bbox 

        cropped_image = image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
        
        if cropped_image.size == 0: 
            return bbox

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        blur = cv2.GaussianBlur(binary, (5, 5), 20)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(blur, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        thr = cv2.threshold(eroded, 110, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours]).astype(np.int32)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            
            x_min = int(x_min + x_min_crop)
            y_min = int(y_min + y_min_crop)
            x_max = int(x_max + x_min_crop)
            y_max = int(y_max + y_min_crop)

            horizontal_pad = int((self.config.horizontal_padding_ratio / 100) * (x_max - x_min))
            vertical_pad = int((self.config.vertical_padding_ratio / 100) * (y_max - y_min))

            horizontal_pad = max(self.config.min_horizontal_pad, min(horizontal_pad, self.config.max_horizontal_pad))
            vertical_pad = max(self.config.min_vertical_pad, min(vertical_pad, self.config.max_vertical_pad))

            x_min = int(max(0, x_min - horizontal_pad))
            y_min = int(max(0, y_min - vertical_pad))
            x_max = int(min(pil_image.width - 1, x_max + horizontal_pad))
            y_max = int(min(pil_image.height - 1, y_max + vertical_pad))

            return [x_min, x_max, y_min, y_max]
        else:
            return self._convert_numpy_to_python_types(bbox)
        
    def soft_padding(self, box, image_size):
        x1, x2, y1, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_width, img_height = image_size

        box_width = x2 - x1
        box_height = y2 - y1

        pad_horizontal = int(box_width * self.config.horizontal_padding_ratio)
        pad_vertical = int(box_height * self.config.vertical_padding_ratio)

        pad_horizontal = max(self.config.min_horizontal_pad, min(pad_horizontal, self.config.max_horizontal_pad))
        pad_vertical = max(self.config.min_vertical_pad, min(pad_vertical, self.config.max_vertical_pad))

        x1 = int(max(0, x1 - pad_horizontal))
        y1 = int(max(0, y1 - pad_vertical))
        x2 = int(min(img_width - 1, x2 + pad_horizontal))
        y2 = int(min(img_height - 1, y2 + pad_vertical))

        return x1, x2, y1, y2

    def draw_boxes_and_save(self, image_path, bboxes, save_folder):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size

        processed_bboxes = self._convert_numpy_to_python_types(bboxes)

        if isinstance(processed_bboxes, list) and len(processed_bboxes) > 0 and "horizontal_list" in processed_bboxes[0]:
            original_boxes = processed_bboxes[0]["horizontal_list"]

            shrinked_boxes = [self.shrink_bbox(image, box) for box in original_boxes]
            
            if any((box[0] >= box[1] or box[2] >= box[3]) for box in shrinked_boxes):
                 shrinked_boxes = original_boxes

            for box in shrinked_boxes:
                x1, x2, y1, y2 = self.soft_padding(box, (img_width, img_height))
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            processed_bboxes[0]["horizontal_list"] = shrinked_boxes

        os.makedirs(save_folder, exist_ok=True)
        image.save(os.path.join(save_folder, image_path.name))
        
    def get_text_detections(self, image_folder, output_json_path, vis_folder, log_file): 
        image_folder_path = Path(image_folder)
        image_paths = []
        extensions = ["jpg", "jpeg", "png", "bmp", "gif", "tiff"]
        for ext in extensions:
            image_paths.extend(image_folder_path.glob(f"*.{ext}"))
            image_paths.extend(image_folder_path.glob(f"*.{ext.upper()}"))
        image_paths.sort(key=lambda x: str(x.name)) # Use str for simple sorting as natural_sort_key is in processor

        results = {}
        successful_detections = 0
        failed_detections = 0

        with open(log_file, 'a', encoding='utf-8') as log: 
            log.write("\n\n" + "=" * 50 + "\n")
            log.write("STARTING TEXT DETECTION LOG\n")
            log.write("=" * 50 + "\n\n")
            log.write("PROCESSING IMAGES FOR TEXT DETECTION:\n")
            log.write("-" * 50 + "\n")

            for image_path in tqdm(image_paths, desc="Detecting text"):
                try:
                    base64_img = self.encode_image_to_base64(image_path)
                    self._apply_api_delay() 
                    response = requests.get(
                        self.detection_api_url, 
                        headers=self.detection_headers, 
                        json={"img": f"data:image/jpeg;base64,{base64_img}"}
                    )
                    response.raise_for_status()
                    bboxes = response.json()
                    
                    converted_bboxes = self._convert_numpy_to_python_types(bboxes)
                    results[image_path.name] = converted_bboxes
                    
                    self.draw_boxes_and_save(image_path, converted_bboxes, vis_folder)
                    successful_detections += 1
                    log.write(f"✓ Detected text for: {image_path.name}\n") 
                    
                except Exception as e:
                    failed_detections += 1
                    error_message = f"✗ Detection failed for image: {image_path.name} - {e}"
                    print(error_message)
                    log.write(f"{error_message}\n") 
                    results[image_path.name] = {"error": str(e)}

            log.write(f"\nTotal Successful Detections: {successful_detections}\n") 
            log.write(f"Total Failed Detections: {failed_detections}\n") 
            log.write(f"Total Images Processed for Detection: {successful_detections + failed_detections}\n") 
            log.write("\n" + "=" * 50 + "\n")
            log.write("TEXT DETECTION LOG END\n")

        os.makedirs(Path(output_json_path).parent, exist_ok=True)
        os.makedirs(vis_folder, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved detection results to: {output_json_path}")
        print(f"🖼️ Saved visualized images to: {vis_folder}")
   
    def get_bboxes(self, boxes):
        new_boxes = []
        for box in boxes:
            new_boxes.append([box[0], box[2], box[1], box[3]]) 
        return new_boxes

    def post_process_detections(self, detection_json_path, output_json_path):
        try:
            with open(detection_json_path, "r") as f:
                detection_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Detection results JSON file not found at {detection_json_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {detection_json_path}")
            return

        processed_data = {}
        for image_name, entries in detection_data.items():
            if isinstance(entries, list) and len(entries) > 0 and "horizontal_list" in entries[0]:
                boxes = entries[0]["horizontal_list"]
                processed_data[image_name] = self.get_bboxes(boxes)
            else:
                processed_data[image_name] = [] 

        os.makedirs(Path(output_json_path).parent, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(processed_data, f, indent=2)

        print(f"✅ Post-processed detection results saved to: {output_json_path}")