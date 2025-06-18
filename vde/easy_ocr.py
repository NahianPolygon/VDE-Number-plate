import os
import json
import cv2
import numpy as np
import easyocr
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from config.config import Config
import re

class EasyOCRRecognizer:
    def __init__(self, config: Config):
        self.config = config
        self.reader = easyocr.Reader(self.config.easy_ocr_languages)

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

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def draw_ocr_boxes_and_save(self, image_path, ocr_results, save_path):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        for (bbox, text, prob) in ocr_results:
            bbox = np.array(bbox).astype(int)
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
            draw.text((x_min, y_min - 15), f"{text} ({prob:.2f})", fill="blue")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    def process_images_for_ocr(self, image_folder, output_json_path, vis_folder, log_file):
        image_folder_path = Path(image_folder)
        image_paths = []
        extensions = ["jpg", "jpeg", "png", "bmp", "gif", "tiff"]
        for ext in extensions:
            image_paths.extend(image_folder_path.glob(f"*.{ext}"))
            image_paths.extend(image_folder_path.glob(f"*.{ext.upper()}"))
        image_paths.sort(key=lambda x: self.natural_sort_key(x.name))

        results = {}
        successful_ocrs = 0
        failed_ocrs = 0

        with open(log_file, 'a', encoding='utf-8') as log:
            log.write("\n\n" + "=" * 50 + "\n")
            log.write("STARTING EASYOCR RECOGNITION LOG\n")
            log.write("=" * 50 + "\n\n")
            log.write("PROCESSING IMAGES FOR EASYOCR:\n")
            log.write("-" * 50 + "\n")

            for image_path in tqdm(image_paths, desc="Running EasyOCR"):
                try:
                    ocr_results = self.reader.readtext(str(image_path))
                    
                    processed_results = []
                    for (bbox, text, prob) in ocr_results:
                        bbox_flat = [int(min(p[0] for p in bbox)), int(min(p[1] for p in bbox)),
                                     int(max(p[0] for p in bbox)), int(max(p[1] for p in bbox))]
                        
                        processed_results.append(
                            self._convert_numpy_to_python_types({
                                "bbox": bbox_flat,
                                "raw_bbox": bbox,
                                "text": text,
                                "confidence": float(prob)
                            })
                        )
                    
                    results[image_path.name] = {"easy_ocr_results": processed_results}

                    vis_output_path = os.path.join(vis_folder, image_path.name)
                    self.draw_ocr_boxes_and_save(image_path, ocr_results, vis_output_path)

                    successful_ocrs += 1
                    log.write(f"✓ EasyOCR processed: {image_path.name}\n")
                except Exception as e:
                    failed_ocrs += 1
                    error_message = f"✗ EasyOCR failed for image: {image_path.name} - {e}"
                    print(error_message)
                    log.write(f"{error_message}\n")
                    results[image_path.name] = {"error": str(e)}

            log.write(f"\nTotal Successful EasyOCR Runs: {successful_ocrs}\n")
            log.write(f"Total Failed EasyOCR Runs: {failed_ocrs}\n")
            log.write(f"Total Images Processed by EasyOCR: {successful_ocrs + failed_ocrs}\n")
            log.write("\n" + "=" * 50 + "\n")
            log.write("EASYOCR RECOGNITION LOG END\n")

        os.makedirs(Path(output_json_path).parent, exist_ok=True)
        os.makedirs(vis_folder, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ EasyOCR results saved to: {output_json_path}")
        print(f"🖼️ EasyOCR visualizations saved to: {vis_folder}")