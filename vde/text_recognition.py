import os
import json
import base64
import requests
import time
import re
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from config.config import Config

class TextRecognizer:
    def __init__(self, config: Config):
        self.config = config
        self.recognition_api_url = self.config.recognition_api_url
        self.recognition_headers = self.config.recognition_headers
        self.request_delay_seconds = self.config.request_delay_seconds

    def _apply_api_delay(self):
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)

    def image_to_base64(self, image_path):
        img = Image.open(image_path).convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def natural_sort_key(self, filename):
        parts = re.split(r'(\d+)', filename)
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    def process_text_recognition(self, image_folder, bbox_json_file, recognition_output_file, log_file): 
        try:
            with open(bbox_json_file, 'r', encoding='utf-8') as f:
                bbox_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Bounding box JSON file not found at {bbox_json_file}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {bbox_json_file}")
            return

        results = {}
        successful_recognitions = 0
        failed_recognitions = 0
        not_found_images = 0

        with open(log_file, 'a', encoding='utf-8') as log: 
            log.write("\n\n" + "=" * 50 + "\n")
            log.write("STARTING TEXT RECOGNITION LOG\n")
            log.write("=" * 50 + "\n\n")
            log.write("PROCESSING IMAGES FOR TEXT RECOGNITION:\n")
            log.write("-" * 50 + "\n")

            sorted_image_names = sorted(bbox_data.keys(), key=self.natural_sort_key)

            for image_name in tqdm(sorted_image_names, desc="Processing images for recognition", unit="image"):
                bboxes_raw = bbox_data[image_name] 
                
                unique_bboxes_tuples = set(tuple(b) for b in bboxes_raw)
                bboxes_to_send = [list(b) for b in unique_bboxes_tuples]
                
                image_path = os.path.join(image_folder, image_name)
                if not os.path.exists(image_path):
                    not_found_images += 1
                    log.write(f"? Image not found for recognition: {image_name}\n") 
                    continue 

                img_str = self.image_to_base64(image_path)
                payload = {"img": f"data:image/jpeg;base64,{img_str}", "bboxes": bboxes_to_send}

                try:
                    self._apply_api_delay() 
                    response = requests.get(self.recognition_api_url, headers=self.recognition_headers, json=payload)
                    response.raise_for_status()
                    recognition_results = response.json()
                    
                    print(f"DEBUG: Raw API recognition_results for {image_name}: {json.dumps(recognition_results, indent=2, ensure_ascii=False)}")
                    
                    deduplicated_results_as_tuples = set()
                    for item in recognition_results:
                        if isinstance(item, list):
                            hashable_item = tuple(tuple(sorted(d.items())) for d in item)
                            deduplicated_results_as_tuples.add(hashable_item)
                        else:
                            hashable_item = tuple(sorted(item.items()))
                            deduplicated_results_as_tuples.add((hashable_item,))

                    cleaned_results = [
                        [dict(sorted_item_tuple) for sorted_item_tuple in inner_tuple]
                        for inner_tuple in sorted(list(deduplicated_results_as_tuples))
                    ]
                    
                    results[image_name] = {"bboxes": bboxes_to_send, "recognized_texts": cleaned_results}
                    successful_recognitions += 1
                    log.write(f"✓ Recognized text for: {image_name}\n") 
                except requests.exceptions.RequestException as e:
                    failed_recognitions += 1
                    error_message = f"✗ Recognition failed for image: {image_name} - {e}"
                    print(error_message)
                    log.write(f"{error_message}\n") 
                    results[image_name] = {"error": str(e)}

            log.write(f"\nTotal Successful Recognitions: {successful_recognitions}\n") 
            log.write(f"Total Failed Recognitions: {failed_recognitions}\n") 
            log.write(f"Total Images Not Found for Recognition: {not_found_images}\n") 
            log.write(f"Total Images Attempted for Recognition: {successful_recognitions + failed_recognitions + not_found_images}\n") 
            log.write("\n" + "=" * 50 + "\n")
            log.write("TEXT RECOGNITION LOG END\n")

        os.makedirs(Path(recognition_output_file).parent, exist_ok=True)
        
        with open(recognition_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ Processing complete! Recognition results saved to: {recognition_output_file}")
        print(f"Total images processed for recognition: {len(results)}")
