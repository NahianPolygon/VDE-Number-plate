import os
import json
import shutil
import time
import re
from tqdm import tqdm
from vde.easy_ocr import EasyOCRRecognizer
from config.config import Config
from vde.yolo import YOLODetector
from vde.edge import EdgeDetector
from vde.perspective import PerspectiveCorrector
from vde.text_detection import TextDetector
from vde.text_recognition import TextRecognizer

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.yolo_detector = YOLODetector(self.config)
        self.edge_detector = EdgeDetector()
        self.perspective_corrector = PerspectiveCorrector()
        self.text_detector = TextDetector(self.config)
        self.text_recognizer = TextRecognizer(self.config)
        self.easy_ocr_recognizer = EasyOCRRecognizer(self.config)

    def _clear_folder(self, folder_path):
        if os.path.exists(folder_path):
            print(f"Clearing folder: {folder_path}")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            os.makedirs(folder_path, exist_ok=True) 
        else:
            os.makedirs(folder_path, exist_ok=True) 

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def run_full_pipeline(self):
        print("=" * 60)
        print("STARTING FULL DOCUMENT PROCESSING PIPELINE")
        print("=" * 60)

        if os.path.exists(self.config.log_file):
            with open(self.config.log_file, 'w') as f:
                f.truncate(0)

        yolo_detected_cropped_image_paths = []

        print("\nClearing previous output directories...")
        self._clear_folder(self.config.yolo_cropped_vehicles_folder)
        self._clear_folder(self.config.yolo_detection_vis_folder)
        self._clear_folder(self.config.edge_output_folder)
        self._clear_folder(self.config.successful_folder)
        self._clear_folder(self.config.unsuccessful_folder)
        self._clear_folder(self.config.visualization_folder)
        self._clear_folder(self.config.corrected_output_folder)
        self._clear_folder(self.config.detection_vis_folder)

        if self.config.run_yolo_detection:
            print("\n0. Running YOLOv8 Vehicle Detection...")
            if self.yolo_detector.model is None:
                print("Skipping YOLOv8 detection as the model failed to load.")
            else:
                input_images = [os.path.join(self.config.input_folder, f)
                                for f in os.listdir(self.config.input_folder)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                input_images.sort(key=self.natural_sort_key) 

                if self.config.limit is not None:
                    input_images = input_images[:self.config.limit]
                    print(f"    (YOLO processing limited to the first {len(input_images)} images from {self.config.input_folder})")

                all_yolo_detections_log = [] 
                
                os.makedirs(self.config.yolo_cropped_vehicles_folder, exist_ok=True)
                os.makedirs(self.config.yolo_detection_vis_folder, exist_ok=True)

                with open(self.config.log_file, 'a', encoding='utf-8') as log:
                    log.write("\n\n" + "=" * 50 + "\n")
                    log.write("STARTING YOLOv8 DETECTION LOG\n")
                    log.write("=" * 50 + "\n\n")
                    log.write("PROCESSING IMAGES FOR YOLO DETECTION:\n")
                    log.write("-" * 50 + "\n")

                    yolo_success_count = 0
                    yolo_fail_count = 0

                    for img_path in tqdm(input_images, desc="YOLO Detecting and Cropping"):
                        original_filename = os.path.basename(img_path)
                        detections = self.yolo_detector.detect_and_crop_vehicles(
                            img_path,
                            self.config.yolo_cropped_vehicles_folder,
                            self.config.yolo_detection_vis_folder,
                            all_yolo_detections_log 
                        )
                        if detections:
                            yolo_success_count += 1
                            log.write(f"✓ Detected {len(detections)} vehicles in: {original_filename}\n")
                        else:
                            yolo_fail_count += 1
                            log.write(f"✗ No vehicles detected in: {original_filename}\n")

                        for det in detections:
                            yolo_detected_cropped_image_paths.append(det['cropped_image_path'])
                
                    log.write(f"\nTotal Successful YOLO Detections (at least one vehicle): {yolo_success_count}\n")
                    log.write(f"Total Failed YOLO Detections (no vehicles): {yolo_fail_count}\n")
                    log.write(f"Total Images Processed by YOLO: {yolo_success_count + yolo_fail_count}\n")
                    log.write("\n" + "=" * 50 + "\n")
                    log.write("YOLOv8 DETECTION LOG END\n")

                with open(self.config.yolo_detection_results_file, 'w', encoding='utf-8') as f:
                    json.dump(all_yolo_detections_log, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved detailed YOLO detection results to: {self.config.yolo_detection_results_file}")
                print(f"🖼️ Saved YOLO visualized images to: {self.config.yolo_detection_vis_folder}")
                print(f"✂️ Saved cropped vehicle images to: {self.config.yolo_cropped_vehicles_folder}")

        if self.config.run_edge_detection:
            print("\n1. Running Edge Detection...")
            
            if self.config.run_yolo_detection and yolo_detected_cropped_image_paths:
                edge_detection_input_paths = yolo_detected_cropped_image_paths
                print(f"    (Processing {len(edge_detection_input_paths)} images from YOLO cropped vehicles)")
            else:
                edge_detection_input_paths = [os.path.join(self.config.input_folder, f)
                                              for f in os.listdir(self.config.input_folder)
                                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                edge_detection_input_paths.sort(key=self.natural_sort_key)
                print(f"    (Processing {len(edge_detection_input_paths)} images from original input folder)")

            self.edge_detector.process_folder_edge_detection(
                input_image_paths=edge_detection_input_paths, 
                output_folder=self.config.edge_output_folder,
                successful_folder=self.config.successful_folder,
                unsuccessful_folder=self.config.unsuccessful_folder,
                visualization_folder=self.config.visualization_folder,
                log_file=self.config.log_file,
                coordinates_file=self.config.coordinates_file
            )

        if self.config.run_perspective_correction:
            print("\n2. Running Perspective Correction...")
            self.perspective_corrector.correct_all_images(
                source_directory=self.config.successful_folder,
                coordinates_json_path=self.config.coordinates_file,
                output_directory=self.config.corrected_output_folder
            )

        if self.config.run_text_detection:
            print("\n3. Running Text Detection...")
            self.text_detector.get_text_detections(
                image_folder=self.config.corrected_output_folder,
                output_json_path=self.config.detection_results_file,
                vis_folder=self.config.detection_vis_folder,
                log_file=self.config.log_file
            )

        if self.config.run_post_processing:
            print("\n4. Post-processing Detection Results...")
            self.text_detector.post_process_detections(
                detection_json_path=self.config.detection_results_file,
                output_json_path=self.config.processed_detection_file
            )

        if self.config.run_text_recognition:
            print("\n5. Running Text Recognition...")
            self.text_recognizer.process_text_recognition(
                image_folder=self.config.corrected_output_folder,
                bbox_json_file=self.config.processed_detection_file,
                recognition_output_file=self.config.recognition_results_file,
                log_file=self.config.log_file
            )
        if self.config.run_easy_ocr: 
            print("\n6. Running Text Recognition (EasyOCR)...")
            self.easy_ocr_recognizer.process_images_for_ocr(
                image_folder=self.config.corrected_output_folder,
                output_json_path=self.config.easy_ocr_results_file, 
                vis_folder=self.config.easy_ocr_vis_folder,
                log_file=self.config.log_file
            )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)