import cv2
import numpy as np
import os
import shutil
import json
from tqdm import tqdm

class EdgeDetector:
    def detect_document_edges(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading {image_path}")
            return False, None

        original = image.copy()
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny = cv2.Canny(binary, 350, 1500)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:100]

        doc_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.10 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break
        
        if doc_contour is None:
            return False, None

        if output_path:
            cv2.drawContours(original, [doc_contour], -1, (0, 255, 0), 10)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, original)

        coordinates = doc_contour.reshape(4, 2).tolist()
        return True, coordinates

    def visualize_coordinates(self, image_path, coordinates, output_path):
        image = cv2.imread(image_path)
        if image is None:
            return False

        for i, (x, y) in enumerate(coordinates):
            cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)
            cv2.putText(image, str(i+1), (int(x+15), int(y+15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for i in range(len(coordinates)):
            start_point = (int(coordinates[i][0]), int(coordinates[i][1]))
            end_point = (int(coordinates[(i+1) % len(coordinates)][0]),
                        int(coordinates[(i+1) % len(coordinates)][1]))
            cv2.line(image, start_point, end_point, (255, 0, 0), 3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        return True

    def process_folder_edge_detection(self, input_image_paths, output_folder, successful_folder,
                                    unsuccessful_folder, visualization_folder, log_file, coordinates_file):
        
        for folder in [output_folder, successful_folder, unsuccessful_folder, visualization_folder]:
            os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.dirname(coordinates_file), exist_ok=True)

        successful_count = 0
        unsuccessful_count = 0
        coordinates_data = {}
        
        successful_log_entries = []
        unsuccessful_log_entries = []

        with open(log_file, 'a', encoding='utf-8') as log:
            log.write("\n\n" + "=" * 50 + "\n")
            log.write("STARTING EDGE DETECTION LOG\n")
            log.write("=" * 50 + "\n\n")
            log.write("PROCESSING IMAGES FOR EDGE DETECTION:\n")
            log.write("-" * 50 + "\n")

            for input_path in tqdm(input_image_paths, desc="Edge Detecting"):
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_folder, filename)

                success, coordinates = self.detect_document_edges(input_path, output_path)
                if success:
                    successful_count += 1
                    shutil.copy2(input_path, os.path.join(successful_folder, filename))
                    coordinates_data[filename] = coordinates

                    visualization_path = os.path.join(visualization_folder, filename)
                    self.visualize_coordinates(input_path, coordinates, visualization_path)
                    successful_log_entries.append(f"✓ Detected edges for: {filename}")
                else:
                    unsuccessful_count += 1
                    shutil.copy2(input_path, os.path.join(unsuccessful_folder, filename))
                    unsuccessful_log_entries.append(f"✗ Failed to detect edges for: {filename}")
            
            for entry in successful_log_entries:
                log.write(entry + "\n")
            for entry in unsuccessful_log_entries:
                log.write(entry + "\n")

            log.write(f"\nTotal Successful: {successful_count}\n")
            log.write(f"Total Unsuccessful: {unsuccessful_count}\n")
            log.write(f"Total Processed: {successful_count + unsuccessful_count}\n\n")

            log.write("\n" + "=" * 50 + "\n")
            log.write("EDGE DETECTION LOG END\n")

        with open(coordinates_file, 'w') as coord_file:
            json.dump(coordinates_data, coord_file, indent=2)
        print(f"✅ Edge detection log saved to: {log_file}")
        print(f"✅ Coordinates saved to: {coordinates_file}")