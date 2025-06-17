import cv2
import numpy as np
import os
import json
from tqdm import tqdm

class PerspectiveCorrector:
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def find_image_in_directory(self, source_directory, image_name):
        direct_path = os.path.join(source_directory, image_name)
        if os.path.exists(direct_path):
            return direct_path

        name_without_ext = os.path.splitext(image_name)[0]
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                     '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']

        for ext in extensions:
            test_path = os.path.join(source_directory, name_without_ext + ext)
            if os.path.exists(test_path):
                return test_path

        return None

    def correct_single_image(self, source_directory, output_directory, image_name, coordinates):
        image_path = self.find_image_in_directory(source_directory, image_name)
        if image_path is None:
            print(f"Warning: Image '{image_name}' not found in {source_directory}")
            return False

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return False

        try:
            pts = np.array(coordinates, dtype=np.float32)
            warped = self.four_point_transform(img, pts)

            name_without_ext = os.path.splitext(image_name)[0]
            output_filename = f"corrected_{name_without_ext}.jpg"
            output_path = os.path.join(output_directory, output_filename)

            os.makedirs(output_directory, exist_ok=True)
            success = cv2.imwrite(output_path, warped)

            if success:
                return True
            else:
                print(f"✗ Failed to save corrected image: {output_filename}")
                return False

        except Exception as e:
            print(f"✗ Error processing {image_name} for perspective correction: {str(e)}")
            return False

    def correct_all_images(self, source_directory, coordinates_json_path, output_directory):
        try:
            with open(coordinates_json_path, 'r') as f:
                coordinates_data = json.load(f)
            print(f"Loaded coordinates for {len(coordinates_data)} images from JSON.")
        except FileNotFoundError:
            print(f"Error: JSON file not found at {coordinates_json_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {coordinates_json_path}")
            return

        if not coordinates_data:
            print("No coordinates data loaded. Exiting perspective correction.")
            return

        total_images = len(coordinates_data)
        successful = 0
        failed = 0
        not_found = 0

        print(f"\nStarting perspective correction for {total_images} images...")
        print(f"Source directory: {source_directory}")
        print(f"Output directory: {output_directory}")
        print("-" * 60)

        for i, (image_name, coordinates) in tqdm(enumerate(coordinates_data.items(), 1), 
                                                total=total_images, 
                                                desc="Correcting perspectives"):
            if not isinstance(coordinates, list) or len(coordinates) != 4:
                failed += 1
                continue

            if self.find_image_in_directory(source_directory, image_name) is None:
                not_found += 1
                continue

            if self.correct_single_image(source_directory, output_directory, image_name, coordinates):
                successful += 1
            else:
                failed += 1

        print("-" * 60)
        print(f"Processing complete!")
        print(f"✓ Successfully processed: {successful}")
        print(f"✗ Failed to process: {failed}")
        print(f"? Images not found: {not_found}")
        print(f"Total images in JSON: {total_images}")

        if successful > 0:
            print(f"\n✅ Corrected images saved in: {output_directory}")