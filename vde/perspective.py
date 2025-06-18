import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import re

class PerspectiveCorrector:
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def getContours(self, img, orig):
        biggest = np.array([])
        maxArea = 0
        imgContour = orig.copy()
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        index = None
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 100:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
                if area > maxArea and len(approx) >= 4:
                    if len(approx) > 4:
                        hull = cv2.convexHull(approx)
                        if len(hull) >= 4:
                            distances = []
                            for point in hull:
                                dist = np.sum((hull - point)**2, axis=2)
                                distances.append(np.sum(dist))
                            sorted_indices = np.argsort(distances)[-4:]
                            biggest = hull[sorted_indices]
                        else:
                            biggest = None
                    else:
                        biggest = approx
                    if biggest is not None and len(biggest) == 4 and biggest.size > 0:
                        maxArea = area
                        index = i
                    else:
                        biggest = np.array([])

        warped = None
        if index is not None and len(biggest) == 4 and biggest.size > 0:
            cv2.drawContours(imgContour, contours, index, (0, 255, 0), 2)
            cv2.drawContours(imgContour, [biggest], -1, (255, 0, 0), 3)
            
            src = np.squeeze(biggest).astype(np.float32)
            src = self.order_points(src)
            
            width = max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[2] - src[3]))
            height = max(np.linalg.norm(src[1] - src[2]), np.linalg.norm(src[3] - src[0]))
            
            dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(orig, M, (int(width), int(height)), flags=cv2.INTER_LINEAR)
        
        return biggest, imgContour, warped

    def correct_perspective(self, image_path, output_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path} for perspective correction.")
            return False

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
        imgThres = cv2.erode(imgDial, kernel, iterations=1)
        
        _, _, warped = self.getContours(imgThres, img)

        if warped is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, warped)
            return True
        else:
            print(f"Warning: Could not correct perspective for {os.path.basename(image_path)}. No suitable contour found.")
            return False

    def correct_all_images(self, source_directory, output_directory, log_file):
        os.makedirs(output_directory, exist_ok=True)
        
        image_files = [f for f in os.listdir(source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        image_files.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
        
        successful_corrections = 0
        failed_corrections = 0

        with open(log_file, 'a', encoding='utf-8') as log:
            log.write("\n\n" + "=" * 50 + "\n")
            log.write("STARTING PERSPECTIVE CORRECTION LOG\n")
            log.write("=" * 50 + "\n\n")
            log.write("PROCESSING IMAGES FOR PERSPECTIVE CORRECTION:\n")
            log.write("-" * 50 + "\n")

            for filename in tqdm(image_files, desc="Correcting Perspective"):
                image_path = os.path.join(source_directory, filename)
                output_path = os.path.join(output_directory, filename)
                
                success = self.correct_perspective(image_path, output_path)
                if success:
                    successful_corrections += 1
                    log.write(f"✓ Corrected perspective for: {filename}\n")
                else:
                    failed_corrections += 1
                    log.write(f"✗ Failed to correct perspective for: {filename}\n")

            log.write(f"\nTotal Successful Corrections: {successful_corrections}\n")
            log.write(f"Total Failed Corrections: {failed_corrections}\n")
            log.write(f"Total Images Processed: {successful_corrections + failed_corrections}\n")
            log.write("\n" + "=" * 50 + "\n")
            log.write("PERSPECTIVE CORRECTION LOG END\n")

        print(f"✅ Perspective correction complete. Corrected images saved to: {output_directory}")