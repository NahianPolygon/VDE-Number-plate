# VDE-Number-plate

A Python-based solution for vehicle number plate detection and recognition. This project provides tools and scripts to automatically locate, extract, and interpret vehicle registration numbers from images or video streams. It is designed for developers, researchers, and anyone interested in computer vision and intelligent transportation systems.

---

## Project Structure

```
VDE_OCR/
│
├── test_images/               # Sample images for testing
│
├── api/
│   └── api.py                 # API endpoint to use the model as a service
│
├── vde/
│   ├── __init__.py
│   ├── processor.py           # Image processing utilities
│   ├── yolo.py                # YOLO-based detection
│   ├── edge.py                # Edge detection methods
│   ├── perspective.py         # Perspective transformation methods
│   ├── text_detection.py      # Text detection logic
│   ├── text_recognition.py    # Text recognition logic
│   └── easy_ocr.py            # EasyOCR wrapper
│
├── config/
│   └── config.py              # Configuration settings (e.g., model paths)
│
├── weights/
│   └── best.pt                # Pretrained YOLO weights
│
├── main.py                    # Main script to run detection and recognition
```

---

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/NahianPolygon/VDE-Number-plate.git
   cd VDE-Number-plate/VDE_OCR
   ```

2. **Create a Virtual Environment (Recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
   > Make sure you have [Python 3.7+](https://www.python.org/downloads/) installed.

---

## Usage

### Detect and Recognize Number Plates from an Image

To run detection and recognition on a single image:
```sh
python main.py --image path/to/image.jpg
```

### Process a Directory of Images

To process all images in a folder:
```sh
python main.py --input_dir test_images/
```

### Options

- `--output_dir` : Directory to save result images and text (default: `output/`)
- `--show`       : Display images with results (default: False)

**Example:**
```sh
python main.py --input_dir test_images/ --output_dir results/ --show
```

---

## API Usage

If you want to use the model as an API service:
- See the `api/api.py` file for example endpoints.
- Run the API (assuming FastAPI or Flask is used):
  ```sh
  python api/api.py
  ```
- Interact with the API using HTTP requests (see the API script for details).

---

## Model Weights

- The YOLO weights file (`best.pt`) should be placed in the `weights/` directory.
- If you want to retrain or use different weights, update the path in `config/config.py`.

---

## Dependencies

- [OpenCV](https://opencv.org/) for image processing
- [YOLO](https://github.com/ultralytics/yolov5) for detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition

> **Note:** This project uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) for reading text from number plates. 

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [YOLO](https://github.com/ultralytics/yolov5)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

For questions or support, please open an issue on [GitHub](https://github.com/NahianPolygon/VDE-Number-plate/issues).
