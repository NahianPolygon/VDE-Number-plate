# VDE-Number-plate

A Python-based solution for vehicle number plate detection and recognition, with a comprehensive pipeline that also supports document processing features like edge detection and perspective correction. This project provides tools and scripts to automatically locate, extract, and interpret vehicle registration numbers from images or video streams. It is designed for developers, researchers, and anyone interested in computer vision and intelligent transportation systems.

---

## Project Structure

This seems to be a simple formatting issue, likely caused by a non-standard space character in the original text. The first example you provided contains a non-breaking space character (Unicode `U+00A0`), which can cause display problems. The second example is correctly formatted with standard spaces.

I will provide the correct text that you can directly copy and paste into your README file to fix this. No further searches are required as this is a formatting issue with the provided text.

Here is the corrected version of the project structure block for you to copy and paste:

```
VDE_OCR/
│
├── test_images/                 # Sample images for testing
│
├── api/
│   └── api.py                   # FastAPI endpoint to use the model as a service
│
├── vde/
│   ├── init.py
│   ├── processor.py             # Main pipeline orchestrator
│   ├── yolo.py                  # YOLO-based detection
│   ├── perspective.py           # Perspective transformation methods
│   ├── text_detection.py        # Text detection logic
│   ├── text_recognition.py      # Text recognition logic
│   └── easy_ocr.py              # EasyOCR wrapper for text recognition
│
├── config/
│   └── config.py                # Configuration settings (e.g., model paths, API endpoints)
│
├── weights/
│   └── best.pt                  # Pretrained YOLOv8 weights for number plate detection
│
├── main.py                      # Main script to run the full pipeline on a local folder
└── requirements.txt             # Python dependencies
```

---

## Installation

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/NahianPolygon/VDE-Number-plate.git](https://github.com/NahianPolygon/VDE-Number-plate.git)
    cd VDE-Number-plate/VDE_OCR
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```
    > Make sure you have [Python 3.7+](https://www.python.org/downloads/) installed.

---

## Usage

### Run the Full Pipeline on Local Images

The `main.py` script runs the entire detection and recognition pipeline on images located in the `test_images/` directory. It performs YOLO detection, perspective correction, and text recognition using EasyOCR.

1.  Place your vehicle images in the `VDE_OCR/test_images/` folder.
2.  Run the main script:
    ```sh
    python main.py
    ```

The results, including cropped images, visualizations, and a JSON file with recognized text, will be saved in the `output/` directory.

### Configuration

You can customize the pipeline's behavior by editing the `config/config.py` file. This includes:
- Toggling different pipeline stages (`run_yolo_detection`, `run_easy_ocr`, etc.)
- Changing the EasyOCR language (`easy_ocr_languages = ['bn']` for Bengali)
- Updating the paths to weights or input/output folders.

---

## API Usage

The project includes an API built with FastAPI that allows you to upload an image and receive the processed results.

1.  **Run the API Server**
    ```sh
    python api/api.py
    ```
    This will start the API server, typically at `http://127.0.0.1:8000`.

2.  **Interact with the API**
    - You can use an HTTP client like `curl` or Postman to send a `POST` request with an image file.
    - The main endpoint is `/process-document/`.

    **Example using `curl`:**
    ```sh
    curl -X 'POST' \
      '[http://127.0.0.1:8000/process-document/](http://127.0.0.1:8000/process-document/)' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@/path/to/your/image.jpg'
    ```

The API will return a JSON response containing the recognized text and other processing details.

---

## Model Weights

- The YOLOv8 weights file (`best.pt`) must be placed in the `weights/` directory.
- The `api.py` script attempts to copy these weights to a temporary directory for each API call.

---

## Dependencies

- [OpenCV](https://opencv.org/) for image processing
- [YOLOv8](https://docs.ultralytics.com/) for number plate detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition (supports languages like Bengali)
- [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) for the API server

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [YOLOv8](https://docs.ultralytics.com/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

For questions or support, please open an issue on [GitHub](https://github.com/NahianPolygon/VDE-Number-plate/issues).
