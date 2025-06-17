import os
import shutil
import json
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 

from config.config import Config
from vde.processor import DocumentProcessor

app = FastAPI(
    title="VDE OCR Document Processing API",
    description="API for full document edge detection, perspective correction, text detection, and text recognition.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.post("/process-document/")
async def process_document_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    with tempfile.TemporaryDirectory() as temp_base_path:
        temp_base_path_obj = Path(temp_base_path)

        temp_input_folder = temp_base_path_obj / 'test_images'
        temp_input_folder.mkdir(parents=True, exist_ok=True)

        image_path = temp_input_folder / file.filename
        try:
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

        temp_config = Config(base_path=str(temp_base_path_obj))

        weights_source_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')
        weights_dest_path = temp_base_path_obj / 'weights'
        
        if os.path.exists(weights_source_path):
            shutil.copytree(weights_source_path, weights_dest_path, dirs_exist_ok=True)

        processor = DocumentProcessor(temp_config)

        try:
            processor.run_full_pipeline()

            final_results_path = temp_config.recognition_results_file
            if os.path.exists(final_results_path):
                with open(final_results_path, 'r', encoding='utf-8') as f:
                    recognition_results = json.load(f)
                return JSONResponse(content=recognition_results, status_code=200)
            else:
                raise HTTPException(status_code=500, detail="Recognition results file not found after pipeline execution.")

        except Exception as e:
            error_message = f"Document processing failed: {e}"
            print(f"ERROR: {error_message}") 
            
            log_content = ""
            if os.path.exists(temp_config.log_file):
                try:
                    with open(temp_config.log_file, 'r', encoding='utf-8') as log_f:
                        log_content = log_f.read()
                except Exception as log_read_e:
                    log_content = f"Could not read log file: {log_read_e}"

            raise HTTPException(
                status_code=500,
                detail={"message": error_message, "log_details": log_content}
            )

@app.get("/")
async def read_root():
    return {"message": "VDE OCR API is running!"}