import os
from config.config import Config # CHANGE THIS LINE to explicitly import from config.config
from vde.processor import DocumentProcessor

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    base_output_directory = os.path.join(project_root, "output")
    os.makedirs(base_output_directory, exist_ok=True)

    # Calculate the desired input folder path (test_images at the project root)
    desired_input_folder = os.path.join(project_root, 'test_images')
    
    # Initialize Config, passing the input_folder_override
    config = Config(base_path=base_output_directory, input_folder_override=desired_input_folder)

    # Ensure the input folder actually exists
    os.makedirs(config.input_folder, exist_ok=True)

    # Display where YOLO weights are expected
    weights_folder_path = os.path.join(project_root, 'weights')
    os.makedirs(weights_folder_path, exist_ok=True)
    print(f"Make sure your YOLOv8 'best.pt' weights are placed here: {weights_folder_path}")

    # Display where input images are expected (now correctly set)
    print(f"Put your vehicle images here for processing: {config.input_folder}")
    print("For testing, ensure you have .jpg or .png images in this folder.")
    print("All pipeline outputs (cropped images, visualizations, results) will be saved to:")
    print(f"{config.base_path}")

    print("\nStarting the integrated document processing pipeline...")

    processor = DocumentProcessor(config)

    processor.run_full_pipeline()

    print("\nProcessing complete! Check the 'output' folder for results.")

if __name__ == "__main__":
    main()