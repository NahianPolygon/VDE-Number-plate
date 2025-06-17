import os

class Config:
    def __init__(self, base_path="output", input_folder_override=None):
        self.base_path = base_path
        self._input_folder_override = input_folder_override

        self.detection_api_url = 'http://192.168.12.91:8080/predictions/text_detection'
        self.recognition_api_url = 'http://192.168.12.91:8080/predictions/text_recognizer'
        self.api_key = 'Polygon12'
        self.detection_headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        self.recognition_headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

        self.limit = 10
        self.run_yolo_detection = True
        self.run_edge_detection = True
        self.run_perspective_correction = True
        self.run_text_detection = True
        self.run_post_processing = True
        self.run_text_recognition = True
        self.run_easy_ocr = True

        self.request_delay_seconds = 2.0

        self.horizontal_padding_ratio = 0.50
        self.vertical_padding_ratio = 0.50
        self.min_horizontal_pad = 2
        self.max_horizontal_pad = 10
        self.min_vertical_pad = 2
        self.max_vertical_pad = 8

        self.easy_ocr_languages = ['bn']

    @property
    def input_folder(self):
        if self._input_folder_override:
            return self._input_folder_override
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, 'test_images')

    @property
    def yolo_weights_path(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, 'weights', 'best.pt')

    @property
    def yolo_cropped_vehicles_folder(self):
        return os.path.join(self.base_path, 'yolo_cropped_vehicles')

    @property
    def yolo_detection_vis_folder(self):
        return os.path.join(self.base_path, 'yolo_detection_visuals')

    @property
    def edge_output_folder(self):
        return os.path.join(self.base_path, 'edge_detection_output')

    @property
    def successful_folder(self):
        return os.path.join(self.base_path, 'successful')

    @property
    def unsuccessful_folder(self):
        return os.path.join(self.base_path, 'unsuccessful')

    @property
    def visualization_folder(self):
        return os.path.join(self.base_path, 'visualization')

    @property
    def corrected_output_folder(self):
        return os.path.join(self.base_path, 'corrected_images')

    @property
    def detection_vis_folder(self):
        return os.path.join(self.base_path, 'detection_visuals')

    @property
    def log_file(self):
        return os.path.join(self.base_path, 'processing_log.txt')

    @property
    def coordinates_file(self):
        return os.path.join(self.base_path, 'coordinates.json')

    @property
    def yolo_detection_results_file(self):
        return os.path.join(self.base_path, 'yolo_detections.json')

    @property
    def detection_results_file(self):
        return os.path.join(self.base_path, 'detection_results.json')

    @property
    def processed_detection_file(self):
        return os.path.join(self.base_path, 'processed_detection_results.json')
    
    @property
    def api_recognition_results_folder(self):
        return os.path.join(self.base_path, 'text_recognition_api_results')

    @property
    def recognition_results_file(self):
        return os.path.join(self.api_recognition_results_folder, 'recognition_results.json')
    
    @property
    def easy_ocr_results_folder(self):
        return os.path.join(self.base_path, 'easy_ocr_results')
    
    @property
    def easy_ocr_results_file(self):
        return os.path.join(self.easy_ocr_results_folder, 'easy_ocr_results.json')

    @property
    def easy_ocr_vis_folder(self):
        return os.path.join(self.easy_ocr_results_folder, 'visualizations')


    def to_dict(self):
        return {
            'base_path': self.base_path,
            'input_folder': self.input_folder,
            'yolo_weights_path': self.yolo_weights_path,
            'yolo_cropped_vehicles_folder': self.yolo_cropped_vehicles_folder,
            'yolo_detection_vis_folder': self.yolo_detection_vis_folder,
            'edge_output_folder': self.edge_output_folder,
            'successful_folder': self.successful_folder,
            'unsuccessful_folder': self.unsuccessful_folder,
            'visualization_folder': self.visualization_folder,
            'corrected_output_folder': self.corrected_output_folder,
            'detection_vis_folder': self.detection_vis_folder,
            'log_file': self.log_file,
            'coordinates_file': self.coordinates_file,
            'yolo_detection_results_file': self.yolo_detection_results_file,
            'detection_results_file': self.detection_results_file,
            'processed_detection_file': self.processed_detection_file,
            'recognition_results_file': self.recognition_results_file,
            'easy_ocr_results_file': self.easy_ocr_results_file,
            'easy_ocr_vis_folder': self.easy_ocr_vis_folder,
            'limit': self.limit,
            'run_yolo_detection': self.run_yolo_detection,
            'run_edge_detection': self.run_edge_detection,
            'run_perspective_correction': self.run_perspective_correction,
            'run_text_detection': self.run_text_detection,
            'run_post_processing': self.run_post_processing,
            'run_text_recognition': self.run_text_recognition,
            'run_easy_ocr': self.run_easy_ocr,
            'request_delay_seconds': self.request_delay_seconds,
            'horizontal_padding_ratio': self.horizontal_padding_ratio,
            'vertical_padding_ratio': self.vertical_padding_ratio,
            'min_horizontal_pad': self.min_horizontal_pad,
            'max_horizontal_pad': self.max_horizontal_pad,
            'min_vertical_pad': self.min_vertical_pad,
            'max_vertical_pad': self.max_vertical_pad,
            'easy_ocr_languages': self.easy_ocr_languages,
        }

    def update_api_config(self, detection_url=None, recognition_url=None, api_key=None):
        if detection_url:
            self.detection_api_url = detection_url
        if recognition_url:
            self.recognition_api_url = recognition_url
        if api_key:
            self.api_key = api_key
        
        self.detection_headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        self.recognition_headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}