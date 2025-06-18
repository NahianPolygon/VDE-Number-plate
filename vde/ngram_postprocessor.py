import re
import json
import os
from difflib import SequenceMatcher
from tqdm import tqdm

class NgramPostprocessor:
    def __init__(self, config):
        self.config = config
        self.targets = [
            'ঢাকা মেট্রো', 'চট্ট মেট্রো', 'রাজ মেট্রো', 'সিলেট মেট্রো', 'খুলনা মেট্রো', 'বরিশাল মেট্রো',
            'ঢাকা', 'চট্টগ্রাম', 'রাজশাহী', 'খুলনা', 'বরিশাল', 'সিলেট', 'রংপুর', 'ময়মনসিংহ',
            'কুমিল্লা', 'নরসিংদী', 'ফরিদপুর', 'দিনাজপুর', 'ভোলা', 'নোয়াখালী', 
            'কক্সবাজার', 'বগুড়া', 'সাতক্ষীরা', 'নাটোর', 'টাঙ্গাইল', 'গাজীপুর', 'ফেনী',
            'ঝিনাইদহ', 'লক্ষ্মীপুর', 'ঝালকাঠি', 'পিরোজপুর', 'ব্রাহ্মণবাড়িয়া', 
            'শরীয়তপুর', 'জয়পুরহাট', 'নওগাঁ', 'চাঁপাইনবাবগঞ্জ', 'কুড়িগ্রাম', 
            'ঠাকুরগাঁও', 'পঞ্চগড়', 'সিরাজগঞ্জ', 'রাজবাড়ী', 'কিশোরগঞ্জ', 
            'গোপালগঞ্জ', 'মেহেরপুর', 'চুয়াডাঙ্গা', 'মাদারীপুর',
        ]
        self.max_n = 3
        self.matching_threshold = 0.6
        self.replacement_threshold = config.ngram_replacement_threshold

    def get_best_ngram_match(self, ocr_text): 
        best_match_info = {
            'matched_phrase_in_text': None,
            'matched_target': None,
            'similarity_score': None
        }

        
        words_in_ocr_text = re.split(r'[\s\-]+', ocr_text.strip())
        
        current_best_score = float('-inf')

        for n in range(1, self.max_n + 1):
            for i in range(len(words_in_ocr_text) - n + 1):
                phrase = ' '.join(words_in_ocr_text[i:i + n])
                for target in self.targets:
                    sim = SequenceMatcher(None, phrase, target).ratio()
                    if sim > current_best_score and sim >= self.matching_threshold:
                        best_match_info['matched_phrase_in_text'] = phrase
                        best_match_info['matched_target'] = target
                        best_match_info['similarity_score'] = sim
                        current_best_score = sim
        
        return best_match_info

    def process_and_enrich_results(self, main_recognition_file, easy_ocr_file, output_file, log_file):
        combined_results = {}

        if os.path.exists(main_recognition_file):
            with open(main_recognition_file, 'r', encoding='utf-8') as f:
                main_data = json.load(f)
                for img_name, img_data in main_data.items():
                    combined_results[img_name] = combined_results.get(img_name, {})
                    combined_results[img_name]['main_recognition'] = img_data

        if os.path.exists(easy_ocr_file):
            with open(easy_ocr_file, 'r', encoding='utf-8') as f:
                easy_data = json.load(f)
                for img_name, img_data in easy_data.items():
                    combined_results[img_name] = combined_results.get(img_name, {})
                    combined_results[img_name]['easy_ocr_recognition'] = img_data
        
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write("\n\n" + "=" * 50 + "\n")
            log.write("STARTING N-GRAM SIMILARITY POST-PROCESSING LOG\n")
            log.write("=" * 50 + "\n\n")
            log.write("ENRICHING RECOGNITION RESULTS WITH N-GRAM MATCHES:\n")
            log.write("-" * 50 + "\n")

            for img_name in tqdm(combined_results.keys(), desc="Applying N-gram post-processing"):
                
                
                if 'main_recognition' in combined_results[img_name]:
                    main_data_for_img = combined_results[img_name]['main_recognition']
                    if 'recognized_texts' in main_data_for_img and main_data_for_img['recognized_texts']:
                        processed_main_texts = []
                        for sublist in main_data_for_img['recognized_texts']:
                            processed_sublist = []
                            for text_obj in sublist:
                                if 'text' in text_obj and text_obj['text']:
                                    original_text = text_obj['text']
                                    ngram_match_info = self.get_best_ngram_match(original_text) 
                                    
                                    if ngram_match_info['similarity_score'] is not None and \
                                       ngram_match_info['similarity_score'] >= self.replacement_threshold and \
                                       ngram_match_info['matched_phrase_in_text'] is not None: 
                                        
                                       
                                        text_obj['text'] = original_text.replace(
                                            ngram_match_info['matched_phrase_in_text'],
                                            ngram_match_info['matched_target'],
                                            1
                                        )
                                    
                                processed_sublist.append(text_obj)
                            processed_main_texts.append(processed_sublist)
                        main_data_for_img['recognized_texts'] = processed_main_texts
                    log.write(f"✓ N-gram processed main recognition for: {img_name}\n")

                
                if 'easy_ocr_recognition' in combined_results[img_name]:
                    easy_ocr_data_for_img = combined_results[img_name]['easy_ocr_recognition']
                    if 'easy_ocr_results' in easy_ocr_data_for_img and easy_ocr_data_for_img['easy_ocr_results']:
                        processed_easy_ocr_texts = []
                        for text_obj in easy_ocr_data_for_img['easy_ocr_results']:
                            if 'text' in text_obj and text_obj['text']:
                                original_text = text_obj['text']
                                ngram_match_info = self.get_best_ngram_match(original_text) 
                                
                                if ngram_match_info['similarity_score'] is not None and \
                                   ngram_match_info['similarity_score'] >= self.replacement_threshold and \
                                   ngram_match_info['matched_phrase_in_text'] is not None: 
                                    
                                    
                                    text_obj['text'] = original_text.replace(
                                        ngram_match_info['matched_phrase_in_text'],
                                        ngram_match_info['matched_target'],
                                        1
                                    )
                                
                            processed_easy_ocr_texts.append(text_obj)
                        easy_ocr_data_for_img['easy_ocr_results'] = processed_easy_ocr_texts
                    log.write(f"✓ N-gram processed EasyOCR recognition for: {img_name}\n")
            
            log.write("\n" + "=" * 50 + "\n")
            log.write("N-GRAM SIMILARITY POST-PROCESSING LOG END\n")


        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        print(f"✅ N-gram enriched results saved to: {output_file}")