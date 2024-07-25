import sys
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv
from PIL import Image

class AutoLabel_FLorence2:
    def __init__(self, model_name: str, device: str = None):
        self.checkpoint = model_name
        self.device = device or torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.processor = self.load_model()

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint, trust_remote_code=True).to(self.device)
        processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        return model, processor

    def load_image(self, image_path: str):
        image = Image.open(image_path)
        img_width, img_height = image.size
        return image, img_width, img_height

    def run_inference(self, image: Image, task: str, text: str = ""):
        prompt = task + text
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(generated_text, task=task, image_size=image.size)

    def get_description(self, image: Image):
        task = "<MORE_DETAILED_CAPTION>"
        response = self.run_inference(image=image, task=task)
        return response[task]
    
    def label_by_keyword(self, image: Image, keyword: str):
        labels = []
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        response = self.run_inference(image=image, task=task, text=keyword)
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        if detections['class_name'] is None:
            return labels
        for i in range(len(detections['class_name'])):
            class_name = detections['class_name'][i]
            if any(kw in class_name for kw in keyword.split()):
                labels.append({class_name: detections.xyxy[i].astype(float)})
        return labels

    def label_image(self, image: Image, keyword: str):
        labels = []
        task = "<OD>"
        response = self.run_inference(image=image, task=task)
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        for idx, item in enumerate(detections['class_name']):
            if any(kw in item for kw in keyword.split()):
                labels.append({item: detections.xyxy[idx].astype(float)})
        return labels

    def label_image_all(self, image: Image):
        labels = []
        task = "<OD>"
        response = self.run_inference(image=image, task=task)
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        for idx, item in enumerate(detections['class_name']):
            labels.append({item: detections.xyxy[idx].astype(float)})
        return labels
    
    def get_xyxy(self, res):
        xyxy = []
        for item in res:
            bbox = list(item.keys())
            for key in bbox:
                x_min, y_min, x_max, y_max = item[key]
                line = f"{item} {x_min} {y_min} {x_max} {y_max}"
                xyxy.append(line)
        return xyxy
    
    def convert_to_xywh(self, res, image_width, image_height):
        yolo_lines = []
        for item in res:
            bbox = list(item.keys())
            for key in bbox:
                x_min, y_min, x_max, y_max = item[key]
                width = x_max - x_min
                height = y_max - y_min
                center_x = x_min + width / 2
                center_y = y_min + height / 2
                center_x /= image_width
                center_y /= image_height
                width /= image_width
                height /= image_height
                line = f"0 {center_x} {center_y} {width} {height}"
                yolo_lines.append(line)
        return yolo_lines

    def save_yolo_labels(self, yolo_lines, output_file):
        with open(output_file, 'w') as file:
            for line in yolo_lines:
                file.write(line + '\n')

    def auto_labelMQ(self, file_path: str, folder_path: str, keyword: str):
        image, image_width, image_height = self.load_image(file_path)
        output_file = os.path.join(folder_path, os.path.basename(file_path).replace("images", "labels").replace("jpg", "txt"))
        res = self.label_by_keyword(image, keyword)
        res.extend(self.label_image(image, keyword))
        yolo_lines = self.convert_to_xywh(res, image_width, image_height)
        self.save_yolo_labels(yolo_lines, output_file)

    def auto_label_folder(self, folder_path: str, folder_path_output: str, keyword: str):
        for file in os.listdir(folder_path):
            if file.endswith((".jpg", 'png', 'jpeg', 'gif')):
                self.auto_labelMQ(os.path.join(folder_path, file), folder_path_output, keyword)

