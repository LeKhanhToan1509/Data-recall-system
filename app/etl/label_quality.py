from cleanlab.object_detection.rank import get_label_quality_scores, issues_from_scores
from cleanlab.object_detection.filter import find_label_issues
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any


class CleanLabObjectDetection:
    def __init__(self):
        pass

    @staticmethod
    def processing_empty_label(folder_path: str) -> None:
        for item in os.listdir(folder_path):
            file = os.path.join(folder_path, item)
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    with open(file, 'w') as f:
                        f.write('0 0 0 0 0\n')

    @staticmethod
    def length_processing(label_folder_path: str, predict_folder_path: str) -> None:
        for item in os.listdir(label_folder_path):
            if item not in os.listdir(predict_folder_path):
                with open(os.path.join(predict_folder_path, item), 'w') as f:
                    f.write('0 0 0 0 0\n')

    @staticmethod
    def convert_bbox_to_absolute(xywh: List[float], image_width: int, image_height: int) -> List[float]:
        x_center, y_center, width, height = xywh
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height
        return [x_min, y_min, x_max, y_max]

    def visualize_img_pil(self, img_path: str, data: list, conf_thres=0.7, classes_list: List[str] = [], is_predict=False):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for det in data:
            # Ensure all inputs are converted to the correct types
            class_id = int(det[0])
            x_center, y_center, width, height = map(float, det[1:5])
            conf_score = float(det[5]) if is_predict else 1.0

            if is_predict and conf_score < conf_thres:
                continue
            
            bbox = self.convert_bbox_to_absolute([x_center, y_center, width, height], img.width, img.height)
            x_min, y_min, x_max, y_max = map(int, bbox)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="blue" if is_predict else "green", width=2)
            class_name = classes_list[class_id] if class_id < len(classes_list) else "Unknown"
            text = f"{class_name}-{conf_score:.2f}" if is_predict else class_name
            draw.text((x_min, y_min - 10), text, fill="red", font=font)
        return img

    
    def save_comparison(self, img_path, label, predict, save_path):
        label_img = self.visualize_img_pil(img_path, label, is_predict=False)
        predict_img = self.visualize_img_pil(img_path, predict, is_predict=True)
        
        combined_img = Image.new('RGB', (label_img.width * 2, label_img.height))
        combined_img.paste(label_img, (0, 0))
        combined_img.paste(predict_img, (label_img.width, 0))
        
        combined_img.save(save_path)
        print(f"Comparison image saved at {save_path}")
        
    def validate_lengths(self, label_folder_path: str, image_folder_path: str, predict_folder_path: str) -> None:
        length_label = len(os.listdir(label_folder_path))
        length_image = len(os.listdir(image_folder_path))
        length_predict = len(os.listdir(predict_folder_path))
        if length_label != length_image or length_label != length_predict:
            raise ValueError("The number of labels, images, and predictions are not equal")

    def load_labels(self, label_folder_path: str, image_folder_path: str) -> List[Dict[str, Any]]:
        labels = []
        for annotation_file in os.listdir(label_folder_path):
            image_name = annotation_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_folder_path, image_name)

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            with open(os.path.join(label_folder_path, annotation_file), 'r') as file:
                bboxes = []
                classes = []
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    bbox_absolute = self.convert_bbox_to_absolute([x_center, y_center, width, height], image_width, image_height)
                    bboxes.append(bbox_absolute)
                    classes.append(int(class_id))

                labels.append({
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'labels': np.array(classes),
                    'seg_map': image_name
                })
        return labels

    def load_predictions(self, image_folder_path: str, predict_folder_path: str, num_classes: int) -> List[List[np.ndarray]]:
        predictions = []
        for predict_file in os.listdir(predict_folder_path):
            image_name = predict_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_folder_path, image_name)

            if os.path.isfile(image_path):
                with Image.open(image_path) as img:
                    image_width, image_height = img.size

                file_path = os.path.join(predict_folder_path, predict_file)
                if os.path.isfile(file_path):
                    bboxes_by_class = [np.array([], dtype=np.float32).reshape(0, 5) for _ in range(num_classes)]
                    with open(file_path, 'r') as file:
                        for line in file:
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                class_id, x_center, y_center, width, height, confidence = map(float, parts)
                                class_id = int(class_id)
                                bbox_absolute = self.convert_bbox_to_absolute([x_center, y_center, width, height], image_width, image_height)
                                bbox_with_confidence = bbox_absolute + [confidence]
                                if 0 <= class_id < num_classes:
                                    bboxes_by_class[class_id] = np.append(bboxes_by_class[class_id], [bbox_with_confidence], axis=0)
                    predictions.append(bboxes_by_class)
        return predictions

    def clean_lap(self, label_folder_path: str, predict_folder_path: str, image_folder_path: str, num_classes: int, threshold: float = 0.8) -> List[str]:
        self.processing_empty_label(label_folder_path)
        self.processing_empty_label(predict_folder_path)
        self.length_processing(label_folder_path, predict_folder_path)
        self.validate_lengths(label_folder_path, image_folder_path, predict_folder_path)
        labels = self.load_labels(label_folder_path, image_folder_path)
        predictions = self.load_predictions(image_folder_path, predict_folder_path, num_classes)

        label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)
        scores = get_label_quality_scores(labels, predictions)
        issue_idx = issues_from_scores(scores, threshold=threshold)

        img_paths = []
        for item in issue_idx:
            label = labels[item]
            image_path = os.path.join(image_folder_path, label['seg_map'])
            img_paths.append(image_path)
        return img_paths

    def create_report(self, label_folder_path: str, image_folder_path: str, predict_folder_path: str, report_path : str, classes: List[str] = [], threshold: float = 0.8) -> None:
        number_classes = len(classes)
        image_paths = []
        save_path = ""
        for item in report_path.split("/")[:-1]:
            save_path += item + "/"
        labels = self.load_labels(label_folder_path, image_folder_path)
        predictions = self.load_predictions(image_folder_path = image_folder_path, predict_folder_path = predict_folder_path, num_classes = number_classes)
        scores = get_label_quality_scores(labels, predictions)
        issue_idx = issues_from_scores(scores, threshold = threshold)
        for item in issue_idx:
            label = labels[item]
            image_path = os.path.join(image_folder_path, label['seg_map'])
            file_name = label['seg_map']
            a = file_name.replace('.jpg', '.txt')
            b = file_name.replace('.jpg', '.txt')
            label_path = os.path.join(label_folder_path, a)
            predict_path = os.path.join(predict_folder_path, b)
            label = []
            predict = []
            with open(label_path, 'r') as f:
                for line in f:
                    label.append(line.strip().split())
            label = np.array(label, dtype=np.float32)

            with open(predict_path, 'r') as f:
                for line in f:
                    predict.append(line.strip().split())
            predict = np.array(predict, dtype=np.float32)
            label_img = self.visualize_img_pil(image_path, label, classes_list=classes, is_predict=False)
            predict_img = self.visualize_img_pil(image_path, predict, classes_list=classes, is_predict=True)
            combined_img = Image.new('RGB', (label_img.width * 2 + 20, label_img.height))
            combined_img.paste(label_img, (0, 0))
            combined_img.paste(predict_img, (label_img.width + 20, 0))
            combined_img.save(f'{save_path}{file_name}')
            image_paths.append(f'{save_path}{file_name}')
        images = [Image.open(image).convert('RGB') for image in image_paths]    
        images[0].save(report_path, save_all=True, append_images=images[1:])
        for item in image_paths:
            os.remove(item)

        
            
