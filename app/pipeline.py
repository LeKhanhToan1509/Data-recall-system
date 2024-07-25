import os
import uuid
import json

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient # type: ignore 
from qdrant_client.models import PointStruct, VectorParams, Distance # type: ignore

from configure import Config
from database.database import SessionLocal, engine, Base
import database.models as models
from etl.image_quality import Brightness, Blurriness, Entropy
from serverless.task.autolabel.vlm import AutoLabel_FLorence2
from serverless.task.image_embeded_clip.image_feature import ImageFeatureExtractor
from storage.minio_storage import MinioClientWrapper
from utils import get_file_download_date, crop_image


# Initialize database
Base.metadata.create_all(engine)
db = SessionLocal()

class ImageProcessor:
    def __init__(self, minio_config, db_session, qdrant_url):
        self.auto_label = AutoLabel_FLorence2("microsoft/Florence-2-base", 'cpu')
        self.minio_client = MinioClientWrapper(minio_config['domain'], minio_config['user'], minio_config['password'])
        self.db = db_session
        self.client = QdrantClient(url=qdrant_url, timeout=60.0)  # Increase timeout to 60 seconds
        self.feature_extractor = ImageFeatureExtractor()
        self.feature_size = 512
        self._initialize_qdrant_collections()

    def _initialize_qdrant_collections(self):
        try:
            if not self.client.collection_exists(collection_name="object_collection"):
                self.client.create_collection(
                    collection_name="object_collection",
                    vectors_config=VectorParams(size=self.feature_size, distance=Distance.COSINE)
                )
            if not self.client.collection_exists(collection_name="image_collection"):
                self.client.create_collection(
                    collection_name="image_collection",
                    vectors_config=VectorParams(size=self.feature_size, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"An error occurred while initializing Qdrant collections: {e}")

    def process_image(self, file_path, file_name, task_name):
        img = Image.open(file_path)
        width, height = img.size
        
        self.minio_client.upload_object(bucket_name=task_name, file_path=file_path, object_name=file_name)
        
        id_image = uuid.uuid4().hex
        url_image = self.minio_client.get_url_object(bucket_name=task_name, object_name=file_name)
        
        # Extract feature vector and ensure it is a list
        feature_vector = self.feature_extractor.extract_features_clip(img).tolist()
        
        # Upload image to Qdrant
        qdrant_image = PointStruct(
            id=id_image,
            vector=feature_vector,
            payload={"id_image": id_image}
        )
        try:
            self.client.upsert(collection_name="image_collection", points=[qdrant_image])
        except Exception as e:
            print(f"An error occurred while uploading to Qdrant: {e}")
        
        description = self.auto_label.get_description(img)
        date_time = get_file_download_date(file_path)
        
        dark_score = Brightness.calculate_brightness_score(img)['brightness_perc_95']
        blur_score = Blurriness.calculate_blurriness_score(img)
        light_score = 1 - Brightness.calculate_brightness_score(img)['brightness_perc_5']
        low_information_score = Entropy.calc_entropy_score(img)

        metrics = {
            "dark_score": dark_score,
            "light_score": light_score,
            "low_information_score": low_information_score,
            "blur_score": blur_score
        }

        metadata = {
            "date_time": date_time,
            "local_path": file_path,
            "task": task_name,
            "size": "{}x{}".format(width, height)
        }

        image = models.Image(
            id=id_image,
            url=url_image,
            description=description,
            meta_data=metadata,
            metric=metrics
        )
        self.db.add(image)

        objects = self.auto_label.label_image_all(img)
        id_objects = []
        features = []
        for obj in objects:
            for class_name, bbox in obj.items():
                x1, y1, x2, y2 = bbox
                cr_img = crop_image(img, (x1, y1, x2, y2))
                cr_ft = np.array(self.feature_extractor.extract_features_clip(cr_img)).tolist()
                features.append(cr_ft)
                bbox_json = json.dumps(bbox.tolist())
                id_object = uuid.uuid4().hex
                id_objects.append(id_object)
                obj_instance = models.Object(
                    id=id_object,
                    image_id=id_image,
                    class_name=class_name,
                    bbox=bbox_json
                )
                self.db.add(obj_instance)

        qdrant_objects = [
            PointStruct(
                id=id_object,
                vector=feature,
                payload={"id_image": id_image, "id_object": id_object}
            )
            for feature, id_object in zip(features, id_objects)
        ]
        try:
            self.client.upsert(collection_name="object_collection", points=qdrant_objects)
        except Exception as e:
            print(f"An error occurred while uploading objects to Qdrant: {e}")
        
        self.db.commit()

    def close(self):
        self.db.close()

def process_images_in_folder(folder_path, task_name, minio_config, db_session, qdrant_url):
    processor = ImageProcessor(minio_config, db_session, qdrant_url)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            processor.process_image(file_path, file_name, task_name)
    processor.close()

# Usage example
if __name__ == "__main__":
    minio_config = {
        'domain': Config.minio.MINIO_DOMAIN,
        'user': Config.minio.MINIO_USER,
        'password': Config.minio.MINIO_PASSWORD
    }
    db_session = SessionLocal()
    qdrant_url = "http://192.168.6.161:6333"
    
    # Process all images in a folder
    process_images_in_folder("/home/mq/data_disk2T/Data-System-Recall/app/test/fire2", "fire", minio_config, db_session, qdrant_url)