# dataset.py
import os
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog


def register_dataset(dataset_name, image_dir, mask_dir):
    dataset_dicts = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_id = int(filename.split("_")[1].split(".")[0])
            mask_path = os.path.join(mask_dir, f"mask_{image_id}.png")
            image_path = os.path.join(image_dir, filename)

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            instances = []
            for class_id in range(1, 256):
                binary_mask = np.where(mask == class_id, 1, 0).astype(np.uint8)
                if np.sum(binary_mask) > 0:
                    instances.append({
                        "segmentation": [binary_mask.flatten().tolist()],
                        "category_id": class_id,
                        "iscrowd": 0,
                    })

            dataset_dicts.append({
                "file_name": image_path,
                "height": 256,  # Set appropriate height
                "width": 256,  # Set appropriate width
                "annotations": instances,
            })

    DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=[str(i) for i in range(1, 256)])