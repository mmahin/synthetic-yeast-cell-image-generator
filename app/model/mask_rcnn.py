import os
import numpy as np
import cv2
from mrcnn import utils
from mrcnn import model as MaskRCNN
from mrcnn import visualize
import matplotlib.pyplot as plt

# Import your generated dataset loader
from app.generator.cell_generator import load_generated_data


# Configuration class for Mask R-CNN
class InferenceConfig(MaskRCNN.Config):
    NAME = "yeast_cells"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + 1 (yeast cells)
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


def train_model():
    # Load your generated dataset
    dataset_train, dataset_val = load_generated_data()

    # Create model in training mode
    model = MaskRCNN.MaskRCNN(mode="training", config=InferenceConfig(), model_dir=os.getcwd())

    # Load weights for the COCO model or any pre-trained model
    model.load_weights("mask_rcnn_coco.h5", by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Train the model
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')


def visualize_predictions(model, image):
    results = model.detect([image], verbose=1)
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset_train.class_names, r['scores'])


if __name__ == "__main__":
    train_model()