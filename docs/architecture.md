# Synthetic Yeast Cell Image Generator Architecture

## Overview

The Synthetic Yeast Cell Image Generator is a Python application with a graphical user interface (GUI) for generating, saving, and labeling synthetic images of yeast cells. It also includes functionality for training a Mask R-CNN model on these images, evaluating the model, and visualizing predictions. The application is structured into distinct components to facilitate synthetic image generation, model training, and evaluation, each accessible through the user-friendly GUI.

## Project Structure

```
synthetic-yeast-cell-image-generator/
├── app/
│   ├── data/                            # Directory for generated images and masks
│   ├── generator/
│   │   ├── cell_generator.py            # Generates synthetic yeast images and masks
│   ├── maskrcnn/
│   │   ├── dataset.py                   # Dataset class for Mask R-CNN
│   │   ├── model.py                     # Defines the Mask R-CNN model
│   │   ├── train.py                     # Functions for training and evaluation
│   │   ├── visualize.py                 # Visualization functions for model predictions
│   ├── ui/
│   │   ├── main.py                      # Main single-file GUI application
└── README.md                            # Project documentation
```


## Components

### 1. **Synthetic Image Generation (`app/generator/cell_generator.py`)**

   - **Purpose**: Generates synthetic images of yeast cells with customizable parameters (e.g., cell radius, fluorescence intensity).
   - **Main Functions**:
     - `generate_synthetic_yeast_image`: Creates an image and mask for yeast cells by simulating realistic cell shapes and textures.
     - `save_synthetic_data`: Saves generated images and masks to specified paths.

### 2. **Mask R-CNN Model Components (`app/maskrcnn/`)**

   - **dataset.py**: Defines a `YeastCellDataset` class for loading synthetic images and masks, providing them to the Mask R-CNN model during training.
   - **model.py**: Configures the Mask R-CNN architecture with a customizable backbone for object detection and instance segmentation.
   - **train.py**: Contains functions for:
     - `train_one_epoch`: Trains the model for one epoch and calculates the loss.
     - `evaluate`: Evaluates the model's predictions, calculating Intersection over Union (IoU) metrics.
   - **visualize.py**: Provides functions to visualize predictions made by the Mask R-CNN model on test images.

### 3. **Graphical User Interface (`app/ui/main.py`)**

   - **Purpose**: The GUI provides users with an intuitive interface to configure parameters, generate synthetic images, and train/evaluate the Mask R-CNN model.
   - **Main Panels**:
     - **Preview Panel**: Allows users to set parameters for yeast cell generation (e.g., image size, cell count) and preview the synthetic image and mask.
     - **Generation Panel**: Enables batch generation of synthetic images based on the configured parameters, saving images and masks to a specified directory.
     - **Mask R-CNN Training and Testing Panel**: Facilitates model training by configuring directories and hyperparameters (learning rate, epochs). Users can train, evaluate (calculate IoU), and visualize predictions.

## How the Components Interact

1. **Synthetic Image Generation**: The user configures generation parameters in the **Preview** and **Generation** panels of the UI, using `generate_synthetic_yeast_image` to produce individual or batch synthetic yeast images with corresponding masks.

2. **Training and Evaluation**:
   - The user loads generated images and masks through the **Mask R-CNN Training and Testing Panel**. The **YeastCellDataset** class (in `dataset.py`) reads these images for training or testing.
   - The Mask R-CNN model, defined in `model.py`, processes batches of images from the dataset, updating weights during training or generating predictions during evaluation.
   - Evaluation results (IoU metrics) and predictions are displayed in the UI for model analysis.

## Future Improvements

- **Improving Yeast Cell Design**: Include more shapes and color variations for Yeast cell design.
- **Additional Augmentation Options**: Include more parameters for varied synthetic data generation, enhancing the diversity of the training dataset.
- **Enhanced Visualization**: Support additional visualization options, such as metrics tracking for model accuracy and loss curves during training.
- **Perfecting Deep Learning Model**: Deep learning evaluation, visualization modules need more time. Enabling interrrupt during the training process, saving and loading pretrained models need to be done. 
- **Hyperparameter Tuning**: Provide advanced options in the UI to allow users to fine-tune Mask R-CNN hyperparameters for improved model performance.
- **Improving Based on User Feedback**: The software, modules and UI need to be improved based on user feedback.

## Summary

This application combines synthetic image generation with deep learning training capabilities through Mask R-CNN. The modular architecture, consisting of generation, training, and UI components, supports experimentation and rapid model iteration in an accessible interface.
