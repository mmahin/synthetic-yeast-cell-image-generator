# Synthetic Yeast Cell Image Generator

This project is a GUI-based application for generating synthetic yeast cell images with masks, simulating fluorescence microscopy images. The application provides tools for synthetic image generation, real-time previewing, and training/testing a Mask R-CNN model on the generated dataset.

## Features

- **Unified Interface**: A single, easy-to-use interface for image preview, generation, training, evaluation, and visualization.
- **Preview Panel**: Real-time preview of synthetic images with customizable parameters.
- **Generation Panel**: Generate and save batches of synthetic images and masks.
- **Mask R-CNN Training and Testing Panel**: Train a Mask R-CNN model, evaluate its performance, and visualize results on generated data.

## Installation

### Prerequisites
- Python 3.8 or later
- Install required packages:

```bash
pip install -r requirements.txt
```
## Required Modules

- **torch** and **torchvision**: Used for deep learning and Mask R-CNN model training.
- **numpy**: For numerical operations.
- **opencv-python**: Handles image processing tasks.
- **Pillow**: Used for image manipulation.
- **tkinter**: Provides the GUI interface for the application.

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
## Usage

1. **Run the GUI Application**
   
   Start the application by running the following command:

   ```bash
   python app/ui/main.py

2. **Using the Unified Interface**

- **Preview Panel:**
    - Adjust parameters such as image dimensions, cell radius, fluorescence level, and cell count.
    - Click **Generate Preview** to view a sample synthetic image and its mask.

- **Generation Panel:**
    - Specify the number of images to generate in a batch and choose a save directory.
    - Click **Generate Images** to save synthetic images and masks based on the specified parameters.

- **Mask R-CNN Training and Testing Panel:**
    - Set directories for training images and masks.
    - Adjust training parameters (learning rate, number of epochs).
    - Click **Train Model** to start training.
    - Use **Evaluate Model** to calculate Intersection over Union (IoU) on test images, and **Visualize Predictions** to view the model’s predictions.
## Code Samples

### Generating a Batch of Synthetic Images

The following Python code generates a batch of synthetic yeast cell images and corresponding masks, then saves each pair in specified directories.

### Code

```python
from app.generator.cell_generator import generate_synthetic_yeast_image, save_synthetic_data

def generate_batch(batch_size=10, width=256, height=256, cell_count=15, cell_radius_range=(10, 25)):
    """
    Generates and saves a batch of synthetic yeast cell images and masks.

    Parameters:
        batch_size (int): Number of images to generate in the batch.
        width (int): Width of each generated image.
        height (int): Height of each generated image.
        cell_count (int): Number of yeast cells per image.
        cell_radius_range (tuple): Minimum and maximum radius for yeast cells.
        
    Each generated image and mask is saved in the specified directories.
    """
    for i in range(batch_size):
        # Generate a synthetic yeast cell image and mask
        image, mask = generate_synthetic_yeast_image(
            width=width,
            height=height,
            cell_count=cell_count,
            cell_radius_range=cell_radius_range
        )
        
        # Save the generated image and mask
        save_synthetic_data(
            image, 
            mask, 
            f"app/data/images/synthetic_image_{i + 1}.png", 
            f"app/data/masks/synthetic_mask_{i + 1}.png"
        )
```
## Training the Mask R-CNN Model

This Python code sets up the Mask R-CNN model to train on a dataset of synthetic yeast cell images and masks.

### Code

```python
import torch
from torch.utils.data import DataLoader
from app.maskrcnn.dataset import YeastCellDataset
from app.maskrcnn.model import get_model
from app.maskrcnn.train import train_one_epoch

# Set up dataset and dataloader
dataset = YeastCellDataset("app/data/images", "app/data/masks")
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    train_loss = train_one_epoch(model, data_loader, optimizer, device)
    print(f"Epoch {epoch + 1}, Loss: {train_loss}")
```
# Future Improvements
- **Improving Yeast Cell Design**: Include more shapes and color variations for Yeast cell design.
- **Additional Augmentation Options**: Include more parameters for varied synthetic data generation, enhancing the diversity of the training dataset.
- **Enhanced Visualization**: Support additional visualization options, such as metrics tracking for model accuracy and loss curves during training.
- **Perfecting Deep Learning Model**: Deep learning evaluation, visualization modules need more time. Enabling interrrupt during the training process, saving and loading pretrained models need to be done. 
- **Hyperparameter Tuning**: Provide advanced options in the UI to allow users to fine-tune Mask R-CNN hyperparameters for improved model performance.
- **Improving Based on User Feedback**: The software, modules and UI need to be improved based on user feedback.
## License

This project is licensed under the [MIT License](LICENSE).

## Authors

This project was developed by Md Mahin.

---
```
This README file provides an organized and comprehensive overview of your application, including usage instructions and code samples to guide users through generating synthetic images, training Mask R-CNN, and evaluating model performance.

```
