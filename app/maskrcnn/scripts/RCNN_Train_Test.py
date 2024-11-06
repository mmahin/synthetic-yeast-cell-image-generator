"""
main.py

Script to train and evaluate the Mask R-CNN model on yeast cell images.

Usage:
    python main.py
"""

import torch
from torch.utils.data import DataLoader
from dataset import YeastCellDataset
from model import get_model
from train import train_one_epoch, evaluate
import torch.optim as optim
from visualize import plot_predictions
# Define directories for images and masks, and device configuration
images_dir = '/app/data/output/images'
masks_dir = '/app/data/output/masks/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize dataset, data loader, and model
dataset = YeastCellDataset(images_dir, masks_dir)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Initialize Mask R-CNN model with 2 classes (background + 1 object class)
model = get_model(num_classes=2)
model.to(device)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, data_loader, optimizer, device)
    print(f"Epoch {epoch + 1}, Loss: {train_loss}")

# Evaluation and visualization
evaluate(model, data_loader, device)
plot_predictions(model, dataset, device)