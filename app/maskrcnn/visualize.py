"""
visualize.py

Script to visualize predictions from the trained Mask R-CNN model on yeast cell images.

Usage:
    python visualize.py
"""

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

def plot_predictions(model, dataset, device):
    model.eval()
    img, _ = dataset[0]
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    img = T.ToPILImage()(img)
    plt.imshow(img)
    for mask in prediction['masks']:
        plt.imshow(mask[0].cpu(), alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()