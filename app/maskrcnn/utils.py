"""
utils.py

Utility functions for visualizing predictions from the Mask R-CNN model.
"""

import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

def plot_predictions(model, dataset, device):
    """
    Plots predictions from the model on a sample image.

    Args:
        model (torch.nn.Module): The Mask R-CNN model.
        dataset (YeastCellDataset): Dataset object for accessing images.
        device (torch.device): Device to use for prediction.
    """
    model.eval()
    img, _ = dataset[0]
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    img = T.ToPILImage()(img)
    plt.imshow(img)

    # Overlay masks on the image
    for mask in prediction['masks']:
        plt.imshow(mask[0].cpu(), alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()