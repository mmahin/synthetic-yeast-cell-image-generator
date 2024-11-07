"""
train.py

Defines functions for training and evaluating the Mask R-CNN model on yeast cell data.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
def train_one_epoch(model, data_loader, optimizer, device):
    """
    Trains the Mask R-CNN model for one epoch.

    Args:
        model (torch.nn.Module): The Mask R-CNN model.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to use for training.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()

    return running_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """
    Evaluates the Mask R-CNN model by computing the mean Intersection over Union (IoU) for the masks.

    Args:
        model (torch.nn.Module): The Mask R-CNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average mean IoU for the evaluated images.
    """
    model.eval()  # Set the model to evaluation mode
    total_iou = 0
    num_samples = 0
    low_threshold = 0.1  # Lower threshold to check for predictions

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # Run inference

            # Process each image and corresponding prediction
            for i, output in enumerate(outputs):
                # Check if any masks are predicted
                if 'masks' in output and output['masks'].shape[0] > 0:
                    # Lower the threshold for initial evaluation
                    pred_mask = output['masks'][0, 0] > low_threshold
                    true_mask = targets[i]['masks'][0].to(device)

                    # Compute Intersection over Union (IoU)
                    intersection = (pred_mask & true_mask).float().sum((1, 2))
                    union = (pred_mask | true_mask).float().sum((1, 2))
                    iou = (intersection / union).mean().item()

                    total_iou += iou
                    num_samples += 1
                    print(f"Sample {i+1}: IoU = {iou:.4f}")  # Print IoU for each sample
                else:
                    print("No masks predicted for this image.")

    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    print(f"Average IoU: {avg_iou:.4f}")
    return avg_iou