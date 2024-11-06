"""
dataset.py

Defines the custom dataset class for yeast cell images and masks, with preprocessing for Mask R-CNN.
"""

import torch
from torch.utils.data import Dataset
import cv2
import os

class YeastCellDataset(Dataset):
    """
    Dataset for loading yeast cell images and corresponding binary masks.

    Args:
        images_dir (str): Directory path for input images.
        masks_dir (str): Directory path for mask images.
        transforms (callable, optional): Optional transform to be applied on a sample.

    Returns:
        Tuple of processed image tensor and target dictionary with 'boxes', 'labels', and 'masks'.
    """

    def __init__(self, images_dir, masks_dir=None, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir)) if masks_dir else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        target = None
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
            binary_mask = self.process_mask(cv2.imread(mask_path))
            target = self.create_target(binary_mask)

        # Convert image to tensor format expected by the model
        return torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1), target

    def process_mask(self, mask):
        """
        Processes the mask by creating a binary version, where 0 is background and 1 is cell.

        Args:
            mask (np.array): Original mask image.

        Returns:
            np.array: Binary mask.
        """
        binary_mask = cv2.inRange(mask, (255, 0, 0), (255, 0, 0))
        binary_mask[binary_mask == 255] = 0
        binary_mask[binary_mask != 0] = 1
        return binary_mask

    def create_target(self, binary_mask):
        """
        Creates target dictionary with bounding boxes, labels, and masks for Mask R-CNN.

        Args:
            binary_mask (np.array): Binary mask of the cells.

        Returns:
            dict: Target dictionary with 'boxes', 'labels', and 'masks'.
        """
        num_objs, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        boxes = [[0, 0, 1, 1]] if num_objs == 1 else [[x, y, x+w, y+h] for _, x, y, w, h in stats[1:]]

        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((max(num_objs - 1, 1),), dtype=torch.int64),
            "masks": torch.as_tensor(binary_mask, dtype=torch.uint8).unsqueeze(0),
        }