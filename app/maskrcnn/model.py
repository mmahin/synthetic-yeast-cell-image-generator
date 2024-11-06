"""
model.py

Defines the Mask R-CNN model with a ResNet-50 FPN backbone for segmentation tasks.
"""

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_model(num_classes):
    """
    Initializes a Mask R-CNN model with a ResNet-50 backbone.

    Args:
        num_classes (int): Number of classes for object detection.

    Returns:
        MaskRCNN: Mask R-CNN model.
    """
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    return MaskRCNN(backbone, num_classes=num_classes)