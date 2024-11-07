import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import cv2
import os


class YeastCellDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        binary_mask = self.process_mask(mask)

        # Find connected components for bounding boxes
        num_objs, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        boxes = []
        if num_objs == 1:  # Only background is present
            boxes = [[0, 0, 1, 1]]  # Dummy box
        else:
            for i in range(1, num_objs):  # Exclude background
                x, y, w, h = stats[i]
                boxes.append([x, y, x + w, y + h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((max(num_objs - 1, 1),), dtype=torch.int64)  # At least one label
        masks = torch.as_tensor(binary_mask, dtype=torch.uint8).unsqueeze(0)  # (1, H, W)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        image = F.to_tensor(image)

        return image, target

    def process_mask(self, mask):
        binary_mask = cv2.inRange(mask, (255, 0, 0), (255, 0, 0))
        binary_mask[binary_mask == 255] = 0  # Background to 0
        binary_mask[binary_mask != 0] = 1  # Object regions to 1
        return binary_mask

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model


import torch
import torch.optim as optim


def train_one_epoch(model, data_loader, optimizer, device):
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
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # Process outputs if needed (e.g., to calculate metrics)

# Define directories and parameters
images_dir = '/app/data/output/images/'
masks_dir = '/app/data/output/masks/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize dataset, data loader, and model
dataset = YeastCellDataset(images_dir, masks_dir)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = get_model(num_classes=2)  # Background + 1 object class
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, data_loader, optimizer, device)
    print(f"Epoch {epoch + 1}, Loss: {train_loss}")

import matplotlib.pyplot as plt
import torchvision.transforms as T

def plot_predictions(model, dataset, device):
    model.eval()
    img, _ = dataset[0]
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    img = T.ToPILImage()(img)
    plt.imshow(img)

    # Overlay masks
    for mask in prediction['masks']:
        plt.imshow(mask[0].cpu(), alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()

# Visualize predictions
plot_predictions(model, dataset, device)