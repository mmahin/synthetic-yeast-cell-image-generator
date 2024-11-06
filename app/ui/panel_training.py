import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from app.maskrcnn.dataset import YeastCellDataset
from app.maskrcnn.model import get_model
from app.maskrcnn.train import train_one_epoch
from app.maskrcnn.train import evaluate
from app.maskrcnn.visualize import plot_predictions
import threading


class TrainingPanel:
    def __init__(self, root):
        self.root = root
        self.images_dir = tk.StringVar()
        self.masks_dir = tk.StringVar()
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.num_epochs = tk.IntVar(value=10)

    def create_widgets(self, row_start, column_start):
        # Panel Title
        tk.Label(self.root, text="Training Panel", font=('Arial', 14, 'bold')).grid(row=row_start, column=column_start, columnspan=2)

        # Image Directory and Mask Directory inputs
        tk.Label(self.root, text="Images Directory:").grid(row=row_start+1, column=column_start)
        tk.Entry(self.root, textvariable=self.images_dir).grid(row=row_start+1, column=column_start+1)
        tk.Button(self.root, text="Browse", command=self.browse_images).grid(row=row_start+1, column=column_start+2)

        tk.Label(self.root, text="Masks Directory:").grid(row=row_start+2, column=column_start)
        tk.Entry(self.root, textvariable=self.masks_dir).grid(row=row_start+2, column=column_start+1)
        tk.Button(self.root, text="Browse", command=self.browse_masks).grid(row=row_start+2, column=column_start+2)

        tk.Label(self.root, text="Learning Rate:").grid(row=row_start+3, column=column_start)
        tk.Entry(self.root, textvariable=self.learning_rate).grid(row=row_start+3, column=column_start+1)

        tk.Label(self.root, text="Number of Epochs:").grid(row=row_start+4, column=column_start)
        tk.Entry(self.root, textvariable=self.num_epochs).grid(row=row_start+4, column=column_start+1)

        tk.Button(self.root, text="Start Training", command=self.start_training).grid(row=row_start+5, column=column_start, columnspan=3)

    def browse_images(self):
        folder_selected = filedialog.askdirectory()
        self.images_dir.set(folder_selected)

    def browse_masks(self):
        folder_selected = filedialog.askdirectory()
        self.masks_dir.set(folder_selected)

    def start_training(self):
        thread = threading.Thread(target=self.train_model)
        thread.start()

    def train_model(self):
        # Initialize dataset, model, optimizer, and loss function here
        dataset = YeastCellDataset(self.images_dir.get(), self.masks_dir.get(), transforms=None)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = get_model()
        optimizer = optim.Adam(params=model.parameters(), lr=self.learning_rate.get())

        # Training loop
        for epoch in range(self.num_epochs.get()):
            train_one_epoch(model, optimizer, dataloader)
            evaluate(model, dataloader)

            # Optionally, display the plot here
            plot_predictions(model, dataloader)

        messagebox.showinfo("Training Complete", "Model training is complete.")
