import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from app.maskrcnn.dataset import YeastCellDataset
from app.maskrcnn.model import get_model
from app.maskrcnn.train import train_one_epoch
from app.maskrcnn.train import evaluate
from app.maskrcnn.visualize import plot_predictions
import threading

class SyntheticImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Yeast Cell Image Generator")

        # Default directories and parameters
        self.default_images_dir = r"app/data/images"
        self.default_masks_dir = r"app/data/masks"
        self.images_dir = tk.StringVar(value=self.default_images_dir)
        self.masks_dir = tk.StringVar(value=self.default_masks_dir)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.num_epochs = tk.IntVar(value=10)

        self.create_widgets()

    def create_widgets(self):
        # Training panel
        tk.Label(self.root, text="Training Panel").grid(row=0, column=4, columnspan=2)
        self.create_training_panel()

    def create_training_panel(self):
        # Model Training Configuration
        tk.Label(self.root, text="Images Directory:").grid(row=1, column=4)
        tk.Entry(self.root, textvariable=self.images_dir).grid(row=1, column=5)
        tk.Button(self.root, text="Browse", command=self.browse_images).grid(row=1, column=6)

        tk.Label(self.root, text="Masks Directory:").grid(row=2, column=4)
        tk.Entry(self.root, textvariable=self.masks_dir).grid(row=2, column=5)
        tk.Button(self.root, text="Browse", command=self.browse_masks).grid(row=2, column=6)

        tk.Label(self.root, text="Learning Rate:").grid(row=3, column=4)
        tk.Entry(self.root, textvariable=self.learning_rate).grid(row=3, column=5)

        tk.Label(self.root, text="Number of Epochs:").grid(row=4, column=4)
        tk.Entry(self.root, textvariable=self.num_epochs).grid(row=4, column=5)

        # Start training button
        tk.Button(self.root, text="Train Model", command=self.train_model).grid(row=5, column=4, columnspan=2)

        # Progress monitor for training output
        self.progress_text = tk.Text(self.root, width=40, height=10)
        self.progress_text.grid(row=6, column=4, columnspan=2)

        # Evaluation and Visualization buttons
        self.eval_button = tk.Button(self.root, text="Evaluate Model", command=self.evaluate_model, state=tk.DISABLED)
        self.eval_button.grid(row=7, column=4)

        self.visualize_button = tk.Button(self.root, text="Visualize Predictions", command=self.visualize_predictions,
                                          state=tk.DISABLED)
        self.visualize_button.grid(row=7, column=5)

    def browse_images(self):
        directory = filedialog.askdirectory(initialdir=self.default_images_dir, title="Select Images Directory")
        if directory:
            self.images_dir.set(directory)

    def browse_masks(self):
        directory = filedialog.askdirectory(initialdir=self.default_masks_dir, title="Select Masks Directory")
        if directory:
            self.masks_dir.set(directory)

    def train_model(self):
        images_dir = self.images_dir.get()
        masks_dir = self.masks_dir.get()
        lr = self.learning_rate.get()
        num_epochs = self.num_epochs.get()

        # Check if directories are valid
        if not images_dir or not masks_dir:
            messagebox.showerror("Error", "Please select valid directories for images and masks.")
            return

        # Start a new thread for training
        training_thread = threading.Thread(target=self._train_model, args=(images_dir, masks_dir, lr, num_epochs))
        training_thread.start()

    def _train_model(self, images_dir, masks_dir, lr, num_epochs):
        # Initialize device, dataset, dataloader, model, and optimizer
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = YeastCellDataset(images_dir, masks_dir)
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        model = get_model(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        self.progress_text.insert(tk.END, "Starting training...\n")
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, data_loader, optimizer, device)
            self.progress_text.insert(tk.END, f"Epoch {epoch + 1}, Loss: {train_loss}\n")
            self.progress_text.see(tk.END)
            self.root.update_idletasks()  # This updates the UI during training

        # Enable evaluation and visualization buttons after training
        self.eval_button.config(state=tk.NORMAL)
        self.visualize_button.config(state=tk.NORMAL)
        self.model = model  # Save model for later use
        messagebox.showinfo("Training Complete", "Model training completed.")

    def evaluate_model(self):
        data_loader = DataLoader(YeastCellDataset(self.images_dir.get(), self.masks_dir.get()),
                                 batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        iou_score = evaluate(self.model, data_loader, device)
        messagebox.showinfo("Evaluation", f"IoU Score: {iou_score:.2f}")

    def visualize_predictions(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        plot_predictions(self.model, YeastCellDataset(self.images_dir.get(), self.masks_dir.get()), device)


# Running the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticImageGeneratorApp(root)
    root.mainloop()
