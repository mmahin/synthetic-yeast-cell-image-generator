import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from app.generator.cell_generator import generate_synthetic_yeast_image, save_synthetic_data
from app.maskrcnn.dataset import YeastCellDataset
from app.maskrcnn.model import get_model
from app.maskrcnn.train import train_one_epoch, evaluate
from app.maskrcnn.visualize import plot_predictions

class SyntheticImageGeneratorApp:
    """
    Main application class that integrates the preview, generation,
    and training panels for synthetic yeast cell image generation.

    This application is designed for creating synthetic yeast cell images,
    generating batches of images with masks, and training a Mask R-CNN model
    for object detection and segmentation. The user interface consists of
    three main sections:

    1. Preview Panel: Allows the user to configure and preview synthetic images.
    2. Generation Panel: Configures parameters for generating multiple synthetic images.
    3. Training Panel: Provides configuration for training the Mask R-CNN model
       with options to evaluate and visualize predictions.
    """
    def __init__(self, root):
        """
        Initializes the main application window and its components.

        """
        self.root = root
        self.root.title("Synthetic Yeast Cell Image Generator")

        # Default directories and parameters
        self.default_directory = r"\synthetic-yeast-cell-image-generator\data"
        self.default_images_dir = r"app/data/images"
        self.default_masks_dir = r"app/data/masks"

        # Synthetic image generation parameters
        self.image_width = tk.IntVar(value=256)
        self.image_height = tk.IntVar(value=256)
        self.cell_radius_min = tk.IntVar(value=10)
        self.cell_radius_max = tk.IntVar(value=20)
        self.min_cell_count = tk.IntVar(value=5)
        self.max_cell_count = tk.IntVar(value=15)
        self.fluorescence_level = tk.DoubleVar(value=1.0)
        self.num_images = tk.IntVar(value=1)
        self.save_directory = tk.StringVar(value=self.default_directory)

        # Model training parameters
        self.images_dir = tk.StringVar(value=self.default_images_dir)
        self.masks_dir = tk.StringVar(value=self.default_masks_dir)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.num_epochs = tk.IntVar(value=10)

        # Initialize widgets and layout
        self.create_widgets()

        # Thread control variables
        self.training_thread = None
        self.model = None

    def create_widgets(self):
        """
        Sets up the user interface elements of the preview panel, including
        input fields for configuration values and canvases for image previews.
        Sets up the user interface elements of the generation panel, including
        input fields for configuration values and buttons for image generation.
        Sets up the user interface elements of the training panel, including
        input fields for training configuration values, training controls,
        and evaluation/visualization buttons.
        """
        tk.Label(self.root, text="Synthetic Yeast Cell Image Generator", font=("Arial", 14)).pack(pady=10)

        # Main frame with upper and bottom sections separated by a line
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Upper half frame for preview and generation panels
        upper_frame = tk.Frame(main_frame)
        upper_frame.pack(fill="x", padx=20, pady=5)

        # Lower half frame for training panel
        lower_frame = tk.Frame(main_frame)
        lower_frame.pack(fill="x", padx=20, pady=5)

        # Divider line
        tk.Frame(main_frame, height=2, bd=1, relief="sunken").pack(fill="x", padx=10, pady=10)

        # Preview Panel
        self.create_preview_panel(upper_frame)

        # Generation Panel
        self.create_generation_panel(upper_frame)

        # Training Panel
        self.create_training_panel(lower_frame)

    def create_preview_panel(self, frame):
        """
        A tkinter Frame for configuring and previewing synthetic yeast cell images.

        Allows the user to set parameters for image width, height, cell radius range,
        cell count range, and fluorescence level. A preview button generates a synthetic
        image and mask with the specified parameters, displaying them within the panel.

        Attributes:
            image_width (tk.IntVar): Width of the synthetic image.
            image_height (tk.IntVar): Height of the synthetic image.
            cell_radius_min (tk.IntVar): Minimum radius for yeast cells.
            cell_radius_max (tk.IntVar): Maximum radius for yeast cells.
            min_cell_count (tk.IntVar): Minimum number of cells in the image.
            max_cell_count (tk.IntVar): Maximum number of cells in the image.
            fluorescence_level (tk.DoubleVar): Level of fluorescence for the cells.
            image_canvas (tk.Canvas): Canvas to display the generated image.
            mask_canvas (tk.Canvas): Canvas to display the generated mask.
        """
        preview_frame = tk.LabelFrame(frame, text="Preview Panel", font=("Arial", 12))
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(preview_frame, text="Image Width:").grid(row=0, column=0)
        tk.Entry(preview_frame, textvariable=self.image_width).grid(row=0, column=1)

        tk.Label(preview_frame, text="Image Height:").grid(row=1, column=0)
        tk.Entry(preview_frame, textvariable=self.image_height).grid(row=1, column=1)

        tk.Label(preview_frame, text="Min Cell Radius:").grid(row=2, column=0)
        tk.Entry(preview_frame, textvariable=self.cell_radius_min).grid(row=2, column=1)

        tk.Label(preview_frame, text="Max Cell Radius:").grid(row=3, column=0)
        tk.Entry(preview_frame, textvariable=self.cell_radius_max).grid(row=3, column=1)

        tk.Label(preview_frame, text="Min Cell Count:").grid(row=4, column=0)
        tk.Entry(preview_frame, textvariable=self.min_cell_count).grid(row=4, column=1)

        tk.Label(preview_frame, text="Max Cell Count:").grid(row=5, column=0)
        tk.Entry(preview_frame, textvariable=self.max_cell_count).grid(row=5, column=1)

        tk.Label(preview_frame, text="Fluorescence Level:").grid(row=6, column=0)
        tk.Entry(preview_frame, textvariable=self.fluorescence_level).grid(row=6, column=1)

        tk.Button(preview_frame, text="Generate Preview", command=self.generate_preview).grid(row=7, column=0, columnspan=2, pady=5)
        tk.Label(preview_frame, text="Image").grid(row=8, column=0)
        self.image_canvas = tk.Canvas(preview_frame, width=256, height=256)
        self.image_canvas.grid(row=9, column=0)
        tk.Label(preview_frame, text="Mask").grid(row=8, column=1)
        self.mask_canvas = tk.Canvas(preview_frame, width=256, height=256)
        self.mask_canvas.grid(row=9, column=1)

    def create_generation_panel(self, frame):
        """
        A tkinter Frame for configuring and generating synthetic yeast cell images.

        This panel allows users to set parameters for the number of images to generate
        and the save directory. It provides a button to generate synthetic yeast cell
        images based on configurations set in the Preview Panel.

        Attributes:
            num_images (tk.IntVar): Number of synthetic images to generate.
            save_directory (tk.StringVar): Directory where generated images and masks will be saved.
        """
        generation_frame = tk.LabelFrame(frame, text="Generation Panel", font=("Arial", 12))
        generation_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        tk.Label(generation_frame, text="Number of Images:").grid(row=0, column=0)
        tk.Entry(generation_frame, textvariable=self.num_images).grid(row=0, column=1)

        tk.Label(generation_frame, text="Save Directory:").grid(row=1, column=0)
        tk.Entry(generation_frame, textvariable=self.save_directory).grid(row=1, column=1)

        tk.Button(generation_frame, text="Select Directory", command=self.select_directory).grid(row=2, column=1, pady=5)
        tk.Button(generation_frame, text="Generate Images", command=self.generate_images).grid(row=3, column=0, columnspan=2, pady=5)

    def create_training_panel(self, frame):
        """
        A tkinter Frame for configuring and training a Mask R-CNN model on synthetic yeast cell images.

        This panel allows the user to specify training configurations, initiate training,
        and provides options to evaluate and visualize model performance.

        Attributes:
            images_dir (tk.StringVar): Directory containing training images.
            masks_dir (tk.StringVar): Directory containing mask images.
            learning_rate (tk.DoubleVar): Learning rate for model training.
            num_epochs (tk.IntVar): Number of epochs for training.
            model (torch.nn.Module): The Mask R-CNN model instance.
            training_thread (threading.Thread): Thread to handle training without blocking the UI.
        """
        # Mask RCNN Training and Testing Panel
        training_frame = tk.LabelFrame(frame, text="Mask RCNN Training and Testing Panel", font=("Arial", 12))
        training_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Training Inputs
        tk.Label(training_frame, text="Images Directory:").grid(row=1, column=0)
        tk.Entry(training_frame, textvariable=self.images_dir).grid(row=1, column=1)
        tk.Button(training_frame, text="Browse", command=self.browse_images).grid(row=1, column=2)

        tk.Label(training_frame, text="Masks Directory:").grid(row=2, column=0)
        tk.Entry(training_frame, textvariable=self.masks_dir).grid(row=2, column=1)
        tk.Button(training_frame, text="Browse", command=self.browse_masks).grid(row=2, column=2)

        tk.Label(training_frame, text="Learning Rate:").grid(row=3, column=0)
        tk.Entry(training_frame, textvariable=self.learning_rate).grid(row=3, column=1)

        tk.Label(training_frame, text="Number of Epochs:").grid(row=4, column=0)
        tk.Entry(training_frame, textvariable=self.num_epochs).grid(row=4, column=1)

        # Buttons for Training, Evaluation, and Visualization
        self.train_button = tk.Button(training_frame, text="Train Model", command=self.train_model)
        self.train_button.grid(row=5, column=0, pady=5)

        self.stop_train_button = tk.Button(training_frame, text="End Train", command=self.stop_training, state=tk.DISABLED)
        self.stop_train_button.grid(row=5, column=1, pady=5)

        self.eval_button = tk.Button(training_frame, text="Evaluate Model", command=self.evaluate_model, state=tk.DISABLED)
        self.eval_button.grid(row=6, column=0, pady=5)

        self.visualize_button = tk.Button(training_frame, text="Visualize Predictions", command=self.visualize_predictions, state=tk.DISABLED)
        self.visualize_button.grid(row=6, column=1, pady=5)

        # Message box on the right, spanning the size of two smaller boxes
        self.progress_text = tk.Text(training_frame, width=40, height=10)
        self.progress_text.grid(row=1, column=3, rowspan=5, sticky="nsew", padx=10)

    # Functions for image generation and model training/evaluation

    def generate_preview(self):
        """
        Generates a synthetic image and mask based on user-configured parameters.

        The function uses the `generate_synthetic_yeast_image` to create a synthetic
        image and mask with specified parameters and then updates the preview canvases
        with the generated image and mask.
        """
        width = self.image_width.get()
        height = self.image_height.get()
        cell_count = np.random.randint(self.min_cell_count.get(), self.max_cell_count.get() + 1)
        radius_range = (self.cell_radius_min.get(), self.cell_radius_max.get())
        fluorescence_level = self.fluorescence_level.get()

        # Generate preview image and mask
        image, mask = generate_synthetic_yeast_image(
            width=width,
            height=height,
            cell_count=cell_count,
            fluorescence_level=fluorescence_level,
            cell_radius_range=radius_range
        )
        self.update_preview(image, mask)

    def update_preview(self, image, mask):
        """
        Updates the preview canvases with the generated synthetic image and mask.

        Args:
            image (np.ndarray): The generated synthetic image.
            mask (np.ndarray): The generated mask for the synthetic image.
        """
        # Process and display image
        image_rgb = cv2.cvtColor((image / 256).astype(np.uint8), cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail((256, 256))
        self.image_photo = ImageTk.PhotoImage(pil_image)
        self.image_canvas.create_image(128, 128, image=self.image_photo)

        # Process and display mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if len(mask.shape) == 3 else cv2.cvtColor(mask,
                                                                                                   cv2.COLOR_GRAY2RGB)
        pil_mask = Image.fromarray(mask_rgb)
        pil_mask.thumbnail((256, 256))
        self.mask_photo = ImageTk.PhotoImage(pil_mask)
        self.mask_canvas.create_image(128, 128, image=self.mask_photo)

    def select_directory(self):
        """
        Opens a file dialog for the user to select a directory to save generated images and masks.
        """
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.save_directory.set(selected_dir)

    def generate_images(self):
        """
        Generates synthetic images and masks based on configurations in the preview panel and
        saves them to the specified directory.

        Uses the parameters set in the PreviewPanel to control image properties such as width,
        height, cell count, fluorescence level, and cell radius range.
        """
        width, height = self.image_width.get(), self.image_height.get()
        radius_range = (self.cell_radius_min.get(), self.cell_radius_max.get())
        num_images = self.num_images.get()
        save_path = self.save_directory.get() or self.default_directory

        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)

        for i in range(num_images):
            cell_count = np.random.randint(self.min_cell_count.get(), self.max_cell_count.get() + 1)
            fluorescence_level = self.fluorescence_level.get()
            image, mask = generate_synthetic_yeast_image(
                width=width,
                height=height,
                cell_count=cell_count,
                fluorescence_level=fluorescence_level,
                cell_radius_range=radius_range
            )
            save_synthetic_data(image, mask,
                                os.path.join(save_path, f"images/synthetic_image_{i + 1}.png"),
                                os.path.join(save_path, f"masks/synthetic_mask_{i + 1}.png"))

        messagebox.showinfo("Generation Complete", f"{num_images} images generated and saved to {save_path}")

    def train_model(self):
        """
        Conducts the training process for the Mask R-CNN model over multiple epochs.
        Tracks training loss per epoch and logs the progress in the text box.
        """
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
        """
        Evaluates the model by calculating the Intersection over Union (IoU) score.
        Logs the IoU score in the text box and displays it in a message box.
        """
        data_loader = DataLoader(YeastCellDataset(self.images_dir.get(), self.masks_dir.get()),
                                 batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        iou_score = evaluate(self.model, data_loader, device)
        messagebox.showinfo("Evaluation", f"IoU Score: {iou_score:.2f}")

    def visualize_predictions(self):
        """
        Visualizes model predictions by plotting the images and masks on a separate panel.
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        plot_predictions(self.model, YeastCellDataset(self.images_dir.get(), self.masks_dir.get()), device)

    def stop_training(self):
        """Stops the training process and enables evaluation and visualization of the current model."""
        self.stop_event.set()


    def browse_images(self):
        """
        Opens a file dialog for selecting the directory containing training images.
        """
        directory = filedialog.askdirectory(initialdir=self.default_images_dir, title="Select Images Directory")
        if directory:
            self.images_dir.set(directory)

    def browse_masks(self):
        """
        Opens a file dialog for selecting the directory containing mask images.
        """
        directory = filedialog.askdirectory(initialdir=self.default_masks_dir, title="Select Masks Directory")
        if directory:
            self.masks_dir.set(directory)

# Running the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticImageGeneratorApp(root)
    root.mainloop()
