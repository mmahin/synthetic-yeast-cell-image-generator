import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from app.generator.cell_generator import generate_synthetic_yeast_image, save_synthetic_data

class SyntheticImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Yeast Cell Image Generator")

        # Default directory
        self.default_directory = r"\synthetic-yeast-cell-image-generator\data"

        # Parameters for synthetic image generation
        self.image_width = tk.IntVar(value=256)
        self.image_height = tk.IntVar(value=256)
        self.cell_radius_min = tk.IntVar(value=10)
        self.cell_radius_max = tk.IntVar(value=20)
        self.min_cell_count = tk.IntVar(value=5)
        self.max_cell_count = tk.IntVar(value=15)
        self.fluorescence_level = tk.DoubleVar(value=1.0)
        self.num_images = tk.IntVar(value=1)
        self.save_directory = tk.StringVar(value=self.default_directory)

        self.create_widgets()

    def create_widgets(self):
        # Unified Panel
        tk.Label(self.root, text="Synthetic Yeast Cell Image Generator", font=("Arial", 14)).grid(row=0, column=0, columnspan=4, pady=10)

        # Preview Panel
        tk.Label(self.root, text="Preview Panel", font=("Arial", 12)).grid(row=1, column=0, columnspan=2)
        self.create_preview_panel()

        # Generation Panel
        tk.Label(self.root, text="Generation Panel", font=("Arial", 12)).grid(row=1, column=2, columnspan=2)
        self.create_generation_panel()

    def create_preview_panel(self):
        # Image dimension inputs
        tk.Label(self.root, text="Image Width:").grid(row=2, column=0)
        tk.Entry(self.root, textvariable=self.image_width).grid(row=2, column=1)

        tk.Label(self.root, text="Image Height:").grid(row=3, column=0)
        tk.Entry(self.root, textvariable=self.image_height).grid(row=3, column=1)

        # Cell radius range
        tk.Label(self.root, text="Min Cell Radius:").grid(row=4, column=0)
        tk.Entry(self.root, textvariable=self.cell_radius_min).grid(row=4, column=1)

        tk.Label(self.root, text="Max Cell Radius:").grid(row=5, column=0)
        tk.Entry(self.root, textvariable=self.cell_radius_max).grid(row=5, column=1)

        # Cell count range and fluorescence level
        tk.Label(self.root, text="Min Cell Count:").grid(row=6, column=0)
        tk.Entry(self.root, textvariable=self.min_cell_count).grid(row=6, column=1)

        tk.Label(self.root, text="Max Cell Count:").grid(row=7, column=0)
        tk.Entry(self.root, textvariable=self.max_cell_count).grid(row=7, column=1)

        tk.Label(self.root, text="Fluorescence Level:").grid(row=8, column=0)
        tk.Entry(self.root, textvariable=self.fluorescence_level).grid(row=8, column=1)

        # Preview generation button
        tk.Button(self.root, text="Generate Preview", command=self.generate_preview).grid(row=9, column=0, columnspan=2)

        # Image and mask preview labels
        tk.Label(self.root, text="Image Preview").grid(row=10, column=0)
        tk.Label(self.root, text="Mask Preview").grid(row=10, column=1)

        # Canvas for previewing image and mask
        self.image_canvas = tk.Canvas(self.root, width=256, height=256)
        self.image_canvas.grid(row=11, column=0)

        self.mask_canvas = tk.Canvas(self.root, width=256, height=256)
        self.mask_canvas.grid(row=11, column=1)

    def create_generation_panel(self):
        # Number of images and save directory
        tk.Label(self.root, text="Number of Images:").grid(row=2, column=2)
        tk.Entry(self.root, textvariable=self.num_images).grid(row=2, column=3)

        tk.Label(self.root, text="Save Directory:").grid(row=3, column=2)
        tk.Entry(self.root, textvariable=self.save_directory).grid(row=3, column=3)

        tk.Button(self.root, text="Select Directory", command=self.select_directory).grid(row=4, column=3)
        tk.Button(self.root, text="Generate Images", command=self.generate_images).grid(row=5, column=2, columnspan=2)

    def generate_preview(self):
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
        # Process and display image
        image_rgb = cv2.cvtColor((image / 256).astype(np.uint8), cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail((256, 256))
        self.image_photo = ImageTk.PhotoImage(pil_image)
        self.image_canvas.create_image(128, 128, image=self.image_photo)

        # Process and display mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if len(mask.shape) == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        pil_mask = Image.fromarray(mask_rgb)
        pil_mask.thumbnail((256, 256))
        self.mask_photo = ImageTk.PhotoImage(pil_mask)
        self.mask_canvas.create_image(128, 128, image=self.mask_photo)

    def select_directory(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.save_directory.set(selected_dir)

    def generate_images(self):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticImageGeneratorApp(root)
    root.mainloop()
