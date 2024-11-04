import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from app.generator.cell_generator import generate_synthetic_yeast_image, save_synthetic_data

class SyntheticImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Yeast Cell Image Generator")

        # Parameters for the synthetic image generation
        self.image_width = tk.IntVar(value=256)
        self.image_height = tk.IntVar(value=256)
        self.cell_count = tk.IntVar(value=15)
        self.cell_radius_min = tk.IntVar(value=10)
        self.cell_radius_max = tk.IntVar(value=25)

        self.create_widgets()

    def create_widgets(self):
        # Create UI elements for parameters
        tk.Label(self.root, text="Image Width:").grid(row=0, column=0)
        tk.Entry(self.root, textvariable=self.image_width).grid(row=0, column=1)

        tk.Label(self.root, text="Image Height:").grid(row=1, column=0)
        tk.Entry(self.root, textvariable=self.image_height).grid(row=1, column=1)

        tk.Label(self.root, text="Number of Cells:").grid(row=2, column=0)
        tk.Entry(self.root, textvariable=self.cell_count).grid(row=2, column=1)

        tk.Label(self.root, text="Min Cell Radius:").grid(row=3, column=0)
        tk.Entry(self.root, textvariable=self.cell_radius_min).grid(row=3, column=1)

        tk.Label(self.root, text="Max Cell Radius:").grid(row=4, column=0)
        tk.Entry(self.root, textvariable=self.cell_radius_max).grid(row=4, column=1)

        tk.Button(self.root, text="Generate Image", command=self.generate_image).grid(row=5, columnspan=2)

        # Labels for image and mask previews
        tk.Label(self.root, text="Generated Image Preview").grid(row=6, column=0, columnspan=2)
        tk.Label(self.root, text="Generated Mask Preview").grid(row=6, column=2, columnspan=2)

        # Canvas for previewing the generated image
        self.image_canvas = tk.Canvas(self.root, width=256, height=256)
        self.image_canvas.grid(row=7, column=0, columnspan=2)

        # Canvas for previewing the mask
        self.mask_canvas = tk.Canvas(self.root, width=256, height=256)
        self.mask_canvas.grid(row=7, column=2, columnspan=2)

    def generate_image(self):
        try:
            # Get parameter values from the UI
            width = self.image_width.get()
            height = self.image_height.get()
            count = self.cell_count.get()
            radius_range = (self.cell_radius_min.get(), self.cell_radius_max.get())

            # Generate synthetic image and mask (OpenCV format)
            image, mask = generate_synthetic_yeast_image(
                width=width, height=height,
                cell_count=count,
                cell_radius_range=radius_range
            )



            # Scale image from uint16 to uint8 for display
            image_scaled = (image / 256).astype(np.uint8)  # Dividing by 256 scales 0–65535 to 0–255
            # Convert OpenCV image to RGB for display with PIL
            image_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_image.thumbnail((256, 256), Image.ANTIALIAS)
            self.image_photo = ImageTk.PhotoImage(pil_image)

            # Update image canvas with the new image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(128, 128, image=self.image_photo)

            # Check mask shape and handle accordingly
            if len(mask.shape) == 2:  # Single channel mask
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            elif len(mask.shape) == 3:  # Already an RGB mask
                mask_rgb = mask  # No need for conversion
            else:
                raise ValueError("Invalid mask shape. Expected 2D or 3D array.")
            pil_mask = Image.fromarray(mask_rgb)
            pil_mask.thumbnail((256, 256), Image.ANTIALIAS)
            self.mask_photo = ImageTk.PhotoImage(pil_mask)
            # Update mask canvas with the new mask
            self.mask_canvas.delete("all")
            self.mask_canvas.create_image(128, 128, image=self.mask_photo)
            # Optionally save the generated images
            save_synthetic_data(image, mask)

        except Exception as e:
            print(str(e))
            ...

if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticImageGeneratorApp(root)
    root.mainloop()