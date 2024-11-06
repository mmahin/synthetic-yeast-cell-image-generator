import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from app.generator.cell_generator import generate_synthetic_yeast_image


class PreviewPanel:
    def __init__(self, root):
        self.root = root
        self.image_width = tk.IntVar(value=256)
        self.image_height = tk.IntVar(value=256)
        self.cell_radius_min = tk.IntVar(value=10)
        self.cell_radius_max = tk.IntVar(value=25)

    def create_widgets(self, row_start, column_start):
        # Panel Title
        tk.Label(self.root, text="Preview Panel", font=('Arial', 14, 'bold')).grid(row=row_start, column=column_start, columnspan=2)

        # Image Preview Parameters
        tk.Label(self.root, text="Image Width:").grid(row=row_start+1, column=column_start)
        tk.Entry(self.root, textvariable=self.image_width).grid(row=row_start+1, column=column_start+1)

        tk.Label(self.root, text="Image Height:").grid(row=row_start+2, column=column_start)
        tk.Entry(self.root, textvariable=self.image_height).grid(row=row_start+2, column=column_start+1)

        tk.Label(self.root, text="Min Cell Radius:").grid(row=row_start+3, column=column_start)
        tk.Entry(self.root, textvariable=self.cell_radius_min).grid(row=row_start+3, column=column_start+1)

        tk.Label(self.root, text="Max Cell Radius:").grid(row=row_start+4, column=column_start)
        tk.Entry(self.root, textvariable=self.cell_radius_max).grid(row=row_start+4, column=column_start+1)

        tk.Button(self.root, text="Generate Preview", command=self.generate_preview).grid(row=row_start+5, column=column_start, columnspan=2)

        # Image and mask preview labels
        tk.Label(self.root, text="Image Preview").grid(row=row_start+6, column=column_start)
        tk.Label(self.root, text="Mask Preview").grid(row=row_start+6, column=column_start+1)

        # Canvas for previewing image and mask
        self.image_canvas = tk.Canvas(self.root, width=256, height=256)
        self.image_canvas.grid(row=row_start+7, column=column_start)

        self.mask_canvas = tk.Canvas(self.root, width=256, height=256)
        self.mask_canvas.grid(row=row_start+7, column=column_start+1)

    def generate_preview(self):
        width = self.image_width.get()
        height = self.image_height.get()
        count = np.random.randint(self.cell_radius_min.get(), self.cell_radius_max.get())
        radius_range = (self.cell_radius_min.get(), self.cell_radius_max.get())

        # Generate preview image and mask
        image, mask = generate_synthetic_yeast_image(width=width, height=height, cell_count=count, cell_radius_range=radius_range)
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