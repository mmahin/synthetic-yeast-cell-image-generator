import tkinter as tk
from tkinter import filedialog
import os
from app.generator.cell_generator import save_synthetic_data
import numpy as np


class GenerationPanel:
    def __init__(self, root):
        self.root = root
        self.num_images = tk.IntVar(value=1)
        self.save_directory = tk.StringVar(value="")

    def create_widgets(self, row_start, column_start):
        # Panel Title
        tk.Label(self.root, text="Generation Panel", font=('Arial', 14, 'bold')).grid(row=row_start, column=column_start, columnspan=2)

        tk.Label(self.root, text="Number of Images:").grid(row=row_start+1, column=column_start)
        tk.Entry(self.root, textvariable=self.num_images).grid(row=row_start+1, column=column_start+1)

        tk.Label(self.root, text="Save Directory:").grid(row=row_start+2, column=column_start)
        tk.Entry(self.root, textvariable=self.save_directory).grid(row=row_start+2, column=column_start+1)

        tk.Button(self.root, text="Select Directory", command=self.select_directory).grid(row=row_start+3, column=column_start+1)
        tk.Button(self.root, text="Generate Images", command=self.generate_images).grid(row=row_start+4, column=column_start, columnspan=2)

    def select_directory(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.save_directory.set(selected_dir)

    def generate_images(self):
        num_images = self.num_images.get()
        save_path = self.save_directory.get() or "default_directory"

        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)

        for i in range(num_images):
            count = np.random.randint(10, 25)
            image, mask = generate_synthetic_yeast_image(width=256, height=256, cell_count=count, cell_radius_range=(10, 25))
            save_synthetic_data(image, mask, os.path.join(save_path, f"images/synthetic_image_{i + 1}.png"),
                                os.path.join(save_path, f"masks/synthetic_mask_{i + 1}.png"))