import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
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

        # Canvas for previewing the generated image
        self.canvas = tk.Canvas(self.root, width=256, height=256)
        self.canvas.grid(row=6, columnspan=2)

    def generate_image(self):
        try:
            # Get parameter values from the UI
            width = self.image_width.get()
            height = self.image_height.get()
            count = self.cell_count.get()
            radius_range = (self.cell_radius_min.get(), self.cell_radius_max.get())

            # Generate synthetic image and mask (OpenCV format)
            image, mask = generate_synthetic_yeast_image(
                width=256, height=256,
                cell_count=count,
                cell_radius_range=radius_range
            )

            # Convert OpenCV image to PIL format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            pil_image = Image.fromarray(image_rgb)

            # Resize for preview
            pil_image.thumbnail((256, 256), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(pil_image)

            # Update the canvas with the new image
            self.canvas.delete("all")  # Clear any existing image
            self.canvas.create_image(128, 128, image=self.photo)

            # Optionally save the generated images
            save_synthetic_data(pil_image, mask)

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticImageGeneratorApp(root)
    root.mainloop()