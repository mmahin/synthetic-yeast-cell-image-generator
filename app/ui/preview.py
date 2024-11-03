import tkinter as tk
from PIL import Image, ImageTk
import cv2


class ImagePreviewer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.photo = None

    def update_preview(self, image):
        """
        Update the image preview on the canvas.

        Parameters:
            image (numpy ndarray): The synthetic image in OpenCV format to preview.
        """
        # Convert OpenCV image (NumPy array) to PIL Image
        pil_image = Image.fromarray(cv2.convertScaleAbs(image))

        # Resize the PIL Image to fit the canvas
        pil_image = pil_image.resize((256, 256), Image.ANTIALIAS)

        # Convert PIL Image to ImageTk format for Tkinter display
        self.photo = ImageTk.PhotoImage(pil_image)

        # Clear previous image and display the new one
        self.canvas.delete("all")
        self.canvas.create_image(128, 128, image=self.photo)

    def clear_preview(self):
        """Clear the preview canvas."""
        self.canvas.delete("all")