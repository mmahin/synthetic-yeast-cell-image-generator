import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
import random

def draw_circle_with_intensity_variation(img, center, radius, base_color, noise_level):
    """
    Draws a circle with intensity variation on an image.

    Parameters:
        img (numpy.ndarray): Image array where the circle is drawn.
        center (tuple): Center of the circle as (x, y).
        radius (int): Radius of the circle.
        base_color (tuple): Base color of the circle in (B, G, R) format.
        noise_level (int): Range for random noise variation in intensity.

    This function creates a circular pattern on the image with small, random
    variations in intensity for a realistic effect.
    """
    angles = np.linspace(0, 2 * np.pi, num=100)
    for angle in angles:
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        intensity_variation = np.random.randint(-noise_level, noise_level)
        color_with_variation = (
            base_color[0],
            np.clip(base_color[1] + intensity_variation, 0, 65535),
            np.clip(base_color[2] + intensity_variation, 0, 65535)
        )
        img[y, x] = color_with_variation

def draw_noisy_circle(img, mask, center, radius, color, mask_color, noise_level):
    """
    Draws a noisy circle on both an image and a mask.

    Parameters:
        img (numpy.ndarray): Image array where the circle is drawn.
        mask (numpy.ndarray): Mask array where the circle is drawn.
        center (tuple): Center of the circle as (x, y).
        radius (int): Radius of the circle.
        color (tuple): Color of the circle in the image in (B, G, R) format.
        mask_color (tuple): Color for the circle in the mask in (B, G, R) format.
        noise_level (float): Standard deviation for noise in circle's edges.

    This function draws a circle with noise around its edges to simulate a
    natural, irregular boundary.
    """
    angles = np.linspace(0, 2 * np.pi, num=100)
    x_points = center[0] + radius * np.cos(angles)
    y_points = center[1] + radius * np.sin(angles)
    x_points += np.random.normal(0, noise_level, x_points.shape)
    y_points += np.random.normal(0, noise_level, y_points.shape)
    x_points = np.clip(x_points, 0, img.shape[1] - 1).astype(np.int32)
    y_points = np.clip(y_points, 0, img.shape[0] - 1).astype(np.int32)
    points = np.array([x_points, y_points]).T.reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color)
    cv2.fillPoly(mask, [points], mask_color)

def interpolate_color(start_color, end_color, steps):
    """
    Interpolates between two colors to generate a gradient of colors.

    Parameters:
        start_color (tuple): Starting color in (B, G, R) format.
        end_color (tuple): Ending color in (B, G, R) format.
        steps (int): Number of intermediate colors to generate.

    Returns:
        list: List of interpolated colors between start_color and end_color.
    """
    colors = []
    for i in range(steps):
        red = int(start_color[0] + (end_color[0] - start_color[0]) * (i / (steps - 1)))
        green = int(start_color[1] + (end_color[1] - start_color[1]) * (i / (steps - 1)))
        blue = int(start_color[2] + (end_color[2] - start_color[2]) * (i / (steps - 1)))
        colors.append((blue, green, red))  # BGR format
    return colors

def generate_synthetic_yeast_image(width=256, height=256, cell_count=10, fluorescence_level=1,
                                   cell_radius_range=(10, 20), noise_level=0.01):
    """
    Generates a synthetic yeast cell image and corresponding binary mask.

    Parameters:
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        cell_count (int): Number of yeast cells to generate in the image.
        fluorescence_level (float): Intensity level for fluorescence effect.
        cell_radius_range (tuple): Min and max radius of yeast cells.
        noise_level (float): Standard deviation for noise added to the cell edges.

    Returns:
        tuple: Generated synthetic yeast cell image and corresponding mask.
    """
    img = np.zeros((width, height, 3), dtype=np.uint16)
    mask = np.zeros((width, height, 3), dtype=np.uint8)
    mask[:] = (255, 0, 0)
    color_variations = interpolate_color((0, 0, 0), (255, 255, 0), cell_count)

    for cell in range(cell_count):
        x = np.random.randint(0, img.shape[1])
        y = np.random.randint(0, img.shape[0])
        radius = np.random.randint(cell_radius_range[0], cell_radius_range[1])
        intensity_factor = np.random.uniform(0.1, 1.0)
        yellow = (0, 65535 * fluorescence_level, 65535 * fluorescence_level)
        draw_noisy_circle(img, mask, (x, y), radius, yellow, color_variations[cell], noise_level)

    return img, mask

def save_synthetic_data(image, mask, image_path="data/output/synthetic_image.png",
                        mask_path="data/output/synthetic_mask.png"):
    """
    Saves synthetic yeast cell image and corresponding mask to specified paths.

    Parameters:
        image (numpy.ndarray): Synthetic yeast cell image.
        mask (numpy.ndarray): Corresponding binary mask.
        image_path (str): File path to save the synthetic image.
        mask_path (str): File path to save the mask.

    This function saves the generated synthetic image and mask to the disk at
    the specified file paths, creating directories if necessary.
    """
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)
    print(f"Synthetic image saved at {image_path}")
    print(f"Synthetic mask saved at {mask_path}")

def load_generated_data(image_dir="data/output/images", mask_dir="data/output/masks"):
    """
    Loads previously generated synthetic yeast cell images and masks from disk.

    Parameters:
        image_dir (str): Directory where images are stored.
        mask_dir (str): Directory where masks are stored.

    Returns:
        tuple: Numpy arrays containing loaded images and masks.

    This function loads images and their corresponding masks from specified
    directories and returns them as numpy arrays.
    """
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('.png', '_mask.png'))
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)
