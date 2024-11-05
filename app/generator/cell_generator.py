import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
import os

def draw_circle_with_intensity_variation(img, center, radius, base_color, noise_level):
    # Create an array of points around the circle
    angles = np.linspace(0, 2 * np.pi, num=100)
    for angle in angles:
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))

        # Randomly adjust intensity
        intensity_variation = np.random.randint(-noise_level, noise_level)
        color_with_variation = (
            base_color[0],
            np.clip(base_color[1] + intensity_variation, 0, 65535),
            np.clip(base_color[2] + intensity_variation, 0, 65535)
        )

        img[y, x] = color_with_variation
def draw_noisy_circle(img, mask, center, radius, color,mask_color,  noise_level):
    # Create an array of points around the circle
    angles = np.linspace(0, 2 * np.pi, num=100)
    x_points = center[0] + radius * np.cos(angles)
    y_points = center[1] + radius * np.sin(angles)

    # Add noise to the points
    x_points += np.random.normal(0, noise_level, x_points.shape)
    y_points += np.random.normal(0, noise_level, y_points.shape)

    # Convert to integer coordinates
    x_points = np.clip(x_points, 0, img.shape[1] - 1).astype(np.int32)
    y_points = np.clip(y_points, 0, img.shape[0] - 1).astype(np.int32)

    # Draw the noisy filled shape
    points = np.array([x_points, y_points]).T.reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color)
    cv2.fillPoly(mask, [points],  mask_color)  # Draw on uint8 mask (deep blue)


def interpolate_color(start_color , end_color , steps):
    # Generate a list of interpolated colors between start_color and end_color
    colors = []
    for i in range(steps):
        # Interpolate each channel
        red = int(start_color[0] + (end_color[0] - start_color[0]) * (i / (steps - 1)))
        green = int(start_color[1] + (end_color[1] - start_color[1]) * (i / (steps - 1)))
        blue = int(start_color[2] + (end_color[2] - start_color[2]) * (i / (steps - 1)))
        colors.append((blue, green, red))  # BGR format
    return colors
def generate_synthetic_yeast_image(width=256, height=256, cell_count=10,
                                    fluorescence_level=1,
                                    cell_radius_range=(10, 20), noise_level=0.01 ):
    """
    Generates a synthetic yeast cell image and corresponding binary mask.

    Parameters:
        image_size (tuple): Size of the generated image (width, height).
        cell_count (int): Number of yeast cells to generate in the image.
        cell_radius_range (tuple): Min and max radius of the yeast cells.

    Returns:
        tuple: PIL Image for synthetic image and mask.
    """
    # Create blank arrays for the synthetic image (uint16) and the mask (uint8)
    img = np.zeros((width, height, 3), dtype=np.uint16)
    # Create a blank uint8 image
    mask = np.zeros((width, height, 3), dtype=np.uint8)

    # Set the image to deep blue (B, G, R)
    mask[:] = (255, 0, 0)

    color_variations = interpolate_color((0, 0, 0), (255, 255, 0), cell_count)

    # Generate random yeast cells
    for cell in range(cell_count):
        # Random position and radius
        x = np.random.randint(0, img.shape[1])
        y = np.random.randint(0, img.shape[0])
        radius = np.random.randint(cell_radius_range[0], cell_radius_range[1])

        # Random intensity factor for the yellow color
        intensity_factor = np.random.uniform(0.1, 1.0)

        # Calculate yellow color with varying intensity
        yellow = (0, 65535*fluorescence_level , 65535*fluorescence_level)

        # Draw filled circle (simulating a yeast cell) on both the image and mask
        # Draw the circle with varying intensity
        #cv2.circle(img, (x, y), radius, yellow, -1)
        draw_noisy_circle(img, mask, (x, y), radius, yellow, color_variations[cell], noise_level)
        #draw_circle_with_intensity_variation(img, (x, y), radius, yellow, noise_level)

    return img, mask


def save_synthetic_data(image, mask, image_path="data/output/synthetic_image.png",
                        mask_path="data/output/synthetic_mask.png"):
    """
    Saves synthetic yeast image and mask to specified file paths.

    Parameters:
        image (numpy.ndarray): Synthetic yeast cell image.
        mask (numpy.ndarray): Corresponding binary mask.
        image_path (str): File path to save the synthetic image.
        mask_path (str): File path to save the mask.
    """
    # Ensure the directories exist
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    # Save the image and mask
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)
    print(f"Synthetic image saved at {image_path}")
    print(f"Synthetic mask saved at {mask_path}")


def load_generated_data():
    # Paths to your generated data
    image_dir = 'data/output/images/'  # Adjust this path as needed
    mask_dir = 'data/output/masks/'  # Adjust this path as needed

    images = []
    masks = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):  # Assuming you save images as PNG
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('.png', '_mask.png'))  # Adjust accordingly

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load mask with unchanged flag

            images.append(image)
            masks.append(mask)

    # Convert to numpy arrays
    return np.array(images), np.array(masks)