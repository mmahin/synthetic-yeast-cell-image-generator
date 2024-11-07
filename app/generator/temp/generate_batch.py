import os
from app.generator.cell_generator import generate_synthetic_yeast_image, save_synthetic_data


def generate_batch(batch_size=10, width=256, height=256, cell_count=15, cell_radius_range=(10, 25),
                   save_dir="data/output"):
    """
    Generates a batch of synthetic yeast cell images and masks, saving each pair with unique filenames.

    Parameters:
        batch_size (int): Number of images to generate in the batch.
        width (int): Width of each generated image.
        height (int): Height of each generated image.
        cell_count (int): Number of yeast cells in each image.
        cell_radius_range (tuple): Range for yeast cell radii (min, max).
        save_dir (str): Directory to save generated images and masks.
    """
    images_dir = os.path.join(save_dir, "images")
    masks_dir = os.path.join(save_dir, "masks")

    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for i in range(batch_size):
        # Generate a synthetic image and mask
        image, mask = generate_synthetic_yeast_image(
            width=width,
            height=height,
            cell_count=cell_count,
            cell_radius_range=cell_radius_range
        )

        # Save the generated images and masks with unique filenames
        image_path = os.path.join(images_dir, f"synthetic_image_{i + 1}.png")
        mask_path = os.path.join(masks_dir, f"synthetic_mask_{i + 1}.png")

        save_synthetic_data(image, mask, image_path=image_path, mask_path=mask_path)

        print(f"Generated and saved {image_path} and {mask_path}")


# Generate a batch of 10 synthetic images and masks
generate_batch(batch_size=10, width=256, height=256, cell_count=15, cell_radius_range=(10, 25))
