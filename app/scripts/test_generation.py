from app.generator.cell_generator import generate_synthetic_yeast_image, save_synthetic_data

# Generate a synthetic yeast cell image and corresponding mask
image, mask = generate_synthetic_yeast_image(width=256, height=256, cell_count=15, cell_radius_range=(10, 25))

# Save the generated images to files
save_synthetic_data(image, mask, image_path="data/output/images/synthetic_image.png", mask_path="data/output/masks/synthetic_mask.png")

