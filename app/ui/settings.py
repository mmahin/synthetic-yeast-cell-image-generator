# Default settings for the synthetic image generator

DEFAULT_IMAGE_WIDTH = 256
DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_CELL_COUNT = 15
DEFAULT_MIN_RADIUS = 10
DEFAULT_MAX_RADIUS = 25

# Define ranges or limits for input parameters if needed
RADIUS_RANGE = (5, 50)  # Example limits for radius
CELL_COUNT_RANGE = (1, 100)  # Example limits for the number of cells

def get_default_settings():
    """
    Returns a dictionary of default settings.
    """
    return {
        "image_width": DEFAULT_IMAGE_WIDTH,
        "image_height": DEFAULT_IMAGE_HEIGHT,
        "cell_count": DEFAULT_CELL_COUNT,
        "min_radius": DEFAULT_MIN_RADIUS,
        "max_radius": DEFAULT_MAX_RADIUS,
    }