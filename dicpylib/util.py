import numpy as np
from PIL import Image

def load_image_as_grayscale(image_path):
    """
    Load an image from the specified file path and convert it to grayscale.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    np.array: The image as a grayscale numpy array.
    """
    image = Image.open(image_path).convert('L')
    return np.array(image)

def extract_subset(image, center_x, center_y, subset_radius_x, subset_radius_y):
    """
    Extract a subset of the image centered at (center_x, center_y) with the specified radii.

    Parameters:
    image (np.array): The input image.
    center_x (int): The x-coordinate of the center point.
    center_y (int): The y-coordinate of the center point.
    subset_radius_x (int): The radius of the subset in the x-direction.
    subset_radius_y (int): The radius of the subset in the y-direction.

    Returns:
    np.array: The extracted subset of the image.
    """
    center_x = int(center_x)
    center_y = int(center_y)
    x_start = max(center_x - subset_radius_x, 0)
    x_end = min(center_x + subset_radius_x + 1, image.shape[1])
    y_start = max(center_y - subset_radius_y, 0)
    y_end = min(center_y + subset_radius_y + 1, image.shape[0])
    
    subset_width = 2 * subset_radius_x + 1
    subset_height = 2 * subset_radius_y + 1
    
    # initialization value for when the subset exceeds image boundaries
    # 200 is the experimental best result
    subset_init_val = 200

    subset = np.zeros((subset_height, subset_width)) + subset_init_val
    subset[(y_start - (center_y - subset_radius_y)):(y_end - (center_y - subset_radius_y)),
           (x_start - (center_x - subset_radius_x)):(x_end - (center_x - subset_radius_x))] = \
           image[y_start:y_end, x_start:x_end]
    
    return subset

def generate_poi_queue(start_point, end_point, num_poi_x, num_poi_y):
    """
    Generate a queue of Points of Interest (POI) between the start and end points.

    Parameters:
    start_point (tuple): The starting point (x, y).
    end_point (tuple): The ending point (x, y).
    num_poi_x (int): The number of POIs in the x-direction.
    num_poi_y (int): The number of POIs in the y-direction.

    Returns:
    list: A list of POI dictionaries.
    """
    x_coords = np.linspace(start_point[0], end_point[0], num_poi_x, dtype=int)
    y_coords = np.linspace(start_point[1], end_point[1], num_poi_y, dtype=int)
    
    poi_queue = []
    for y in y_coords:
        for x in x_coords:
            poi = {
                'x': int(x),
                'y': int(y),
                'u': 0,
                'v': 0,
                'u0': 0,
                'v0': 0,
                'zncc': 0,
                'iteration': 0,
                'convergence': 0,
                'exx': 0,
                'eyy': 0,
                'exy': 0,
            }
            poi_queue.append(poi)
    return poi_queue

def zero_mean_norm(matrix):
    """
    Perform zero-mean normalization on the given matrix.

    Parameters:
    matrix (np.array): The input matrix.

    Returns:
    float: The norm of the matrix after zero-mean normalization.
    """
    mean = matrix.mean()
    matrix -= mean
    norm = np.linalg.norm(matrix, ord=2)
    return np.sqrt(norm)
