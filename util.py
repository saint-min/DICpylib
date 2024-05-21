import numpy as np
from PIL import Image

def load_image_as_grayscale(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image)

def extract_subset(image, center_x, center_y, subset_radius_x, subset_radius_y):
    center_x = int(center_x)
    center_y = int(center_y)
    x_start = max(center_x - subset_radius_x, 0)
    x_end = min(center_x + subset_radius_x + 1, image.shape[1])
    y_start = max(center_y - subset_radius_y, 0)
    y_end = min(center_y + subset_radius_y + 1, image.shape[0])
    
    subset_width = 2 * subset_radius_x + 1
    subset_height = 2 * subset_radius_y + 1
    
    subset = np.zeros((subset_height, subset_width))
    subset[(y_start - (center_y - subset_radius_y)):(y_end - (center_y - subset_radius_y)),
           (x_start - (center_x - subset_radius_x)):(x_end - (center_x - subset_radius_x))] = \
           image[y_start:y_end, x_start:x_end]
    
    return subset

def generate_poi_queue(start_point, end_point, num_poi_x, num_poi_y):
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
    mean = matrix.mean()
    matrix -= mean
    norm = np.linalg.norm(matrix, ord=2)
    return np.sqrt(norm)