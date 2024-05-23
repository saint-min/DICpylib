from dicpylib.util import load_image_as_grayscale, generate_poi_queue
from dicpylib.fftcc import fftcc_compute_poi_queue
import pandas as pd

if __name__ == "__main__":
    # Define file paths for the reference and target images
    ref_img_path = './img/Test_2_reference.tif'
    tar_img_path = './img/Test_2_deformed.tif'
    
    # Load the reference and target images as grayscale
    ref_img = load_image_as_grayscale(ref_img_path)
    tar_img = load_image_as_grayscale(tar_img_path)

    # Define parameters for the POI (Points of Interest) queue generation
    start_point = (40, 40)  # Top-left corner
    end_point = (590, 1020)  # Bottom-right corner
    num_poi_x = (end_point[0] - start_point[0]) // 10 + 1  # Number of POIs in the x-direction
    num_poi_y = (end_point[1] - start_point[1]) // 10 + 1  # Number of POIs in the y-direction
    subset_radius_x = 200  # Subset radius in the x-direction
    subset_radius_y = 200  # Subset radius in the y-direction

    # Generate a queue of POIs
    poi_queue = generate_poi_queue(start_point, end_point, num_poi_x, num_poi_y)

    # Compute the FFT-based cross-correlation for the POI queue
    poi_queue = fftcc_compute_poi_queue(poi_queue, ref_img, tar_img, subset_radius_x, subset_radius_y)

    # Save the results to a CSV file
    df = pd.DataFrame(poi_queue)
    df.to_csv('fftcc_output.csv', index=True, encoding='utf-8')
