from dicpylib.util import load_image_as_grayscale, generate_poi_queue
from dicpylib.fftcc import fftcc_compute_poi_queue

if __name__ == "__main__":
    ref_img_path = './img/oht_cfrp_0.bmp'
    tar_img_path = './img/oht_cfrp_4.bmp'
    
    ref_img = load_image_as_grayscale(ref_img_path)
    tar_img = load_image_as_grayscale(tar_img_path)

    # Define parameters
    start_point = (30, 30)
    end_point = (220, 620)
    num_poi_x = 100
    num_poi_y = 200
    subset_radius_x = 16
    subset_radius_y = 16

    # Generate POI queue
    poi_queue = generate_poi_queue(start_point, end_point, num_poi_x, num_poi_y)

    # fftcc results
    poi_queue = fftcc_compute_poi_queue(poi_queue, ref_img, tar_img, subset_radius_x, subset_radius_y)

    for point in poi_queue:
        print(point)