import numpy as np
from scipy.fft import fft2, ifft2
from multiprocessing import Pool, cpu_count
from .util import extract_subset

def fftcc_compute_single_poi(poi, ref_img, tar_img, subset_radius_x, subset_radius_y):
    """
    Compute the displacement and zero-mean normalized cross-correlation (ZNCC) for a single Point of Interest (POI).

    Parameters:
    poi (dict): A dictionary containing POI data.
    ref_img (np.array): The reference image.
    tar_img (np.array): The target image.
    subset_radius_x (int): The radius of the subset in the x-direction.
    subset_radius_y (int): The radius of the subset in the y-direction.

    Returns:
    dict: The updated POI dictionary with computed displacement and ZNCC.
    """
    # Initial displacement from the POI (Point of Interest)
    initial_displacement = np.array([poi['u'], poi['v']])
    
    # Set subset size
    subset_width = 2 * subset_radius_x + 1
    subset_height = 2 * subset_radius_y + 1

    # Extract reference subset
    ref_subset = extract_subset(image=ref_img,
                                center_x=poi['x'],
                                center_y=poi['y'],
                                subset_radius_x=subset_radius_x,
                                subset_radius_y=subset_radius_y)
    
    # Extract target subset
    tar_subset = extract_subset(image=tar_img,
                                center_x=poi['x'] + initial_displacement[0],
                                center_y=poi['y'] + initial_displacement[1],
                                subset_radius_x=subset_radius_x,
                                subset_radius_y=subset_radius_y)

    # Calculate the mean of the subsets
    ref_mean = np.mean(ref_subset)
    tar_mean = np.mean(tar_subset)

    # Zero-mean operation on the subsets
    ref_subset -= ref_mean
    tar_subset -= tar_mean

    # Calculate the norm of the subsets
    ref_norm = np.sqrt(np.sum(ref_subset**2))
    tar_norm = np.sqrt(np.sum(tar_subset**2))

    if ref_norm == 0 or tar_norm == 0:
        # If the norm is zero, correlation cannot be computed
        poi['zncc'] = 0
        return poi

    # Normalize the subsets
    ref_subset /= ref_norm
    tar_subset /= tar_norm

    # 2D Fourier Transform of the subsets
    ref_freq = fft2(ref_subset)
    tar_freq = fft2(tar_subset)

    # Perform complex multiplication in frequency domain
    zncc_freq = np.zeros_like(ref_freq, dtype=np.complex128)
    zncc_freq.real = (ref_freq.real * tar_freq.real) + (ref_freq.imag * tar_freq.imag)
    zncc_freq.imag = (ref_freq.real * tar_freq.imag) - (ref_freq.imag * tar_freq.real)
    
    # Perform inverse Fourier Transform to get the cross-correlation
    zncc = ifft2(zncc_freq).real

    # Get the maximum ZNCC value
    max_zncc = np.max(zncc)
    local_displacement_v, local_displacement_u = np.unravel_index(np.argmax(zncc), zncc.shape)

    # Adjust displacement if it exceeds subset radius
    if local_displacement_u > subset_radius_x:
        local_displacement_u -= subset_width
    if local_displacement_v > subset_radius_y:
        local_displacement_v -= subset_height

    # Store the final results back into the POI dictionary
    final_displacement = np.array([local_displacement_u, local_displacement_v]) + initial_displacement
    poi['u'] = final_displacement[0]
    poi['v'] = final_displacement[1]
    poi['u0'] = initial_displacement[0]
    poi['v0'] = initial_displacement[1]
    poi['zncc'] = max_zncc

    return poi

def fftcc_compute_poi_queue(poi_queue, ref_img, tar_img, subset_radius_x, subset_radius_y):
    """
    Compute the displacements and ZNCC for a queue of Points of Interest (POIs) using multiprocessing.

    Parameters:
    poi_queue (list): A list of POI dictionaries.
    ref_img (np.array): The reference image.
    tar_img (np.array): The target image.
    subset_radius_x (int): The radius of the subset in the x-direction.
    subset_radius_y (int): The radius of the subset in the y-direction.

    Returns:
    list: A list of updated POI dictionaries with computed displacements and ZNCC.
    """
    # Determine the number of threads to use
    num_threads = min(cpu_count(), len(poi_queue))
    # Prepare arguments for multiprocessing
    args = [(poi, ref_img, tar_img, subset_radius_x, subset_radius_y) for poi in poi_queue]

    # Use multiprocessing Pool to compute the POI queue in parallel
    with Pool(processes=num_threads) as pool:
        results = pool.starmap(fftcc_compute_single_poi, args)

    return results
