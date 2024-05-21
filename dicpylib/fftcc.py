import numpy as np
from scipy.fft import fft2, ifft2
from multiprocessing import Pool, cpu_count
from util import extract_subset

def fftcc_compute_single_poi(poi, ref_img, tar_img, subset_radius_x, subset_radius_y):
    initial_displacement = np.array([poi['u'], poi['v']])
    
    # Set subset size
    subset_width = 2 * subset_radius_x + 1
    subset_height = 2 * subset_radius_y + 1

    ref_subset = extract_subset(ref_img, poi['x'], poi['y'], subset_radius_x, subset_radius_y)
    tar_subset = extract_subset(tar_img, poi['x'] + initial_displacement[0], poi['y'] + initial_displacement[1], subset_radius_x, subset_radius_y)

    ref_mean = np.mean(ref_subset)
    tar_mean = np.mean(tar_subset)

    # Zero-mean operation
    ref_subset -= ref_mean
    tar_subset -= tar_mean

    # Norm
    ref_norm = np.sqrt(np.sum(ref_subset**2))
    tar_norm = np.sqrt(np.sum(tar_subset**2))

    if ref_norm == 0 or tar_norm == 0:
        # If norm is zero, correlation cannot be computed
        poi['zncc'] = 0
        return poi

    # Normalize subsets
    ref_subset /= ref_norm
    tar_subset /= tar_norm

    # 2D Fourier Transform
    ref_freq = np.fft.fft2(ref_subset)
    tar_freq = np.fft.fft2(tar_subset)

    # Perform complex multiplication
    zncc_freq = np.zeros_like(ref_freq, dtype=np.complex128)
    zncc_freq.real = (ref_freq.real * tar_freq.real) + (ref_freq.imag * tar_freq.imag)
    zncc_freq.imag = (ref_freq.real * tar_freq.imag) - (ref_freq.imag * tar_freq.real)
    
    # Perform inverse Fourier Transform
    zncc = np.fft.ifft2(zncc_freq).real

    # Get max ZNCC
    max_zncc = np.max(zncc)
    local_displacement_v, local_displacement_u = np.unravel_index(np.argmax(zncc), zncc.shape)

    if local_displacement_u > subset_radius_x:
        local_displacement_u -= subset_width
    if local_displacement_v > subset_radius_y:
        local_displacement_v -= subset_height

    # Store the final results
    final_displacement = np.array([local_displacement_u, local_displacement_v]) + initial_displacement
    poi['u'] = final_displacement[0]
    poi['v'] = final_displacement[1]
    poi['u0'] = initial_displacement[0]
    poi['v0'] = initial_displacement[1]
    poi['zncc'] = max_zncc

    return poi

def fftcc_compute_poi_queue(poi_queue, ref_img, tar_img, subset_radius_x, subset_radius_y):
    num_threads = min(cpu_count(), len(poi_queue))
    args = [(poi, ref_img, tar_img, subset_radius_x, subset_radius_y) for poi in poi_queue]

    with Pool(processes=num_threads) as pool:
        results = pool.starmap(fftcc_compute_single_poi, args)

    return results
