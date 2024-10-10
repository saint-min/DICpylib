import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def compute_strain_single_poi(poi, neighbors):
    """
    Compute the strain for a single Point of Interest (POI) based on its neighbors.

    Parameters:
    poi (dict): A dictionary containing POI data.
    neighbors (list): A list of neighboring POIs.

    Returns:
    dict: Updated POI dictionary with computed strain values.
    """
    # Check if there are enough neighbors for strain computation
    if len(neighbors) < 2:
        # Not enough neighbors to compute strain
        poi['exx'] = 0.0
        poi['eyy'] = 0.0
        poi['exy'] = 0.0
        return poi

    # Extract neighbor coordinates and displacements
    neighbor_coords = np.array([(neighbor['x'] - poi['x'], neighbor['y'] - poi['y']) for neighbor in neighbors])
    neighbor_displacements = np.array([(neighbor['u'], neighbor['v']) for neighbor in neighbors])

    # Construct coefficient matrix and displacement vectors
    A = np.zeros((len(neighbors), 3))
    A[:, 0] = 1
    A[:, 1:3] = neighbor_coords

    u_vector = neighbor_displacements[:, 0]
    v_vector = neighbor_displacements[:, 1]

    # Solve for displacement gradients using least squares
    u_gradient, _, _, _ = np.linalg.lstsq(A, u_vector, rcond=None)
    v_gradient, _, _, _ = np.linalg.lstsq(A, v_vector, rcond=None)

    # Compute strain components
    exx = u_gradient[1]
    eyy = v_gradient[2]
    exy = 0.5 * (u_gradient[2] + v_gradient[1])

    # Update POI with strain values
    poi['exx'] = exx
    poi['eyy'] = eyy
    poi['exy'] = exy

    return poi

def compute_strain(poi_queue, neighbor_count=50, search_radius=None):
    """
    Compute the strain at each Point of Interest (POI) based on its neighbors using multiprocessing.

    Parameters:
    poi_queue (list): A list of POI dictionaries containing coordinates and displacements.
    neighbor_count (int): Number of neighbors to use for strain computation.
    search_radius (float): Radius to search for neighbors. If None, use k nearest neighbors.

    Returns:
    list: Updated POI queue with computed strain values.
    """
    # Convert POI queue to numpy array for KD-Tree construction
    poi_coords = np.array([(poi['x'], poi['y']) for poi in poi_queue])

    # Construct KD-Tree for nearest neighbor search
    tree = cKDTree(poi_coords)

    # Prepare arguments for multiprocessing
    args = []
    for i, poi in enumerate(poi_queue):
        if search_radius is not None:
            # Find neighbors within a specific radius
            indices = tree.query_ball_point(poi_coords[i], r=search_radius)
            neighbors = [poi_queue[idx] for idx in indices if idx != i]
        else:
            # Find k nearest neighbors
            _, indices = tree.query(poi_coords[i], k=neighbor_count)
            neighbors = [poi_queue[idx] for idx in indices if idx != i]

        args.append((poi, neighbors))

    # Determine the number of threads to use
    num_threads = min(cpu_count(), len(poi_queue))

    # Use multiprocessing Pool to compute the POI queue in parallel
    with Pool(processes=num_threads) as pool:
        results = pool.starmap(compute_strain_single_poi, tqdm(args, total=len(poi_queue)))

    return results