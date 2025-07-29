
import numpy as np
from scipy.spatial import cKDTree

def generate_square(center, side, plane="xy", num_points_per_side=25):
    """Generate a square grid of points in the specified plane centered at 'center'."""
    lin = np.linspace(-side/2, side/2, num_points_per_side)
    grid = np.array(np.meshgrid(lin, lin)).reshape(2, -1).T
    square = np.zeros((grid.shape[0], 3))

    if plane == "xy":
        square[:, 0:2] = grid
        square[:, 2] = 0
    elif plane == "xz":
        square[:, 0] = grid[:, 0]
        square[:, 2] = grid[:, 1]
    elif plane == "yz":
        square[:, 1:3] = grid
        square[:, 0] = 0
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'.")

    return square + center

def compute_fit_error(robot_points, square_points):
    """Compute mean squared distance from each robot point to its nearest square point."""
    tree = cKDTree(square_points)
    dists, _ = tree.query(robot_points)
    return np.sqrt(np.mean(dists**2))

def optimize_square_center(robot_points, side=0.1, plane="xy", num_steps=10):
    """
    Find the square center that minimizes distance from robot trajectory to square.
    The search region is bounded by the robot trajectory itself.
    """
    mins = robot_points.min(axis=0)
    maxs = robot_points.max(axis=0)

    x_vals = np.linspace(mins[0], maxs[0], num_steps)
    y_vals = np.linspace(mins[1], maxs[1], num_steps)
    z_vals = np.linspace(mins[2], maxs[2], num_steps)

    best_error = np.inf
    best_center = None

    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                center = np.array([x, y, z])
                square = generate_square(center, side, plane=plane)
                error = compute_fit_error(robot_points, square)
                if error < best_error:
                    best_error = error
                    best_center = center

    return best_center, best_error

def optimize_square_center_hierarchical(robot_points, side=0.1, plane="xy", 
                                        coarse_steps=10, fine_steps=15, refine_factor=0.1):
    """
    Hierarchical search: coarse grid followed by refined fine grid near best coarse center.
    refine_factor = size of fine region as fraction of full bounds
    """
    # 1. Coarse search
    mins = robot_points.min(axis=0)
    maxs = robot_points.max(axis=0)

    coarse_x = np.linspace(mins[0], maxs[0], coarse_steps)
    coarse_y = np.linspace(mins[1], maxs[1], coarse_steps)
    coarse_z = np.linspace(mins[2], maxs[2], coarse_steps)

    best_error = np.inf
    best_center = None

    for x in coarse_x:
        for y in coarse_y:
            for z in coarse_z:
                center = np.array([x, y, z])
                square = generate_square(center, side, plane=plane)
                error = compute_fit_error(robot_points, square)
                if error < best_error:
                    best_error = error
                    best_center = center

    # 2. Refined search around best_center
    range_xyz = maxs - mins
    half_size = refine_factor * range_xyz / 2
    local_mins = best_center - half_size
    local_maxs = best_center + half_size

    fine_x = np.linspace(local_mins[0], local_maxs[0], fine_steps)
    fine_y = np.linspace(local_mins[1], local_maxs[1], fine_steps)
    fine_z = np.linspace(local_mins[2], local_maxs[2], fine_steps)

    refined_best_error = best_error
    refined_best_center = best_center

    for x in fine_x:
        for y in fine_y:
            for z in fine_z:
                center = np.array([x, y, z])
                square = generate_square(center, side, plane=plane)
                error = compute_fit_error(robot_points, square)
                if error < refined_best_error:
                    refined_best_error = error
                    refined_best_center = center

    return refined_best_center, refined_best_error
