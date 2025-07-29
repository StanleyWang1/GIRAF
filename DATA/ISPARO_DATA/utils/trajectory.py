import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

def dense_square(origin, side_len, pts_per_edge=25):
    ox, oy, oz = origin
    L = side_len / 2
    A = np.array([ox - L, oy, oz - L])
    B = np.array([ox + L, oy, oz - L])
    C = np.array([ox + L, oy, oz + L])
    D = np.array([ox - L, oy, oz + L])
    return np.vstack([
        np.linspace(A, B, pts_per_edge, endpoint=False),
        np.linspace(B, C, pts_per_edge, endpoint=False),
        np.linspace(C, D, pts_per_edge, endpoint=False),
        np.linspace(D, A, pts_per_edge, endpoint=False)
    ])

def average_loop(traj, num_bins=150, min_pts=3):
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    data_2d = PCA(n_components=2).fit_transform(traj)
    theta = (np.arctan2(data_2d[:, 1], data_2d[:, 0]) + 2*np.pi) % (2*np.pi)
    bins = np.linspace(0, 2*np.pi, num_bins+1)
    avg_points = []

    for i in range(num_bins):
        mask = (theta >= bins[i]) & (theta < bins[i+1])
        if np.sum(mask) >= min_pts:
            avg_points.append([np.mean(x[mask]), np.mean(y[mask]), np.mean(z[mask])])
    return np.array(avg_points)

def average_linear_bins(traj, num_bins=500, min_pts=1):
    """
    Splits a trajectory into `num_bins` sequential chunks and averages each chunk.

    Parameters:
        traj (np.ndarray): Nx3 array of 3D points.
        num_bins (int): Number of segments to divide the data into.
        min_pts (int): Minimum number of points required to compute an average in a bin.

    Returns:
        np.ndarray: Mx3 array of averaged points (M ≤ num_bins depending on min_pts).
    """
    N = len(traj)
    bin_size = max(N // num_bins, 1)
    avg_points = []

    for i in range(0, N, bin_size):
        chunk = traj[i:i + bin_size]
        if len(chunk) >= min_pts:
            avg_points.append(np.mean(chunk, axis=0))

    return np.array(avg_points)



def average_loop_3d(traj, num_samples=100, radius=0.05, min_pts=5):
    """
    Robust 3D loop averaging by resampling along a smoothed trajectory
    and averaging nearby raw points.

    Parameters:
    - traj: Nx3 trajectory
    - num_samples: number of points in the averaged trajectory
    - radius: radius to look for neighboring points
    - min_pts: minimum number of points to compute a valid average

    Returns:
    - Mx3 averaged trajectory (M ≤ num_samples)
    """
    traj = np.asarray(traj)
    tree = cKDTree(traj)

    # Fit 3D spline to smooth the path
    tck, _ = splprep(traj.T, s=0.001, per=True)
    u_fine = np.linspace(0, 1, num_samples)
    x_smooth, y_smooth, z_smooth = splev(u_fine, tck)
    smooth_path = np.vstack([x_smooth, y_smooth, z_smooth]).T

    avg_points = []
    for pt in smooth_path:
        idx = tree.query_ball_point(pt, r=radius)
        if len(idx) >= min_pts:
            local_pts = traj[idx]
            avg_points.append(np.mean(local_pts, axis=0))

    return np.array(avg_points)

def crop_motion_segment(traj, threshold=1e-3, window=5):
    """
    Crops a trajectory to exclude stationary periods at start and end.

    Parameters:
    - traj (Nx3): 3D trajectory as a NumPy array
    - threshold: minimum displacement to consider as "motion"
    - window: how many steps to use when computing movement

    Returns:
    - cropped_traj: trimmed trajectory array
    """
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    motion = np.convolve(diffs, np.ones(window), mode='valid')
    moving = np.where(motion > threshold)[0]

    if len(moving) == 0:
        return traj

    start_idx = max(0, moving[0])
    end_idx = min(len(traj) - 1, moving[-1] + window)
    return traj[start_idx:end_idx + 1]



def sliding_pca_average(trials, reference_traj=None, radius=0.05, step=1, fallback_index=0):
    """
    Compute average trajectory using sliding local PCA.

    Parameters:
    - trials: list of Nx3 numpy arrays (multiple 3D trajectories)
    - reference_traj: optional Nx3 array. If None, average_loop is used on fallback_index trial
    - radius: float, radius to consider nearby points
    - step: int, sampling step along the reference
    - fallback_index: which trial to use for average_loop fallback

    Returns:
    - avg_traj: Mx3 numpy array, the average trajectory
    """
    if reference_traj is None:
        reference_traj = average_loop(trials[fallback_index])

    all_points = np.vstack(trials)
    avg_points = []

    for i in range(0, len(reference_traj), step):
        ref_point = reference_traj[i]
        distances = np.linalg.norm(all_points - ref_point, axis=1)
        local_points = all_points[distances <= radius]

        if len(local_points) < 3:
            continue

        mean = np.mean(local_points, axis=0)
        centered = local_points - mean
        pca = PCA(n_components=2)
        pca.fit(centered)

        projected = pca.transform(centered)
        avg_2d = np.mean(projected, axis=0)

        avg_3d = mean + avg_2d @ pca.components_
        avg_points.append(avg_3d)

    return np.array(avg_points)
