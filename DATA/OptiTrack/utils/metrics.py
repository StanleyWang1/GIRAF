import numpy as np
from scipy.spatial import cKDTree

def compute_nearest_distances(goal_traj, actual_traj):
    """
    For each point in `actual_traj`, find the distance to the nearest point in `goal_traj`.
    """
    tree = cKDTree(goal_traj)
    distances, _ = tree.query(actual_traj)
    return distances

def compute_error_stats(goal_traj, actual_traj):
    """
    Returns a dictionary with RMSE, mean, max, and percentile errors based on nearest distances.
    """
    distances = compute_nearest_distances(goal_traj, actual_traj)
    return {
        "mean": np.mean(distances),
        "max": np.max(distances),
        "rmse": np.sqrt(np.mean(distances**2)),
        "perc_90": np.percentile(distances, 90),
        "perc_95": np.percentile(distances, 95),
    }

def local_max_errors(goal_traj, robot_traj):
    """
    For each point in goal_traj, find the maximum distance to robot points closest to it,
    or the distance to the closest robot point if none were closest.
    """
    tree_goal = cKDTree(goal_traj)
    _, closest = tree_goal.query(robot_traj)

    tree_robot = cKDTree(robot_traj)
    radii = np.zeros(len(goal_traj))

    for i in range(len(goal_traj)):
        assigned = robot_traj[closest == i]
        if len(assigned):
            radii[i] = np.max(np.linalg.norm(assigned - goal_traj[i], axis=1))
        else:
            dist, _ = tree_robot.query(goal_traj[i])
            radii[i] = dist
    return radii


