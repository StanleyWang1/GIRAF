# ISPARO_DATA/utils/plot.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import special_ortho_group
from scipy.linalg import eigh

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    span = np.max(limits[:,1] - limits[:,0])
    ax.set_xlim3d([centers[0]-span/2, centers[0]+span/2])
    ax.set_ylim3d([centers[1]-span/2, centers[1]+span/2])
    ax.set_zlim3d([centers[2]-span/2, centers[2]+span/2])

def hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.set_title("")

def plot_task_tube_spheres(goal, robot, radii, ax=None,
                   goal_color='crimson', robot_color='orange', tube_color='orange'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(*goal.T, '--', label='Goal', color=goal_color, linewidth=3)
    ax.plot(*robot.T, '-', label='Robot', color=robot_color, linewidth=1)

    u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
    for p, r in zip(goal, radii):
        if r < 1e-4: continue
        x = p[0] + r*np.cos(u)*np.sin(v)
        y = p[1] + r*np.sin(u)*np.sin(v)
        z = p[2] + r*np.cos(v)
        ax.plot_surface(x, y, z, alpha=0.05, color=tube_color, linewidth=0)

    set_axes_equal(ax)
    hide_axes(ax)
    #ax.legend()

def plot_task_tube_ellipsoids(goal, robot, radii=None, ax=None,
                               goal_color='black', robot_color='orange', tube_color='orange',
                               scale=1.0, radius=0.1, min_pts=5):
    from scipy.spatial import cKDTree
    from scipy.linalg import eigh

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(*goal.T, '--', label='Goal', color=goal_color, linewidth=2)
    ax.plot(*robot.T, '-', label='Robot', color=robot_color, linewidth=.85)

    tree = cKDTree(robot)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    for i, center in enumerate(goal):
        idx = tree.query_ball_point(center, r=radius)
        if len(idx) < min_pts:
            continue

        local_pts = robot[idx]
        diffs = local_pts - np.mean(local_pts, axis=0)
        cov = np.cov(diffs.T)

        eigvals, eigvecs = eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        axes = scale * np.sqrt(np.maximum(eigvals, 1e-8))  # avoid sqrt(0)

        z = axes[0] * np.cos(u) * np.sin(v)
        y = axes[1] * np.sin(u) * np.sin(v)
        x = axes[2] * np.cos(v)

        pts = np.stack([x, y, z], axis=-1)
        rotated = pts @ eigvecs.T
        x_ellip = rotated[..., 0] + center[0]
        y_ellip = rotated[..., 1] + center[1]
        z_ellip = rotated[..., 2] + center[2]

        ax.plot_surface(x_ellip, y_ellip, z_ellip, color=tube_color, alpha=0.1, linewidth=0)

    set_axes_equal(ax)
    hide_axes(ax)
    #ax.legend()
