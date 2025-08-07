import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_full_hemisphere_with_shaded_xz_slice(r_min=0.6, r_max=1.8, color='mediumslateblue', alpha=0.25):
    # === Hemisphere mesh ===
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi / 2, 100)
    theta, phi = np.meshgrid(theta, phi)

    x = r_max * np.sin(phi) * np.cos(theta)
    y = r_max * np.sin(phi) * np.sin(theta)
    z = r_max * np.cos(phi)

    # Inner dome mesh
    x_inner_dome = r_min * np.sin(phi) * np.cos(theta)
    y_inner_dome = r_min * np.sin(phi) * np.sin(theta)
    z_inner_dome = r_min * np.cos(phi)

    # === Set up plot ===
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # === Plot outer and inner domes ===
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')
    ax.plot_surface(x_inner_dome, y_inner_dome, z_inner_dome, color=color, alpha=alpha, edgecolor='none')

    # === Create red shaded slice in XZ plane from 0° to 90° ===
    angles = np.linspace(0, np.pi / 2, 100)
    x_inner = r_min * np.sin(angles)
    z_inner = r_min * np.cos(angles)
    x_outer = r_max * np.sin(angles)
    z_outer = r_max * np.cos(angles)
    y_vals = np.zeros_like(angles)

    # Build quads between inner and outer arcs
    verts = []
    for i in range(len(angles) - 1):
        quad = [
            [x_inner[i], y_vals[i], z_inner[i]],
            [x_inner[i+1], y_vals[i+1], z_inner[i+1]],
            [x_outer[i+1], y_vals[i+1], z_outer[i+1]],
            [x_outer[i], y_vals[i], z_outer[i]]
        ]
        verts.append(quad)

    poly = Poly3DCollection(verts, color='mediumslateblue', alpha=0.4)
    ax.add_collection3d(poly)

    # === Plot formatting ===
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Full 3D Hemisphere with Inner Dome and Shaded XZ Slice (0° to 90°)')
    ax.set_box_aspect([1, 1, 0.6])
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()

# Run it
plot_full_hemisphere_with_shaded_xz_slice()
