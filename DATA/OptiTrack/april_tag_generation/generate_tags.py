import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import os
from scipy.ndimage import rotate

# === CONFIG ===
tag_filenames = [
    "tag36_11_00012.png",
    "tag36_11_00013.png",
    "tag36_11_00014.png",
    "tag36_11_00015.png"
]

# Rotate each tag (in degrees, counter-clockwise)
tag_orientations_deg = [180, 180, 180, 180]

# Flip each tag (flip_x, flip_y)
flip_flags = [
    (True, False),
    (True, False),
    (True, False),
    (True, False)
]

tag_size_cm = 6.0
spacing_x_cm = 11.0
spacing_y_cm = 11.0
margin_cm = 0.0
dpi = 300

# US Letter size in cm
page_width_in = 11
page_height_in = 8.5
page_width_cm = page_width_in * 2.54
page_height_cm = page_height_in * 2.54

rows, cols = 2, 2
assert len(tag_filenames) == rows * cols

# === SETUP CANVAS ===
fig, ax = plt.subplots(figsize=(page_width_in, page_height_in), dpi=dpi)
ax.set_xlim(0, page_width_cm)
ax.set_ylim(0, page_height_cm)
ax.set_aspect('equal')
ax.axis('off')

# Compute grid origin
grid_width_cm = (cols - 1) * spacing_x_cm + tag_size_cm - 15
grid_height_cm = (rows - 1) * spacing_y_cm + tag_size_cm - 7
origin_x = (page_width_cm - grid_width_cm) / 2
origin_y = (page_height_cm - grid_height_cm) / 2

# === PLACE TAGS ===
for idx, fname in enumerate(tag_filenames):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing file: {fname}")
    img = mpimg.imread(fname)

    # Flip
    flip_x, flip_y = flip_flags[idx]
    if flip_x:
        img = np.fliplr(img)
    if flip_y:
        img = np.flipud(img)

    # Rotate image using scipy
    angle = tag_orientations_deg[idx]
    if angle != 0:
        img = rotate(img, angle, reshape=False)

    # Grid position
    row = idx // cols
    col = idx % cols
    x_center = origin_x + col * spacing_x_cm
    y_center = origin_y + (rows - 1 - row) * spacing_y_cm
    x0 = x_center - tag_size_cm / 2
    y0 = y_center - tag_size_cm / 2

    # Draw tag image
    ax.imshow(img, extent=[x0, x0 + tag_size_cm, y0, y0 + tag_size_cm], origin='lower', cmap='gray')

    # Draw label
    tag_id = fname.split("_")[-1].split(".")[0]
    ax.text(x_center, y0 - 0.3, f"ID {int(tag_id)}", ha='center', va='top', fontsize=8)

    # Border
    ax.add_patch(patches.Rectangle(
        (x0, y0), tag_size_cm, tag_size_cm,
        linewidth=0.3, edgecolor='black', facecolor='none'
    ))

# === SAVE ===
output_file = "tags12-15.pdf"
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
print(f"Saved: {output_file}")
