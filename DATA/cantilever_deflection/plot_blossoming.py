import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# Provided data
x = np.array([0, -0.244, -0.577, -0.92, -1.552, -3.601, -5.495, -8.262, -13.132, -16.573,
              -21.979, -26.31, -29.619, -30, -29.602, -29.234, -27.771, -26.973, -26.291,
              -25.44, -23.036, -20.26, -17.114, -13.574, -9.659, -6.567, -3.617, -0.901, 0])
y = np.array([0.04445, 0.05715, 0.0762, 0.09525, 0.12954, 0.20955, 0.31496, 0.45212, 0.7239,
              0.90805, 1.21285, 1.4732, 1.67894, 1.69545, 1.69545, 1.69545, 1.69545, 1.68656,
              1.64846, 1.59512, 1.4605, 1.29794, 1.10998, 0.89662, 0.65532, 0.46355, 0.2794,
              0.10414, 0.04572])

# Combine into points for convex hull
points = np.column_stack((x, y))
hull = ConvexHull(points)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'ko', label='Measured Data')

# Draw convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], color='#008080')

# Fill convex hull
plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='#5fd7d7', alpha=0.3, label='Uncertainty Region')

# Custom labels
plt.xlabel("Boom Drive Motor [rad]")
plt.ylabel("Boom Length [m]")
plt.title("Hysteresis Loop with Convex Hull")
plt.legend()
plt.grid(True)
# plt.axis('equal')
plt.tight_layout()
plt.ylim(0, 1.8)
plt.show()