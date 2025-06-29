import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from giraf_2d_scipy import solve_static_feasibility_2d_scipy

def solve_static_feasibility_2d(load, giraf_keypoints, m=50, mu=0.7):
    ## Constants
    g = 9.81  # Acceleration due to gravity (m/s^2)
    ## Unpack load and giraf keypoints
    px, py = load
    ax, ay = giraf_keypoints['A']
    bx, by = giraf_keypoints['B']
    cx, cy = giraf_keypoints['C']
    ## Decision variables
    f1x = cp.Variable()
    f1y = cp.Variable()
    f2x = cp.Variable()
    f2y = cp.Variable()
    ## Constraints
    constraints = [
        f1x + f2x + px == 0,
        f1y + f2y - m*g + py == 0,
        ax * f1y - ay * f1x + bx * f2y - by * f2x + cx * py - cy * px == 0,
        f1y >= 0,
        f2y >= 0,
        f1x <= mu * f1y,
        f1x >= -mu * f1y,
        f2x <= mu * f2y,
        f2x >= -mu * f2y
    ]
    ## Solve feasibility problem
    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve()
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        # Feasible solution found
        return float(f1x.value), float(f1y.value), float(f2x.value), float(f2y.value)
    else:
        # No feasible solution found
        return None, None, None, None

def plot_static_feasibility_region_DEPRECATED(giraf_keypoints, solve_func, m=50, mu=0.7, px_range=(-500, 500), py_range=(-500, 500), resolution=50):
    # Unpack keypoints
    ax, ay = giraf_keypoints['A']
    bx, by = giraf_keypoints['B']
    cx, cy = giraf_keypoints['C']

    # Create grid
    px_vals = np.linspace(*px_range, resolution)
    py_vals = np.linspace(*py_range, resolution)
    PX, PY = np.meshgrid(px_vals, py_vals)
    feasible = np.zeros_like(PX, dtype=bool)

    # Check feasibility at each grid point
    for i in range(PX.shape[0]):
        for j in range(PX.shape[1]):
            px = PX[i, j]
            py = PY[i, j]
            f1x, f1y, f2x, f2y = solve_func((px, py), giraf_keypoints, m=m, mu=mu)
            print(f"({i}, {j})")
            feasible[i, j] = f1x is not None

    # Plot
    plt.figure(figsize=(6, 6))
    plt.contourf(PX, PY, feasible, levels=1, colors=["white", "skyblue"])
    plt.xlabel("px [N]")
    plt.ylabel("py [N]")
    plt.title("Feasible Region for External Force (px, py)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def ray_cast_feasibility(solve_func, giraf_keypoints, m=50, mu=0.7, num_directions=100, alpha_bounds=(0, 200), tol=1e-1):
    angles = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    boundary_points = []

    for idx, theta in enumerate(angles):
        # print(f"Solving direction {idx} / {num_directions}")
        direction = np.array([np.cos(theta), np.sin(theta)])

        # Binary search for max feasible alpha
        lo, hi = alpha_bounds
        while hi - lo > tol:
            mid = (lo + hi) / 2
            px, py = mid * direction
            f1x, f1y, f2x, f2y = solve_func((px, py), giraf_keypoints, m, mu)
            if f1x is not None:
                lo = mid
            else:
                hi = mid

        boundary_points.append(lo * direction)

    return np.array(boundary_points)

def plot_feasible_boundary(giraf_keypoints, solve_func, m=50, mu=0.7, r_max=500):
    boundary = ray_cast_feasibility(solve_func, giraf_keypoints, m, mu, alpha_bounds=(0, r_max))
    X, Y = boundary[:, 0], boundary[:, 1]

    # Circle for reference boundary
    theta_circle = np.linspace(0, 2*np.pi, 300)
    circle_x = r_max * np.cos(theta_circle)
    circle_y = r_max * np.sin(theta_circle)

    plt.figure(figsize=(6,6))
    plt.fill(X, Y, color='skyblue', alpha=0.8, edgecolor='k')
    plt.plot(circle_x, circle_y, 'r--', linewidth=1, label='Search Bound (r = 200)')
    plt.plot(0, 0, 'ro', label='Origin')
    plt.xlabel("px [N]")
    plt.ylabel("py [N]")
    # plt.title("Feasible Force Boundary via Ray Casting")
    plt.axis("equal")
    plt.grid(True)
    # plt.legend()
    plt.savefig("angle_60_1m.png", dpi=300, bbox_inches='tight')  # Save before showing
    plt.show()


def plot_feasible_boundary_polar(giraf_keypoints, solve_func, m=50, mu=0.7, r_max=500):
    boundary = ray_cast_feasibility(solve_func, giraf_keypoints, m, mu)
    X, Y = boundary[:, 0], boundary[:, 1]

    # Convert to polar coordinates
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)

    # Sort by angle for cleaner fill
    sorted_indices = np.argsort(theta)
    theta = theta[sorted_indices]
    r = r[sorted_indices]

    # Plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(theta, r, 'k-', linewidth=1)
    ax.fill(theta, r, color='skyblue', alpha=0.7)
    ax.set_title("Feasible Force Boundary (Polar)", va='bottom')
    ax.set_rmax(r_max)
    ax.grid(True)
    plt.show()

def main():
    # load = [15, 0] # [px, py]
    giraf_keypoints = {'A' : [-0.3, -0.3], 'B': [0.3, -0.3], 'C': [0.5, 1.066]}

    # plot_feasible_boundary(giraf_keypoints, solve_static_feasibility_2d)
    plot_feasible_boundary(giraf_keypoints, solve_static_feasibility_2d_scipy)
    # plot_feasible_boundary_polar(giraf_keypoints, solve_static_feasibility_2d_scipy)


if __name__ == "__main__":
    main()
