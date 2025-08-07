import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_trajectory_csv(filename):
    return np.loadtxt(filename, delimiter=',')

def compute_nearest_distances(goal_traj, actual_traj):
    tree = cKDTree(goal_traj)
    distances, _ = tree.query(actual_traj)
    return distances

def plot_overlaid_error_histograms(goal_files, robot_files, labels, colors):
    plt.figure(figsize=(10, 6))

    for goal_file, robot_file, label, color in zip(goal_files, robot_files, labels, colors):
        goal = load_trajectory_csv(goal_file)
        robot = load_trajectory_csv(robot_file)
        distances = compute_nearest_distances(goal, robot)

        mean_err = np.mean(distances)
        perc_95 = np.percentile(distances, 95)

        # Plot the KDE curve
        sns.kdeplot(distances, label=label, fill=True, linewidth=2, alpha=0.3, color=color)

        # Optional: mark statistics
        # plt.axvline(mean_err, color=color, linestyle='--', linewidth=2)
        # plt.axvline(perc_95, color=color, linestyle='-', linewidth=2)

    plt.xlabel("Distance to Nearest Goal Point [m]")
    plt.ylabel("Density")
    plt.title("LENGTH VS AVG")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(0, 0.05)
    plt.xlabel("Distance to Nearest Goal Point [m]", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title("LENGTH VS SQUARE", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.show()
    

if __name__ == "__main__":
    goal_csvs = [
        "data/goals/OPTIMAL_SQUARE.csv",
        "data/goals/OPTIMAL_SQUARE.csv",
        "data/goals/OPTIMAL_SQUARE.csv"
    ]
    robot_csvs = [
        "data/processed/SHORT_CONVERTED.csv",
        "data/processed/MEDIUM_CONVERTED.csv",
        "data/processed/LONG_CONVERTED.csv"
    ]
    labels = ["SHORT", "MEDIUM", "LONG"]
    colors = ["orange", "lightseagreen", "mediumslateblue"]

    plot_overlaid_error_histograms(goal_csvs, robot_csvs, labels, colors)
