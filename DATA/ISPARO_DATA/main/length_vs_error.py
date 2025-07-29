import matplotlib.pyplot as plt
import numpy as np

# === Define boom lengths (x-axis) ===
lengths = np.array([0.6, 0.9, 1.2, 1.5, 1.8])

# === Error data for each speed ===
# Format: one list per speed (y-values)

# SQUARE ACCURACY DATA
error_17 = [3.3, 4.2, 6.0, 11.2, 13.4]
error_33 = [4.7, 6.1, 8.9, 14.5, 16.4]
error_50 = [5.9, 8.8, 11.9, 16.3, 21.9]
error_67 = [7.2, 10.7, 16.8, 20.7, 22.7]
error_80 = [8.7, 15.7, 19.2, 21.9, 29.1]

# AVG TRIAL PRECISION DATA
# error_17 = [1.6, 1.7, 3.2, 2.9, 4.5]
# error_33 = [1.9, 3.1, 4.8, 5.2, 7.5]
# error_50 = [2.3, 5.6, 7.5, 6.8, 8.5]
# error_67 = [2.8, 4.6, 10.0, 9.2, 10.6]
# error_80 = [3.1, 9.9, 10.5, 10.5, 13.9]


# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(lengths, error_17, label='17 mm/s', marker='o')
plt.plot(lengths, error_33, label='33 mm/s', marker='o')
plt.plot(lengths, error_50, label='50 mm/s', marker='o')
plt.plot(lengths, error_67, label='67 mm/s', marker='o')
plt.plot(lengths, error_80, label='80 mm/s', marker='o')

plt.xlabel("Boom Length (m)", fontsize=14)
plt.ylabel("Error (mm)", fontsize=14)
plt.title("Boom Length vs Error for Different Speeds", fontsize=16)
plt.grid(True)
plt.legend(title="Speed")
plt.tight_layout()
plt.savefig("plots/length_vs_error_by_speed.png", dpi=300)
plt.show()
