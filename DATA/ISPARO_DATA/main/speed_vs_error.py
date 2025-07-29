import matplotlib.pyplot as plt
import numpy as np

# === Define speeds (x-axis) ===
speeds = np.array([17, 33, 50, 67, 80])

# === Error data for each boom length ===
# Format: one list per boom length (y-values)

# SQUARE ACCURACY DATA
error_24in = [3.3, 4.7, 5.9, 7.2, 8.7]
error_36in = [4.2, 6.1, 8.8, 10.7, 15.7]
error_48in = [6.0, 8.9, 11.9, 16.8, 19.2]
error_60in = [11.2, 14.5, 16.3, 20.7, 21.9]
error_72in = [13.4, 16.4, 21.9, 22.7, 29.1]

# AVG TRIAL PRECISION DATA
# error_24in = [1.6, 1.9, 2.3, 2.8, 3.1]
# error_36in = [1.7, 3.1, 5.6, 4.6, 9.9]
# error_48in = [3.2, 4.8, 7.5, 10.0, 10.5]
# error_60in = [2.9, 5.2, 6.8, 9.2, 10.5]
# error_72in = [4.5, 7.5, 8.5, 10.6, 13.9]

# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(speeds, error_24in, label='0.6 m', marker='o')
plt.plot(speeds, error_36in, label='0.9 m', marker='o')
plt.plot(speeds, error_48in, label='1.2 m', marker='o')
plt.plot(speeds, error_60in, label='1.5 m', marker='o')
plt.plot(speeds, error_72in, label='1.8 m', marker='o')

plt.xlabel("Speed (mm/s)", fontsize=14)
plt.ylabel("Error (mm)", fontsize=14)
plt.title("Speed vs Error for Different Boom Lengths", fontsize=16)
plt.grid(True)
plt.legend(title="Boom Length")
plt.tight_layout()
plt.savefig("plots/speed_vs_error_by_length.png", dpi=300)
plt.show()
