import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

# Given system parameters
wn = 12.763  # natural frequency [rad/s]
z = 0.0194   # damping ratio
b = 1.0      # input coupling

# Define system matrices
A = np.array([[0, 1],
              [-wn**2, -2*z*wn]])
C = np.array([[-wn**2, -2*z*wn]])

# (1) Compute system poles
system_poles = np.linalg.eigvals(A)
print("System poles:")
print(system_poles)

# (2) Design observer poles 5x and 6x faster
real_parts = np.real(system_poles)
observer_poles = np.array([5 * real_parts[0], 6 * real_parts[1]])
print("\nDesired observer poles:")
print(observer_poles)

# Compute observer gain
L = place_poles(A.T, C.T, observer_poles).gain_matrix.T
observer_poles_actual = np.linalg.eigvals(A - L @ C)
print("\nActual observer poles (from A - LC):")
print(observer_poles_actual)

print("\nObserver gain matrix L:")
print(L)

# (3) Plot poles
plt.figure(figsize=(8, 6))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.plot(np.real(system_poles), np.imag(system_poles), 'bo', label='System Poles')
plt.plot(np.real(observer_poles_actual), np.imag(observer_poles_actual), 'rx', label='Observer Poles')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.title('System vs Observer Poles')
plt.grid(True)
# plt.axis('equal')
plt.legend()
plt.show()
