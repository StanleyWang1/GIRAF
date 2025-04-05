import pandas as pd
import matplotlib.pyplot as plt

# Constants
rho = 0.188  # Linear density of boom in kg/mm
g = 9.81  # Acceleration due to gravity in m/s^2
endcap_weight = 0.25  # Endcap weight in N

# Load the CSV data
data = pd.read_csv('./DATA/cantilever_deflection/deflection_test_040425.csv')

# Convert units
boom_length = data['Boom Extension (mm)'] / 1000  # Convert mm to meters
applied_force = data['Applied Force (N)']
deflection = data['Deflection (mm)'] / 1000  # Convert mm to meters

# Calculate bending terms without endcap weight
endpoint_bending_term_no_endcap = (applied_force * boom_length**3) / 3
weight_bending_term = (rho * g * boom_length**4) / 8
total_bending_no_endcap = endpoint_bending_term_no_endcap + weight_bending_term

# Calculate bending terms with endcap weight
endpoint_bending_term_with_endcap = ((applied_force + endcap_weight) * boom_length**3) / 3
total_bending_with_endcap = endpoint_bending_term_with_endcap + weight_bending_term

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(total_bending_no_endcap, deflection, 'o', label='Endcap Weight = 0 N', color='blue')
plt.plot(total_bending_with_endcap, deflection, 'o', label='Endcap Weight = 0.25 N', color='red')
plt.xlabel('Force × Length³ (N·m³)')
plt.ylabel('Deflection (m)')
plt.title('Deflection vs. Force × Length³')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
