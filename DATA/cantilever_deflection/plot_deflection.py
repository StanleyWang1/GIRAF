import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['text.usetex'] = False

# Linear density of boom
rho = 0.188 # [kg/m]
g = 9.81 # [m/s^2]
endcap_weight = 0.25 # [N] - approximation of endcap weight (25 g))

# Load the CSV
data = pd.read_csv('./DATA/cantilever_deflection/deflection_test_040425.csv')
# Parse data into variables
boom_length = data['Boom Extension (mm)'] / 1000 # [m]
endpoint_force = data['Applied Force (N)'] # [N]
deflection = data['Deflection (mm)'] / 1000 # [m]

# Calculate bending terms
endpoint_bending_term = (endpoint_force * boom_length**3) / 3 # bending induced by applied endpoint load
weight_bending_term = (rho * g * boom_length**4) / 8 # bending induced by weight of boom
correction_factor = (endcap_weight * boom_length**3) / 3 # correction factor from mass of endcap

# Calculate bending with correction factor from endcap
total_bending = endpoint_bending_term + weight_bending_term
total_bending_corrected = total_bending + correction_factor

# Calculate flexural rigidity
slope, _, _, _ = np.linalg.lstsq(total_bending_corrected.to_numpy()[:, np.newaxis], deflection.to_numpy(), rcond=None)
flexural_rigidity = 1/slope # [N m^2]
print(str(flexural_rigidity) + ' Nm^2')
# Plot Deflection vs Force * Length^3
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(total_bending_corrected, deflection, 'o', label='experimental data')
ax.plot(total_bending_corrected, total_bending_corrected / flexural_rigidity, '--r', label='flexural rigidity fit line')


ax.set_xlabel(r'$\frac{1}{3}FL^3 + \frac{1}{8}\rho g L^4$ [Nm$^3$]', fontname='Times New Roman', fontsize=14)
ax.set_ylabel('Deflection [m]', fontname='Times New Roman', fontsize=14)
# ax.set_title('Deflection vs Bending Term')
ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='0.8')
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='0.9')
legend = ax.legend()
# legend.set_title('Legend Title', prop={'family': 'Times New Roman', 'size': 14})
for text in legend.get_texts():
    text.set_fontfamily('Times New Roman')
    text.set_fontsize(14)
plt.tight_layout()

# plt.savefig('./DATA/cantilever_deflection/flexural_rigidity_fit.png', format='png')
plt.show()
