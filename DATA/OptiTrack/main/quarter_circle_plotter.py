# plot_example.py

from interp import get_error, plot_quarter_circle_for_speed, plot_circle_with_hemisphere
import matplotlib.pyplot as plt

# === Example 1: Query the error value
length = 1.5    # meters
speed = 60       # mm/s
angle = 30       # degrees
error_val = get_error(length, speed, angle)
print(f"Interpolated error at l={length}m, s={speed}mm/s, θ={angle}° = {error_val:.2f} mm")

fig = plot_circle_with_hemisphere(50)
fig.savefig(f"quarter_circle_{length}hemisphere.png", dpi=300)
print("✅ Saved plot for hemisphere at 50 mm/s")
plt.show()  # Close to avoid memory buildup

# === Example 2: Plot quarter circles for custom speeds 17-80 mm/s
custom_speeds = [20, 50, 80]  # mm/s

for spd in custom_speeds:
    fig = plot_quarter_circle_for_speed(spd)
    fig.savefig(f"quarter_circle_error_{spd}mmps.png", dpi=300)
    print(f"✅ Saved plot for speed {spd} mm/s")
    plt.show()  # Show the plot
    # plt.close(fig)  # Close to avoid memory buildup

