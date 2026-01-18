import mujoco
import mujoco.viewer
import sys
from pathlib import Path

# Path to your model
model_path = Path("./ReachBot4X/SIM/RB4X/env_flat_w_anchors.xml")

# Resolve path and ensure it exists
model_path = model_path.resolve()
if not model_path.exists():
    print(f"ERROR: Model file not found:\n{model_path}")
    sys.exit(1)

# Load model
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Launch the viewer
print(f"Opening MuJoCo viewer with model:\n{model_path}")
mujoco.viewer.launch(model, data)
