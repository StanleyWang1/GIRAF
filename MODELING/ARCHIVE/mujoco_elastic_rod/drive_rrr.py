import math
import time

import mujoco
import numpy as np

# Try to open an interactive viewer if available; fall back to headless.
try:
    import mujoco.viewer as mjv
    HAS_VIEWER = True
except Exception:
    HAS_VIEWER = False

XML_PATH = "./MODELING/mujoco_elastic_rod/rrr_arm.xml"

# Sinusoid parameters for the base joint (j1)
AMP = math.radians(30.0)   # 30 deg amplitude
FREQ = 0.5                 # 0.5 Hz
KP = 50.0                  # PD gains for torque control
KD = 2.0 * math.sqrt(KP)   # critical-ish damping (tune as desired)

SIM_TIME = 10.0            # seconds

def run():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("[info] DoFs:", model.nv, "| Joints:", [model.joint(i).name for i in range(model.njnt)])
    print("[info] Timestep:", model.opt.timestep)

    # Helper to access joint idx by name
    j1_qpos_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j1")
    j1_qpos_addr = model.jnt_qposadr[j1_qpos_adr]  # index into qpos
    j1_dof_adr = model.jnt_dofadr[j1_qpos_adr]     # index into qvel/ctrl for motor actuator mapping

    # Ensure controls are zeroed
    data.ctrl[:] = 0.0

    # Real-time stepping with or without viewer
    t0 = time.time()
    sim_end = data.time + SIM_TIME

    if HAS_VIEWER:
        with mjv.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

            while data.time < sim_end:
                # Desired sinusoidal POSITION
                q_des = AMP * math.sin(2.0 * math.pi * FREQ * data.time)

                # Current state
                q = data.qpos[j1_qpos_addr]
                qd = data.qvel[j1_dof_adr]

                # PD torque to track sinusoid
                tau = KP * (q_des - q) - KD * qd

                # Apply only to motor on j1 (actuator 0 in this model)
                data.ctrl[0] = tau

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)  # Add this line to slow down the loop

            print("[done] Simulation finished.")
    else:
        print("[warn] mujoco.viewer not available; running headless.")
        nsteps = int(SIM_TIME / model.opt.timestep)
        for _ in range(nsteps):
            q_des = AMP * math.sin(2.0 * math.pi * FREQ * data.time)
            q = data.qpos[j1_qpos_addr]
            qd = data.qvel[j1_dof_adr]
            tau = KP * (q_des - q) - KD * qd
            data.ctrl[0] = tau
            mujoco.mj_step(model, data)

        print("[done] Headless simulation finished.")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("[error]", e)
