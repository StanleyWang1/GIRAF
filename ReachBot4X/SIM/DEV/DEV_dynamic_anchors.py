import sys
import time
import threading
from pathlib import Path

import mujoco
import mujoco.viewer

from attachment_controller import SimulationConfig, RobotAttachmentController


MODEL_REL_PATH = "./ReachBot4X/SIM/RB4X/env_flat_w_dynamic_anchors.xml"


def load_model(path: Path):
    """Load a MuJoCo model and data, exiting with a clear error if the file is missing."""
    model_path = path.resolve()
    if not model_path.exists():
        print(f"ERROR: Model file not found:\n{model_path}")
        sys.exit(1)

    print(f"Loading MuJoCo model:\n{model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def get_equalities(model):
    """Resolve equality constraint IDs for each foot."""
    eq_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot1_anchor"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot2_anchor"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot3_anchor"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "foot4_anchor"),
    }
    for key, eq_id in eq_ids.items():
        if eq_id < 0:
            raise RuntimeError(f"Could not find equality constraint 'foot{key}_anchor'")
    return eq_ids


def get_sites_and_geoms_and_mocap(model):
    """
    Get IDs for:
      - foot tip sites
      - region sites & geoms
      - per-foot anchor mocap bodies (mocap indices)
    """
    # Feet tip sites
    foot_site_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite1"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite2"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite3"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "boomEndSite4"),
    }

    # Anchor REGIONS
    region_site_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site1"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site2"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site3"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "region_site4"),
    }
    region_geom_ids = {
        "1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region1_geom"),
        "2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region2_geom"),
        "3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region3_geom"),
        "4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "region4_geom"),
    }

    # Per-foot ANCHOR mocap bodies
    anchor_mocap_ids = {}
    for key in ("1", "2", "3", "4"):
        body_name = f"foot{key}_anchor_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"Missing body '{body_name}'")
        mocap_id = model.body_mocapid[body_id]
        if mocap_id < 0:
            raise RuntimeError(f"Body '{body_name}' is not marked as mocap='true'")
        anchor_mocap_ids[key] = mocap_id

    # Sanity checks for sites/geoms
    for label, sid in foot_site_ids.items():
        if sid < 0:
            raise RuntimeError(f"Missing foot site boomEndSite{label}")
    for label, sid in region_site_ids.items():
        if sid < 0:
            raise RuntimeError(f"Missing region site region_site{label}")
    for label, gid in region_geom_ids.items():
        if gid < 0:
            raise RuntimeError(f"Missing region geom region{label}_geom")

    return foot_site_ids, region_site_ids, region_geom_ids, anchor_mocap_ids


def set_region_color(model, geom_id, rgba):
    """Update the RGBA color of a region geom."""
    model.geom_rgba[geom_id] = rgba


def start_input_thread(
    controller: RobotAttachmentController,
    data,
    model,
    eq_ids,
    region_geom_ids,
    config: SimulationConfig,
):
    """
    Background thread:
      - "1"/"2"/"3"/"4": detach that foot from its region (if attached).
      - re-attach happens automatically when the foot exits and re-enters a region.
    """
    def input_loop():
        print("Input thread ready. Type 1/2/3/4 + Enter to detach a foot from its region.")
        print("To re-attach: move the foot out of all regions, then back into one.")
        while True:
            try:
                cmd = input().strip()
            except EOFError:
                break

            if cmd in controller.get_foot_keys():
                # Manual detach: deactivate equality and free region
                if controller.detach_foot(cmd):
                    region_key = controller.get_foot_region(cmd)
                    if region_key is not None:
                        # Deactivate equality
                        eq_id = eq_ids[cmd]
                        data.eq_active[eq_id] = 0

                        # Visual: back to inactive color
                        set_region_color(
                            model,
                            region_geom_ids[region_key],
                            config.region_inactive_rgba,
                        )
                        print(f"[MANUAL] foot {cmd} detached from region {region_key}")
                else:
                    print(f"[MANUAL] foot {cmd} is not attached to any region.")
            elif cmd.lower() in {"q", "quit", "exit"}:
                print("Input thread exiting on user request.")
                break
            else:
                print('Unrecognized command. Use 1, 2, 3, 4, or "q" to quit input thread.')

    thread = threading.Thread(target=input_loop, daemon=True)
    thread.start()
    return thread


def run_simulation(
    model,
    data,
    eq_ids,
    foot_site_ids,
    region_site_ids,
    region_geom_ids,
    anchor_mocap_ids,
    config: SimulationConfig = None,
):
    """
    Main simulation loop:

    - Starts with no attachments (all eq_active = 0)
    - Regions are visible spheres in world
    - Auto-attach:
        * Foot must be free
        * Region must be free
        * Foot must be within REGION_RADIUS
        * foot_inhibit_autoattach must be False
    - Manual detach:
        * sets eq_active = 0
        * frees region
        * sets foot_inhibit_autoattach = True
        * auto-attach resumes only after foot exits all regions
    """
    if config is None:
        config = SimulationConfig()

    # Initialize attachment controller
    controller = RobotAttachmentController(config)
    controller.reset()

    # Start with all equalities OFF, and regions in inactive color
    for eq_id in eq_ids.values():
        data.eq_active[eq_id] = 0
    for rkey, gid in region_geom_ids.items():
        set_region_color(model, gid, config.region_inactive_rgba)

    # Start input thread
    start_input_thread(
        controller,
        data,
        model,
        eq_ids,
        region_geom_ids,
        config,
    )

    print("Opening MuJoCo viewer.")
    print("All feet start FREE. Regions are translucent gray spheres.")
    print("Move a foot tip into a region to auto-attach.")
    print("Type 1/2/3/4 + Enter to detach that foot from its region.")

    sim_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Advance physics by one timestep
            mujoco.mj_step(model, data)

            # --- Auto-attach logic ---
            for fkey in controller.get_foot_keys():
                # Skip if this foot is already attached
                if controller.is_foot_attached(fkey):
                    continue

                fsid = foot_site_ids[fkey]
                foot_pos = data.site_xpos[fsid]

                # Build region position dict
                region_positions = {}
                for rkey in controller.get_region_keys():
                    rsid = region_site_ids[rkey]
                    region_positions[rkey] = data.site_xpos[rsid]

                # Update inhibit state (may re-enable auto-attach if foot exited all regions)
                controller.update_inhibit_state(fkey, foot_pos, region_positions)

                # Check if foot should auto-attach
                if not controller.should_autoattach(fkey):
                    continue

                # Find nearest free region within radius
                best_region = controller.find_best_region(foot_pos, region_positions)

                if best_region is not None:
                    # Attach foot to region
                    if controller.attach_foot_to_region(fkey, best_region):
                        # Update physics: teleport anchor mocap and activate equality
                        region_site_id = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_SITE, f"region_site{best_region}"
                        )
                        region_pos = data.site_xpos[region_site_id].copy()

                        mocap_id = anchor_mocap_ids[fkey]
                        data.mocap_pos[mocap_id] = region_pos

                        eq_id = eq_ids[fkey]
                        data.eq_active[eq_id] = 1

                        # Visual: make the region solid/active
                        set_region_color(
                            model,
                            region_geom_ids[best_region],
                            config.region_active_rgba,
                        )

                        print(f"[AUTO] foot {fkey} attached to region {best_region}")

            # Render
            viewer.sync()

            # --- Real-time factor control ---
            sim_time = data.time
            wall_time = time.time() - sim_start
            if wall_time <= 0.0:
                continue

            rtf = sim_time / wall_time
            if rtf > config.target_rtf:
                desired_wall = sim_time / config.target_rtf
                sleep_time = desired_wall - wall_time
                if sleep_time > 0.0:
                    time.sleep(min(sleep_time, 0.01))


def main():
    config = SimulationConfig(
        region_radius=0.1,
        target_rtf=1.0,  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = half-speed
    )
    
    model, data = load_model(Path(MODEL_REL_PATH))
    eq_ids = get_equalities(model)
    (
        foot_site_ids,
        region_site_ids,
        region_geom_ids,
        anchor_mocap_ids,
    ) = get_sites_and_geoms_and_mocap(model)

    run_simulation(
        model,
        data,
        eq_ids,
        foot_site_ids,
        region_site_ids,
        region_geom_ids,
        anchor_mocap_ids,
        config=config,
    )


if __name__ == "__main__":
    main()
