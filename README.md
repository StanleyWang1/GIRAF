# GIRAF — Greatly Increased Reach ANYmal Functionality

This repository contains code for control, modeling, and experimentation with a novel long-reach manipulator boom design.

## Directory Overview

- **Base directory**  
  Contains core scripts and drivers for teleoperation of the boom in various modes.

- **ARCHIVE**  
  Currently inactive functions/features saved for reference.

- **CABLE_MANIP_CONTROL**  
  Autonomous cable routing trajectories using AprilTag perception (iSpaRo25).

- **CAMERA**  
  Code for interfacing with and controlling Oak-D S2 camera sensors.

- **DATA**  
  Experimental data collected and associated analysis scripts.
  - `cantilever_deflection/` - static endpoint load/deflection measurements
  - `deployer_blossoming/` - empirical measurements of deployer spool blossoming
  - `observer_damping/` - (DRAFT) experimental implementation of Luenberger observer for boom vibration stabilization
  - `OptiTrack/` - all core OptiTrack motion trajectories (for iSpaRo 25)
  - `resonance_drop_impulse/` - (ARCHIVED) free vibrations measured by dropping the boom
  - `resonance_step_velocity/` - free vibrations measured by upwards velocity impulse in boom pitch motor

- **FORCE_CONTROL**
  Code for endpoint force sensing and control
  - `BOTA/` - BOTA 6-axis FT sensor packages

- **MODELING**  
  Code for model development and simulation.
  - `boom_resonance/` — Frequency analysis of free vibrations for deployable boom.
  - `elastica/` - Elastic boom modeling using Cosserat rod (PyElastica)
  - `giraf_stance/` — Static and quasi-static stance stability analysis of full GIRAF system with extended boom.
  - `trajectory_optimization/` — (DRAFT) speed-variations of d3_dot along trajectory

- **TELEOP**  
  Code for initial teleoperation and experiments with 6 DoF boom
---

Feel free to open an issue or contact us for questions or collaboration.
