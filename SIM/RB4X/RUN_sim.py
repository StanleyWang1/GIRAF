from pathlib import Path

from robot import RB4X

def main():
    # Initialize the RB4X robot with the path to the Mujoco model
    model_path = Path("./SIM/RB4X/models/vertical_climbing.xml")
    robot = RB4X(model_path)
    print(robot.anchor_manager['feet'])

if __name__ == "__main__":
    main()
    