import argparse
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.animation import FuncAnimation

AXIS_DEADZONE = 0.15
AXIS_MAPPING = {
    "LX": (0, False),
    "LY": (1, True),
    "RX": (2, False),
    "RY": (3, True),
}
TRIGGER_MAPPING = {
    "LT": 4,
    "RT": 5,
}
BUTTON_MAPPING = {
    "AB": 0,
    "BB": 1,
    "XB": 2,
    "YB": 3,
    "LB": 4,
    "RB": 5,
    "MENULEFT": 6,
    "MENURIGHT": 7,
}

def joystick_connect():
    """Initialize pygame and connect to the first available joystick."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    return js

def joystick_read(js):
    """Read all Xbox controller inputs and apply deadzone to analog values.

    NOTE: Axis mappings verified for new red Xbox controller on Stanley's gaming laptop.
    """

    def apply_deadzone(value, deadzone=AXIS_DEADZONE):
        return 0 if -deadzone < value < deadzone else value

    pygame.event.pump()
    data = {}

    for name, (axis_index, invert) in AXIS_MAPPING.items():
        value = apply_deadzone(js.get_axis(axis_index))
        data[name] = -value if invert else value

    for name, axis_index in TRIGGER_MAPPING.items():
        data[name] = apply_deadzone((js.get_axis(axis_index) + 1) / 2)

    for name, button_index in BUTTON_MAPPING.items():
        data[name] = js.get_button(button_index)

    return data

class JoystickGUI:
    """Real-time GUI visualization of joystick state using matplotlib."""
    
    def __init__(self):
        self.js = None
        self.current_data = {}
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("Xbox Controller Visualization", fontsize=16, fontweight='bold')
        
        # Left stick (subplot 1)
        self.ax_left = self.fig.add_subplot(2, 3, 1)
        self.ax_left.set_xlim(-1.2, 1.2)
        self.ax_left.set_ylim(-1.2, 1.2)
        self.ax_left.set_aspect('equal')
        self.ax_left.set_title("Left Stick")
        self.ax_left.grid(True, alpha=0.3)
        self.ax_left.axhline(0, color='k', linewidth=0.5)
        self.ax_left.axvline(0, color='k', linewidth=0.5)
        self.circle_left = patches.Circle((0, 0), 0.1, color='blue', alpha=0.5)
        self.ax_left.add_patch(self.circle_left)
        self.text_left = self.ax_left.text(0, -1.0, "LX: 0.00, LY: 0.00", ha='center', fontsize=10)
        
        # Right stick (subplot 2)
        self.ax_right = self.fig.add_subplot(2, 3, 2)
        self.ax_right.set_xlim(-1.2, 1.2)
        self.ax_right.set_ylim(-1.2, 1.2)
        self.ax_right.set_aspect('equal')
        self.ax_right.set_title("Right Stick")
        self.ax_right.grid(True, alpha=0.3)
        self.ax_right.axhline(0, color='k', linewidth=0.5)
        self.ax_right.axvline(0, color='k', linewidth=0.5)
        self.circle_right = patches.Circle((0, 0), 0.1, color='red', alpha=0.5)
        self.ax_right.add_patch(self.circle_right)
        self.text_right = self.ax_right.text(0, -1.0, "RX: 0.00, RY: 0.00", ha='center', fontsize=10)
        
        # Triggers (subplot 3)
        self.ax_triggers = self.fig.add_subplot(2, 3, 3)
        self.ax_triggers.set_xlim(-0.5, 2.5)
        self.ax_triggers.set_ylim(-0.1, 1.2)
        self.ax_triggers.set_title("Triggers")
        self.ax_triggers.set_xticks([0.5, 1.5])
        self.ax_triggers.set_xticklabels(['LT', 'RT'])
        self.bar_lt = self.ax_triggers.bar([0.5], [0], width=0.3, color='green', alpha=0.5)
        self.bar_rt = self.ax_triggers.bar([1.5], [0], width=0.3, color='orange', alpha=0.5)
        
        # Buttons (subplot 4-6)
        self.ax_buttons = self.fig.add_subplot(2, 3, (4, 6))
        self.ax_buttons.set_xlim(-0.5, 6.5)
        self.ax_buttons.set_ylim(-0.5, 3.5)
        self.ax_buttons.axis('off')
        self.ax_buttons.set_title("Buttons (colored = pressed)", loc='left', fontsize=11)
        
        # Button layout
        self.button_rects = {}
        self.button_texts = {}
        button_layout = {
            'AB': (5, 2, 'A', 'green'),
            'BB': (5, 1, 'B', 'red'),
            'XB': (4, 2, 'X', 'blue'),
            'YB': (4, 3, 'Y', 'yellow'),
            'LB': (0.5, 2.5, 'LB', 'purple'),
            'RB': (5.5, 2.5, 'RB', 'cyan'),
            'MENULEFT': (2.5, 0.5, 'Menu', 'gray'),
            'MENURIGHT': (2.5, 3, 'Start', 'gray'),
        }
        
        for button, (x, y, label, color) in button_layout.items():
            rect = patches.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, 
                                    linewidth=2, edgecolor='black', facecolor='white')
            self.ax_buttons.add_patch(rect)
            text = self.ax_buttons.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
            self.button_rects[button] = (rect, color)
            self.button_texts[button] = text
        
        plt.tight_layout()
        
    def connect_joystick(self):
        """Connect to joystick."""
        self.js = joystick_connect()
    
    def update(self, frame):
        """Update GUI with current joystick data."""
        if self.js is None:
            return
        
        self.current_data = joystick_read(self.js)
        
        # Update left stick
        lx = self.current_data['LX']
        ly = self.current_data['LY']
        self.circle_left.set_center((lx, ly))
        self.text_left.set_text(f"LX: {lx:.2f}, LY: {ly:.2f}")
        
        # Update right stick
        rx = self.current_data['RX']
        ry = self.current_data['RY']
        self.circle_right.set_center((rx, ry))
        self.text_right.set_text(f"RX: {rx:.2f}, RY: {ry:.2f}")
        
        # Update triggers
        lt = self.current_data['LT']
        rt = self.current_data['RT']
        self.bar_lt[0].set_height(lt)
        self.bar_rt[0].set_height(rt)
        
        # Update buttons
        for button, (rect, base_color) in self.button_rects.items():
            if self.current_data[button]:
                rect.set_facecolor(base_color)
                rect.set_edgecolor('yellow')
                rect.set_linewidth(3)
            else:
                rect.set_facecolor('white')
                rect.set_edgecolor('black')
                rect.set_linewidth(2)
    
    def run(self):
        """Run the GUI."""
        self.connect_joystick()
        print("Joystick GUI running... (Close the window to exit)")
        ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)
        plt.show()


def joystick_disconnect(js):
    """Clean up pygame joystick."""
    pygame.quit()

def main():
    """Test loop: continuously read and display controller inputs."""
    parser = argparse.ArgumentParser(description="Xbox controller input reader")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    args = parser.parse_args()
    
    if args.gui:
        gui = JoystickGUI()
        gui.run()
    else:
        try:
            js = joystick_connect()
            print("Xbox controller connected. Reading inputs... (Press Ctrl+C to exit)")
            print("-" * 60)
            
            while True:
                data = joystick_read(js)
                print(data)
                time.sleep(0.01)  # 50ms update rate
        except KeyboardInterrupt:
            print("\nExiting...")
        except RuntimeError as e:
            print(f"Error: {e}")
        finally:
            joystick_disconnect(js)

if __name__ == "__main__":
    main()
