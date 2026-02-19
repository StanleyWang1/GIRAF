import pygame
import time
import tkinter as tk
from tkinter import ttk
import threading

def joystick_connect():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    return js

def joystick_read(js):
    def apply_deadzone(value, deadzone=0.2):
        return 0 if -deadzone < value < deadzone else value

    pygame.event.pump()

    return {
        "LX": apply_deadzone(js.get_axis(0)),
        "LY": apply_deadzone(-js.get_axis(1)),
        "RX": apply_deadzone(js.get_axis(2)),
        "RY": apply_deadzone(js.get_axis(3)),
        "LT": apply_deadzone((js.get_axis(4) + 1) / 2),
        "RT": apply_deadzone((js.get_axis(5) + 1) / 2),
        "AB": js.get_button(0),
        "BB": js.get_button(1),
        "XB": js.get_button(2),
        "YB": js.get_button(3),
        "LB": js.get_button(4),
        "RB": js.get_button(5),
        "MENULEFT": js.get_button(6),  # replace with your button numbers
        "MENURIGHT": js.get_button(7)
    }

def joystick_disconnect(js):
    js.quit()

class JoystickGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Joystick Visualizer")
        self.root.geometry("800x600")
        self.root.configure(bg="#2b2b2b")
        
        self.running = True
        self.js = None
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Title
        title = tk.Label(main_frame, text="Controller Input Visualizer", 
                        font=("Arial", 18, "bold"), fg="white", bg="#2b2b2b")
        title.pack(pady=(0, 20))
        
        # Axes frame
        axes_frame = tk.LabelFrame(main_frame, text="Analog Axes", 
                                   font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b")
        axes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.axis_labels = {}
        self.axis_bars = {}
        
        axes = ["LX", "LY", "RX", "RY", "LT", "RT"]
        for i, axis in enumerate(axes):
            row = i // 2
            col = i % 2
            
            frame = tk.Frame(axes_frame, bg="#2b2b2b")
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            label = tk.Label(frame, text=f"{axis}:", font=("Arial", 10, "bold"),
                           fg="white", bg="#2b2b2b", width=4)
            label.pack(side=tk.LEFT, padx=(0, 5))
            
            canvas = tk.Canvas(frame, width=150, height=20, bg="#1a1a1a", highlightthickness=0)
            canvas.pack(side=tk.LEFT, padx=(0, 5))
            self.axis_bars[axis] = canvas
            
            value_label = tk.Label(frame, text="0.00", font=("Courier", 10),
                                  fg="#00ff00", bg="#2b2b2b", width=6)
            value_label.pack(side=tk.LEFT)
            self.axis_labels[axis] = value_label
        
        axes_frame.columnconfigure(0, weight=1)
        axes_frame.columnconfigure(1, weight=1)
        
        # Buttons frame
        buttons_frame = tk.LabelFrame(main_frame, text="Buttons", 
                                      font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b")
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        
        self.button_labels = {}
        
        buttons = ["AB", "BB", "XB", "YB", "LB", "RB", "MENULEFT", "MENURIGHT"]
        button_colors = {
            "AB": "#00ff00",  # Green (A button on Xbox)
            "BB": "#ff0000",  # Red (B button)
            "XB": "#0088ff",  # Blue (X button)
            "YB": "#ffff00",  # Yellow (Y button)
            "LB": "#888888",  # Gray
            "RB": "#888888",  # Gray
            "MENULEFT": "#ffffff",
            "MENURIGHT": "#ffffff"
        }
        
        for i, button in enumerate(buttons):
            row = i // 4
            col = i % 4
            
            color = button_colors.get(button, "#888888")
            btn_frame = tk.Frame(buttons_frame, bg="#1a1a1a", relief=tk.RAISED, borderwidth=2)
            btn_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            label = tk.Label(btn_frame, text=button, font=("Arial", 12, "bold"),
                           fg=color, bg="#1a1a1a", width=10, height=3)
            label.pack(expand=True, fill=tk.BOTH)
            self.button_labels[button] = (btn_frame, label, color)
        
        for i in range(4):
            buttons_frame.columnconfigure(i, weight=1)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Connecting to joystick...", 
                                     font=("Arial", 10), fg="#ffaa00", bg="#2b2b2b")
        self.status_label.pack(pady=(10, 0))
        
    def update_display(self, data):
        """Update the GUI with joystick data"""
        # Update axes
        for axis in ["LX", "LY", "RX", "RY", "LT", "RT"]:
            value = data.get(axis, 0)
            self.axis_labels[axis].config(text=f"{value:.2f}")
            
            # Draw bar
            canvas = self.axis_bars[axis]
            canvas.delete("all")
            
            # Draw background bar
            canvas.create_rectangle(0, 0, 150, 20, fill="#1a1a1a", outline="")
            
            # Draw value bar
            if axis in ["LT", "RT"]:  # Triggers are 0 to 1
                bar_width = int(value * 150)
                canvas.create_rectangle(0, 0, bar_width, 20, fill="#00ff00", outline="")
            else:  # Sticks are -1 to 1
                center = 75
                if value >= 0:
                    bar_start = center
                    bar_end = center + int(value * 75)
                else:
                    bar_start = center + int(value * 75)
                    bar_end = center
                canvas.create_rectangle(bar_start, 0, bar_end, 20, fill="#00aaff", outline="")
                # Draw center line
                canvas.create_line(center, 0, center, 20, fill="#666666", width=2)
        
        # Update buttons
        for button in ["AB", "BB", "XB", "YB", "LB", "RB", "MENULEFT", "MENURIGHT"]:
            pressed = data.get(button, 0)
            btn_frame, label, color = self.button_labels[button]
            
            if pressed:
                btn_frame.config(bg=color, relief=tk.SUNKEN)
                label.config(bg=color, fg="#000000")
            else:
                btn_frame.config(bg="#1a1a1a", relief=tk.RAISED)
                label.config(bg="#1a1a1a", fg=color)
    
    def joystick_loop(self):
        """Background thread to read joystick data"""
        try:
            self.js = joystick_connect()
            self.status_label.config(text="Connected!", fg="#00ff00")
            
            while self.running:
                data = joystick_read(self.js)
                self.root.after(0, self.update_display, data)
                time.sleep(0.01)  # 100 Hz update rate
                
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="#ff0000")
            print(f"Joystick error: {e}")
        finally:
            if self.js:
                joystick_disconnect(self.js)
    
    def on_closing(self):
        """Handle window close event"""
        self.running = False
        self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        # Start joystick thread
        joystick_thread = threading.Thread(target=self.joystick_loop, daemon=True)
        joystick_thread.start()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Run GUI main loop
        self.root.mainloop()

def main():
    gui = JoystickGUI()
    gui.run()

if __name__ == "__main__":
    main()
    
