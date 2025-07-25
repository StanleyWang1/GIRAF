import pygame
import time

def joystick_connect():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    return js

def joystick_read(js):
    def apply_deadzone(value, deadzone=0.25):
        return 0 if -deadzone < value < deadzone else value

    pygame.event.pump()

    # Get D-pad (hat) input
    hat_x, hat_y = js.get_hat(0)  # 0 = first hat

    return {
        "LX": apply_deadzone(js.get_axis(0)),
        "LY": apply_deadzone(js.get_axis(1)),
        "RX": apply_deadzone(js.get_axis(3)),
        "RY": apply_deadzone(js.get_axis(4)),
        "LT": apply_deadzone((js.get_axis(2) + 1) / 2),
        "RT": apply_deadzone((js.get_axis(5) + 1) / 2),
        "AB": js.get_button(0),
        "BB": js.get_button(1),
        "XB": js.get_button(2),
        "YB": js.get_button(3),
        "LB": js.get_button(4),
        "RB": js.get_button(5),
        "DPAD_X": hat_x,  # -1 = left, 0 = neutral, 1 = right
        "DPAD_Y": hat_y  # -1 = down, 0 = neutral, 1 = up
    }

def joystick_disconnect(js):
    js.quit()

def main():
    js = joystick_connect()
    while True:
        data = joystick_read(js)
        print(data)
        time.sleep(0.005)

if __name__ == "__main__":
    main()
    
