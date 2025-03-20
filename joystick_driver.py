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
    def apply_deadzone(value, deadzone=0.1):
        return 0 if -deadzone < value < deadzone else value

    pygame.event.pump()
    return {
        "LX": apply_deadzone(js.get_axis(0)),
        "LY": apply_deadzone(js.get_axis(1)),
        "LT": apply_deadzone((js.get_axis(2) + 1) / 2),
        "RT": apply_deadzone((js.get_axis(5) + 1) / 2),
        "XB": js.get_button(2),
        "LB": js.get_button(4),
        "RB": js.get_button(5),
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
    