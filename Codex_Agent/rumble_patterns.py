"""Advanced rumble pattern examples using XInput."""
import ctypes
import math
import time
from ctypes import wintypes

# Load XInput DLL
try:
    xinput = ctypes.windll.xinput1_4
except:
    try:
        xinput = ctypes.windll.xinput1_3
    except:
        xinput = ctypes.windll.xinput9_1_0

class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [("wLeftMotorSpeed", wintypes.WORD), ("wRightMotorSpeed", wintypes.WORD)]

def set_vibration(controller_id, left, right):
    """Set vibration (0-65535 for each motor)."""
    vibration = XINPUT_VIBRATION(int(left), int(right))
    xinput.XInputSetState(controller_id, ctypes.byref(vibration))

def fade_in_out(duration=2.0, controller_id=0):
    """Fade rumble in and out smoothly."""
    print("Fade in/out pattern...")
    steps = 50
    dt = duration / steps
    
    for i in range(steps):
        # Triangle wave: 0 -> 1 -> 0
        if i < steps // 2:
            strength = i / (steps // 2)
        else:
            strength = (steps - i) / (steps // 2)
        
        value = int(strength * 65535)
        set_vibration(controller_id, value, value)
        time.sleep(dt)
    
    set_vibration(controller_id, 0, 0)

def pulse_wave(frequency=2.0, duration=3.0, controller_id=0):
    """Pulsating rumble at specified frequency (Hz)."""
    print(f"Pulsing at {frequency} Hz...")
    dt = 0.02  # 50 Hz update rate
    elapsed = 0
    
    while elapsed < duration:
        # Sine wave modulation
        strength = (math.sin(2 * math.pi * frequency * elapsed) + 1) / 2
        value = int(strength * 65535)
        set_vibration(controller_id, value, value)
        time.sleep(dt)
        elapsed += dt
    
    set_vibration(controller_id, 0, 0)

def alternating_motors(duration=2.0, controller_id=0):
    """Alternate between left and right motors."""
    print("Alternating motors...")
    steps = 40
    dt = duration / steps
    
    for i in range(steps):
        phase = (i / steps) * 2 * math.pi
        left = int(((math.sin(phase) + 1) / 2) * 65535)
        right = int(((math.cos(phase) + 1) / 2) * 65535)
        set_vibration(controller_id, left, right)
        time.sleep(dt)
    
    set_vibration(controller_id, 0, 0)

def impact_simulation(intensity=1.0, controller_id=0):
    """Simulate a sharp impact with decay."""
    print("Impact simulation...")
    steps = 30
    dt = 0.02
    
    for i in range(steps):
        # Exponential decay
        strength = intensity * math.exp(-i / 5)
        value = int(min(strength, 1.0) * 65535)
        set_vibration(controller_id, value, value)
        time.sleep(dt)
    
    set_vibration(controller_id, 0, 0)

def heartbeat(bpm=60, duration=5.0, controller_id=0):
    """Simulate heartbeat pattern."""
    print(f"Heartbeat at {bpm} BPM...")
    beat_interval = 60.0 / bpm
    elapsed = 0
    
    while elapsed < duration:
        # First beat (lub)
        set_vibration(controller_id, 40000, 50000)
        time.sleep(0.1)
        set_vibration(controller_id, 0, 0)
        time.sleep(0.1)
        
        # Second beat (dub)
        set_vibration(controller_id, 35000, 40000)
        time.sleep(0.08)
        set_vibration(controller_id, 0, 0)
        
        # Wait for next heartbeat
        time.sleep(beat_interval - 0.28)
        elapsed += beat_interval
    
    set_vibration(controller_id, 0, 0)

def ramp_test(controller_id=0):
    """Test full range of intensities."""
    print("Testing intensity range...")
    steps = 20
    
    for i in range(steps + 1):
        strength = i / steps
        value = int(strength * 65535)
        print(f"  {int(strength * 100)}%")
        set_vibration(controller_id, value, value)
        time.sleep(0.3)
    
    set_vibration(controller_id, 0, 0)

def main():
    """Run demo patterns."""
    controller_id = 0
    
    print("=== Rumble Pattern Demos ===\n")
    
    # Test 1: Ramp
    ramp_test(controller_id)
    time.sleep(1)
    
    # Test 2: Fade
    fade_in_out(duration=2.0, controller_id=controller_id)
    time.sleep(1)
    
    # Test 3: Pulse
    pulse_wave(frequency=4.0, duration=2.0, controller_id=controller_id)
    time.sleep(1)
    
    # Test 4: Alternating
    alternating_motors(duration=2.0, controller_id=controller_id)
    time.sleep(1)
    
    # Test 5: Impact
    impact_simulation(intensity=1.0, controller_id=controller_id)
    time.sleep(1)
    
    # Test 6: Heartbeat
    heartbeat(bpm=80, duration=4.0, controller_id=controller_id)
    
    print("\nAll patterns complete!")

if __name__ == "__main__":
    main()
