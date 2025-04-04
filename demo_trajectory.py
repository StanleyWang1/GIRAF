import pyCandle
import time
import math
import sys
 
## --------------------------------------------------
# Demo RRP Manipulator Trajectory
# Uses impedance mode to drive all joints with a sinusoidal trajectory
# Start robot with boom fully retracted and pointed ~45deg upwards
## --------------------------------------------------

# Create CANdle object and set FDCAN baudrate to 1Mbps
candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
 
## --------------------------------------------------
# INITIALIZE MOTORS
## --------------------------------------------------
# Define Motor ID
ROLL_ID = 11
PITCH_ID = 12
BOOM_ID = 13
# Add motors to MD80 controller
candle.addMd80(ROLL_ID)
candle.addMd80(PITCH_ID)
candle.addMd80(BOOM_ID)
# Motor index in MD80
ROLL = 0
PITCH = 1
BOOM = 2


## --------------------------------------------------
# CONTROL PARAMETERS
## --------------------------------------------------
# Set kp and kd gains
kp = 1000
kd = 50
max_torque = 25.0
# zero encoders and enable impedance mode
for md in candle.md80s:
    candle.controlMd80SetEncoderZero(md)                #  Reset encoder at current position
    candle.controlMd80Mode(md, pyCandle.IMPEDANCE)      # Set mode to impedance control
    candle.controlMd80Enable(md, True)                  # Enable the drive
# set impedance controller gains
candle.md80s[ROLL].setImpedanceControllerParams(kp, kd)
candle.md80s[PITCH].setImpedanceControllerParams(kp, kd)
candle.md80s[BOOM].setImpedanceControllerParams(kp, kd)
# set maximum motor output torque
candle.md80s[ROLL].setMaxTorque(max_torque)
candle.md80s[PITCH].setMaxTorque(max_torque)
candle.md80s[BOOM].setMaxTorque(max_torque)


## --------------------------------------------------
# CONTROL LOOP
## --------------------------------------------------
# begin update loop (it starts in the background)
candle.begin()
 
t = 0.0
dt = 0.005
 
for i in range(5000):
    candle.md80s[ROLL].setTargetPosition(0.6*math.sin(t))
    candle.md80s[PITCH].setTargetPosition(0.5*math.sin(t))
    candle.md80s[BOOM].setTargetPosition(-5 + 5*math.cos(t))
 
    # for md in candle.md80s:
    #     md.setTargetPosition(math.sin(t) * 2.0)
    # candle.md80s[0].setTargetPosition(5.0)
 
    # EXTENSION
    # POSITIVE : IN
    # NEGATIVE : OUT
    # candle.md80s[0].setTargetPosition(15*math.cos(t/3) - 15)
 
    # candle.md80s[0].setTargetPosition(1)
 
    # PITCH
    # POSITIVE : UP - standard config
    # NEGATIVE : DOWN - standard config
    # candle.md80s[1].setTargetPosition(0)
 
    # candle.md80s[1].setTargetPosition(max(-0.001*i, -1))
    # candle.md80s[1].setTargetPosition(min(0.001*i, 0.7))
    # candle.md80s[1].setTargetPosition(-0.5*math.cos(t) + 0.5)
 
 
    # candle.md80s[0].setTargetVelocity(0.1)
 
    # display position, velocity, and torque
    pos = str('%.2f' % candle.md80s[ROLL].getPosition())
    vel = str('%.2f' % candle.md80s[ROLL].getVelocity())
    torque = str('%.2f' % candle.md80s[ROLL].getTorque())
 
    print(" Position: " + pos + " Velocity: " + vel + " Torque: " + torque)
    t = t + dt
    time.sleep(0.005) # Add some delay
 
# Time for joints to stabilize to starting reference (zero)
for i in range(500):
    candle.md80s[ROLL].setTargetPosition(0)
    candle.md80s[PITCH].setTargetPosition(0)
    candle.md80s[BOOM].setTargetPosition(0)
    time.sleep(0.005) # Add some delay

print("DONE")

# Close the update loop
candle.end()
 
sys.exit("EXIT SUCCESS")
