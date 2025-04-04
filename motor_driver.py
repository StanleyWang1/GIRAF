import pyCandle
import warnings

def motor_connect():
    kp = 1000
    kd = 50
    max_torque = 25.0

    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)

    ids = {"ROLL": 11, "PITCH": 12, "BOOM": 13}
    for motor_id in ids.values():
        candle.addMd80(motor_id)
    motors = {name: i for i, name in enumerate(ids.keys())}
    
    for md in candle.md80s:
        candle.controlMd80SetEncoderZero(md)
        candle.controlMd80Mode(md, pyCandle.IMPEDANCE)
        candle.controlMd80Enable(md, True)

    for i in motors.values():
        candle.md80s[i].setImpedanceControllerParams(1000, 50)
        candle.md80s[i].setMaxTorque(10.0)
    
    candle.begin()
    
    return candle, motors

def motor_status(candle, motors):
    error_flags = {
        0: "Main encoder error",
        1: "Output encoder error",
        2: "Calibration encoder error",
        3: "MOSFET bridge error",
        4: "Hardware error",
        5: "Communication error",
        6: "Motion error"
    }

    for name, index in motors.items():
        motor = candle.md80s[index]
        status = motor.getQuickStatus()
        for bit, message in error_flags.items():
            if status & (1 << bit):
                warnings.warn(f"Motor {name} (ID {motor.getId()}) {message}.", RuntimeWarning)
        if status & (1 << 15):
            print(f"Motor {name} (ID {motor.getId()}) has reached its target position or velocity.")

# def motor_status(candle, motors):
#     for name, index in motors.items():
#         motor = candle.md80s[index]
#         status = motor.getQuickStatus()
#         print(status)

def motor_drive(candle, motors, roll, pitch, boom):
    candle.md80s[motors["ROLL"]].setTargetPosition(roll)
    candle.md80s[motors["PITCH"]].setTargetPosition(pitch)
    candle.md80s[motors["BOOM"]].setTargetPosition(boom)

def motor_disconnect(candle):
    candle.end()
