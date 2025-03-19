import pyCandle

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
        candle.md80s[i].setMaxTorque(25.0)

    candle.begin()

    return candle, motors

def motor_drive(candle, motors, roll, pitch, boom):
    candle.md80s[motors["ROLL"]].setTargetPosition(roll)
    candle.md80s[motors["PITCH"]].setTargetPosition(pitch)
    candle.md80s[motors["BOOM"]].setTargetPosition(boom)

def motor_disconnect(candle):
    candle.end()
