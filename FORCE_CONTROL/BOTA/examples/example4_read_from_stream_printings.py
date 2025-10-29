import os
import time
import signal
import bota_driver
import sys
from collections import deque

# Flag for graceful shutdown
stop_flag = False

def signal_handler(signum, frame):
    global stop_flag
    stop_flag = True

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#################
## CONFIG FILE ##
#################

# Bota Serial Binary Gen0
# config_path = os.path.join(project_root, "bota_driver_config", "bota_binary_gen0.json")

# Bota Serial Binary
# config_path = os.path.join(project_root, "bota_driver_config", "bota_binary.json")

# Bota Serial Socket
# config_path = os.path.join(project_root, "bota_driver_config", "bota_socket.json")

##################
## DRIVER USAGE ##
##################

try:
    # Create driver instance
    bota_ft_sensor_driver = bota_driver.BotaDriver(config_path)
    
    # Get driver version information
    print(f" >>>>>>>>>>> BotaDriver version: {bota_ft_sensor_driver.get_driver_version_string()} <<<<<<<<<<<< ")

    # Transition driver from UNCONFIGURED to INACTIVE state
    if not bota_ft_sensor_driver.configure():
        raise RuntimeError("Failed to configure driver")

    # Uncomment to tare the sensor
    if not bota_ft_sensor_driver.tare():
        raise RuntimeError("Failed to tare sensor")

    # Transition driver from INACTIVE to ACTIVE state
    if not bota_ft_sensor_driver.activate():
        raise RuntimeError("Failed to activate driver")
    
    ########################
    ## CONTROL LOOP START ##
    ########################

    # Define the example duration
    EXAMPLE_DURATION = 10.0  # seconds

    # Define the printing frequency (lower than the sensor update frequency to not overload the console)
    PRINTING_FREQUENCY = 1.0  # Hz

    start_time = time.perf_counter()  # High-resolution start time
    last_print_time = start_time  # Track when we last printed
    
    # For computing average update rate
    max_samples = 100
    reading_times = deque(maxlen=max_samples)
    last_reading_time = time.perf_counter()

    while time.perf_counter() - start_time < EXAMPLE_DURATION and not stop_flag:
        # Read frame
        bota_frame = bota_ft_sensor_driver.read_frame_blocking()
        
        # Record the time it took to read the frame
        current_reading_time = time.perf_counter()
        reading_duration = current_reading_time - last_reading_time
        last_reading_time = current_reading_time
        reading_times.append(reading_duration)
        
        # Extract the data from the bota_frame
        status = bota_frame.status
        force = bota_frame.force  
        torque = bota_frame.torque
        timestamp = bota_frame.timestamp
        temperature = bota_frame.temperature
        acceleration = bota_frame.acceleration
        angular_rate = bota_frame.angular_rate

        # Print data only at the specified printing rate
        current_time = time.perf_counter()
        if current_time - last_print_time >= 1.0/PRINTING_FREQUENCY:
            # Calculate average sensor update frequency
            if reading_times:
                total_time = sum(reading_times)
                avg_frequency = len(reading_times) / total_time if total_time > 0 else 0
            else:
                avg_frequency = 0

            print("----------------------------")
            print(f"Status: [throttled={status.throttled}, overrange={status.overrange}, invalid={status.invalid}, raw={status.raw}]")
            print(f"Force: [{force[0]}, {force[1]}, {force[2]}] N")
            print(f"Torque: [{torque[0]}, {torque[1]}, {torque[2]}] Nm")
            print(f"Acceleration: [{acceleration[0]}, {acceleration[1]}, {acceleration[2]}] m/s²")
            print(f"Angular Rate: [{angular_rate[0]}, {angular_rate[1]}, {angular_rate[2]}] rad/s")
            print(f"Temperature: {temperature} °C")
            print(f"Timestamp: {timestamp}")
            print(f"Average update rate: {avg_frequency:.2f} Hz (last {len(reading_times)} samples)")
            print("----------------------------")
            last_print_time = current_time

        #################################
        ## YOUR CONTROL LOOP CODE HERE ##
        #################################

    # Transition driver from ACTIVE to INACTIVE state
    if not bota_ft_sensor_driver.deactivate():
        raise RuntimeError("Failed to deactivate driver")
    
    # Shutdown the driver
    if not bota_ft_sensor_driver.shutdown():
        raise RuntimeError("Failed to shutdown driver")
    
    print("Completion WITHOUT errors.")

except Exception as e:
    print(f"FATAL: {e}")
    print("Completion WITH errors.")
    
finally:
    print("EXITING PROGRAM")
    sys.exit(0)

