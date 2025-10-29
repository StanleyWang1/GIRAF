# bota_driver_py_example for Ubuntu
This folder contains scripts to run the examples in a Ubuntu machine, profiting from the Linux packages.

## Requirements
- **OS**: Any Ubuntu machine with python 3.8+.
- **FT Sensor**: Any Bota System FT sensor from generations Z (GEN0) or A (GENA).

## Setup Python environment 
When working with Python, it is recommended to create an isolated virtual environment. You can install the utility 'venv' with the following command:

```bash
sudo apt install python3-venv
```
We have setup a bash file to easily create virtual environments. 
If you have more than one python version installed in your system, it will ask with which one you would like to proceed.

```bash
./scripts/ubuntu/01_create_venv.sh 
```

If you prefer to do it manually, you can call this command:

```bash
python<VERSION> -m venv <VENV_DIRECTORY>

# `<VERSION>` is the version of Python you want to use (e.g. 3.8, 3.9, etc.) 
# `<VENV_DIRECTORY>` is the directory where you want to create the virtual environment.
```

**ADDITIONAL: Bota Serial Binary** 

Only if you are working with a Bota Binary communication interfaces, both Gen0 and GenA, you will need to make sure the user running the Python scripts belong to the dialout group. 

We have setup a bash file to easily add the user to the dialout group. You can run it with the following command:

```bash
sudo ./scripts/ubuntu/01b_add_user_to_dialout.sh 
```

Alternatively, you can do it manually with the following command:

```bash
sudo usermod -a -G dialout $(whoami)
```

After running the script/command, you will need to log out and log in again for the changes to take effect.

**ADDITIONAL: CANOpen over EtherCAT** 

Only if you are working with a CANopen over EtherCAT sensor, both Gen0 and GenA, you will need to setup special capabilities to access the network interface hardware for the executable, in this case the Python interpreter. The best way to do this is to subtitute the symlink to the system-wide Python interpreter in the virtual environemnt with a local copy, and then assign to this local copy of the python interpreter the correct capabilities. 

We have setup a bash file to easily do this. You can run it with the following command:

```bash
sudo ./scripts/ubuntu/01b_set_network_capabilities.sh 
```

Alternatively, if you do not mind running the python scripts with sudo rights, you can skip this step and run the scripts with sudo.


## Run the examples scripts

> Before using the examples, inspect the source files and uncomment the communication interface you are using. 

### Example 1: Reading from the buffer

To run the example, we have setup a bash file, you can run it with the following command:

```bash
./scripts/ubuntu/02_run_example1_from_buffer.sh 
```

### Example 2: Reading from the stream

To build and run the example, we have setup a bash file, you can run it with the following command:

```bash
./scripts/ubuntu/02_run_example2_read_from_stream.sh 
```

### Example 3: Reading from the buffer with pritings

To build and run the example, we have setup a bash file, you can run it with the following command:

```bash
./scripts/ubuntu/02_run_example3_read_from_buffer_printings.sh 
```

### Example 4: Reading from the stream with pritings

To build and run the example, we have setup a bash file, you can run it with the following command:
```bash
./scripts/ubuntu/02_run_example4_read_from_stream_printings.sh 
```

## [HELP] Setting up communication interface in configuration file

### Network interface for CANopen over EtherCAT
You can list all the available network interfaces with the following command:

```bash
ip link show
```
This will output a list of all the network interfaces available in your system. Look for the one that corresponds to your CANopen over EtherCAT network. If you are not sure which one the sensor is connected to, you can try to run the examples and see which one works, it will not hurt.

Then modify the JSON configuration file as in this example:
```json
    ...
        "communication_interface_name": "CANopen_over_EtherCAT_gen0",
        "communication_interface_params": {
            "network_interface": "eth0"
        },
    ...
```

### Serial port for Bota Binary
You can list all the available serial ports with the following command:
```bash
ls /dev/tty*
```
This will output a list of all the serial ports available in your system. Look for the one that corresponds to your Bota Binary sensor. Usually, it will be something like `/dev/ttyUSB0` (if using RS422 with adapater) or `/dev/ttyACM0` (if using native USB). 

Then modify the JSON configuration file as in this example:
```json
    ...
        "communication_interface_name": "Bota_Binary_gen0",
        "communication_interface_params": {
            "serial_port": "/dev/ttyUSB0"
        },
    ...
```
