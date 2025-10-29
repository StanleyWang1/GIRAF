# bota_driver_py_example for Windows
This folder contains scripts to run the examples in a Windows machine.
## Requirements
- **OS**: Windows 10 or 11 3.12+.
- **FT Sensor**: Any Bota System FT sensor from generations Z (GEN0) or A (GENA).

## Setup Python environment 
When working with Python, it is recommended to create an isolated virtual environment. 

We have setup a bat file to easily create virtual environments. 
If you have more than one python version installed in your system, it will ask with which one you would like to proceed.

```bash
./scripts/ubuntu/01_create_venv.bat
```

If you prefer to do it manually, you can call this command:

```bash
py -<VERSION> -m venv <VERSION>

# `<VERSION>` is the version of Python you want to use (e.g. 3.12, 3.13, etc.) 
# `<VENV_DIRECTORY>` is the directory where you want to create the virtual environment.
```

## Run the examples scripts

> Before using the examples, inspect the source files and uncomment the communication interface you are using. 

### Example 1: Reading from the buffer

To run the example, we have setup a bash file, you can run it with the following command:

```bash
./scripts/ubuntu/02_run_example1_from_buffer.bat 
```

### Example 2: Reading from the stream

To build and run the example, we have setup a bash file, you can run it with the following command:

```bash
./scripts/ubuntu/02_run_example2_read_from_stream.bat 
```

### Example 3: Reading from the buffer with pritings

To build and run the example, we have setup a bash file, you can run it with the following command:

```bash
./scripts/ubuntu/02_run_example3_read_from_buffer_printings.bat 
```

### Example 4: Reading from the stream with pritings

To build and run the example, we have setup a bash file, you can run it with the following command:
```bash
./scripts/ubuntu/02_run_example4_read_from_stream_printings.bat 
```

## [HELP] Setting up communication interface in configuration file

### External dependencies for CANopen over EtherCAT
To be able to access EtherCAT devices from Windows, it is neccesary to install some version of the PCAP library. 

You can find one [here](https://npcap.com/).

### Network interface for CANopen over EtherCAT
You can follow the next steps to find the name of the interface:

1. Install Wireshark. It can be downloaded from this [link](https://www.wireshark.org/download.html).

2. Open a terminal and run the following command:

```bash
& "C:\Program Files\Wireshark\dumpcap.exe" -D
```

Example output:

```bash
1. \Device\NPF_{D97F26C7-C6A3-4CFC-98B9-D615DF8ABE69} (Bluetooth Network Connection)
2. \Device\NPF_{B5620C5D-23F6-41BA-B903-6AC4EC8A15A0} (Wi-Fi)
3. \Device\NPF_{0444E422-3353-4340-B1B8-60CD0D755514} (Ethernet)
4. \Device\NPF_{DDF4C2E1-7CC1-43C6-82BE-BA27728A2E9A} (Local Area Connection* 2)
5. \Device\NPF_{9B05D097-EC87-4F6A-880A-B49B386C65FA} (Local Area Connection* 1)
6. \Device\NPF_{A7C8887D-93EA-42B5-A98B-8F1648D616B4} (Ethernet 3)
7. \Device\NPF_{7A41DD8E-6266-4E10-83EF-BAB575953D05} (Ethernet 2)
8. \Device\NPF_Loopback (Adapter for loopback traffic capture)
```

3. Then modify the JSON configuration file as in this example:
```json
	...
        "communication_interface_name": "CANopen_over_EtherCAT_gen0",
        "communication_interface_params": {
            "network_interface": "\\Device\\NPF_{0444E422-3353-4340-B1B8-60CD0D755514}"
        },
    ...
```

IMPORTANT: The "\\"s must be duplicated to assure a proper parsing of the file.

### Serial port for Bota Binary
You can open the device manager in Windows and look for the COM port where the sensor is connected.
It is usually something like "COM3" or "COM4". 

Then modify the JSON configuration file as in this example:
```json
    ...
        "communication_interface_name": "Bota_Serial_Binary_gen0",
        "communication_interface_params": {
            "serial_port": "COM3"
        },
    ...
```


