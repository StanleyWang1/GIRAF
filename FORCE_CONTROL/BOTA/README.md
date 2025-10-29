# bota_driver_py_example
This is a collection of examples of how to use the Python Driver for Bota System FT sensors.

## About
- **Author**: Raul Cruz-Oliver (support@botasys.com) 
- **Date**: 10 June 2025 
- **Place**: Zurich, Switzerland 

## Documentation

A complete documentation of the Python driver can be found in [Bota Systems FT Stack](https://code.botasys.com/en/latest/layer1/driver_py.html). We recommend reading the documentation before diving into the examples. 

This examples setup virtual environments to help getting started. However, it should be not forgot that the driver is available in PyPi and can be installed with:

```bash
pip install bota-driver
```

## Platforms
- **Linux x86_64** examples. Please refer to the folder [scripts/ubuntu/README.md](scripts/ubuntu/README.md) for detailed explanations on how to get started.

- **Linux aarch64** examples. Please refer to the folder [scripts/ubuntu/README.md](scripts/ubuntu/README.md) for detailed explanations on how to get started.

- **Windows** examples. Please refer to the folder [scripts/windows/README.md](scripts/windows/README.md) for detailed explanations on how to get started.

## Contents
### Example 1: Reading from the buffer
This example shows how to obtain the last available measurement from the buffer with non-blocking calls.

The example file can be found in **/bota_driver_py_example/examples/example1_read_from_buffer.py**.

### Example 2: Reading from the stream
This example shows how to read from the stream, returning only when a new measurement is available.

The source file can be found in **/bota_driver_py_example/examples/example2_read_from_stream.py**.

### Example 3: Reading from the buffer with printings
Similar to Example 1, but with printings to the console. 

The source file can be found in **/bota_driver_py_example/examples/example3_read_from_buffer_printings.py**.

### Example 4: Reading with from the stream with printings
Similar to Example 2, but with printings to the console. 

The source file can be found in **/bota_driver_py_example/examples/example4_read_from_stream_printings.py**.

## License
BSD3. See [LICENSE](LICENSE) file.

