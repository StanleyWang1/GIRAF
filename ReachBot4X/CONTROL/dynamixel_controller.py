"""Low-level Dynamixel motor controller software."""

import logging

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


class DynamixelController:
    """A class to control Dynamixel motors using the Dynamixel SDK."""

    def __init__(
        self, device_name: str, baudrate: int, protocol_version: float
    ) -> None:
        """
        Initializes the DynamixelController class.

        Args:
            device_name: The name of the device port.
            baudrate: The baudrate for communication.
            protocol_version: The version of the communication protocol.
        """
        self.device_name: str = device_name
        self.baudrate: int = baudrate
        self.protocol_version: float = protocol_version

        # Initialize PortHandler and PacketHandler
        self.port_handler: PortHandler = PortHandler(self.device_name)
        self.packet_handler: PacketHandler = PacketHandler(self.protocol_version)

        self.open_port()
        self.set_baudrate()

    def open_port(self) -> None:
        """Opens the communication port."""
        if not self.port_handler.openPort():
            raise RuntimeError("Failed to open the port")

    def set_baudrate(self) -> None:
        """Sets the baudrate for the communication port."""
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError("Failed to change the baudrate")

    def write(
        self, dxl_id: int, command_type: tuple[int, int], command_value: int
    ) -> bool:
        """
        Writes a value for a specified type of command to a specific motor ID.

        Args:
            dxl_id: The ID of the Dynamixel motor.
            command_type: (address, byte_length).
            command_value: The value to be written.

        Returns:
            True if the write was successful, False otherwise.
        """
        address, length = command_type
        if length == 1:
            dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, address, command_value
            )
        elif length == 2:
            dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                self.port_handler, dxl_id, address, command_value
            )
        elif length == 4:
            dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
                self.port_handler, dxl_id, address, command_value
            )
        else:
            logging.error("Invalid byte length: %d", length)
            return False

        if dxl_comm_result != COMM_SUCCESS:
            logging.error(
                "Communication error on motor %d: %s",
                dxl_id,
                self.packet_handler.getTxRxResult(dxl_comm_result),
            )
            return False
        if dxl_error != 0:
            logging.error(
                "Packet error on motor %d: %s",
                dxl_id,
                self.packet_handler.getRxPacketError(dxl_error),
            )
            return False
        return True

    def read(self, dxl_id: int, command_type: tuple[int, int]) -> int | bool:
        """
        Reads a value for a specified type of command to a specific motor ID.

        Args:
            dxl_id: The ID of the Dynamixel motor.
            command_type: (address, byte_length).

        Returns:
            The value read from the motor, or False if there was an error.
        """
        address, length = command_type
        if length == 1:
            dxl_value, dxl_comm_result, dxl_error = self.packet_handler.read1ByteTxRx(
                self.port_handler, dxl_id, address
            )
        elif length == 2:
            dxl_value, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
                self.port_handler, dxl_id, address
            )
        elif length == 4:
            dxl_value, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
                self.port_handler, dxl_id, address
            )
        else:
            logging.error("Invalid byte length: %d", length)
            return False

        if dxl_comm_result != COMM_SUCCESS:
            logging.error(
                "Communication error on motor %d: %s",
                dxl_id,
                self.packet_handler.getTxRxResult(dxl_comm_result),
            )
            return False
        if dxl_error != 0:
            logging.error(
                "Packet error on motor %d: %s",
                dxl_id,
                self.packet_handler.getRxPacketError(dxl_error),
            )
            return False
        return dxl_value

    def close_port(self) -> None:
        """Closes the communication port."""
        self.port_handler.closePort()


def main() -> None:
    """Demo driver for Dynamixel motor."""

    controller = DynamixelController("/dev/ttyUSB0", 57600, 2.0)
    try:
        # Example: Enable torque (address and length depend on motor model)
        controller.write(1, (64, 1), 1)
        position = controller.read(1, (132, 4))
        print(f"Position: {position}")
    except Exception as e:  # TODO: catch more general exception
        print(f"Error: {e}")
    finally:
        controller.close_port()


# Example usage
if __name__ == "__main__":
    main()
