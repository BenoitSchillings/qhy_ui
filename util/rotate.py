import serial
import time
import sys

class RotatorController:
    """
    A class to control the WandererRotator Mini camera rotator.
    """
    STEPS_PER_DEGREE = 1142
    # HANDSHAKE_COMMAND = "1500001" # Handshake removed
    SERIAL_PORT = '/dev/ttyUSB0' # Default serial port
    BAUD_RATE = 19200
    TIMEOUT = 15 # seconds for serial read timeout

    def __init__(self, port=None):
        """
        Initializes the RotatorController.

        Args:
            port (str, optional): The serial port to connect to.
                                    Defaults to '/dev/ttyUSB0'.
        """
        self.port = port if port else self.SERIAL_PORT
        self.serial_connection = None
        print(f"RotatorController initialized for port: {self.port}, baud rate: {self.BAUD_RATE}")

    def connect(self):
        """
        Establishes a serial connection with the rotator.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            self.serial_connection = serial.Serial(
                self.port,
                self.BAUD_RATE,
                timeout=self.TIMEOUT
            )
            print(f"Successfully connected to {self.port}")
            time.sleep(2) # Wait for the connection to stabilize
            return True
        except serial.SerialException as e:
            print(f"Error: Could not connect to serial port {self.port}. Details: {e}")
            self.serial_connection = None
            return False

    def disconnect(self):
        """
        Closes the serial connection.
        """
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print(f"Disconnected from {self.port}")
        else:
            print("No active connection to disconnect.")

    def send_command(self, command_str):
        """
        Sends a command string to the rotator.

        Args:
            command_str (str): The command to send.

        Returns:
            bool: True if command was sent successfully, False otherwise.
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("Error: Not connected to rotator. Please connect first.")
            return False
        try:
            self.serial_connection.write(command_str.encode('ascii'))
            print(f"Sent command: {command_str}")
            time.sleep(0.1) # Small delay after sending
            return True
        except serial.SerialException as e:
            print(f"Error sending command: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while sending command: {e}")
            return False

    def read_feedback(self):
        """
        Reads feedback from the rotator.

        Returns:
            str: The feedback string from the rotator, or None if an error occurs or timeout.
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("Error: Not connected to rotator.")
            return None
        try:
            # Wait for feedback as rotator sends it after action.
            # readline() will wait until a newline is received or timeout occurs.
            print("Waiting for feedback...")
            feedback_bytes = self.serial_connection.readline()
            if feedback_bytes:
                feedback_str = feedback_bytes.decode('ascii').strip()
                print(f"Received raw feedback: '{feedback_str}'")
                return feedback_str
            else:
                print("No feedback received (timeout).")
                return None
        except serial.SerialTimeoutException:
            print("Timeout waiting for feedback.")
            return None
        except serial.SerialException as e:
            print(f"Error reading feedback: {e}")
            return None
        except UnicodeDecodeError as e:
            print(f"Error decoding feedback (likely not ASCII): {feedback_bytes}. Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while reading feedback: {e}")
            return None

    # handshake method removed as per request

    def attempt_rotation_and_check_feedback(self, degrees):
        """
        Attempts to rotate the camera and checks if any feedback is received.

        Args:
            degrees (float): The number of degrees to rotate.

        Returns:
            bool: True if any feedback was received after sending the command, False otherwise.
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("Error: Not connected to rotator. Please connect first.")
            return False

        steps_to_move = int(degrees * self.STEPS_PER_DEGREE)
        command_to_send = str(steps_to_move)

        print(f"Preparing to rotate by {degrees} degrees ({steps_to_move} steps).")
        if not self.send_command(command_to_send):
            return False # Failed to send command

        print("Rotation command sent. Checking for any feedback...")
        feedback = self.read_feedback() # This will use the serial timeout

        if feedback is not None:
            print(f"Successfully received feedback: '{feedback}'")
            # Further parsing of feedback can be done here if needed in the future
            # For now, just confirming something came back is enough.
            return True
        else:
            print("Did not receive any feedback after sending rotation command.")
            return False

def main():
    """
    Main function to control the rotator with a command-line argument for rotation.
    """
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <degrees>")
        print("Example: python your_script_name.py 10.5")
        sys.exit(1)

    try:
        angle = float(sys.argv[1])
    except ValueError:
        print("Error: Rotation angle must be a number.")
        sys.exit(1)

    # You might need to change '/dev/ttyUSB0' to the correct port for your system
    # e.g., 'COM3' on Windows.
    rotator = RotatorController() # Default port is '/dev/ttyUSB0'
    # rotator = RotatorController(port='COM3') # Example for Windows

    if not rotator.connect():
        print("Exiting due to connection failure.")
        return

    # No handshake performed

    try:
        print(f"\nAttempting to rotate by {angle} degrees...")
        feedback_received = rotator.attempt_rotation_and_check_feedback(angle)

        if feedback_received:
            print("Confirmation: Feedback was received from the rotator.")
        else:
            print("Confirmation: No feedback was received from the rotator after the command.")

    except Exception as e:
        print(f"An unexpected error occurred during rotation: {e}")
    finally:
        rotator.disconnect()
        print("Program finished.")

if __name__ == "__main__":
    main()
