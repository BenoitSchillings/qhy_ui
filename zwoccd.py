from ctypes import *
import numpy as np
import time
import zwoasi as asi
import sys

class zwoasi_wrapper():
    def __init__(self, cam_name, live_mode=False):
        self.sdk = asi
        self.live = live_mode
        self.mode = 1 if live_mode else 0
        self.bpp = c_uint(16)  # Default to 16-bit mode
        self.exposureMS = 100  # 100ms default exposure
        self.cam = None
        self.start_time = 0

        self.connect(self.mode, cam_name)
        self.ClearBuffers()

    # ... [previous methods remain unchanged] ...

    def SetFanSpeed(self, speed):
        """
        Set the fan speed.
        :param speed: Fan speed (0-100). 0 is off, 100 is full speed.
        """
        if asi.ASI_FAN_ON in self.cam.get_controls():
            self.cam.set_control_value(asi.ASI_FAN_ON, speed)
            print(f"Set fan speed to {speed}")
        else:
            print("Camera does not support fan control")

    def GetFanSpeed(self):
        """
        Get the current fan speed.
        :return: Current fan speed (0-100) or None if not supported.
        """
        if asi.ASI_FAN_ON in self.cam.get_controls():
            return self.cam.get_control_value(asi.ASI_FAN_ON)[0]
        else:
            print("Camera does not support fan control")
            return None

    def SetTemperature(self, temp):
        if asi.ASI_TARGET_TEMP in self.cam.get_controls():
            self.cam.set_control_value(asi.ASI_TARGET_TEMP, temp)
            # Automatically turn on the fan when cooling is enabled
            if asi.ASI_FAN_ON in self.cam.get_controls():
                self.cam.set_control_value(asi.ASI_FAN_ON, 100)  # Set fan to full speed
            print(f"Set target temperature to {temp}°C and turned on fan")
        else:
            print("Camera does not support temperature control")

    def GetTemperature(self):
        if asi.ASI_TEMPERATURE in self.cam.get_controls():
            return self.cam.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
        else:
            return None

    # ... [rest of the methods remain unchanged] ...

    def close(self):
        if self.cam:
            # Turn off the fan before closing the camera
            if asi.ASI_FAN_ON in self.cam.get_controls():
                self.cam.set_control_value(asi.ASI_FAN_ON, 0)
            self.cam.close()
        asi.close()  # Close the SDK

# Example usage:
if __name__ == "__main__":
    cam_name = "ASI224"  # Replace with your camera model
    camera = zwoasi_wrapper(cam_name, live_mode=False)
    
    camera.SetExposure(100)  # 100ms exposure
    camera.SetGain(10)
    
    # Set and check fan speed
    camera.SetFanSpeed(50)  # Set fan to 50% speed
    print(f"Current fan speed: {camera.GetFanSpeed()}%")
    
    # Set target temperature (this will also turn on the fan)
    camera.SetTemperature(0)  # Set target temperature to 0°C
    
    camera.Begin()
    
    for i in range(10):  # Capture 10 frames
        frame = camera.GetSingleFrame()
        if frame is not None:
            print(f"Captured frame {i+1}, shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"Current temperature: {camera.GetTemperature()}°C")
        time.sleep(1)
    
    camera.close()

