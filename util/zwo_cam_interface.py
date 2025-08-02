from ctypes import *
import numpy as np
import time
import zwoasi as asi
import sys

class zwoasi_wrapper():
    def __init__(self, temp, exp, gain, crop, cam_name, binning=1, live=True):
        self.sdk = asi
        self.live = live
        self.mode = 1 if live else 0
        self.bpp = 16  # Default to 16-bit mode
        self.exposureMS = exp*1000
        self.gain = gain
        self.temp = temp
        self.crop = crop
        self.binning = binning
        self.cam = None
        self.dt = exp
        self.start_time = 0
        self.cooling_target = temp
        self.cooler_supported = True
        self.connect(self.mode, cam_name)
        self.ClearBuffers()
        self.initialize_cooling()
        # Set initial parameters
        self.SetTemperature(temp)
        self.SetExposure(self.exposureMS)
        self.SetGain(gain)
        self.SetBinning(self.binning)
        self.SetUSB(70)
        self.SetFanSpeed(60)
        self.SetSpeed()

        # Handle crop
        if crop is not None and isinstance(crop, (int, float)):
            self.SetCrop(crop)

    def connect(self, mode, cam_name):
        try:
            self.sdk.init()
        except Exception as e:
            print(f"Failed to initialize ZWO ASI SDK: {e}")
            sys.exit(1)

        num_cameras = self.sdk.get_num_cameras()
        if num_cameras == 0:
            print("No cameras found")
            sys.exit(1)

        for i in range(num_cameras):
            self.cam_info = self.sdk._get_camera_property(i)
            if cam_name in self.cam_info['Name']:
                self.cam = self.sdk.Camera(i)
                print(f"FOUND: {self.cam_info['Name']}")
                break

        if self.cam is None:
            print(f"Camera {cam_name} not found")
            sys.exit(1)

        print(f"Open camera: {self.cam_info['Name']}")
        #self.cam.set_control_value(asi.ASI_BIN, 3)

        self.cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 70)
        
        if self.live:
            self.cam.start_video_capture()

        self.image_size_x = self.cam_info['MaxWidth']
        self.image_size_y = self.cam_info['MaxHeight']
        self.sizex = self.image_size_x
        self.sizey = self.image_size_y
        self.roi_w = self.sizex
        self.roi_h = self.sizey

        self.imgdata = np.zeros((self.sizey, self.sizex), dtype=np.uint16)
        self.SetBit(self.bpp)

    def GetName(self):
        return self.cam_info['Name']

    def GetSize(self):
        return self.sizex, self.sizey

    def SetStreamMode(self, mode):
        self.mode = mode
        if mode == 1:
            self.cam.start_video_capture()
        else:
            self.cam.stop_video_capture()

    def size_x(self):
        return self.sizex

    def size_y(self):
        return self.sizey

    def start(self):
        if self.live:
            self.cam.start_video_capture()
        else:
            self.start_time = time.time()
        print(f"Camera started in {'live' if self.live else 'single frame'} mode")

    def SetExposure(self, exposureMS):
        self.exposureMS = exposureMS
        self.dt = exposureMS / 1000.0
        self.cam.set_control_value(asi.ASI_EXPOSURE, int(exposureMS * 1000))
        print(f"Set exposure to {self.cam.get_control_value(asi.ASI_EXPOSURE)[0] / 1000}")

    def get_exposure(self):
        return self.dt

    def SetGain(self, gain):
        self.gain = gain
        self.cam.set_control_value(asi.ASI_GAIN, int(gain))
        print(f"Set gain to {self.cam.get_control_value(asi.ASI_GAIN)[0]}")

    def SetBinning(self, binning):
        self.binning = binning
        #self.cam.set_control_value(asi.ASI_BIN, int(binning))
        #print(f"Set binning to {self.cam.get_control_value(asi.ASI_BIN)[0]}")

    def SetOffset(self, offset):
        self.cam.set_control_value(asi.ASI_OFFSET, int(offset))



    def SetSpeed(self):
        self.cam.set_control_value(asi.ASI_HIGH_SPEED_MODE, int(0))


    def SetUSB(self, bandwidth):
        self.cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, int(bandwidth))
        print(f"Set USB bandwidth to {self.cam.get_control_value(asi.ASI_BANDWIDTHOVERLOAD)[0]}")

    def SetBit(self, bpp):
        self.bpp = bpp
        if bpp == 8:
            self.cam.set_image_type(asi.ASI_IMG_RAW8)
        elif bpp == 16:
            self.cam.set_image_type(asi.ASI_IMG_RAW16)
        height = self.roi_h if hasattr(self, 'roi_h') else self.sizey
        width = self.roi_w if hasattr(self, 'roi_w') else self.sizex
        self.imgdata = np.zeros((height, width), dtype=np.uint8 if bpp == 8 else np.uint16)

    def SetFanSpeed(self, speed):
        if asi.ASI_FAN_ON in self.cam.get_controls():
            self.cam.set_control_value(asi.ASI_FAN_ON, speed)
            print(f"Set fan speed to {speed}")
        else:
            print("Camera does not support fan control")

    def GetFanSpeed(self):
        if asi.ASI_FAN_ON in self.cam.get_controls():
            return self.cam.get_control_value(asi.ASI_FAN_ON)[0]
        else:
            print("Camera does not support fan control")
            return None

    def initialize_cooling(self):
        try:
            if asi.ASI_COOLER_ON in self.cam.get_controls():
                self.cooler_supported = True
                self.SetCooling(True)
                self.SetTemperature(self.cooling_target)
            else:
                self.cooler_supported = False
                print("Camera does not support cooling")
                self.SetCooling(True)
                self.SetTemperature(self.cooling_target)
                self.cooler_supported = True
        except:
            print("not cool")

    def SetCooling(self, on):
        try:
            if self.cooler_supported:
                self.cam.set_control_value(asi.ASI_COOLER_ON, 1 if on else 0)
                self.cooler_on = on
                print(f"Cooler turned {'on' if on else 'off'}")
            else:
                print("Cooling not supported")
        except:
            print("not cool")

    def SetTemperature(self, temp):
        try:
            if self.cooler_supported:
                self.cam.set_control_value(asi.ASI_COOLER_ON, 1)
                self.cooling_target = temp
                self.cam.set_control_value(asi.ASI_TARGET_TEMP, int(temp))
                print(f"Set target temperature to {temp}°C")
            else:
                print("Cooling not supported")
        except:
            print("not cool")


    def GetTemperature(self):
        if self.cooler_supported:
            return self.cam.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
        else:
            return None

    def GetCoolerPower(self):
        if self.cooler_supported:
            return self.cam.get_control_value(asi.ASI_COOLER_POWER_PERC)[0]
        else:
            return None

    def monitor_cooling(self):
        if self.cooler_supported:
            current_temp = self.GetTemperature()
            cooler_power = self.GetCoolerPower()
            print(f"Current Temperature: {current_temp:.1f}°C, Target: {self.cooling_target}°C, Cooler Power: {cooler_power}%")
            return current_temp, cooler_power
        else:
            print("Cooling not supported")
            return None, None


    def SetCrop(self, crop):
        self.sizex = int(self.image_size_x * crop)
        self.sizey = int(self.image_size_y * crop)
        # Adjust sizes to be multiples of 8
        self.sizex = (self.sizex // 8) * 8
        self.sizey = (self.sizey // 8) * 8
        ddx = self.image_size_x - self.sizex
        ddy = self.image_size_y - self.sizey
        ddx = ddx // 2
        ddy = ddy // 2
        self.SetROI(ddx, ddy, self.sizex, self.sizey)

    def SetROI(self, x0, y0, roi_w, roi_h):
        self.roi_w = int(roi_w)
        self.roi_h = int(roi_h)
        self.cam.set_roi(start_x=int(x0), start_y=int(y0), width=self.roi_w, height=self.roi_h)
        self.imgdata = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8 if self.bpp == 8 else np.uint16)


    def GetSingleFrame(self):
        if self.live:
            return self.GetLiveFrame()
            
        # Check if we're not already exposing and need to start a new exposure
        if not hasattr(self, 'exposure_started'):
            self.exposure_started = False
            
        # If we haven't started an exposure and we're ready to start one
        if not self.exposure_started and self.exposing_done():
            try:
                self.cam.start_exposure()
                self.exposure_started = True
                return None
            except Exception as e:
                print(f"Failed to start exposure: {e}")
                self.exposure_started = False
                return None
                
        # If we're in the middle of an exposure, check status
        if self.exposure_started:
            status = self.cam.get_exposure_status()
            
            if status == asi.ASI_EXP_WORKING:
                # Still exposing
                return None
                
            elif status == asi.ASI_EXP_SUCCESS:
                try:
                    # Get the image data
                    buffer = self.cam.get_data_after_exposure()
                    
                    # Convert bytearray to numpy array
                    if self.bpp == 8:
                        img = np.frombuffer(buffer, dtype=np.uint8)
                    else:  # 16-bit
                        img = np.frombuffer(buffer, dtype=np.uint16)
                    
                    # Reshape the array to 2D
                    img = img.reshape((self.roi_h, self.roi_w))
                    
                    # Store the image and update timing
                    self.imgdata = img
                    self.start_time = time.time()
                    
                    # Reset exposure state
                    self.exposure_started = False
                    
                    return self.imgdata
                    
                except Exception as e:
                    print(f"Failed to retrieve image data: {e}")
                    self.exposure_started = False
                    return None
                    
            else:  # ASI_EXP_FAILED or other status
                print(f"Exposure failed with status: {status}")
                self.exposure_started = False
                return None
                
        return None

    def GetLiveFrame(self):
        try:
            buffer = self.cam.capture_video_frame(timeout=1300)
            # Convert bytearray to numpy array
            if self.bpp == 8:
                img = np.frombuffer(buffer, dtype=np.uint8)
            else:  # 16-bit
                img = np.frombuffer(buffer, dtype=np.uint16)
            
            # Reshape the array to 2D
            img = img.reshape((self.roi_h, self.roi_w))
            
            self.imgdata = img
            return self.imgdata
        except asi.ZWO_Error:
            print("start") 
            self.cam.start_video_capture()
            return None

    def name(self):
        return self.cam_info['Name']

    def get_frame(self):
        self.frame = self.GetSingleFrame()
        return self.frame

    def Begin(self):
        if self.live:
            self.cam.start_video_capture()
        else:
            self.start_time = time.time()


    def StopLive(self):
        self.cam.stop_video_capture()

    def GetStatus(self):
        if self.live:
            return [1, 0, 0, 0]  # Simulating QHYCCD status format
        elif self.exposing_done():
            return [0, 1, 0, 0]
        else:
            return [0, 0, 1, 0]

    def ClearBuffers(self):
        if self.live:
            self.cam.stop_video_capture()
            time.sleep(0.1)
            self.cam.start_video_capture()
        else:
            self.cam.stop_exposure()
        print("Buffers cleared")

    def exposing_done(self):
        return True
        if self.start_time == 0:
            return True
        return time.time() > (self.start_time + self.exposureMS / 1000.0)

    def close(self):
        if self.cam:
            if asi.ASI_COOLER_ON in self.cam.get_controls():
                self.cam.set_control_value(asi.ASI_COOLER_ON, 0)  # Turn off cooler
            #if asi.ASI_FAN_ON in self.cam.get_controls():
            #    self.cam.set_control_value(asi.ASI_FAN_ON, 0)  # Turn off fan
            self.cam.close()
        #self.sdk.close()

# Example usage:
if __name__ == "__main__":
    cam_name = "ASI178MM"  # Replace with your camera model
    camera = zwoasi_wrapper(temp=-10, exp=100, gain=200, crop=None, cam_name=cam_name, live=True)
    
    #print(f"Camera size: {camera.GetSize()}")
    
    # Set and check fan speed
    camera.SetFanSpeed(50)  # Set fan to 50% speed
    print(f"Current fan speed: {camera.GetFanSpeed()}%")
    
    camera.Begin()
    
    for i in range(10):  # Capture 10 frames
        frame = camera.GetSingleFrame()
        if frame is not None:
            print(f"Captured frame {i+1}, shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"Current temperature: {camera.GetTemperature()}°C")
        time.sleep(1)
    
    camera.close()
