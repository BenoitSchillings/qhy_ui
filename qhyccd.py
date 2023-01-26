#!/bin/python3
import ctypes
from ctypes import *
import numpy as np
import time
from libqhy import *

from sys import platform

import sys
import cv2

"""
Basic functions to control qhyccd camera

| Version | Commit
| 0.1     | initial version @2020/07/02 hf

#TODO: fail to change steammode inside sdk
"""
800
class qhyccd():
    def __init__(self, cam_name):
        # create sdk handle
        #self.sdk= CDLL()
        self.tmp = CDLL('/usr/local/lib/libopencv_core.so', mode=ctypes.RTLD_GLOBAL)
        self.tmp = CDLL('/usr/local/lib/libopencv_imgproc.so', mode=ctypes.RTLD_GLOBAL)

        self.sdk= CDLL('/usr/local/lib/libqhyccd.so.23.1.11.17')

        self.sdk.GetQHYCCDParam.restype = c_double
        self.sdk.OpenQHYCCD.restype = ctypes.POINTER(c_uint32)
        # ref: https://www.qhyccd.com/bbs/index.php?topic=6356.0
        self.mode = 1 # set default mode to stream mode, otherwise set 0 for single frame mode
        self.bpp = c_uint(16) # 8 bit
        self.exposureMS = 100 # 100ms
        self.connect(self.mode, cam_name)
        self.ClearBuffers()
        self.start_time = 0
        self.live = False

    def GetModeName(self, mode_number):
        mode_char_array_32 = c_char*32
        mode = mode_char_array_32()
        self.sdk.GetQHYCCDReadModeName(self.cam, mode_number, mode)
        return mode.value

    def GetName(self):
        return str(self.name)


    def connect(self, mode, cam_name):
        ret = -1
        
        self.sdk.InitQHYCCDResource()
        self.sdk.ScanQHYCCD()
        type_char_array_32 = c_char*32
        self.id = type_char_array_32()
        #self.sdk.SetQHYCCDLogLevel(1)

        for id in range(3):
            self.sdk.GetQHYCCDId(c_int(id), self.id)    # open the first camera
            name = str(self.id.value)
            if (cam_name in name):
                print("FOUND")
                break

        #self.sdk.GetQHYCCDId(c_int(1), self.id)        
        print("Open camera:", self.id)
        self.name = self.id.value
        self.cam = self.sdk.OpenQHYCCD(self.id)
        self.sdk.StopQHYCCDLive(self.cam)   #here
        print(self.GetModeName(1))
 
        self.sdk.SetQHYCCDReadMode(self.cam, 1)
        self.sdk.SetQHYCCDStreamMode(self.cam, 0)  
        self.sdk.InitQHYCCD(self.cam)

        # Get Camera Parameters
        self.chipw = c_double()
        self.chiph = c_double()
        self.w = c_uint()
        self.h = c_uint()
        self.pixelw = c_double()
        self.pixelh = c_double() 
        self.channels = c_uint32(1)
        self.sdk.GetQHYCCDChipInfo(self.cam, byref(self.chipw), byref(self.chiph),
                byref(self.w), byref(self.h), byref(self.pixelw),
                byref(self.pixelh), byref(self.bpp))
        self.roi_w = self.w
        self.roi_h = self.h
        


        self.imgdata = (ctypes.c_uint8 * self.w.value* self.h.value)()
        self.SetExposure( 10)
        self.SetBit(self.bpp.value)
        
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_USBTRAFFIC, c_double(1))
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_TRANSFERBIT, self.bpp)
        err = self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_DDR, 0)
        print("err", err)
        # Maximum fan speed
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_MANULPWM, c_double(255))
        self.sdk.CancelQHYCCDExposingAndReadout(self.cam)
        #self.sdk.SetQHYCCDStreamMode(self.cam, 1)  


    def GetSize(self):
        tx = c_uint()
        ty = c_uint()

        sx = c_uint()
        sy = c_uint()

        self.sdk.GetQHYCCDEffectiveArea(self.cam, byref(tx), byref(ty),
                                        byref(sx), byref(sy))
        
        self.image_start_x = tx.value
        self.image_start_y = ty.value
        self.image_size_x = sx.value
        self.image_size_y = sy.value

        print(self.image_start_x, self.image_start_y, self.image_size_x, self.image_size_y)


        self.sdk.GetQHYCCDOverScanArea(self.cam, byref(tx), byref(ty),
                                        byref(sx), byref(sy))
        
        self.overscan_start_x = tx.value
        self.overscan_start_y = ty.value
        self.overscan_size_x = sx.value
        self.overscan_size_y = sy.value


        print(self.overscan_start_x, self.overscan_start_y, self.overscan_size_x, self.overscan_size_y)


    def SetStreamMode(self, mode):
        """ TODO: Unable to change"""
        self.sdk.CloseQHYCCD(self.cam)
        self.mode = mode
        self.connect(mode)

    """Set camera exposure in ms, return actual exposure after setting """
    def SetExposure(self, exposureMS):
        # sdk exposure uses us as unit
        self.exposureMS = exposureMS # input ms
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_EXPOSURE, c_double(exposureMS*1000))
        self.ClearBuffers()
        print("Set exposure to", 
                self.sdk.GetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_EXPOSURE)/1000)

    """ Set camera gain """
    def SetGain(self, gain):
        self.gain = gain
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_GAIN, c_double(gain))
        print("Set gain to", 
                self.sdk.GetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_GAIN)/1)


    def SetOffset(self, offset):
       self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_OFFSET, c_double(offset))
        


    """ Set camera gain """
    def SetUSB(self, usbrate):
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_USBTRAFFIC, c_double(usbrate))
        print("Set usb to", 
                self.sdk.GetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_USBTRAFFIC)/1)
    
    """ Set camera depth """
    def SetBit(self, bpp):
        self.bpp.value = bpp
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_TRANSFERBIT, c_double(bpp))

    def SetTemperature(self, temp):
    	self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_COOLER, c_double(temp))
        
    def GetTemperature(self):
       return self.sdk.GetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_CURTEMP)

    def GetDDR(self):
       return self.sdk.GetQHYCCDParam(self.cam, CONTROL_ID.DDR_BUFFER_CAPACITY)

    def exposing_done(self):
        t0 = time.time()
        #print(self.start_time, t0)
        if (self.start_time == 0):
            return True
        if (t0 > (self.start_time + self.exposureMS/1000.0)):
            return True
        return False
        
        
    def SetDDR(self, value):
        self.sdk.SetQHYCCDParam(self.cam, CONTROL_ID.CONTROL_DDR, c_double(value))
        
        
    """ Set camera ROI """
    def SetROI(self, x0, y0, roi_w, roi_h):
        self.roi_w =  c_uint(roi_w)
        self.roi_h =  c_uint(roi_h)
        # update buffer to recive camera image
        if self.bpp.value == 16:
            self.imgdata = (ctypes.c_uint16 * roi_w * roi_h)()
            self.sdk.SetQHYCCDResolution(self.cam, x0, y0, self.roi_w, self.roi_h)
        else: # 8 bit
            self.imgdata = (ctypes.c_uint8 * roi_w * roi_h)()
            self.sdk.SetQHYCCDResolution(self.cam, x0, y0, self.roi_w, self.roi_h)

    """ Exposure and return single frame """
    def GetSingleFrame(self):
        if (self.live):
            return GetLiveFrame()

        if (self.exposing_done()):
            #print("exposing done")
            ret = self.sdk.GetQHYCCDSingleFrame(self.cam, byref(self.roi_w), byref(self.roi_h), byref(self.bpp),byref(self.channels), self.imgdata)
            
            self.sdk.ExpQHYCCDSingleFrame(self.cam)
            self.start_time = time.time()
            return np.asarray(self.imgdata) #.reshape([self.roi_h.value, self.roi_w.value])

        #print("still exposing")
        return None

  

    def Begin(self):
        """ Begin live mode"""
        if (self.live):
            self.sdk.SetQHYCCDStreamMode(self.cam, 1)  # Live mode
            self.sdk.CancelQHYCCDExposingAndReadout(self.cam)
            self.sdk.BeginQHYCCDLive(self.cam)
        else:
            self.sdk.ExpQHYCCDSingleFrame(self.cam)
            self.start_time = time.time()

    
    def GetLiveFrame(self):
        """ Return live image """
        result = self.sdk.GetQHYCCDLiveFrame(self.cam, byref(self.roi_h), byref(self.roi_w), 
                byref(self.bpp), byref(self.channels), self.imgdata)
        if (result < 0):
            return None
        return np.asarray(self.imgdata)

    def StopLive(self):
        """ Stop live mode, change to single frame """
        self.sdk.StopQHYCCDLive(self.cam)
        #self.sdk.SetQHYCCDStreamMode(self.cam, 0)  # Single Mode


    def GetStatus(self):
        status = (ctypes.c_uint8 * 4)()

        ret = self.sdk.GetQHYCCDCameraStatus(self.cam, status)
        print("ret ", ret, status)
        return status 

    def ClearBuffers(self):
        self.sdk.SetQHYCCDWriteFPGA(self.cam, 0, 63, 0)
        self.sdk.SetQHYCCDWriteFPGA(self.cam, 0, 63, 1)
        self.sdk.SetQHYCCDWriteFPGA(self.cam , 0, 63, 0)


    """ Relase camera and close sdk """
    def close(self):
        self.sdk.CloseQHYCCD(self.cam)
        self.sdk.ReleaseQHYCCDResource()
