#!/usr/bin/env python3
# Generated by usbrply
# cmd: /Library/Frameworks/Python.framework/Versions/3.10/bin/usbrply

import binascii
import time
import usb1
import time
import math


def validate_read(expected, actual, msg):
    if expected != actual:
        print('Failed %s' % msg)
        print('  Expected; %s' % binascii.hexlify(expected,))
        print('  Actual:   %s' % binascii.hexlify(actual,))
        #raise Exception("failed validate: %s" % msg)

def hex_string_to_formatted_binary_array(hex_string):
    return bytes.fromhex(hex_string)


def interruptRead(dev, endpoint, size, timeout=None):
    return dev.interruptRead(endpoint, size,
                timeout=(1000 if timeout is None else timeout))

def interruptWrite(dev, endpoint, data, timeout=None):
    #print(data)
    dev.interruptWrite(endpoint, data, timeout=(1000 if timeout is
None else timeout))

def create_byte_array_for_rot(rotation):
  """
  Creates a byte array with the specified rotation.

  Args:
    rotation: The rotation value.

  Returns:
    A byte array.
  """
  low_byte = rotation & 0xFF
  high_byte = (rotation >> 8) & 0xFF
  return bytearray([0x6b, low_byte, high_byte, 0x0d, low_byte, high_byte, 0x00, 0x00])



def rotate_to_angle(dev, angle):


    # Generated by usbrply
    # Source: Windows pcap (USBPcap)
    # cmd: /Library/Frameworks/Python.framework/Versions/3.10/bin/usbrply

    # PCapGen device hi: selected device 2
    # Generated from packet 7/8

    iangle = int(angle * 29)


    interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("672474038fb02795"))
    result = interruptRead(dev, 0x81, 8)
    #print(result.hex())
    rot_array = create_byte_array_for_rot(iangle)
    #print(rot_array.hex())
    interruptWrite(dev, 0x02,rot_array)



    for i in range(100):
        interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("6875f90538db760b"))
        result = interruptRead(dev, 0x81, 8)
        print(result.hex())
        time.sleep(0.01)

def c2_byte(v):
    if (v < 0):
        return 0x80 - v
    return v


def create_byte_array_for_move(dx, dy):
  """
  Creates a byte array with the specified rotation.

  Args:
    rotation: The rotation value.

  Returns:
    A byte array.
  """
  return bytearray([0x32, c2_byte(dx), c2_byte(dy), 0x00, 0x81, 0x7a, 0x90, 0x90])




def move_ao(dev, dx, dy):
    mov_array = create_byte_array_for_move(dx, dy)
    #print(mov_array.hex())
    interruptWrite(dev, 0x02,mov_array)

import signal

def build_motor_move_array(motor_idx, delta):
  return bytearray([0x61, c2_byte(delta), c2_byte(motor_idx), 0x00, 0xc7, 0xdd, 0x42, 0x1a])



def move_motors(dev, m1, m2, m3, m4):
    mov_1 = build_motor_move_array(0, m1)
    mov_2 = build_motor_move_array(1, m2)
    mov_3 = build_motor_move_array(2, m3)
    mov_4 = build_motor_move_array(3, m4)
    #print((mov_1.hex()))
    #print((mov_2.hex()))
    #print((mov_3.hex()))
    #print((mov_4.hex()))

    dt = 0.0
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if (m1 != 0):
            interruptWrite(dev, 0x02,mov_1)
            #time.sleep(dt)

        if (m2 != 0):
            interruptWrite(dev, 0x02,mov_2)
            #time.sleep(dt)

        if (m3 != 0):    
            interruptWrite(dev, 0x02,mov_3)
            #time.sleep(dt)

        if (m4 != 0):
            interruptWrite(dev, 0x02,mov_4)
            #time.sleep(dt)
    #print("move motor done")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        signal.signal(signal.SIGINT, original_handler)  # Re-enable Ctrl+C


def b_2_i(v):
    if (v > 128):
        return v - 256
    return v

def get_ao_pos(dev):
    
    interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("30000000c7dd421a"))
    interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("3916421ac7dd421a"))
    interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("62000000c7dd421a"))

    
    result = interruptRead(dev, 0x81, 8)
    print(b_2_i(result[0]), b_2_i(result[1]), b_2_i(result[2]), b_2_i(result[3]))
    return b_2_i(result[0]), b_2_i(result[1]), b_2_i(result[2]), b_2_i(result[3])

def open_dev(vid_want, pid_want, usbcontext=None):
    if usbcontext is None:
        usbcontext = usb1.USBContext()

    print("Scanning for devices...")
    for udev in usbcontext.getDeviceList(skip_on_error=True):
        vid = udev.getVendorID()
        pid = udev.getProductID()
        if (vid, pid) == (vid_want, pid_want):
            print("Found device")
            print("Bus %03i Device %03i: ID %04x:%04x" % (
                udev.getBusNumber(),
                udev.getDeviceAddress(),
                vid,
                pid))
            return udev.open()
    raise Exception("could not connect to AO unit")


def get_homing_status(dev):
    #interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("63000000c7dd421a"))
    #interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("36050000c7dd421a"))
    interruptWrite(dev, 0x02, hex_string_to_formatted_binary_array("34000000c7dd421a"))
    result = interruptRead(dev, 0x81, 8)
    return result[0]



def home(dev):
    m1, m2, m3, m4 = get_ao_pos(dev)
    print("p0 = ", m1, m2, m3, m4)
    move_motors(dev, m1, m2, m3, m4)
    time.sleep(0.2)
    m1, m2, m3, m4 = get_ao_pos(dev)
    print("p1 = ", m1, m2, m3, m4)
    status = get_homing_status(dev)
   
    count = 100

    while(count > 0 and status != 0):
        print(count, hex(status))
        count = count - 1 
        #m1, m2, m3, m4 = get_ao_pos(dev)
        print("new pos = ", m1, m2, m3, m4)
        if (status & 0x80 != 0):
            move_m4 = -7
        else:
            move_m4 = 7

        if (status & 0x40 != 0):
            move_m3 = -7
        else:
            move_m3 = 7

        if (status & 0x20 != 0):
            move_m2 = -7
        else:
            move_m2 = 7

        if (status & 0x10 != 0):
            move_m1 = -7
        else:
            move_m1 = 7

        print("new move is ", move_m1, move_m2, move_m3, move_m4)
        move_motors(dev, move_m1, move_m2, move_m3, move_m4)

        status = get_homing_status(dev)


    m1, m2, m3, m4 = get_ao_pos(dev)
    print("initial pos = ", m1, m2, m3, m4)
    move_motors(dev, m1, m2, m3, m4)
    time.sleep(1)
    m1, m2, m3, m4 = get_ao_pos(dev)
    print("initial pos = ", m1, m2, m3, m4)

    return status


    get_ao_pos(dev)



class ao:
    def __init__(self):
        self.do_init()
        self.home()

    def home(self):
        home(self.dev)


    def do_init(self):
        vid_want = 0x03EB
        pid_want = 0x2013

        usbcontext = usb1.USBContext()
        self.dev = open_dev(vid_want, pid_want, usbcontext)
        
        self.dev.resetDevice()

        try:
            self.dev.detachKernelDriver(0)
        except:
            print("already detached")

        self.dev.claimInterface(0)
        
        #self.home()

        self.m1 = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0

        self.ax = 0
        self.ay = 0

    def clip_motor_pos(self, v):
        MAX = 43
        if (v < -MAX):
            v = -MAX
        if (v > MAX):
            v = MAX

        return v

    def  set_motors(self, target_m1, target_m2, target_m3, target_m4):
        target_m1 = self.clip_motor_pos(target_m1)
        target_m2 = self.clip_motor_pos(target_m2)
        target_m3 = self.clip_motor_pos(target_m3)
        target_m4 = self.clip_motor_pos(target_m4)

        delta_m1 = target_m1 - self.m1
        delta_m2 = target_m2 - self.m2
        delta_m3 = target_m3 - self.m3
        delta_m4 = target_m4 - self.m4

        #print("move motor by ", delta_m1, delta_m2, delta_m3, delta_m4)
        move_motors(self.dev, delta_m1, delta_m2, delta_m3, delta_m4)

        self.m1 = target_m1
        self.m2 = target_m2
        self.m3 = target_m3
        self.m4 = target_m4
        #print("now motor at ", self.m1, self.m2, self.m3, self.m4)


    def move_motors(self, dm1, dm2, dm3, dm4):
        self.target_m1 = self.m1 + dm1
        self.target_m2 = self.m2 + dm2
        self.target_m3 = self.m3 + dm3
        self.target_m4 = self.m4 + dm4

        self.set_motors(self.target_m1, self.target_m2, self.target_m3, self.target_m4)
        
# motors are set as
# m1 at 2 pm
# m2 at 5 pm
# m3 at 7 pm
# m4 at 10 pm
# let's say that x axis is the m1,m3 line
# let's say that y axis is the m2,m4 line 
# these are at 45 degres from the guider, but this is easy mapping



    def motor_to_xy(self, m1, m2, m3, m4):
        tx = m1 - m3
        ty = m2 - m4

        return tx, ty

    def xy_to_motor(self, tx, ty):
        m1 = tx
        m2 = ty
        m3 = 0
        m4 = 0

        ddx = tx // 2
        ddy = ty // 2

        m1 -= ddx 
        m3 -= ddx 
        m2 -= ddy
        m4 -= ddy

        return m1, m2, m3, m4


    def set_ao(self, tx, ty):
        print("set ao ", tx, ty)
        m1, m2, m3, m4 = self.xy_to_motor(tx, ty)
        self.set_motors(m1, m2, m3, m4)
        self.ax, self.ay = self.motor_to_xy(self.m1, self.m2, self.m3, self.m4) 

    def get_ao(self):
        return self.ax, self.ay

    def move_ao(self, dx, dy):
        self.set_ao(self.ax + dx, self.ay + dy)

    def goto(self, x, y):
        print("goto", x, y)
        self.set_ao(x, y)

    def zero(self):
        self.goto(0,0)
        time.sleep(0.3)

    def move(self, x,y):
        self.move_ao(x, y)


    def rotate_to_angle(self, angle):
        rotate_to_angle(self.dev, angle)

    def circle_test(self, speed, diameter):
        alpha = 0.0
        #rotate_to_angle(self.dev, 90)
        while(True):
            alpha = alpha + speed
            x = math.sin(alpha) * diameter
            y = math.cos(alpha) * diameter

            self.set_ao(int(round(x)), int(round(y)))
            time.sleep(0.02)

    def close(self):
        print("close ao")


if __name__ == "__main__":
    ao = ao()
    ao.home()

    #m1,m2,m3, m4 = ao.xy_to_motor(20, 19)
    #print("cp ", m1,m2,m3,m4)
    #print(ao.motor_to_xy(m1,m2,m3,m4))

    #for i in range(300):
    #    ao.move_ao(3, 0)
    #    time.sleep(0.03)
    #    ao.move_ao(-3, 0)
    #    time.sleep(0.03)

    #ao.circle_test(0.01, 30)

   
