

import usb.core
import usb.util
import re
import time

NEWFOCUS_COMMAND_REGEX = re.compile("([0-9]{0,1})([a-zA-Z?]{2,})([0-9+-]*)")
MOTOR_TYPE = {
        "0":"No motor connected",
        "1":"Motor Unknown",
        "2":"'Tiny' Motor",
        "3":"'Standard' Motor"
        }

class Controller(object):
    """Picomotor Controller

    Example:

        >>> controller = Controller(idProduct=0x4000, idVendor=0x104d)
        >>> controller.command('VE?')
        
        >>> controller.start_console()
    """


    def __init__(self, idProduct, idVendor):
        """Initialize the Picomotor class with the spec's of the attached device

        Call self._connect to set up communication with usb device and endpoints 
        
        Args:
            idProduct (hex): Product ID of picomotor controller
            idVendor (hex): Vendor ID of picomotor controller
        """
        self.idProduct = idProduct
        self.idVendor = idVendor
        self._connect()

    def _connect(self):
        """Connect class to USB device 

        Find device from Vendor ID and Product ID
        Setup taken from [1]

        Raises:
            ValueError: if the device cannot be found by the Vendor ID and Product
                ID
            Assert False: if the input and outgoing endpoints can't be established
        """
        # find the device
        self.dev = usb.core.find(
                        idProduct=self.idProduct,
                        idVendor=self.idVendor
                        )
       
        if self.dev is None:
            raise ValueError('Device not found')

        # set the active configuration. With no arguments, the first
        # configuration will be the active one
        self.dev.set_configuration()

        # get an endpoint instance
        cfg = self.dev.get_active_configuration()
        intf = cfg[(0,0)]

        self.ep_out = usb.util.find_descriptor(
            intf,
            # match the first OUT endpoint
            custom_match = \
            lambda e: \
                usb.util.endpoint_direction(e.bEndpointAddress) == \
                usb.util.ENDPOINT_OUT)

        self.ep_in = usb.util.find_descriptor(
            intf,
            # match the first IN endpoint
            custom_match = \
            lambda e: \
                usb.util.endpoint_direction(e.bEndpointAddress) == \
                usb.util.ENDPOINT_IN)

        assert (self.ep_out and self.ep_in) is not None
        


    def send_command(self, usb_command, get_reply=False):
        """Send command to USB device endpoint
        
        Args:
            usb_command (str): Correctly formated command for USB driver
            get_reply (bool): query the IN endpoint after sending command, to 
                get controller's reply

        Returns:
            Character representation of returned hex values if a reply is 
                requested
        """
        self.ep_out.write(usb_command)
        if get_reply:
            return self.ep_in.read(100)
            

    def parse_command(self, newfocus_command):
        """Convert a NewFocus style command into a USB command

        Args:
            newfocus_command (str): of the form xxAAnn
                > The general format of a command is a two character mnemonic (AA). 
                Both upper and lower case are accepted. Depending on the command, 
                it could also have optional or required preceding (xx) and/or 
                following (nn) parameters.
                cite [2 - 6.1.2]
        """
        m = NEWFOCUS_COMMAND_REGEX.match(newfocus_command)

        # Check to see if a regex match was found in the user submitted command
        if m:

            # Extract matched components of the command
            driver_number, command, parameter = m.groups()


            usb_command = command

            # Construct USB safe command
            if driver_number:
                usb_command = '1>{driver_number} {command}'.format(
                                                    driver_number=driver_number,
                                                    command=usb_command
                                                    )
            if parameter:
                usb_command = '{command} {parameter}'.format(
                                                    command=usb_command,
                                                    parameter=parameter
                                                    )

            usb_command += '\r'

            return usb_command
        else:
            print("ERROR! Command {} was not a valid format".format(
                                                            newfocus_command
                                                            ))


    def parse_reply(self, reply):
        """Take controller's reply and make human readable

        Args:
            reply (list): list of bytes returns from controller in hex format

        Returns:
            reply (str): Cleaned string of controller reply
        """

        # convert hex to characters 
        reply = ''.join([chr(x) for x in reply])
        return reply.rstrip()


    def command(self, newfocus_command):
        """Send NewFocus formated command

        Args:
            newfocus_command (str): Legal command listed in usermanual [2 - 6.2] 

        Returns:
            reply (str): Human readable reply from controller
        """
        usb_command = self.parse_command(newfocus_command)

        # if there is a '?' in the command, the user expects a response from
        # the driver
        if '?' in newfocus_command:
            get_reply = True
        else:
            get_reply = False

        reply = self.send_command(usb_command, get_reply)

        # if a reply is expected, parse it
        if get_reply:
            return self.parse_reply(reply)




def init_pico():
    idProduct =  '0x4000'
    idVendor =  '0x104d'

    idProduct = int(idProduct, 16)
    idVendor = int(idVendor, 16)
    global controller 
 # Initialize controller and start console
    controller = Controller(idProduct=idProduct, idVendor=idVendor)
    speed = 1100
    controller.command("1va" + str(speed))  
    controller.command("2va" + str(speed))
    controller.command("3va" + str(speed))

def move_pico(speed, count):
    controller.command("1va" + str(speed))
    controller.command("1pr" + str(count))
   

def wait_with_timeout(motor):
    max_attempts = 13150
    while True:
        max_attempts -= 1
        status = moving(motor)
        #print(motor, status)
        time.sleep(0.0023)
        if status != '0' or max_attempts <= 0:
            break


def move3(speed, d1, d2, d3):
    controller.command("1va" + str(speed))    
    controller.command("2va" + str(speed))
    controller.command("3va" + str(speed))

    controller.command("1pr" + str(d1))
    wait_with_timeout(1)

    if (d2 != 0):
        controller.command("2pr" + str(d2))
        wait_with_timeout(2)

    if (d3 != 0):
        controller.command("3pr" + str(d3))
        wait_with_timeout(3)

def moving(motor):
    if motor not in [1, 2, 3]:
        raise ValueError("Motor number must be 1, 2, or 3")
    status = controller.command(f"{motor}MD?")
    return status[2]


def pos():
    result = controller.command("1tp?")
    return result
    
