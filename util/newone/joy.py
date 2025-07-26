
import inputs
import time


MIN_V = 35


class JSTest(object):
    """Simple joystick test class."""
    def __init__(self, gamepad=None):
        self.btn_state = {}

        self.old_btn_state = self.btn_state
        
        self.abs_state = {}
        self.abs_state["Absolute-ABS_X"] = 0
        self.abs_state["Absolute-ABS_Y"] = 0
        self.abs_state["Absolute-ABS_RX"] = 0
        self.abs_state["Absolute-ABS_RY"] = 0
        self.abs_state["Key-BTN_NORTH"] = 0
        self.abs_state["Key-BTN_SOUTH"] = 0
        self.abs_state["Key-BTN_EAST"] = 0
        self.abs_state["Key-BTN_WEST"] = 0
        self.old_abs_state = {}

        self._other = 0
        self.gamepad = gamepad
        if not gamepad:
            self._get_gamepad()

    def _get_gamepad(self):
        """Get a gamepad object."""
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

    def handle_unknown_event(self, event, key):
        return
        

    def process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        key = event.ev_type + '-' + event.code
        abbv = key
        #print(key)
        self.abs_state[abbv] = event.state
        
        self.output_state(event.ev_type, abbv)


    
    def output_state(self, ev_type, abbv):
        print(abbv)
        """Print out the output state."""
        if ev_type == 'Key':
            try:
                    if self.btn_state[abbv] != self.old_btn_state[abbv]:

                        return
            except:
                print("unknown button")

        if abbv[0] == 'H':

            return




    def process_events(self):
        """Process available events."""
        try:
            events = self.gamepad.read()
        
        except EOFError:
            events = []
        for event in events:
            self.process_event(event)



import skyx

jstest = JSTest()

sky = skyx.sky6RASCOMTele()
sky.Connect()


def main():
    """Process all events forever."""

    while 1:
        jstest.process_events()


import threading
import time


old_v1 = -1
old_v2 = -1


def joy1(v1, v2):
    global old_v1
    global old_v2


    if (abs(v1) < MIN_V):
        v1 = 0
    if (abs(v2) < MIN_V):
        v2 = 0

    if (old_v1 != v1 or old_v2 != v2):
        sky.rate((-v1 / 34.0), (v2 / 34.0))

    old_v1 = v1
    old_v2 = v2
    

    #print("j1", v1, v2)


def joy2(v1, v2):
    if (abs(v1) < MIN_V):
        v1 = 0
    if (abs(v2) < MIN_V):
        v2 = 0
    #time.sleep(0.2)
    #print("j2", v1, v2)

 
def save_state():
    p0 = sky.GetRaDec()
    print(p0)
    return 0

def restore_state():
    return 0
 
         
         
def thread_function():
    # This will be the function executed every 50ms.
    joy1(jstest.abs_state["Absolute-ABS_X"], jstest.abs_state["Absolute-ABS_Y"])
    joy2(jstest.abs_state["Absolute-ABS_RX"], jstest.abs_state["Absolute-ABS_RY"])

    if (jstest.abs_state["Key-BTN_NORTH"] != 0):
        save_state()

    if (jstest.abs_state["Key-BTN_EAST"] != 0):
        restore_state()
 


def start_thread():
    threading.Timer(0.05, start_thread).start()  # Timer takes time in seconds.
    thread_function()

  # Call the function once to start the timer.

if __name__ == "__main__":
    start_thread()
    main()
