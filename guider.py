

import logging as log
from util import *
import pickle





class guider:
    def __init__(self, mount, camera):
        log.info("init")
        self.reset()
        self.mount = mount
        self.camera = camera
        self.guide_inited = 0
        #self.filter = KalmanFilter(0.1)
        self.gain_x = 110.0
        self.gain_y = 110.0
        self.center_x = 0
        self_center_y = 0
        self.cheat_move_x = 0.0
        self.cheat_move_y = 0.0

        N = 4
        self.last_x = LastNValues(N)
        self.last_y = LastNValues(N)
        self.load_state("guide.data")


    def fbump(self, dx, dy):

        self.cheat_move_x += dx / 50.0
        self.cheat_move_y += dy / 50.0

        if not (self.mount is None):
            print(dx, dy)
            if (np.abs(dx) < 3250.0 and np.abs(dy) < 3250):
                print("job")
                self.mount.jog(dx/3600.0,dy/3600.0)

    def start_calibrate(self):
        log("calibrate")
        self.cal_state = 20

    def stop_calibrate(self):
        self.cal_state = 0

    def start_guide(self):
        self.is_guiding = 1

    def stop_guide(self):
        self.is_guiding = 0

    def save_state(self, filename):
        settings = {}

        settings['mount_dx1'] = self.mount_dx1
        settings['mount_dx2'] = self.mount_dx2
        settings['mount_dy1'] = self.mount_dy1
        settings['mount_dy2'] = self.mount_dy2

        settings['gain_x'] = self.gain_x
        settings['gain_y'] = self.gain_y

        with open(filename, "wb") as f:
            pickle.dump(settings, f)

    def load_state(self, filename):
        """Load the state of the object from a file.

        Arguments:
        filename -- the name of the file to load the state from
        """
        try:
            with open(filename, "rb") as f:
                settings = pickle.load(f)
                self.mount_dx1 = settings['mount_dx1']
                self.mount_dx2 = settings['mount_dx2']
                self.mount_dy1 = settings['mount_dy1']
                self.mount_dy2 = settings['mount_dy2']

                self.gain_x = settings['gain_x']
                self.gain_y = settings['gain_y']
               
        except Exception as e:
            log.critical("An error occurred while loading the state:", e)
            self.reset()

    def reset(self):
        """Reset the object's state to its default values."""
        self.cal_state = 0
        self.is_guiding = 0
        
        self.mount_dx1 = 0
        self.mount_dy1 = 0
        self.mount_dx2 = 0
        self.mount_dy2 = 0
        self.guide_state = 0
        self.cal_state = 0



    def new_pos(self, x, y):
        log.info("new pos %d %d", x, y)

    def set_pos(self, x, y):
        log.info("set pos %d %d", x, y)

    def calibrate(self):
        self.cal_state = 40
        self.guide_state = 0


    def guide(self):
        self.guide_state = 1

    def handle_calibrate(self, x, y):
        N = 1500
        if (self.cal_state == 40):
            self.pos_x0 = x
            self.pos_y0 = y
            self.fbump(-N, 0.0001)
            log.info("Move Left")

        if (self.cal_state == 30):
            self.pos_x1 = x
            self.pos_y1 = y
            self.fbump(N, 0.0001)
            log.info("Move Right")


        if (self.cal_state == 20):
            self.pos_x2 = x
            self.pos_y2 = y
            self.fbump(0.0001, -N)
            log.info("Move Up")


        if (self.cal_state == 10):
            self.pos_x3 = x
            self.pos_y3 = y
            self.fbump(0.0001, N)
            log.info("Move Down")


        if (self.cal_state == 1):
            self.calc_calibration()

        self.cal_state = self.cal_state - 1
        if (self.cal_state < 0):
            self.cal_state = 0

    def calc_calibration(self):
        log.info("calc cal")
        self.mount_dx1 = self.pos_x1 - self.pos_x0       
        self.mount_dy1 = self.pos_y1 - self.pos_y0

        self.mount_dx2 = self.pos_x3 - self.pos_x2      
        self.mount_dy2 = self.pos_y3 - self.pos_y2

        self.save_state("guide.data")


    def calibrate_state(self):
        return cal_state

    def distance(self, x, y):
        return np.sqrt(x*x+y*y)


    def offset(self, dx, dy):
        self.center_x = self.center_x + dx
        self.center_y = self.center_y + dy

        log.info("new guide position %f %f", self.center_x, self_center_y)

    def handle_guide(self, x, y):
        if (self.guide_inited == 0):
            self.center_x = x
            self.center_y = y
            self.guide_inited = 1
        else:
            dx = x - self.center_x
            dy = y - self.center_y

            #ipc.set_val("guide_error", [dx,dy])
            self.dis = self.distance(dx,dy)

            if (self.dis > 20.0):
                return
            #self.filter.update(GPoint(dx,dy))
            #val = self.filter.value()

            #log.info("e0", self.filter.value())

            self.last_x.add_value(dx)
            self.last_y.add_value(dy)

            if (self.last_x.same_sign()):
                self.gain_x = self.gain_x + 0.1
            else:
                self.gain_x = self.gain_x - 0.1
                
            if (self.last_y.same_sign()):
                self.gain_y = self.gain_y + 0.1
            else:
                self.gain_y = self.gain_y - 0.1
                             

            tx = 3.0*self.error_to_tx(self.gain_x * dx, self.gain_y * dy)
            ty = 3.0*self.error_to_ty(self.gain_x *dx, self.gain_y * dy)

            log.info("ERROR %f %f %f %f", dx, dy, tx, ty)
            self.fbump(tx, ty)
            #self.mount(bump, tx, ty)

        log.info("get guide point %f %f", x, y)

    def pos_handler(self, x, y):
        log.info("handler %f %f", x, y)
        if self.cal_state != 0:
            self.handle_calibrate(x, y)
        if self.guide_state != 0:
            self.handle_guide(x, y)

            
    def error_to_tx(self, mx, my):
        num = (self.mount_dy2 * mx) - (self.mount_dx2 * my)
        den = (self.mount_dx1 * self.mount_dy2) - (self.mount_dx2 * self.mount_dy1)

        return num / den

    def error_to_ty(self, mx, my):
        num = (self.mount_dy1 * mx) - (self.mount_dx1 * my)
        den = (self.mount_dx2 * self.mount_dy1) - (self.mount_dx1 * self.mount_dy2)

        return num / den

