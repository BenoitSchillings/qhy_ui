

import logging as log
from util import *
import pickle
from orion_ao import ao
import time



class guider:
    def __init__(self, mount, camera):
        log.info("init")
        self.reset()
        self.ao = ao()
        self.mount = mount
        self.camera = camera
        self.guide_inited = 0
        #self.filter = KalmanFilter(0.1)
        self.gain_x = 210.0
        self.gain_y = 210.0
        self.center_x = 0
        self.center_y = 0
        self.cheat_move_x = 0.0
        self.cheat_move_y = 0.0
        self.mount_cal_state_count = 0
        self.ao_cal_state_count = 0
        self.ao_calibrated = 0
        self.guide_inited_ao = 0
        N = 2
        self.last_x = LastNValues(N)
        self.last_y = LastNValues(N)
        self.load_state("guide.data")
        #self.ao_calibrated = 1
        self.last_ao_move_time = self.current_milli_time()


    def current_milli_time(self):
        return time.time() * 1000.0



    def fbump_ao(self, dx, dy):
        self.ao.goto(round(dx), round(dy))
        self.last_ao_move_time = self.current_milli_time()

    def fmove_ao(self, dx, dy):
        self.ao.move(round(dx), round(dy))
        if (np.abs(dx) > 1 or np.abs(dy) > 1):
            self.last_ao_move_time = self.current_milli_time()

    def reset_ao(self):
        self.fbump_ao(0,0)
        self.last_ao_move_time = self.current_milli_time()
        time.sleep(0.3)
        self.guide_inited_ao = -5


    def fbump_mount(self, dx, dy):

        self.cheat_move_x += dx / 50.0
        self.cheat_move_y += dy / 50.0
        print("p0")
        if not (self.mount is None):
            print("p1")
            print(dx, dy)
            if (np.abs(dx) < 3250.0 and np.abs(dy) < 3250):
                print("job")
                print("LOG MOVE", dx, dy)
                self.mount.jog(dx/3600.0,dy/3600.0)

    def start_calibrate_mount(self):
        log.info("calibrate")
        self.cal_state_mount = 20

    def stop_calibrate_mount(self):
        self.cal_state_mount = 0

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

        settings['ao_dx1'] = self.ao_dx1
        settings['ao_dx2'] = self.ao_dx2
        settings['ao_dy1'] = self.ao_dy1
        settings['ao_dy2'] = self.ao_dy2


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

                self.ao_dx1 = settings['ao_dx1']
                self.ao_dx2 = settings['ao_dx2']
                self.ao_dy1 = settings['ao_dy1']
                self.ao_dy2 = settings['ao_dy2']

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

        self.ao_dx1 = 0
        self.ao_dy1 = 0
        self.ao_dx2 = 0
        self.ao_dy2 = 0


        self.guiding_with_mount = 0
        self.cal_state_mount = 0



    def new_pos(self, x, y):
        log.info("new pos %d %d", x, y)

    def set_pos(self, x, y):
        log.info("set pos %d %d", x, y)

    def calibrate_mount(self):
        self.mount_cal_state_count = 40
        self.guiding_with_mount = 0

    def calibrate_ao(self):
        self.ao_cal_state_count = 40
        self.ao_calibrated = 0
        print("Calibrate AO !!")

    def guide(self):
        self.is_guiding = 1
        self.ao_calibrated = 1
        self.guide_inited_ao = 0


    def handle_calibrate_aox(self, x, y):
        N1 = 15
        print("handle cal pos", x, y,self.ao_cal_state_count)
        if (self.ao_cal_state_count%4 == 0):
            self.ao_pos_x0 = x
            self.ao_pos_y0 = y
            self.fbump_ao(N1, 0)
            log.info("Move Left")

        if (self.ao_cal_state_count%4 == 1):
            self.ao_pos_x1 = x
            self.ao_pos_y1 = y
            self.fbump_ao(0, 0)
            log.info("Move Right")


        if (self.ao_cal_state_count%4 == 2):
            self.ao_pos_x2 = x
            self.ao_pos_y2 = y
            self.fbump_ao(0, N1)
            log.info("Move Up")


        if (self.ao_cal_state_count%4 == 3):
            self.ao_pos_x3 = x
            self.ao_pos_y3 = y
            self.fbump_ao(0, 0)
            log.info("Move Down")


        if (self.ao_cal_state_count == 1):
            self.calc_calibration_ao()

        self.ao_cal_state_count = self.ao_cal_state_count - 1
        if (self.ao_cal_state_count < 0):
            self.ao_cal_state_count = 0


    def handle_calibrate_ao(self, x, y):
        N = 180
        print("handle cal pos", x, y,)
        if (self.ao_cal_state_count == 40):
            self.ao_pos_x0 = x
            self.ao_pos_y0 = y
            self.fbump_ao(N, 0)
            log.info("Move Left")

        if (self.ao_cal_state_count == 30):
            self.ao_pos_x1 = x
            self.ao_pos_y1 = y
            self.fbump_ao(0, 0)
            log.info("Move Right")


        if (self.ao_cal_state_count == 20):
            self.ao_pos_x2 = x
            self.ao_pos_y2 = y
            self.fbump_ao(0, N)
            log.info("Move Up")


        if (self.ao_cal_state_count == 10):
            self.ao_pos_x3 = x
            self.ao_pos_y3 = y
            self.fbump_ao(0, 0)
            log.info("Move Down")


        if (self.ao_cal_state_count == 1):
            self.calc_calibration_ao()

        self.ao_cal_state_count = self.ao_cal_state_count - 1
        if (self.ao_cal_state_count < 0):
            self.ao_cal_state_count = 0



    def handle_calibrate_mount(self, x, y):
        N = 1500
        if (self.mount_cal_state_count == 40):
            self.mount_pos_x0 = x
            self.mount_pos_y0 = y
            self.fbump_mount(-N, 0.0001)
            log.info("Move Left")

        if (self.mount_cal_state_count == 30):
            self.mount_pos_x1 = x
            self.mount_pos_y1 = y
            self.fbump_mount(N, 0.0001)
            log.info("Move Right")


        if (self.mount_cal_state_count == 20):
            self.mount_pos_x2 = x
            self.mount_pos_y2 = y
            self.fbump_mount(0.0001, -N)
            log.info("Move Up")


        if (self.mount_cal_state_count == 10):
            self.mount_pos_x3 = x
            self.mount_pos_y3 = y
            self.fbump_mount(0.0001, N)
            log.info("Move Down")


        if (self.mount_cal_state_count == 1):
            self.calc_calibration_mount()

        self.mount_cal_state_count = self.mount_cal_state_count - 1
        if (self.mount_cal_state_count < 0):
            self.mount_cal_state_count = 0

    def calc_calibration_mount(self):
        log.info("calc cal mount")
        self.mount_dx1 = self.mount_pos_x1 - self.mount_pos_x0       
        self.mount_dy1 = self.mount_pos_y1 - self.mount_pos_y0

        self.mount_dx2 = self.mount_pos_x3 - self.mount_pos_x2      
        self.mount_dy2 = self.mount_pos_y3 - self.mount_pos_y2

        self.save_state("guide.data")

    def calc_calibration_ao(self):
        log.info("calc cal ao")
        self.ao_dx1 = self.ao_pos_x1 - self.ao_pos_x0       
        self.ao_dy1 = self.ao_pos_y1 - self.ao_pos_y0

        self.ao_dx2 = self.ao_pos_x3 - self.ao_pos_x2      
        self.ao_dy2 = self.ao_pos_y3 - self.ao_pos_y2

        self.save_state("guide.data")
        self.ao_calibrated = 1


    def mount_calibrate_state(self):
        return mount_cal_state_count

    def distance(self, x, y):
        return np.sqrt(x*x+y*y)


    def offset(self, dx, dy):
        self.center_x = self.center_x + dx
        self.center_y = self.center_y + dy

        log.info("new guide position %f %f", self.center_x, self.center_y)


    def handle_guide_ao(self, x, y):
        if (self.guide_inited_ao <= 0):
            self.center_x = x
            self.center_y = y
            self.guide_inited_ao = self.guide_inited_ao + 1
        else:
            dx = x - self.center_x
            dy = y - self.center_y

            #ipc.set_val("guide_error", [dx,dy])
            self.dis = self.distance(dx,dy)
            
            if (self.dis > 50.0):
                return 0,0

            self.last_x.add_value(dx)
            self.last_y.add_value(dy)

            tx = 1.0*self.error_to_tx_ao(dx, dy)
            ty = 1.0*self.error_to_ty_ao(dx, dy)

            #log.info("ERROR %f %f %f %f | dis %f", dx, dy, tx, ty, self.dis)
            self.fmove_ao(90.0*-tx, 90.0*-ty)
            #self.mount(bump, tx, ty)

            return dx, dy



    def handle_guide_mount(self, x, y):
        if (self.guide_inited_mount == 0):
            self.center_x = x
            self.center_y = y
            self.guide_inited_mount = 1
        else:
            dx = x - self.center_x
            dy = y - self.center_y

            #ipc.set_val("guide_error", [dx,dy])
            self.dis = self.distance(dx,dy)

            if (self.dis > 20.0):
                return

            self.last_x.add_value(dx)
            self.last_y.add_value(dy)

            tx = 1.0*self.error_to_tx_mount(dx, dy)
            ty = 1.0*self.error_to_ty_mount(dx, dy)

            log.info("ERROR %f %f %f %f", dx, dy, tx, ty)
            self.fbump_mount(tx, ty)
            #self.mount(bump, tx, ty)

        log.info("get guide point %f %f", x, y)

    def drizzle(self, dx, dy):
        self.center_x = self.center_x + dx 
        self.center_y = self.center_y + dy 

        
    def pos_handler(self, x, y):
        if (self.mount_cal_state_count != 0):
            print("handle mount ", x, y)
            self.handle_calibrate_mount(x , y)


        if self.ao_cal_state_count != 0:
            print("handle ", x, y)
            self.handle_calibrate_ao(x, y)

        #print("guide_ao = ", self.ao_calibrated)

        if self.ao_calibrated != 0:
            return self.handle_guide_ao(x, y)

        return 0,0

            
    def error_to_tx_mount(self, mx, my):
        num = (self.mount_dy2 * mx) - (self.mount_dx2 * my)
        den = (self.mount_dx1 * self.mount_dy2) - (self.mount_dx2 * self.mount_dy1)

        return num / den

    def error_to_ty_mount(self, mx, my):
        num = (self.mount_dy1 * mx) - (self.mount_dx1 * my)
        den = (self.mount_dx2 * self.mount_dy1) - (self.mount_dx1 * self.mount_dy2)

        return num / den

            
    def error_to_tx_ao(self, mx, my):
        num = (self.ao_dy2 * mx) - (self.ao_dx2 * my)
        den = (self.ao_dx1 * self.ao_dy2) - (self.ao_dx2 * self.ao_dy1)

        return num / den

    def error_to_ty_ao(self, mx, my):
        num = (self.ao_dy1 * mx) - (self.ao_dx1 * my)
        den = (self.ao_dx2 * self.ao_dy1) - (self.ao_dx1 * self.ao_dy2)

        return num / den

