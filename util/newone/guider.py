import logging
import pickle
import time
import numpy as np

# Configure logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
log.addHandler(file_handler)
log.addHandler(stream_handler)

class guider:
    def __init__(self, mount, camera):
        log.info("Initializing Guider")
        self.reset()
        self.mount = mount
        self.camera = camera
        self.gain_x = 210.0
        self.gain_y = 210.0
        self.center_x = 0
        self.center_y = 0
        self.mount_calibrated = 0
        self.guide_inited = 0
        self.mount_cal_state_count = 0
        self.mount_dx1 = 0
        self.mount_dy1 = 0
        self.mount_dx2 = 0
        self.mount_dy2 = 0
        self.guide_gain = 0.7
        self.load_state("guide.data")

    def close(self):
        pass

    def start_guide(self):
        self.is_guiding = 1
        self.guide_inited = 0

    def stop_guide(self):
        self.is_guiding = 0

    def save_state(self, filename):
        settings = {
            'gain_x': self.gain_x,
            'gain_y': self.gain_y,
            'mount_dx1': self.mount_dx1,
            'mount_dy1': self.mount_dy1,
            'mount_dx2': self.mount_dx2,
            'mount_dy2': self.mount_dy2
        }
        with open(filename, "wb") as f:
            pickle.dump(settings, f)

    def load_state(self, filename):
        try:
            with open(filename, "rb") as f:
                settings = pickle.load(f)
                self.gain_x = settings['gain_x']
                self.gain_y = settings['gain_y']
                self.mount_dx1 = settings.get('mount_dx1', 0)
                self.mount_dy1 = settings.get('mount_dy1', 0)
                self.mount_dx2 = settings.get('mount_dx2', 0)
                self.mount_dy2 = settings.get('mount_dy2', 0)
                self.mount_calibrated = 1
        except Exception as e:
            log.critical("An error occurred while loading the state:", e)
            self.reset()

    def reset(self):
        self.is_guiding = 0
        self.mount_dx1 = 0
        self.mount_dy1 = 0
        self.mount_dx2 = 0
        self.mount_dy2 = 0

    def calibrate_mount(self):
        self.mount_cal_state_count = 40
        self.mount_calibrated = 0
        log.info("Calibrating Mount")

    def handle_calibrate_mount(self, x, y):
        N = 0.4
        log.info(f"Handling mount calibration position: {x}, {y}")
        if self.mount_cal_state_count == 40:
            self.mount_pos_x0, self.mount_pos_y0 = x, y
            self.mount.jog(N, 0)
            log.info("Move Mount X")
        elif self.mount_cal_state_count == 30:
            self.mount_pos_x1, self.mount_pos_y1 = x, y
            self.mount.jog(-N, 0)
            log.info("Move Mount Back")
        elif self.mount_cal_state_count == 20:
            self.mount_pos_x2, self.mount_pos_y2 = x, y
            self.mount.jog(0, N)
            log.info("Move Mount Y")
        elif self.mount_cal_state_count == 10:
            self.mount_pos_x3, self.mount_pos_y3 = x, y
            self.mount.jog(0, -N)
            log.info("Move Mount Back")
        elif self.mount_cal_state_count == 1:
            self.calc_calibration_mount()

        self.mount_cal_state_count = max(0, self.mount_cal_state_count - 1)

    def calc_calibration_mount(self):
        log.info("Calculating Mount calibration")
        self.mount_dx1 = self.mount_pos_x1 - self.mount_pos_x0
        self.mount_dy1 = self.mount_pos_y1 - self.mount_pos_y0
        self.mount_dx2 = self.mount_pos_x3 - self.mount_pos_x2
        self.mount_dy2 = self.mount_pos_y3 - self.mount_pos_y2
        self.save_state("guide.data")
        self.mount_calibrated = 1

    def handle_guide(self, x, y):
        if self.guide_inited == 0:
            self.center_x, self.center_y = x, y
            self.guide_inited = 1
        else:
            dx = x - self.center_x
            dy = y - self.center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 20.0:
                log.info("Too far")
                return 0, 0

            # Mount correction logic
            tx = self.guide_gain * self.error_to_tx_mount(dx, dy)
            ty = self.guide_gain * self.error_to_ty_mount(dx, dy)
            
            log.info("ERROR %f %f, correction %f %f", dx, dy, tx, ty)
            self.fbump_mount(tx / 200.0, ty / 200.0)
            return dx, dy
        return 0, 0

    def guide(self):
        self.is_guiding = 1
        self.guide_inited = 0

    def fbump_mount(self, dx, dy):
        if self.mount is not None:
            log.info(f"Bumping mount: dx={dx}, dy={dy}")
            self.mount.jog(dx, dy)

    def pos_handler(self, x, y):
        if self.mount_cal_state_count != 0:
            log.info(f"Handling mount calibration: {x}, {y}")
            self.handle_calibrate_mount(x, y)
        elif self.is_guiding:
            return self.handle_guide(x, y)
        return 0, 0

    def offset(self, dx, dy):
        self.center_x += dx
        self.center_y += dy
        log.info(f"New guide position: {self.center_x}, {self.center_y}")

    def error_to_tx_mount(self, mx, my):
        den = (self.mount_dx1 * self.mount_dy2) - (self.mount_dx2 * self.mount_dy1)
        if abs(den) < 1e-6: return 0
        num = (self.mount_dy2 * mx) - (self.mount_dx2 * my)
        return num / den

    def error_to_ty_mount(self, mx, my):
        den = (self.mount_dx2 * self.mount_dy1) - (self.mount_dx1 * self.mount_dy2)
        if abs(den) < 1e-6: return 0
        num = (self.mount_dy1 * mx) - (self.mount_dx1 * my)
        return num / den
