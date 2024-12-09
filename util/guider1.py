import logging
import pickle
import time
import numpy as np
from pico import ao
from util import LastNValues

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
        self.ao = ao()
        self.mount = mount
        self.camera = camera
        self.gain_x = 210.0
        self.gain_y = 210.0
        self.center_x = 0
        self.center_y = 0
        self.ao_cal_state_count = 0
        self.ao_calibrated = 0
        self.mount_calibrated = 0
        self.guide_inited_ao = 0
        self.centering_target_x = None
        self.centering_target_y = None
        self.centering_state = 0  # 0: not centering, 1: centering in progress
        self.pixels_per_ao_x = 1
        self.pixels_per_ao_y = 1
        self.mount_dx1 = 0  # Mount X movement for unit X command
        self.mount_dy1 = 0  # Mount Y movement for unit X command
        self.mount_dx2 = 0  # Mount X movement for unit Y command
        self.mount_dy2 = 0  # Mount Y movement for unit Y command
        self.ao_limit = 8  # AO deflection limit before using mount correction

        self.load_state("guide.data")
        self.last_ao_move_time = self.current_milli_time()

    def close(self):
        self.ao.goto(0, 0)

    def current_milli_time(self):
        return time.time() * 1000.0

    def fbump_ao(self, dx, dy):
        self.ao.goto(round(dx), round(dy))
        self.last_ao_move_time = self.current_milli_time()

    def clip(self, val):
        return max(-10, min(10, val))

    def fmove_ao(self, dx, dy):
        dx = self.clip(dx)
        dy = self.clip(dy)
        self.ao.move(round(dx), round(dy))
        if abs(dx) > 1 or abs(dy) > 1:
            self.last_ao_move_time = self.current_milli_time()

        ax, ay = self.ao.get_ao()
        log.info("*** tip-tilt *** %f %f", ax, ay)

    def reset_ao(self):
        self.fbump_ao(0, 0)
        self.last_ao_move_time = self.current_milli_time()
        time.sleep(0.3)
        self.guide_inited_ao = -5

    def start_guide(self):
        self.is_guiding = 1

    def stop_guide(self):
        self.is_guiding = 0

    def save_state(self, filename):
        settings = {
            'ao_dx1': self.ao_dx1,
            'ao_dx2': self.ao_dx2,
            'ao_dy1': self.ao_dy1,
            'ao_dy2': self.ao_dy2,
            'gain_x': self.gain_x,
            'gain_y': self.gain_y,
            'pixels_per_ao_x': self.pixels_per_ao_x,
            'pixels_per_ao_y': self.pixels_per_ao_y,
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
                self.ao_dx1 = settings['ao_dx1']
                self.ao_dx2 = settings['ao_dx2']
                self.ao_dy1 = settings['ao_dy1']
                self.ao_dy2 = settings['ao_dy2']
                self.gain_x = settings['gain_x']
                self.gain_y = settings['gain_y']
                self.pixels_per_ao_x = settings.get('pixels_per_ao_x', 1)
                self.pixels_per_ao_y = settings.get('pixels_per_ao_y', 1)
                self.mount_dx1 = settings.get('mount_dx1', 0)
                self.mount_dy1 = settings.get('mount_dy1', 0)
                self.mount_dx2 = settings.get('mount_dx2', 0)
                self.mount_dy2 = settings.get('mount_dy2', 0)
        except Exception as e:
            log.critical("An error occurred while loading the state:", e)
            self.reset()

    def reset(self):
        self.is_guiding = 0
        self.ao_dx1 = 0
        self.ao_dy1 = 0
        self.ao_dx2 = 0
        self.ao_dy2 = 0

    def calibrate_ao(self):
        self.ao_cal_state_count = 40
        self.ao_calibrated = 0
        log.info("Calibrating AO")

    def calibrate_mount(self):
        self.mount_cal_state_count = 40
        self.mount_calibrated = 0
        log.info("Calibrating Mount")

    def handle_calibrate_ao(self, x, y):
        N = 50  # Calibration step size
        log.info(f"Handling calibration position: {x}, {y}")
        if self.ao_cal_state_count == 40:
            self.ao_pos_x0, self.ao_pos_y0 = x, y
            self.fbump_ao(N, 0)
            log.info("Move Left")
        elif self.ao_cal_state_count == 30:
            self.ao_pos_x1, self.ao_pos_y1 = x, y
            self.fbump_ao(0, 0)
            log.info("Move Right")
        elif self.ao_cal_state_count == 20:
            self.ao_pos_x2, self.ao_pos_y2 = x, y
            self.fbump_ao(0, N)
            log.info("Move Up")
        elif self.ao_cal_state_count == 10:
            self.ao_pos_x3, self.ao_pos_y3 = x, y
            self.fbump_ao(0, 0)
            log.info("Move Down")
        elif self.ao_cal_state_count == 1:
            self.calc_calibration_ao()

        self.ao_cal_state_count = max(0, self.ao_cal_state_count - 1)

    def handle_calibrate_mount(self, x, y):
        N = 0.001  # Small mount movement for calibration
        log.info(f"Handling mount calibration position: {x}, {y}")
        if self.mount_cal_state_count == 40:
            self.mount_pos_x0, self.mount_pos_y0 = x, y
            self.mount.jog(N, 0)
            log.info("Move Mount X")
        elif self.mount_cal_state_count == 30:
            self.mount_pos_x1, self.mount_pos_y1 = x, y
            self.mount.jog(0, 0)
            log.info("Move Mount Back")
        elif self.mount_cal_state_count == 20:
            self.mount_pos_x2, self.mount_pos_y2 = x, y
            self.mount.jog(0, N)
            log.info("Move Mount Y")
        elif self.mount_cal_state_count == 10:
            self.mount_pos_x3, self.mount_pos_y3 = x, y
            self.mount.jog(0, 0)
            log.info("Move Mount Back")
        elif self.mount_cal_state_count == 1:
            self.calc_calibration_mount()

        self.mount_cal_state_count = max(0, self.mount_cal_state_count - 1)

    def calc_calibration_ao(self):
        log.info("Calculating AO calibration")
        self.ao_dx1 = self.ao_pos_x1 - self.ao_pos_x0
        self.ao_dy1 = self.ao_pos_y1 - self.ao_pos_y0
        self.ao_dx2 = -(self.ao_pos_x3 - self.ao_pos_x2)
        self.ao_dy2 = -(self.ao_pos_y3 - self.ao_pos_y2)
        
        # Calculate pixels per unit AO movement
        self.pixels_per_ao_x = self.ao_dx1 / 50  # 50 is the N value used in calibration
        self.pixels_per_ao_y = self.ao_dy2 / 50
        
        self.save_state("guide.data")
        self.ao_calibrated = 1

    def calc_calibration_mount(self):
        log.info("Calculating Mount calibration")
        self.mount_dx1 = self.mount_pos_x1 - self.mount_pos_x0
        self.mount_dy1 = self.mount_pos_y1 - self.mount_pos_y0
        self.mount_dx2 = self.mount_pos_x3 - self.mount_pos_x2
        self.mount_dy2 = self.mount_pos_y3 - self.mount_pos_y2
        self.save_state("guide.data")
        self.mount_calibrated = 1

    def calculate_mount_correction(self, ao_x, ao_y):
        # First, convert AO position to pixel error
        pixel_error_x = ao_x * self.pixels_per_ao_x
        pixel_error_y = ao_y * self.pixels_per_ao_y

        log.info(f"AO position (x, y): ({ao_x}, {ao_y})")
        log.info(f"Pixel error (x, y): ({pixel_error_x}, {pixel_error_y})")

        # Now use the mount calibration to determine required mount movement
        det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
        if abs(det) < 1e-6:
            log.warning("Mount calibration matrix is singular")
            return 0, 0
        
        inv_dx1 = self.mount_dy2 / det
        inv_dy1 = -self.mount_dy1 / det
        inv_dx2 = -self.mount_dx2 / det
        inv_dy2 = self.mount_dx1 / det

        mount_x = inv_dx1 * pixel_error_x + inv_dx2 * pixel_error_y
        mount_y = inv_dy1 * pixel_error_x + inv_dy2 * pixel_error_y

        log.info(f"Calculated mount correction (x, y): ({mount_x}, {mount_y})")

        return mount_x, mount_y

    def handle_guide_ao(self, x, y):
        if self.guide_inited_ao <= 0:
            self.center_x, self.center_y = x, y
            self.guide_inited_ao += 1
        else:
            dx, dy = x - self.center_x, y - self.center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 20.0:
                log.info("Too far")
                return 0, 0

            # Get current AO position
            ao_x, ao_y = self.ao.get_ao()

            log.info(f"Current AO position: ({ao_x}, {ao_y})")
            log.info(f"Pixel error: ({dx}, {dy})")

            # Check if AO correction is beyond the limit
            if abs(ao_x) > self.ao_limit or abs(ao_y) > self.ao_limit:
                # Calculate and apply mount correction
                mount_x, mount_y = self.calculate_mount_correction(ao_x, ao_y)
                self.fbump_mount(mount_x, mount_y)
                # Reset AO to center
                self.fmove_ao(0, 0)
            else:
                # Convert pixel error to AO units for smaller corrections
                ao_dx = dx / self.pixels_per_ao_x
                ao_dy = dy / self.pixels_per_ao_y

                # Apply a damping factor to avoid overcorrection
                damping_factor = 0.33
                ao_dx *= damping_factor
                ao_dy *= damping_factor

                log.info(f"AO correction: ({ao_dx}, {ao_dy})")
                self.fmove_ao(-1.0*ao_dx, 1.0*ao_dy)

            return dx, dy

    def guide(self):
        self.is_guiding = 1
        self.ao_calibrated = 1
        self.guide_inited_ao = 0

    def fbump_mount(self, dx, dy):
        if self.mount is not None:
            log.info(f"Bumping mount: dx={dx}, dy={dy}")
            self.mount.jog(dx, dy)

    def pos_handler(self, x, y):
        if self.ao_cal_state_count != 0:
            log.info(f"Handling AO calibration: {x}, {y}")
            self.handle_calibrate_ao(x, y)
        elif self.mount_cal_state_count != 0:
            log.info(f"Handling mount calibration: {x}, {y}")
            self.handle_calibrate_mount(x, y)
        elif self.centering_state == 1:
            return self.handle_centering(x, y)
        elif self.ao_calibrated != 0 and self.mount_calibrated != 0:
            return self.handle_guide_ao(x, y)

        return 0, 0

    def new_pos(self, x, y):
        log.info(f"New position: {x}, {y}")

    def set_pos(self, x, y):
        log.info(f"Set position: {x}, {y}")

    def offset(self, dx, dy):
        self.center_x += dx
        self.center_y += dy
        log.info(f"New guide position: {self.center_x}, {self.center_y}")

    def drizzle(self, dx, dy):
        self.center_x += dx
        self.center_y += dy

def start_centering(self, target_x, target_y):
        """
        Start the centering process to move the AO to a target position.
        """
        self.centering_target_x = target_x
        self.centering_target_y = target_y
        self.centering_state = 1
        log.info(f"Starting centering process to target ({target_x}, {target_y})")

    def handle_centering(self, x, y):
        dx = self.centering_target_x - x
        dy = self.centering_target_y - y
        distance = np.sqrt(dx*dx + dy*dy)

        if distance < 5:  # We're close enough
            log.info(f"Centering complete. Current position: ({x}, {y})")
            self.centering_state = 0
            self.center_x, self.center_y = x, y
            return 0, 0

        # Calculate step size (max step of 10)
        step_x = np.clip(dx / self.pixels_per_ao_x, -10, 10)
        step_y = np.clip(dy / self.pixels_per_ao_y, -10, 10)

        log.info(f"Centering: current ({x}, {y}), target ({self.centering_target_x}, {self.centering_target_y}), move ({step_x}, {step_y})")
        self.fmove_ao(-step_x, -step_y)  # Negative because we want to move opposite to the error

        return dx, dy

    def move_ao_to_position(self, target_x, target_y, timeout=30):
        """
        Move the AO to set the current bright pixel at a given x,y position.
        
        :param target_x: The target x-coordinate
        :param target_y: The target y-coordinate
        :param timeout: Maximum time in seconds to attempt the move
        :return: True if successful, False if timed out
        """
        self.start_centering(target_x, target_y)
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.centering_state == 0:
                log.info(f"Successfully moved to target position: ({self.center_x}, {self.center_y})")
                return True
            time.sleep(0.1)  # Wait a bit before next check

        log.warning(f"Timed out while trying to reach position ({target_x}, {target_y})")
        self.centering_state = 0  # Reset centering state
        return False