# guider.py (Complete version with Kalman Filter and all original methods)

import logging
import pickle
import time
import numpy as np
import os

# --- Configure Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    try:
        # Use mode='w' to overwrite the log file for each new run
        file_handler = logging.FileHandler('mount_guide.log', mode='w')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
    except Exception as e:
        print(f"Error setting up logger for guider: {e}")
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)

# --- Kalman Filter Implementation ---
class KalmanFilter:
    """A simple Kalman filter for tracking a 1D state (position and velocity)."""
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.F = np.array([[1, dt], [0,  1]])      # State transition matrix
        self.H = np.array([[1, 0]])               # Observation matrix
        self.Q = np.array([[(dt**4)/4, (dt**3)/2], # Process noise covariance
                           [(dt**3)/2,  dt**2]]) * process_noise
        self.R = np.array([[measurement_noise]])  # Measurement noise covariance
        self.reset(position=0)

    def reset(self, position):
        """Resets the filter's state to a new initial position."""
        self.x = np.array([[position], [0]])  # State vector [position, velocity]
        self.P = np.eye(2) * 500              # Initial covariance (high uncertainty)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x

class guider:
    """
    Handles telescope mount guiding with a persistent Kalman Filter.
    """
    def __init__(self, mount, camera, loop_interval_s=0.2):
        log.info("Initializing Mount Guider with Persistent Kalman Filter")
        self.mount = mount
        self.camera = camera
        self.loop_interval_s = loop_interval_s
        self.kalman_state_filename = "kalman_state.data"
        self.mount_cal_filename = "mount_guide.data"

        # --- Guiding Parameters ---
        self.mount_gain_x = 0.5
        self.mount_gain_y = 0.5
        self.max_mount_correction_pix = 25.0
        self.guide_loop_delay_ms = loop_interval_s * 1000.0

        # --- Kalman Filter Tuning ---
        kalman_measurement_noise = 0.5 
        kalman_process_noise = 0.001

        self.kf_x = KalmanFilter(dt=loop_interval_s, process_noise=kalman_process_noise, measurement_noise=kalman_measurement_noise)
        self.kf_y = KalmanFilter(dt=loop_interval_s, process_noise=kalman_process_noise, measurement_noise=kalman_measurement_noise)

        self.reset()
        self.load_calibration_state()
        self.load_kalman_state()

    def current_milli_time(self):
        return time.time() * 1000.0

    def reset(self):
        log.info("Resetting Guider state")
        self.is_guiding = False
        self.center_x, self.center_y = 0, 0
        self.guide_inited_mount = 0
        self.mount_calibrated = False
        self.mount_cal_state_count = 0
        self.mount_pos_x0, self.mount_pos_y0 = 0, 0
        self.mount_pos_x1, self.mount_pos_y1 = 0, 0
        self.mount_pos_x2, self.mount_pos_y2 = 0, 0
        self.mount_pos_x3, self.mount_pos_y3 = 0, 0
        self.mount_dx1, self.mount_dy1 = 0, 0
        self.mount_dx2, self.mount_dy2 = 0, 0
        self.last_mount_bump_time = self.current_milli_time()
        self.guiding_errors_x, self.guiding_errors_y = [], []
        self.last_rms_calculation_time = self.current_milli_time()
        self.rms_x, self.rms_y = 0.0, 0.0
        self.rms_update_interval_ms = 10000

        if hasattr(self, 'kf_x'):
            self.kf_x.reset(position=0)
            self.kf_y.reset(position=0)

    def start_guide(self):
        if not self.mount_calibrated:
            log.warning("Mount not calibrated. Cannot start guiding.")
            return
        log.info("Starting mount guiding")
        self.is_guiding = True
        self.guide_inited_mount = 0
        self.guiding_errors_x.clear()
        self.guiding_errors_y.clear()
        self.last_rms_calculation_time = self.current_milli_time()

    def guide(self):
        self.start_guide()

    def stop_guide(self):
        log.info("Stopping mount guiding")
        self.is_guiding = False
        if len(self.guiding_errors_x) > 1:
            with np.errstate(invalid='ignore'):
                self.rms_x = np.sqrt(np.mean(np.square(self.guiding_errors_x)))
                self.rms_y = np.sqrt(np.mean(np.square(self.guiding_errors_y)))
            log.info(f"Final Guiding RMS: X={self.rms_x:.3f} px, Y={self.rms_y:.3f} px")
        self.guiding_errors_x.clear()
        self.guiding_errors_y.clear()

    def save_calibration_state(self):
        if not self.mount_calibrated: return
        log.info(f"Saving mount calibration state to {self.mount_cal_filename}")
        settings = {
            'mount_dx1': self.mount_dx1, 'mount_dy1': self.mount_dy1,
            'mount_dx2': self.mount_dx2, 'mount_dy2': self.mount_dy2,
            'mount_gain_x': self.mount_gain_x, 'mount_gain_y': self.mount_gain_y,
        }
        try:
            with open(self.mount_cal_filename, "wb") as f: pickle.dump(settings, f)
        except Exception as e:
            log.error(f"Error saving calibration state: {e}", exc_info=True)

    def load_calibration_state(self):
        if not os.path.exists(self.mount_cal_filename): return
        log.info(f"Loading mount calibration state from {self.mount_cal_filename}")
        try:
            with open(self.mount_cal_filename, "rb") as f: settings = pickle.load(f)
            self.mount_dx1 = settings.get('mount_dx1', 0)
            self.mount_dy1 = settings.get('mount_dy1', 0)
            self.mount_dx2 = settings.get('mount_dx2', 0)
            self.mount_dy2 = settings.get('mount_dy2', 0)
            self.mount_gain_x = settings.get('mount_gain_x', 0.5)
            self.mount_gain_y = settings.get('mount_gain_y', 0.5)
            det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
            if abs(det) > 1e-3:
                self.mount_calibrated = True
                log.info("Mount calibration successfully loaded.")
            else:
                log.warning("Loaded calibration data is invalid. Please re-calibrate.")
                self.mount_calibrated = False
        except Exception as e:
            log.error(f"Error loading calibration state: {e}. Using defaults.", exc_info=True)
            self.reset()
            
    def save_kalman_state(self):
        log.info(f"Saving Kalman filter state to {self.kalman_state_filename}")
        try:
            kalman_states = {
                'x_state': {'x': self.kf_x.x, 'P': self.kf_x.P},
                'y_state': {'x': self.kf_y.x, 'P': self.kf_y.P}
            }
            with open(self.kalman_state_filename, "wb") as f: pickle.dump(kalman_states, f)
        except Exception as e:
            log.error(f"Failed to save Kalman filter state: {e}", exc_info=True)

    def load_kalman_state(self):
        if not os.path.exists(self.kalman_state_filename):
            log.info("No saved Kalman filter state found. Starting fresh.")
            return
        log.info(f"Loading Kalman filter state from {self.kalman_state_filename}")
        try:
            with open(self.kalman_state_filename, "rb") as f: kalman_states = pickle.load(f)
            self.kf_x.x = kalman_states['x_state']['x']
            self.kf_x.P = kalman_states['x_state']['P']
            self.kf_y.x = kalman_states['y_state']['x']
            self.kf_y.P = kalman_states['y_state']['P']
            log.info("Kalman filter state successfully restored.")
        except Exception as e:
            log.error(f"Failed to load Kalman state: {e}. Starting fresh.", exc_info=True)
            self.kf_x.reset(position=0)
            self.kf_y.reset(position=0)

    def calibrate_mount(self, N=3):
        if self.is_guiding:
            log.warning("Cannot calibrate mount while guiding is active.")
            return
        self.reset()
        self.mount_cal_state_count = 40 
        self._calibration_jog_amount = N
        log.info(f"Starting Mount Calibration with jog amount N={N}")

    def handle_calibrate_mount(self, x, y):
        # This original method is restored verbatim
        N = self._calibration_jog_amount
        if self.mount_cal_state_count == 40:
            self.mount_pos_x0, self.mount_pos_y0 = x, y
            log.info(f"Calib Step 1/4: Initial pos ({x:.2f}, {y:.2f}). Jogging +X...")
            time.sleep(0.5); self.fbump_mount(N, 0)
        elif self.mount_cal_state_count == 30:
            self.mount_pos_x1, self.mount_pos_y1 = x, y
            log.info(f"Calib Step 2/4: Pos after +X jog ({x:.2f}, {y:.2f}). Returning...")
            time.sleep(0.5); self.fbump_mount(-N, 0)
        elif self.mount_cal_state_count == 20:
            self.mount_pos_x2, self.mount_pos_y2 = x, y
            log.info(f"Calib Step 3/4: Pos after return ({x:.2f}, {y:.2f}). Jogging +Y...")
            time.sleep(0.5); self.fbump_mount(0, N)
        elif self.mount_cal_state_count == 10:
            self.mount_pos_x3, self.mount_pos_y3 = x, y
            log.info(f"Calib Step 4/4: Pos after +Y jog ({x:.2f}, {y:.2f}). Returning...")
            time.sleep(0.5); self.fbump_mount(0, -N)
        elif self.mount_cal_state_count == 1:
            log.info(f"Final pos measurement ({x:.2f}, {y:.2f}). Calculating...")
            self.calc_calibration_mount()
        if self.mount_cal_state_count > 0: self.mount_cal_state_count -= 1

    def calc_calibration_mount(self):
        # This original method is restored verbatim
        log.info("Calculating Mount calibration vectors")
        N = self._calibration_jog_amount
        self.mount_dx1 = (self.mount_pos_x1 - self.mount_pos_x0) / N
        self.mount_dy1 = (self.mount_pos_y1 - self.mount_pos_y0) / N
        self.mount_dx2 = (self.mount_pos_x3 - self.mount_pos_x2) / N
        self.mount_dy2 = (self.mount_pos_y3 - self.mount_pos_y2) / N
        det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
        move_x_dist = np.sqrt((self.mount_pos_x1 - self.mount_pos_x0)**2 + (self.mount_pos_y1 - self.mount_pos_y0)**2)
        move_y_dist = np.sqrt((self.mount_pos_x3 - self.mount_pos_x2)**2 + (self.mount_pos_y3 - self.mount_pos_y2)**2)
        if abs(det) < 1e-1 or move_x_dist < 2.0 or move_y_dist < 2.0:
            log.error("Mount calibration FAILED. Check jog amount N or star detection.")
            self.mount_calibrated = False
        else:
            log.info("Mount calibration SUCCESSFUL.")
            self.mount_calibrated = True
            self.save_calibration_state()
        self.mount_cal_state_count = 0
        del self._calibration_jog_amount

    def calculate_mount_correction(self, pixel_error_x, pixel_error_y):
        # This original method is restored verbatim
        if not self.mount_calibrated: return 0, 0
        det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
        if abs(det) < 1e-6: return 0, 0
        target_pixel_move_x = -pixel_error_x
        target_pixel_move_y = -pixel_error_y
        inv_m11, inv_m12 = self.mount_dy2 / det, -self.mount_dx2 / det
        inv_m21, inv_m22 = -self.mount_dy1 / det, self.mount_dx1 / det
        mount_jog_x = inv_m11 * target_pixel_move_x + inv_m12 * target_pixel_move_y
        mount_jog_y = inv_m21 * target_pixel_move_x + inv_m22 * target_pixel_move_y
        return mount_jog_x, mount_jog_y

    def calculate_rms_if_needed(self):
        # This original method is restored verbatim
        if self.current_milli_time() - self.last_rms_calculation_time > self.rms_update_interval_ms:
            if len(self.guiding_errors_x) > 10:
                with np.errstate(invalid='ignore'):
                    self.rms_x = np.sqrt(np.mean(np.square(self.guiding_errors_x)))
                    self.rms_y = np.sqrt(np.mean(np.square(self.guiding_errors_y)))
                log.info(f"Guiding RMS: X={self.rms_x:.3f} px, Y={self.rms_y:.3f} px")
            self.last_rms_calculation_time = self.current_milli_time()
            
    def get_guide_rms(self):
        return self.rms_x, self.rms_y

    def handle_guide_mount(self, raw_x, raw_y):
        if not self.is_guiding or not self.mount_calibrated: return 0, 0
        if self.guide_inited_mount == 0:
            self.center_x, self.center_y = raw_x, raw_y
            self.kf_x.reset(position=raw_x); self.kf_y.reset(position=raw_y)
            self.guide_inited_mount = 1
            log.info(f"Guiding initialized. Target ({self.center_x:.2f}, {self.center_y:.2f})")
            return 0, 0
        self.kf_x.update(raw_x); self.kf_y.update(raw_y)
        filtered_x, filtered_y = self.kf_x.x[0, 0], self.kf_y.x[0, 0]
        log.debug(f"Raw:({raw_x:7.2f},{raw_y:7.2f}) -> Filtered:({filtered_x:7.2f},{filtered_y:7.2f})")
        dx = filtered_x - self.center_x
        dy = filtered_y - self.center_y
        self.guiding_errors_x.append(dx); self.guiding_errors_y.append(dy)
        self.calculate_rms_if_needed()
        if np.sqrt(dx**2 + dy**2) < 0.1: return dx, dy
        if self.current_milli_time() - self.last_mount_bump_time < self.guide_loop_delay_ms: return dx, dy
        mount_x, mount_y = self.calculate_mount_correction(dx, dy)
        corrected_mount_x = mount_x * self.mount_gain_x
        corrected_mount_y = mount_y * self.mount_gain_y
        if abs(corrected_mount_x) > 1e-5 or abs(corrected_mount_y) > 1e-5:
            log.info(f"Correction for filtered error ({dx:.2f},{dy:.2f}) -> Command({corrected_mount_x:.3f},{corrected_mount_y:.3f})")
            self.fbump_mount(corrected_mount_x, corrected_mount_y)
            self.last_mount_bump_time = self.current_milli_time()
        return dx, dy

    def fbump_mount(self, dx, dy):
        if self.mount:
            try: self.mount.bump(dx, dy)
            except Exception as e: log.error(f"Failed to send jog: {e}", exc_info=True)

    def pos_handler(self, x, y):
        if self.mount_cal_state_count > 0:
            self.handle_calibrate_mount(x, y); return 0, 0
        elif self.is_guiding:
            return self.handle_guide_mount(x, y)
        elif self.guide_inited_mount > 0:
            return x - self.center_x, y - self.center_y
        return 0, 0

    def set_pos(self, x, y):
        log.info(f"Manual Set Position: New target is ({x:.2f}, {y:.2f})")
        self.center_x, self.center_y = x, y
        if self.is_guiding:
            self.guide_inited_mount = 1
            self.kf_x.reset(position=x); self.kf_y.reset(position=y)
            self.guiding_errors_x.clear(); self.guiding_errors_y.clear()

    def offset(self, dx, dy):
        log.info(f"Applying guide target offset ({dx:.2f}, {dy:.2f})")
        new_center_x, new_center_y = self.center_x + dx, self.center_y + dy
        self.set_pos(new_center_x, new_center_y)

    def drizzle(self, dx, dy):
        self.offset(dx, dy)

    def close(self):
        log.info("Closing guider.")
        if self.is_guiding: self.stop_guide()
        self.save_kalman_state()
        self.save_calibration_state()

# ==============================================================================
# Simulation Block
# ==============================================================================
if __name__ == "__main__":
    class MockMount:
        def __init__(self): self.true_dx1, self.true_dy1, self.true_dx2, self.true_dy2 = 15.0, 2.0, -1.0, 12.0
        def bump(self, dx, dy):
            global mock_star_x, mock_star_y
            mock_star_x += self.true_dx1 * dx + self.true_dx2 * dy
            mock_star_y += self.true_dy1 * dx + self.true_dy2 * dy

    class MockCamera:
        def __init__(self): self.true_x, self.true_y = 200.0, 250.0
        def get_star_position(self):
            self.true_x += 0.02; self.true_y -= 0.01
            return self.true_x + np.random.randn()*0.7, self.true_y + np.random.randn()*0.7

    print("\n" + "="*20 + " Persistent Kalman Filter Guider Simulation " + "="*20)
    if os.path.exists("kalman_state.data"): os.remove("kalman_state.data")

    FRAME_INTERVAL_S = 0.5
    mock_mount = MockMount(); mock_camera = MockCamera()
    mock_star_x, mock_star_y = mock_camera.true_x, mock_camera.true_y

    print("\n--- First Run (No saved state) ---")
    guider1 = Guider(mock_mount, mock_camera, loop_interval_s=FRAME_INTERVAL_S)
    guider1.mount_calibrated = True # Skip calibration for this test
    guider1.start_guide()
    
    print("  Frame | Raw Position      | Filtered Position | Filtered Error")
    print("-" * 65)
    for i in range(20):
        raw_x, raw_y = mock_camera.get_star_position()
        dx, dy = guider1.pos_handler(raw_x, raw_y)
        if guider1.guide_inited_mount:
            fx, fy = guider1.kf_x.x[0,0], guider1.kf_y.x[0,0]
            print(f" {i+1:5d} | ({raw_x:6.2f}, {raw_y:6.2f}) | ({fx:6.2f}, {fy:6.2f}) | ({dx:5.2f}, {dy:5.2f})")
        time.sleep(0.02)
    guider1.close()
    print("--- First run finished and state saved. ---")

    print("\n--- Second Run (Loading saved state) ---")
    guider2 = Guider(mock_mount, mock_camera, loop_interval_s=FRAME_INTERVAL_S)
    guider2.mount_calibrated = True
    guider2.start_guide()
    
    print("  Frame | Raw Position      | Filtered Position | Filtered Error")
    print("-" * 65)
    for i in range(20):
        raw_x, raw_y = mock_camera.get_star_position()
        dx, dy = guider2.pos_handler(raw_x, raw_y)
        if guider2.guide_inited_mount:
            fx, fy = guider2.kf_x.x[0,0], guider2.kf_y.x[0,0]
            print(f" {i+1:5d} | ({raw_x:6.2f}, {raw_y:6.2f}) | ({fx:6.2f}, {fy:6.2f}) | ({dx:5.2f}, {dy:5.2f})")
        time.sleep(0.02)
    guider2.close()