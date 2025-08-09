# guider.py

import logging
import pickle
import time
import numpy as np

# --- Configure Logging ---
# This setup provides detailed logging for debugging and monitoring guider performance.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Prevent adding handlers multiple times if the module is reloaded
if not log.handlers:
    try:
        file_handler = logging.FileHandler('ao_guide.log') # Log to a file
        stream_handler = logging.StreamHandler() # Log to the console
        
        # A detailed formatter helps trace execution flow
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
    except Exception as e:
        print(f"Error setting up logger for guider: {e}")
        # Fallback to basic config if file handler fails
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        log = logging.getLogger(__name__)


class guider:
    """
    A class to handle AO guiding, including calibration,
    state persistence, and logic to avoid chasing atmospheric seeing.
    """
    def __init__(self, ao, camera):
        log.info("Initializing AO Guider")
        self.ao = ao
        self.camera = camera

        # --- Guiding Parameters ---
        self.ao_gain_x = 0.7
        self.ao_gain_y = 0.7
        self.max_ao_correction_pix = 25.0  # Max error before skipping correction
        self.guide_loop_delay_ms = 200  # Minimum time between AO corrections

        # Any star movement smaller than this magnitude is ignored as seeing/noise.
        self.seeing_threshold_pix = 0.25 

        # The 'reset' method initializes all state variables
        self.reset() 

        # Load previous calibration and gains if available
        self.load_state("ao_guide.data")

    def current_milli_time(self):
        """Returns the current time in milliseconds."""
        return time.time() * 1000.0

    def reset(self):
        """Resets all guiding and calibration state variables to their defaults."""
        log.info("Resetting Guider state")
        self.is_guiding = False
        self.center_x = 0
        self.center_y = 0
        self.guide_inited_ao = 0

        # Calibration state
        self.ao_calibrated = False
        self.ao_cal_state_count = 0
        self.ao_pos_x0, self.ao_pos_y0 = 0, 0
        self.ao_pos_x1, self.ao_pos_y1 = 0, 0
        self.ao_pos_x2, self.ao_pos_y2 = 0, 0
        self.ao_pos_x3, self.ao_pos_y3 = 0, 0
        self.ao_dx1, self.ao_dy1 = 0, 0
        self.ao_dx2, self.ao_dy2 = 0, 0
        
        self.last_ao_bump_time = self.current_milli_time()

        # Guiding RMS stats
        self.guiding_errors_x = []
        self.guiding_errors_y = []
        self.last_rms_calculation_time = self.current_milli_time()
        self.rms_x = 0.0
        self.rms_y = 0.0
        self.rms_update_interval_ms = 10000  # 10 seconds

    def start_guide(self):
        """Enables the guiding flag and resets guiding statistics."""
        if not self.ao_calibrated:
            log.warning("AO not calibrated. Cannot start guiding.")
            return
        log.info("Starting AO guiding")
        self.is_guiding = True
        self.guide_inited_ao = 0  # Re-initialize center on start
        
        # Reset RMS stats
        self.guiding_errors_x.clear()
        self.guiding_errors_y.clear()
        self.last_rms_calculation_time = self.current_milli_time()
        self.rms_x = 0.0
        self.rms_y = 0.0

    def guide(self):
        """Convenience method to start guiding."""
        self.start_guide()

    def stop_guide(self):
        """Disables the guiding flag and calculates final RMS."""
        log.info("Stopping AO guiding")
        self.is_guiding = False

        # Calculate final RMS for the session
        if len(self.guiding_errors_x) > 1:
            with np.errstate(invalid='ignore'):
                self.rms_x = np.sqrt(np.mean(np.square(self.guiding_errors_x)))
                self.rms_y = np.sqrt(np.mean(np.square(self.guiding_errors_y)))
            log.info(f"Final Guiding RMS: X={self.rms_x:.3f} px, Y={self.rms_y:.3f} px (over {len(self.guiding_errors_x)} samples)")
        
        self.guiding_errors_x.clear()
        self.guiding_errors_y.clear()

    def save_state(self, filename):
        """Saves the AO calibration data and gains to a file."""
        if not self.ao_calibrated:
            log.warning("AO not calibrated. Skipping saving state.")
            return
        settings = {
            'ao_dx1': self.ao_dx1, 'ao_dy1': self.ao_dy1,
            'ao_dx2': self.ao_dx2, 'ao_dy2': self.ao_dy2,
            'ao_gain_x': self.ao_gain_x, 'ao_gain_y': self.ao_gain_y,
        }
        try:
            with open(filename, "wb") as f:
                pickle.dump(settings, f)
            log.info(f"Guider state saved to {filename}")
        except Exception as e:
            log.error(f"Error saving guider state to {filename}: {e}", exc_info=True)

    def load_state(self, filename):
        """Loads AO calibration data and gains from a file."""
        try:
            with open(filename, "rb") as f:
                settings = pickle.load(f)
            
            self.ao_dx1 = settings.get('ao_dx1', 0)
            self.ao_dy1 = settings.get('ao_dy1', 0)
            self.ao_dx2 = settings.get('ao_dx2', 0)
            self.ao_dy2 = settings.get('ao_dy2', 0)
            self.ao_gain_x = settings.get('ao_gain_x', 0.7)
            self.ao_gain_y = settings.get('ao_gain_y', 0.7)

            det = self.ao_dx1 * self.ao_dy2 - self.ao_dx2 * self.ao_dy1
            if abs(det) > 1e-3:
                self.ao_calibrated = True
                log.info(f"Guider state loaded from {filename}. AO is considered calibrated.")
                log.info(f"Loaded cal: dx1={self.ao_dx1:.2f}, dy1={self.ao_dy1:.2f}, dx2={self.ao_dx2:.2f}, dy2={self.ao_dy2:.2f}")
            else:
                log.warning(f"Loaded state from {filename}, but calibration values seem invalid (det={det:.4f}). AO requires re-calibration.")
                self.ao_calibrated = False
        except FileNotFoundError:
            log.warning(f"State file {filename} not found. Using default values. AO requires calibration.")
            self.reset()
        except Exception as e:
            log.error(f"Error loading guider state from {filename}: {e}. Resetting to defaults.", exc_info=True)
            self.reset()

    def calibrate_ao(self, N=3):
        """Starts the AO calibration sequence."""
        if self.is_guiding:
            log.warning("Cannot calibrate AO while guiding is active.")
            return
        self.reset() # Reset all state before starting
        self.ao_cal_state_count = 40 
        self.ao_calibrated = False
        self._calibration_jog_amount = N
        logging.getLogger('aoscale').info(f"--- START AO CALIBRATION --- Jog Amount N={N}")
        log.info(f"Starting AO Calibration with jog amount N={N}")

    def handle_calibrate_ao(self, x, y):
        """Processes a single step in the AO calibration state machine."""
        N = self._calibration_jog_amount
        logging.getLogger('aoscale').info(f"Step {self.ao_cal_state_count}: Star at ({x:.2f}, {y:.2f})")
        log.debug(f"Handling AO calibration step {self.ao_cal_state_count} at pixel: ({x:.2f}, {y:.2f})")

        if self.ao_cal_state_count == 40:
            self.ao_pos_x0, self.ao_pos_y0 = x, y
            log.info(f"Calib Step 1/4: Recorded initial position ({x:.2f}, {y:.2f}). Jogging AO +X...")
            time.sleep(0.5)
            self.fbump_ao(N, 0)
        elif self.ao_cal_state_count == 30:
            self.ao_pos_x1, self.ao_pos_y1 = x, y
            log.info(f"Calib Step 2/4: Position after +X jog ({x:.2f}, {y:.2f}). Returning to center...")
            time.sleep(0.5)
            self.fbump_ao(-N, 0)
        elif self.ao_cal_state_count == 20:
            self.ao_pos_x2, self.ao_pos_y2 = x, y
            log.info(f"Calib Step 3/4: Position after return ({x:.2f}, {y:.2f}). Jogging AO +Y...")
            time.sleep(0.5)
            self.fbump_ao(0, N)
        elif self.ao_cal_state_count == 10:
            self.ao_pos_x3, self.ao_pos_y3 = x, y
            log.info(f"Calib Step 4/4: Position after +Y jog ({x:.2f}, {y:.2f}). Returning to center...")
            time.sleep(0.5)
            self.fbump_ao(0, -N)
        elif self.ao_cal_state_count == 1:
            log.info(f"Final position measurement ({x:.2f}, {y:.2f}). Calculating calibration...")
            self.calc_calibration_ao()

        if self.ao_cal_state_count > 0:
            self.ao_cal_state_count -= 1
            
    def calc_calibration_ao(self):
        """Calculates the AO calibration matrix based on recorded positions."""
        log.info("Calculating AO calibration vectors")
        N = self._calibration_jog_amount

        # Pixel change vector for the X jog, correcting for the 100x scaling in pico_AO.bump
        self.ao_dx1 = (self.ao_pos_x1 - self.ao_pos_x0) / (N * 100)
        self.ao_dy1 = (self.ao_pos_y1 - self.ao_pos_y0) / (N * 100)

        # Pixel change vector for the Y jog, correcting for the 100x scaling in pico_AO.bump
        self.ao_dx2 = (self.ao_pos_x3 - self.ao_pos_x2) / (N * 100)
        self.ao_dy2 = (self.ao_pos_y3 - self.ao_pos_y2) / (N * 100)

        logging.getLogger('aoscale').info(f"CALC AO: N={N}, P0=({self.ao_pos_x0:.2f}, {self.ao_pos_y0:.2f}), P1=({self.ao_pos_x1:.2f}, {self.ao_pos_y1:.2f}), P2=({self.ao_pos_x2:.2f}, {self.ao_pos_y2:.2f}), P3=({self.ao_pos_x3:.2f}, {self.ao_pos_y3:.2f})")
        logging.getLogger('aoscale').info(f"CALC AO RESULT: ao_dx1={self.ao_dx1:.4f}, ao_dy1={self.ao_dy1:.4f}, ao_dx2={self.ao_dx2:.4f}, ao_dy2={self.ao_dy2:.4f}")

        log.info(f"Jog X ({N:.2f} units) -> dPix/unit: dX1={self.ao_dx1:.2f}, dY1={self.ao_dy1:.2f}")
        log.info(f"Jog Y ({N:.2f} units) -> dPix/unit: dX2={self.ao_dx2:.2f}, dY2={self.ao_dy2:.2f}")

        det = self.ao_dx1 * self.ao_dy2 - self.ao_dx2 * self.ao_dy1
        log.info(f"Calibration matrix determinant: {det:.4f}")

        move_x_dist = np.sqrt((self.ao_pos_x1 - self.ao_pos_x0)**2 + (self.ao_pos_y1 - self.ao_pos_y0)**2)
        move_y_dist = np.sqrt((self.ao_pos_x3 - self.ao_pos_x2)**2 + (self.ao_pos_y3 - self.ao_pos_y2)**2)
        min_move_pixels = 2.0

        if abs(det) < 1e-1 or move_x_dist < min_move_pixels or move_y_dist < min_move_pixels:
            log.error(f"AO calibration FAILED: det={det:.4f}, X move={move_x_dist:.2f}px, Y move={move_y_dist:.2f}px. Check jog amount N or star detection.")
            self.ao_calibrated = False
            self.ao_dx1 = self.ao_dy1 = self.ao_dx2 = self.ao_dy2 = 0
        else:
            log.info("AO calibration SUCCESSFUL.")
            self.ao_calibrated = True
            self.save_state("ao_guide.data")

        self.ao_cal_state_count = 0
        del self._calibration_jog_amount

    def calculate_ao_correction(self, pixel_error_x, pixel_error_y):
        """Calculates required AO jog to correct a given pixel error."""
        if not self.ao_calibrated:
            return 0, 0
        
        det = self.ao_dx1 * self.ao_dy2 - self.ao_dx2 * self.ao_dy1
        if abs(det) < 1e-6:
            log.warning("AO calibration matrix determinant is near zero. Cannot invert.")
            return 0, 0

        # We want a pixel change opposite to the error
        target_pixel_move_x = -pixel_error_x
        target_pixel_move_y = -pixel_error_y

        # Invert the calibration matrix to find the required AO jog
        inv_m11 = self.ao_dy2 / det
        inv_m12 = -self.ao_dx2 / det
        inv_m21 = -self.ao_dy1 / det
        inv_m22 = self.ao_dx1 / det

        ao_jog_x = inv_m11 * target_pixel_move_x + inv_m12 * target_pixel_move_y
        ao_jog_y = inv_m21 * target_pixel_move_x + inv_m22 * target_pixel_move_y

        return ao_jog_x, ao_jog_y

    def calculate_rms_if_needed(self):
        """Calculates RMS of guiding errors periodically."""
        current_time = self.current_milli_time()
        if current_time - self.last_rms_calculation_time > self.rms_update_interval_ms:
            if len(self.guiding_errors_x) > 10:
                with np.errstate(invalid='ignore'):
                    self.rms_x = np.sqrt(np.mean(np.square(self.guiding_errors_x)))
                    self.rms_y = np.sqrt(np.mean(np.square(self.guiding_errors_y)))
                log.info(f"Guiding RMS: X={self.rms_x:.3f} px, Y={self.rms_y:.3f} px")
                
                # Keep last ~2 intervals of data to prevent list from growing forever
                max_samples = 2 * int(self.rms_update_interval_ms / (self.guide_loop_delay_ms + 1))
                if len(self.guiding_errors_x) > max_samples:
                    self.guiding_errors_x = self.guiding_errors_x[-max_samples:]
                    self.guiding_errors_y = self.guiding_errors_y[-max_samples:]
            
            self.last_rms_calculation_time = current_time

    def get_guide_rms(self):
        """Returns the last calculated RMS guiding error for X and Y."""
        return self.rms_x, self.rms_y

    def handle_guide_ao(self, x, y):
        """
        Handles guiding logic. Corrects for any error above the seeing threshold.
        """
        if not self.is_guiding or not self.ao_calibrated:
            return 0, 0

        if self.guide_inited_ao == 0:
            self.center_x, self.center_y = x, y
            self.guide_inited_ao = 1
            log.info(f"Guiding initialized. Target center set to ({self.center_x:.2f}, {self.center_y:.2f})")
            self.reset_guiding_state()
            return 0, 0

        dx = x - self.center_x
        dy = y - self.center_y
        error_magnitude = np.sqrt(dx*dx + dy*dy)
        
        self.guiding_errors_x.append(dx)
        self.guiding_errors_y.append(dy)
        self.calculate_rms_if_needed()

        if error_magnitude <= self.seeing_threshold_pix:
            log.debug(f"Error ({error_magnitude:.2f}) is within seeing threshold. No action.")
            return dx, dy

        # --- Correction Block ---
        if error_magnitude > self.max_ao_correction_pix:
            log.warning(f"Guide error ({error_magnitude:.1f}px) exceeds limit ({self.max_ao_correction_pix:.1f}px). Skipping.")
            return dx, dy

        dt = self.current_milli_time() - self.last_ao_bump_time
        if dt < self.guide_loop_delay_ms:
            log.debug(f"Skipping AO bump: dt={dt:.0f}ms < {self.guide_loop_delay_ms}ms")
            return dx, dy

        ao_x, ao_y = self.calculate_ao_correction(dx, dy)
        corrected_ao_x = ao_x * self.ao_gain_x
        corrected_ao_y = ao_y * self.ao_gain_y

        if abs(corrected_ao_x) > 1e-4 or abs(corrected_ao_y) > 1e-4:
            log.debug(f"Applying AO correction: JogX={corrected_ao_x:.4f}, JogY={corrected_ao_y:.4f}")
            self.fbump_ao(corrected_ao_x, corrected_ao_y)
            self.last_ao_bump_time = self.current_milli_time()
        
        return dx, dy

    def fbump_ao(self, dx, dy):
        """Sends a jog command to the AO, with error handling."""
        if self.ao is not None:
            try:
                self.ao.bump(dx, dy)
            except Exception as e:
                log.error(f"Failed to send jog command to AO: {e}", exc_info=True)
        else:
            log.warning("fbump_ao called but AO object is None.")

    def pos_handler(self, x, y):
        """Primary handler for incoming star positions. Routes to calibration or guiding."""
        log.debug(f"pos_handler received position: ({x:.2f}, {y:.2f})")
        if self.ao_cal_state_count > 0:
            self.handle_calibrate_ao(x, y)
            return 0, 0
        elif self.is_guiding:
            return self.handle_guide_ao(x, y)
        else: # Not guiding and not calibrating
            if self.guide_inited_ao > 0: # Report drift from last center if guiding was stopped
                return x - self.center_x, y - self.center_y
            return 0, 0

    def reset_guiding_state(self):
        """Resets timers and states associated with an active guide session."""
        self.last_ao_bump_time = self.current_milli_time()
        self.last_rms_calculation_time = self.current_milli_time()
        self.guiding_errors_x.clear()
        self.guiding_errors_y.clear()

    def set_pos(self, x, y):
        """Manually sets the target guide star position."""
        log.info(f"Manual Set Position: New target is ({x:.2f}, {y:.2f})")
        self.center_x = x
        self.center_y = y
        if self.is_guiding:
            self.guide_inited_ao = 1
            self.reset_guiding_state()

    def offset(self, dx, dy):
        """Applies an offset to the current guide target (for dithering)."""
        old_center_x, old_center_y = self.center_x, self.center_y
        self.center_x += dx
        self.center_y += dy
        log.info(f"Applied offset ({dx:.2f}, {dy:.2f}). Target moved from ({old_center_x:.2f}, {old_center_y:.2f}) to ({self.center_x:.2f}, {self.center_y:.2f})")
        if self.is_guiding:
            self.reset_guiding_state() # Reset to allow immediate correction to new target

    def close(self):
        """Shuts down the guider gracefully."""
        log.info("Closing guider.")
        self.stop_guide()

# ==============================================================================
# Example Usage and Simulation Block
# ==============================================================================
if __name__ == "__main__":
    
    # --- Mock Objects for Testing ---
    class MockAO:
        def __init__(self):
            # Simulate a AO's true physical response to jog commands
            self.true_dx1 = 15.0  # Pixels X change for +1 unit X jog
            self.true_dy1 = 2.0   # Pixels Y change for +1 unit X jog (axis non-orthogonality)
            self.true_dx2 = -1.0  # Pixels X change for +1 unit Y jog (axis non-orthogonality)
            self.true_dy2 = 12.0  # Pixels Y change for +1 unit Y jog
            print("MOCK AO: Initialized with TRUE response matrix.")

        def jog(self, dx, dy):
            global mock_star_x, mock_star_y
            pixel_change_x = self.true_dx1 * dx + self.true_dx2 * dy
            pixel_change_y = self.true_dy1 * dx + self.true_dy2 * dy
            mock_star_x += pixel_change_x
            mock_star_y += pixel_change_y
            log.debug(f"MOCK AO: Jog ({dx:.4f}, {dy:.4f}). Star moved by ({pixel_change_x:+.2f}, {pixel_change_y:+.2f})px -> New pos ({mock_star_x:.2f}, {mock_star_y:.2f})")

    class MockCamera:
        def get_star_position(self):
            global mock_star_x, mock_star_y
            # Simulate systematic drift (e.g., polar alignment error)
            drift_x = 0.08 
            drift_y = -0.06
            mock_star_x += drift_x
            mock_star_y += drift_y
            # Simulate random measurement noise and seeing
            noise_x = np.random.randn() * 0.15 
            noise_y = np.random.randn() * 0.15
            return mock_star_x + noise_x, mock_star_y + noise_y

    # --- Simulation Setup ---
    print("\n" + "="*20 + " Guider Simulation Start " + "="*20)
    mock_ao = MockAO()
    mock_camera = MockCamera()

    # Global variable for star position shared between mocks
    mock_star_x = 200.0
    mock_star_y = 250.0

    the_guider = guider(mock_ao, mock_camera)

    # --- Run Calibration ---
    print("\n--- Starting Calibration ---")
    the_guider.calibrate_ao(N=0.2)

    max_cal_frames = 50
    frame_count = 0
    while the_guider.ao_cal_state_count > 0 and frame_count < max_cal_frames:
        frame_count += 1
        current_x, current_y = mock_camera.get_star_position()
        log.debug(f"Sim Cal Frame {frame_count}: Star at ({current_x:.2f}, {current_y:.2f})")
        the_guider.pos_handler(current_x, current_y)
        time.sleep(0.05) 

    if not the_guider.ao_calibrated:
        print("\n--- Calibration Failed. Exiting. ---")
    else:
        print("\n--- Calibration Complete ---")
        log.info(f"Guider Calculated dX/JogX = {the_guider.ao_dx1:.2f}, dY/JogX = {the_guider.ao_dy1:.2f}")
        log.info(f"Guider Calculated dX/JogY = {the_guider.ao_dx2:.2f}, dY/JogY = {the_guider.ao_dy2:.2f}")

        # --- Run Guiding ---
        print("\n--- Starting Guiding Simulation ---")
        the_guider.start_guide()
        
        for i in range(150): # Simulate 150 frames
            current_x, current_y = mock_camera.get_star_position()
            print(f"\nSim Guide Frame {i+1}: Star at ({current_x:.2f}, {current_y:.2f})")
            
            dx, dy = the_guider.pos_handler(current_x, current_y)
            
            if the_guider.is_guiding and the_guider.guide_inited_ao > 0:
                rms_x, rms_y = the_guider.get_guide_rms()
                dist = np.sqrt(dx*dx + dy*dy)
                print(f"  -> Target({the_guider.center_x:.2f}, {the_guider.center_y:.2f}), Error({dx:+.2f}, {dy:+.2f}), Dist:{dist:.2f}, RMS(X={rms_x:.2f}, Y={rms_y:.2f})")
            
            time.sleep(0.1)

        the_guider.stop_guide()
        print("\n--- Guiding Stopped ---")

    print("\n" + "="*20 + " Simulation End " + "="*20)