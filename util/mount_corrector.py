import logging
import pickle
import time
import numpy as np
import skyx
import collections

log = logging.getLogger(__name__)

class MountCorrector:
    """
    Handles the low-frequency correction of mount drift by observing the
    accumulated offset of a high-frequency AO system.
    """
    def __init__(self, skyx_instance):
        log.info("Initializing Mount Corrector")
        self.skyx = skyx_instance
        self.IMAGE_SCALE_ARCSEC_PER_PIXEL = 0.1375
        self.reset()
        self.load_state("mount_guide.data")

    def reset(self):
        """Resets all state variables to their defaults."""
        log.info("Resetting Mount Corrector state")
        self.mount_calibrated = False
        self.mount_cal_state_count = 0
        self.mount_pos_x0, self.mount_pos_y0 = 0, 0
        self.mount_pos_x1, self.mount_pos_y1 = 0, 0
        self.mount_pos_x2, self.mount_pos_y2 = 0, 0
        self.mount_pos_x3, self.mount_pos_y3 = 0, 0
        self.mount_dx1, self.mount_dy1 = 0, 0
        self.mount_dx2, self.mount_dy2 = 0, 0

        # New state for tracking rate adjustment
        self.guide_rate_ra_arcsec_per_sec = 0
        self.guide_rate_dec_arcsec_per_sec = 0
        self.correction_history = collections.deque(maxlen=100) # Store last 100 corrections
        self.last_rate_adjustment_time = time.time()
        self.ra_bump_rate = 0.0
        self.dec_bump_rate = 0.0


    def save_state(self, filename):
        """Saves the mount calibration data to a file."""
        if not self.mount_calibrated:
            log.warning("Mount not calibrated. Skipping saving state.")
            return
        settings = {
            'mount_dx1': self.mount_dx1, 'mount_dy1': self.mount_dy1,
            'mount_dx2': self.mount_dx2, 'mount_dy2': self.mount_dy2,
            'guide_rate_ra_arcsec_per_sec': self.guide_rate_ra_arcsec_per_sec,
            'guide_rate_dec_arcsec_per_sec': self.guide_rate_dec_arcsec_per_sec,
        }
        try:
            with open(filename, "wb") as f:
                pickle.dump(settings, f)
            log.info(f"Mount corrector state saved to {filename}")
        except Exception as e:
            log.error(f"Error saving mount corrector state to {filename}: {e}", exc_info=True)

    def load_state(self, filename):
        """Loads mount calibration data from a file."""
        try:
            with open(filename, "rb") as f:
                settings = pickle.load(f)
            
            self.mount_dx1 = settings.get('mount_dx1', 0)
            self.mount_dy1 = settings.get('mount_dy1', 0)
            self.mount_dx2 = settings.get('mount_dx2', 0)
            self.mount_dy2 = settings.get('mount_dy2', 0)
            self.guide_rate_ra_arcsec_per_sec = settings.get('guide_rate_ra_arcsec_per_sec', 0)
            self.guide_rate_dec_arcsec_per_sec = settings.get('guide_rate_dec_arcsec_per_sec', 0)

            det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
            if abs(det) > 1e-3:
                self.mount_calibrated = True
                log.info(f"Mount corrector state loaded from {filename}. Mount is considered calibrated.")
            else:
                log.warning(f"Loaded mount state from {filename}, but calibration seems invalid. Re-calibration needed.")
                self.mount_calibrated = False
        except FileNotFoundError:
            log.warning(f"Mount state file {filename} not found. Using default values. Mount requires calibration.")
            self.reset()
        except Exception as e:
            log.error(f"Error loading mount corrector state from {filename}: {e}. Resetting to defaults.", exc_info=True)
            self.reset()

    def calibrate_mount(self, N=0.5):
        """Starts the mount calibration sequence."""
        self.reset()
        self.mount_cal_state_count = 40 
        self.mount_calibrated = False
        self._calibration_jog_amount = N  # Jog in seconds
        logging.getLogger('aoscale').info(f"--- START MOUNT CALIBRATION --- Jog Duration N={N} seconds")
        log.info(f"Starting Mount Calibration with jog duration N={N} seconds")

    def handle_calibrate_mount(self, x, y):
        """Processes a single step in the mount calibration state machine."""
        N = self._calibration_jog_amount
        logging.getLogger('aoscale').info(f"Step {self.mount_cal_state_count}: Star at ({x:.2f}, {y:.2f})")
        log.debug(f"Handling mount calibration step {self.mount_cal_state_count} at pixel: ({x:.2f}, {y:.2f})")

        if self.mount_cal_state_count == 40:
            self.mount_pos_x0, self.mount_pos_y0 = x, y
            log.info(f"Calib Step 1/4: Recorded initial position ({x:.2f}, {y:.2f}). Jogging mount +RA...")
            self.skyx.bump(N, 0)
        elif self.mount_cal_state_count == 30:
            self.mount_pos_x1, self.mount_pos_y1 = x, y
            log.info(f"Calib Step 2/4: Position after +RA jog ({x:.2f}, {y:.2f}). Returning to center...")
            self.skyx.bump(-N, 0)
        elif self.mount_cal_state_count == 20:
            self.mount_pos_x2, self.mount_pos_y2 = x, y
            log.info(f"Calib Step 3/4: Position after return ({x:.2f}, {y:.2f}). Jogging mount +DEC...")
            self.skyx.bump(0, N)
        elif self.mount_cal_state_count == 10:
            self.mount_pos_x3, self.mount_pos_y3 = x, y
            log.info(f"Calib Step 4/4: Position after +Y jog ({x:.2f}, {y:.2f}). Returning to center...")
            self.skyx.bump(0, -N)
        elif self.mount_cal_state_count == 1:
            log.info(f"Final position measurement ({x:.2f}, {y:.2f}). Calculating calibration...")
            self.calc_calibration_mount()

        if self.mount_cal_state_count > 0:
            self.mount_cal_state_count -= 1

    def calc_calibration_mount(self):
        """Calculates the mount calibration matrix from recorded positions."""
        log.info("Calculating Mount calibration vectors")
        N = self._calibration_jog_amount

        self.mount_dx1 = (self.mount_pos_x1 - self.mount_pos_x0) / N
        self.mount_dy1 = (self.mount_pos_y1 - self.mount_pos_y0) / N
        self.mount_dx2 = (self.mount_pos_x3 - self.mount_pos_x2) / N
        self.mount_dy2 = (self.mount_pos_y3 - self.mount_pos_y2) / N

        logging.getLogger('aoscale').info(f"CALC MOUNT: N={N}, P0=({self.mount_pos_x0:.2f}, {self.mount_pos_y0:.2f}), P1=({self.mount_pos_x1:.2f}, {self.mount_pos_y1:.2f}), P2=({self.mount_pos_x2:.2f}, {self.mount_pos_y2:.2f}), P3=({self.mount_pos_x3:.2f}, {self.mount_pos_y3:.2f})")
        logging.getLogger('aoscale').info(f"CALC MOUNT RESULT: mount_dx1={self.mount_dx1:.4f}, mount_dy1={self.mount_dy1:.4f}, mount_dx2={self.mount_dx2:.4f}, mount_dy2={self.mount_dy2:.4f} (pixels/sec)")

        log.info(f"Jog RA ({N:.2f} sec) -> dPix/sec: dX1={self.mount_dx1:.3f}, dY1={self.mount_dy1:.3f}")
        log.info(f"Jog DEC ({N:.2f} sec) -> dPix/sec: dX2={self.mount_dx2:.3f}, dY2={self.mount_dy2:.3f}")

        # Calculate the effective guide rate in arcseconds/second
        ra_move_pixels_per_sec = np.sqrt(self.mount_dx1**2 + self.mount_dy1**2)
        dec_move_pixels_per_sec = np.sqrt(self.mount_dx2**2 + self.mount_dy2**2)
        self.guide_rate_ra_arcsec_per_sec = ra_move_pixels_per_sec * self.IMAGE_SCALE_ARCSEC_PER_PIXEL
        self.guide_rate_dec_arcsec_per_sec = dec_move_pixels_per_sec * self.IMAGE_SCALE_ARCSEC_PER_PIXEL
        logging.getLogger('aoscale').info(f"CALC GUIDE RATE: RA={self.guide_rate_ra_arcsec_per_sec:.4f} arcsec/s, DEC={self.guide_rate_dec_arcsec_per_sec:.4f} arcsec/s")


        det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
        if abs(det) < 1e-3:
            log.error("Mount calibration FAILED: Matrix determinant is near zero.")
            self.mount_calibrated = False
        else:
            log.info("Mount calibration SUCCESSFUL.")
            self.mount_calibrated = True
            self.save_state("mount_guide.data")
        
        self.mount_cal_state_count = 0

    def calculate_mount_correction(self, pixel_error_x, pixel_error_y):
        """Calculates the required mount jog in seconds to correct a pixel error."""
        if not self.mount_calibrated:
            return 0, 0
        
        det = self.mount_dx1 * self.mount_dy2 - self.mount_dx2 * self.mount_dy1
        if abs(det) < 1e-6:
            return 0, 0

        inv_m11 = self.mount_dy2 / det
        inv_m12 = -self.mount_dx2 / det
        inv_m21 = -self.mount_dy1 / det
        inv_m22 = self.mount_dx1 / det

        # We want to move the star by the opposite of the pixel error
        mount_jog_ra = inv_m11 * -pixel_error_x + inv_m12 * -pixel_error_y
        mount_jog_dec = inv_m21 * -pixel_error_x + inv_m22 * -pixel_error_y

        return mount_jog_ra, mount_jog_dec

    def correct_mount_drift(self, pico_offset_x, pico_offset_y, ao_guider):
        """
        Calculates and applies a mount correction to cancel out the PICO's
        accumulated offset, then resets the PICO.
        """
        if not self.mount_calibrated or not ao_guider.ao_calibrated:
            log.warning("Cannot correct mount drift: Mount or AO is not calibrated.")
            return

        logging.getLogger('aoscale').info(f"--- START MOUNT CORRECTION ---")
        logging.getLogger('aoscale').info(f"INPUT: Pico Offset=({pico_offset_x}, {pico_offset_y})")
        logging.getLogger('aoscale').info(f"INPUT: AO Calib=({ao_guider.ao_dx1:.4f}, {ao_guider.ao_dy1:.4f}, {ao_guider.ao_dx2:.4f}, {ao_guider.ao_dy2:.4f})")
        print("offset is ", pico_offset_x, pico_offset_y)

        # 1. How many pixels did the PICO offset correspond to?
        # This requires the AO calibration matrix (pixels/pico_unit)
        pico_pixel_dx = ao_guider.ao_dx1 * pico_offset_x + ao_guider.ao_dx2 * pico_offset_y
        pico_pixel_dy = ao_guider.ao_dy1 * pico_offset_x + ao_guider.ao_dy2 * pico_offset_y
        logging.getLogger('aoscale').info(f"STEP 1: Calculated Pixel Offset=({pico_pixel_dx:.4f}, {pico_pixel_dy:.4f})")
        print(f"PICO offset ({pico_offset_x}, {pico_offset_y}) corresponds to a star drift of ({pico_pixel_dx:.2f}, {pico_pixel_dy:.2f}) pixels.")

        # 2. What mount jog is needed to correct this pixel drift?
        # The pico_pixel_dx/dy is the CORRECTION the AO is applying.
        # The star's ERROR is the negative of that.
        jog_ra_sec, jog_dec_sec = self.calculate_mount_correction(-pico_pixel_dx, -pico_pixel_dy)
        logging.getLogger('aoscale').info(f"STEP 2: Calculated Mount Jog=({jog_ra_sec:.4f}, {jog_dec_sec:.4f}) seconds")
        print(f"Calculated mount correction: RA={jog_ra_sec:.2f}s, DEC={jog_dec_sec:.2f}s")

        # 3. Apply the correction
        if abs(jog_ra_sec) > 0.01 or abs(jog_dec_sec) > 0.01:
            logging.getLogger('aoscale').info(f"APPLYING CORRECTION: Jogging mount by ({jog_ra_sec:.4f}, {jog_dec_sec:.4f}) seconds.")
            log.info("Applying mount correction.")
            self.skyx.bump(jog_ra_sec, jog_dec_sec)
            # Record the correction for long-term rate adjustment
            self.correction_history.append({'t': time.time(), 'ra_jog': jog_ra_sec, 'dec_jog': jog_dec_sec})

            # Calculate the trend of bumping
            if len(self.correction_history) > 1:
                total_ra_jog = sum(c['ra_jog'] for c in self.correction_history)
                total_dec_jog = sum(c['dec_jog'] for c in self.correction_history)
                time_delta = self.correction_history[-1]['t'] - self.correction_history[0]['t']
                if time_delta > 1.0: # Avoid division by zero and ensure meaningful time has passed
                    self.ra_bump_rate = total_ra_jog / time_delta  # seconds/second
                    self.dec_bump_rate = total_dec_jog / time_delta # seconds/second
                    log.info(f"Updated bump rate trend: RA={self.ra_bump_rate:.4f} s/s, DEC={self.dec_bump_rate:.4f} s/s")

            return True
        else:
            logging.getLogger('aoscale').info(f"SKIPPING CORRECTION: Jog ({jog_ra_sec:.4f}, {jog_dec_sec:.4f}) is below threshold 0.01s.")
            log.info("Mount correction is too small. Skipping.")
            return False

    def proactive_bump(self, time_interval_seconds):
        """Applies a small, proactive bump based on the calculated long-term drift rate."""
        if self.ra_bump_rate == 0.0 and self.dec_bump_rate == 0.0:
            log.info("Proactive bump skipped: No bump rate trend established yet.")
            return

        # Calculate the proactive jog needed for the given time interval
        proactive_ra_jog = self.ra_bump_rate * time_interval_seconds
        proactive_dec_jog = self.dec_bump_rate * time_interval_seconds

        log.info(f"Calculated proactive bump: RA={proactive_ra_jog:.4f}s, DEC={proactive_dec_jog:.4f}s for {time_interval_seconds}s interval.")

        # Apply the correction if it's non-trivial
        if abs(proactive_ra_jog) > 0.01 or abs(proactive_dec_jog) > 0.01:
            log.info(f"Applying proactive bump.")
            self.skyx.bump(proactive_ra_jog, proactive_dec_jog)
        else:
            log.info("Proactive bump is too small. Skipping.")

    def adjust_tracking_rate(self):
        """Analyzes correction history and applies a persistent tracking rate adjustment."""
        if len(self.correction_history) < 10:
            log.info("Not enough correction samples to adjust tracking rate.")
            return

        # Calculate total jog and time duration from history
        total_ra_jog = sum(c['ra_jog'] for c in self.correction_history)
        total_dec_jog = sum(c['dec_jog'] for c in self.correction_history)
        
        time_delta = self.correction_history[-1]['t'] - self.correction_history[0]['t']
        if time_delta < 1.0: # Avoid division by zero
            return

        # Calculate drift rate in jog_seconds / wall_seconds
        drift_ra_jog_per_sec = total_ra_jog / time_delta
        drift_dec_jog_per_sec = total_dec_jog / time_delta

        # Convert drift rate to arcseconds/second using the calibrated guide rate
        # The negative sign is because we want to apply a rate that is OPPOSITE to the drift.
        rate_correction_ra = -1 * drift_ra_jog_per_sec * self.guide_rate_ra_arcsec_per_sec
        rate_correction_dec = -1 * drift_dec_jog_per_sec * self.guide_rate_dec_arcsec_per_sec

        logging.getLogger('aoscale').info(f"--- ADJUST TRACKING RATE ---")
        logging.getLogger('aoscale').info(f"Found {len(self.correction_history)} corrections over {time_delta:.1f}s.")
        logging.getLogger('aoscale').info(f"Total Jog: RA={total_ra_jog:.3f}s, DEC={total_dec_jog:.3f}s")
        logging.getLogger('aoscale').info(f"Drift Rate (jog s/s): RA={drift_ra_jog_per_sec:.4f}, DEC={drift_dec_jog_per_sec:.4f}")
        logging.getLogger('aoscale').info(f"Rate Correction (arcsec/s): RA={rate_correction_ra:.4f}, DEC={rate_correction_dec:.4f}")

        # Apply the new rate to the mount
        # Note: TheSkyX SetTracking adds to the existing rate, so we send the correction directly.
        self.skyx.rate(rate_correction_ra, rate_correction_dec)
        log.info(f"Applied tracking rate correction: RA={rate_correction_ra:.4f}, DEC={rate_correction_dec:.4f} arcsec/s")

        # Clear history so the next adjustment is based on new data
        self.correction_history.clear()
