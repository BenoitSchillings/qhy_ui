
import sys
import argparse
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time
import json
import os

from zwo_cam_interface import zwoasi_wrapper
from skyx import sky6RASCOMTele
from util import compute_centroid_improved, HighValueFinder

class AutoCenterApp(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("AutoCenter Tool")
        self.setGeometry(100, 100, 1000, 1000)

        # --- UI Setup ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.image_view = pg.ImageView()
        self.layout.addWidget(self.image_view)
        self.status_label = QtWidgets.QLabel("Press 'K' to Calibrate, 'C' to Center, 'I' for Idle")
        self.layout.addWidget(self.status_label)

        # --- Core Components ---
        self.camera = None
        self.sky = None
        self.finder = HighValueFinder()
        self.mode = 'idle'  # Modes: idle, calibrating, centering
        self.calibration_data = {}
        self.calibration_step = 'idle'
        self.calib_start_pos = None
        self.calib_move_pixels = 100 # Target pixel movement for calibration

        self.center_target_pos = None
        self.center_tolerance = 2.0 # pixels

        # --- Load Calibration ---
        self.calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autocenter_calibration.json')
        self.load_calibration()

        # --- Connect Devices ---
        self.connect_devices()

        # --- Main Loop Timer ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.main_loop)
        if self.camera:
            self.timer.start(100) # ms

    def connect_devices(self):
        try:
            self.camera = zwoasi_wrapper(
                temp=-10, 
                exp=self.args.exp, 
                gain=self.args.gain, 
                binning=1, 
                crop=None, 
                cam_name=self.args.cam_name, 
                live=True
            )
            self.log(f"Connected to camera: {self.camera.name()}")
        except Exception as e:
            self.log(f"FATAL: Failed to connect to camera: {e}")
            self.camera = None

        try:
            self.sky = sky6RASCOMTele()
            self.sky.Connect()
            self.log("Connected to TheSkyX.")
        except Exception as e:
            self.log(f"Warning: Failed to connect to TheSkyX: {e}. Mount control disabled.")
            self.sky = None

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_K:
            self.start_calibration()
        elif key == QtCore.Qt.Key_C:
            self.start_centering()
        elif key == QtCore.Qt.Key_I:
            self.set_idle_mode()
        elif key == QtCore.Qt.Key_Q:
            self.close()

    def main_loop(self):
        if not self.camera:
            return

        frame = self.camera.get_frame()
        if frame is None:
            return

        star_pos = self.find_brightest_star(frame)
        self.update_display(frame, star_pos)

        if self.mode == 'calibrating':
            self.run_calibration_step(star_pos)
        elif self.mode == 'centering':
            self.run_centering_step(star_pos)

    def find_brightest_star(self, frame):
        try:
            max_y, max_x, _ = self.finder.find_high_value_element(frame[32:-32, 32:-32])
            cy, cx, _ = compute_centroid_improved(frame, max_y + 32, max_x + 32)
            return (cx, cy)
        except Exception as e:
            self.log(f"Star detection failed: {e}")
            return None

    def update_display(self, frame, star_pos):
        self.image_view.setImage(np.rot90(frame), autoRange=False, autoLevels=False)
        if star_pos:
            # Set levels based on a 50x50 cutout around the star
            x, y = int(star_pos[0]), int(star_pos[1])
            cutout = frame[max(0, y-25):y+25, max(0, x-25):x+25]
            if cutout.size > 0:
                min_val, max_val = np.min(cutout), np.max(cutout)
                self.image_view.setLevels(min_val, max_val)
        else:
            self.image_view.autoLevels()

    def set_idle_mode(self):
        self.mode = 'idle'
        self.calibration_step = 'idle'
        self.log("Mode set to IDLE. Press 'K' to Calibrate, 'C' to Center.")

    # --- Calibration Logic ---
    def start_calibration(self):
        if not self.sky:
            self.log("Cannot calibrate: Mount not connected.")
            return
        self.mode = 'calibrating'
        self.calibration_step = 'start'
        self.calibration_data = {}
        self.log("Starting mount calibration...")

    def run_calibration_step(self, star_pos):
        if star_pos is None and self.calibration_step != 'start':
            self.log("Calibration failed: Star lost.")
            self.set_idle_mode()
            return

        jog_amount = 2.0 # seconds of jogging at 1x sidereal
        
        if self.calibration_step == 'start':
            if star_pos is None: return # Wait for a star
            self.calib_start_pos = star_pos
            self.log(f"Calibration start pos: ({star_pos[0]:.1f}, {star_pos[1]:.1f})")
            self.calibration_step = 'move_ra'
            self.log(f"Moving RA+ for {jog_amount}s...")
            self.sky.jog(1.0, 0)
            QtCore.QTimer.singleShot(jog_amount * 1000, self.stop_and_measure_ra)

        # The rest of the steps are handled by callbacks (stop_and_measure_ra, etc.)
    
    def stop_and_measure_ra(self):
        self.sky.jog(0, 0)
        self.log("RA move finished. Measuring position...")
        self.calibration_step = 'measure_ra'
        # Give mount time to settle
        QtCore.QTimer.singleShot(2000, self.measure_ra_position)

    def measure_ra_position(self):
        frame = self.camera.get_frame()
        if frame is None:
            self.log("Calibration failed: No frame.")
            self.set_idle_mode()
            return
        
        star_pos = self.find_brightest_star(frame)
        if star_pos is None:
            self.log("Calibration failed: Lost star after RA move.")
            self.set_idle_mode()
            return

        dx = star_pos[0] - self.calib_start_pos[0]
        dy = star_pos[1] - self.calib_start_pos[1]
        self.calibration_data['ra_vector'] = (dx, dy)
        self.log(f"RA move resulted in pixel shift: (dx={dx:.2f}, dy={dy:.2f})")

        jog_amount = 2.0 # seconds
        self.calibration_step = 'move_dec'
        self.log(f"Moving Dec+ for {jog_amount}s...")
        self.sky.jog(0, 1.0)
        QtCore.QTimer.singleShot(jog_amount * 1000, self.stop_and_measure_dec)

    def stop_and_measure_dec(self):
        self.sky.jog(0, 0)
        self.log("Dec move finished. Measuring position...")
        self.calibration_step = 'measure_dec'
        QtCore.QTimer.singleShot(2000, self.measure_dec_position)

    def measure_dec_position(self):
        frame = self.camera.get_frame()
        if frame is None:
            self.log("Calibration failed: No frame.")
            self.set_idle_mode()
            return
            
        star_pos = self.find_brightest_star(frame)
        if star_pos is None:
            self.log("Calibration failed: Lost star after Dec move.")
            self.set_idle_mode()
            return

        dx = star_pos[0] - self.calib_start_pos[0]
        dy = star_pos[1] - self.calib_start_pos[1]
        self.calibration_data['dec_vector'] = (dx, dy)
        self.log(f"Dec move resulted in pixel shift: (dx={dx:.2f}, dy={dy:.2f})")

        self.finalize_calibration()

    def finalize_calibration(self):
        # We have two vectors:
        # (dx_ra, dy_ra) = self.calibration_data['ra_vector']
        # (dx_dec, dy_dec) = self.calibration_data['dec_vector']
        # We want to find the inverse matrix that maps (err_x, err_y) to (jog_ra, jog_dec)
        
        m = np.array([
            [self.calibration_data['ra_vector'][0], self.calibration_data['dec_vector'][0]],
            [self.calibration_data['ra_vector'][1], self.calibration_data['dec_vector'][1]]
        ])

        try:
            inv_m = np.linalg.inv(m)
            self.calibration_data['inverse_matrix'] = inv_m.tolist()
            self.log("Calibration successful! Matrix calculated.")
            self.save_calibration()
        except np.linalg.LinAlgError:
            self.log("Calibration failed: Could not invert calibration matrix. Moves may be co-linear.")
        
        self.set_idle_mode()

    # --- Centering Logic ---
    def start_centering(self):
        if not self.sky:
            self.log("Cannot center: Mount not connected.")
            return
        if 'inverse_matrix' not in self.calibration_data:
            self.log("Cannot center: Mount not calibrated. Press 'K' first.")
            return
        
        self.mode = 'centering'
        # Note: Camera size might be different if binning is used.
        # This should be handled by getting size from the camera object.
        h, w = self.camera.GetSize() 
        self.center_target_pos = (w / 2.0, h / 2.0)
        self.log(f"Starting centering. Target: {self.center_target_pos[0]:.1f}, {self.center_target_pos[1]:.1f}")

    def run_centering_step(self, star_pos):
        if star_pos is None:
            self.log("Centering failed: Lost star.")
            self.set_idle_mode()
            return

        err_x = self.center_target_pos[0] - star_pos[0]
        err_y = self.center_target_pos[1] - star_pos[1]
        err_dist = math.hypot(err_x, err_y)
        self.log(f"Centering: Star at ({star_pos[0]:.2f}, {star_pos[1]:.2f}), Error: {err_dist:.2f} pixels")

        if err_dist < self.center_tolerance:
            self.log("Centering complete!")
            self.set_idle_mode()
            return

        # Use the inverse matrix to calculate the required correction
        inv_m = np.array(self.calibration_data['inverse_matrix'])
        correction_vector = inv_m @ np.array([err_x, err_y])
        
        # The correction_vector gives us the required move in units of "jog_amount"
        # We apply a gain to prevent overshooting
        gain = 0.7
        jog_ra = correction_vector[0] * gain
        jog_dec = correction_vector[1] * gain

        # Clamp the jog values to a safe maximum
        max_jog = 1.5
        jog_ra = max(-max_jog, min(max_jog, jog_ra))
        jog_dec = max(-max_jog, min(max_jog, jog_dec))

        self.log(f"Jogging by RA={jog_ra:.3f}s, Dec={jog_dec:.3f}s")
        
        # Execute the jog commands sequentially
        if abs(jog_ra) > 0.01:
            self.sky.jog(np.sign(jog_ra), 0)
            time.sleep(abs(jog_ra))
            self.sky.jog(0,0)
        
        if abs(jog_dec) > 0.01:
            self.sky.jog(0, np.sign(jog_dec))
            time.sleep(abs(jog_dec))
            self.sky.jog(0,0)

        time.sleep(1.5) # Settle time after move
        # The main loop will automatically take the next picture and continue the correction

    # --- Utility Methods ---
    def log(self, message):
        print(message)
        self.status_label.setText(message)

    def load_calibration(self):
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
                self.log("Calibration data loaded.")
        except FileNotFoundError:
            self.log("No calibration file found.")
        except Exception as e:
            self.log(f"Could not load calibration data: {e}")

    def save_calibration(self):
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
                self.log("Calibration data saved.")
        except Exception as e:
            self.log(f"Could not save calibration data: {e}")

    def closeEvent(self, event):
        if self.camera:
            self.camera.close()
        super().closeEvent(event)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-Center Tool')
    parser.add_argument('--exp', type=float, default=0.5, help='Exposure time in seconds')
    parser.add_argument('--gain', type=int, default=300, help='Camera gain')
    parser.add_argument('--cam-name', type=str, default="ASI2600MM", help='Name of the imaging camera')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    main_win = AutoCenterApp(args=args)
    main_win.show()
    sys.exit(app.exec_())
