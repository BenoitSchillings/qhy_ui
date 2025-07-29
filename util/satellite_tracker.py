import sys
import argparse
import time
import numpy as np
import requests
from skyfield.api import load, EarthSatellite

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from skyx import sky6RASCOMTele

class PIDController:
    """
    A class to implement a PID control loop.
    """
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = 0.0
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = time.time()

    def update(self, process_variable):
        current_time = time.time()
        delta_time = current_time - self._last_time
        if delta_time == 0: return 0.0

        error = self.setpoint - process_variable
        self._integral += error * delta_time
        derivative = (error - self._last_error) / delta_time
        
        P_term = self.Kp * error
        I_term = self.Ki * self._integral
        D_term = self.Kd * derivative

        output = P_term + I_term + D_term
        self._last_error = error
        self._last_time = current_time
        return output

    def reset(self, setpoint):
        self.setpoint = setpoint
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = time.time()

class SatelliteTrackerApp(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle(f"Satellite Tracker - {self.args.satellite_name}")
        self.setGeometry(100, 100, 800, 600)

        # --- UI Setup ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.layout.addWidget(self.status_label)

        # --- Core Components ---
        self.sky = None
        self.timescale = load.timescale()
        self.observer = None
        self.satellite = None
        
        # PID controllers for RA and Dec
        # These gains are initial guesses and will require tuning.
        self.pid_ra = PIDController(Kp=0.8, Ki=0.2, Kd=0.1)
        self.pid_dec = PIDController(Kp=0.8, Ki=0.2, Kd=0.1)

        # --- Initialization ---
        self.log("Starting Satellite Tracker...")
        self.connect_mount()
        self.setup_observer()
        self.fetch_and_load_tle()

        # --- Main Loop Timer ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tracking_loop)
        if self.sky and self.satellite:
            self.timer.start(200) # Run the loop 5 times per second

    def log(self, message):
        print(message)
        self.status_label.setText(message)

    def connect_mount(self):
        try:
            self.sky = sky6RASCOMTele()
            self.sky.Connect()
            self.log("Connected to TheSkyX.")
        except Exception as e:
            self.log(f"FATAL: Could not connect to TheSkyX: {e}")
            self.sky = None

    def setup_observer(self):
        if not self.sky: return
        try:
            # This is a placeholder. We need to implement getting location from skyx.
            # Using a known location for now.
            lat, lon = 40.7128, -74.0060 # New York City
            self.observer = load.topos(latitude_degrees=lat, longitude_degrees=lon)
            self.log(f"Observer location set to Lat: {lat}, Lon: {lon}")
        except Exception as e:
            self.log(f"Could not set up observer: {e}")

    def fetch_and_load_tle(self):
        try:
            self.log(f"Fetching TLE for {self.args.satellite_name}...")
            url = 'https://celestrak.org/NORAD/elements/gp.php?NAME={}&FORMAT=tle'.format(self.args.satellite_name.replace(' ', '%20'))
            response = requests.get(url)
            response.raise_for_status()
            
            tle_data = response.text.strip().splitlines()
            if len(tle_data) < 3:
                raise ValueError("Invalid TLE data received.")

            self.satellite = EarthSatellite(tle_data[1], tle_data[2], tle_data[0], self.timescale)
            self.log("TLE data loaded successfully.")
        except Exception as e:
            self.log(f"Failed to fetch or parse TLE: {e}")
            self.satellite = None

    def tracking_loop(self):
        if not self.sky or not self.satellite or not self.observer:
            return

        # 1. Predict Satellite Position (Setpoint)
        current_time = self.timescale.now()
        difference = self.satellite - self.observer
        ra, dec, _ = difference.at(current_time).radec()
        
        # Update PID setpoints
        self.pid_ra.reset(ra.hours)
        self.pid_dec.reset(dec.degrees)

        # 2. Measure Telescope Position (Process Variable)
        try:
            current_ra, current_dec = self.sky.GetRaDec()
            current_ra, current_dec = float(current_ra), float(current_dec)
        except Exception as e:
            self.log(f"Could not get mount position: {e}")
            return

        # 3. Calculate PID Correction
        # The PID controllers will output a required *change* in rate
        rate_correction_ra = self.pid_ra.update(current_ra)
        rate_correction_dec = self.pid_dec.update(current_dec)

        # 4. Command Mount
        # We need the satellite's current angular velocity to add our correction to.
        # For now, we'll just use the correction as the rate offset.
        # A more advanced implementation would calculate the true angular velocity.
        base_ra_rate = 15.0 / 3600.0 # Sidereal rate in hours/sec (approx)
        
        # Convert PID output (degrees/sec) to TheSkyX rate (arcseconds/sec)
        final_ra_rate = (base_ra_rate * 3600) + (rate_correction_ra * 3600)
        final_dec_rate = rate_correction_dec * 3600

        self.sky.rate(final_ra_rate, final_dec_rate)

        self.log(f"Target: {ra.hours:.4f}h, {dec.degrees:.4f}° | Actual: {current_ra:.4f}h, {current_dec:.4f}° | Rate Adj: RA={rate_correction_ra*3600:.2f}", Dec={rate_correction_dec*3600:.2f}")

    def closeEvent(self, event):
        if self.sky:
            self.sky.stop() # Revert to sidereal tracking
        super().closeEvent(event)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Satellite Tracker')
    parser.add_argument('satellite_name', type=str, help='Name of the satellite to track (e.g., "ISS (ZARYA)")')
    # Add arguments for observer location later
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    main_win = SatelliteTrackerApp(args=args)
    main_win.show()
    sys.exit(app.exec_())