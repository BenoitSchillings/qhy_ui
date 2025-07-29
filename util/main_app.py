import sys
import argparse
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time
import collections

from zwo_cam_interface import zwoasi_wrapper
from guider import guider
from fli_focuser import focuser
from skyx import sky6RASCOMTele
import mover
from util import compute_centroid_improved, HighValueFinder

class MainApp(QtWidgets.QMainWindow):
    def __init__(self, guider_exposure):
        super().__init__()
        self.setWindowTitle("Unified Astronomy Control")
        self.setGeometry(100, 100, 1600, 900)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.layout.addWidget(self.tabs)

        # Initialize components
        self.camera = None
        self.guider_camera = None
        self.guider = None
        self.focuser = None
        self.sky = None
        self.finder = HighValueFinder()
        self.guider_exposure = guider_exposure

        # Create tabs
        self.create_imaging_tab()
        self.create_guiding_tab()
        self.create_system_tab()

        # Status timer
        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000) # Update every second

    def create_imaging_tab(self):
        self.imaging_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.imaging_tab, "Imaging")
        self.imaging_layout = QtWidgets.QHBoxLayout(self.imaging_tab)

        # Left side: Camera controls
        controls_layout = QtWidgets.QVBoxLayout()
        
        # Camera selection
        self.connect_button = QtWidgets.QPushButton("Connect ASI2600MM")
        self.connect_button.clicked.connect(self.connect_camera)
        controls_layout.addWidget(self.connect_button)

        # Exposure
        self.exp_label = QtWidgets.QLabel("Exposure (s):")
        self.exp_input = QtWidgets.QLineEdit("0.1")
        self.exp_input.editingFinished.connect(self.set_exposure)
        controls_layout.addWidget(self.exp_label)
        controls_layout.addWidget(self.exp_input)

        # Gain
        self.gain_label = QtWidgets.QLabel("Gain:")
        self.gain_input = QtWidgets.QLineEdit("100")
        self.gain_input.editingFinished.connect(self.set_gain)
        controls_layout.addWidget(self.gain_label)
        controls_layout.addWidget(self.gain_input)
        
        # Temperature
        self.temp_label = QtWidgets.QLabel("Target Temp (°C):")
        self.temp_input = QtWidgets.QLineEdit("-10")
        self.temp_input.editingFinished.connect(self.set_temperature)
        controls_layout.addWidget(self.temp_label)
        controls_layout.addWidget(self.temp_input)

        self.cam_temp_label = QtWidgets.QLabel("Camera Temp: N/A")
        self.cooler_power_label = QtWidgets.QLabel("Cooler Power: N/A")
        controls_layout.addWidget(self.cam_temp_label)
        controls_layout.addWidget(self.cooler_power_label)

        # Focuser
        focuser_group = QtWidgets.QGroupBox("Focuser")
        focuser_layout = QtWidgets.QVBoxLayout()
        focuser_group.setLayout(focuser_layout)

        self.connect_focuser_button = QtWidgets.QPushButton("Connect Focuser")
        self.connect_focuser_button.clicked.connect(self.connect_focuser)
        focuser_layout.addWidget(self.connect_focuser_button)

        self.focuser_pos_label = QtWidgets.QLabel("Position: N/A")
        self.focuser_move_input = QtWidgets.QLineEdit("0")
        self.focuser_move_button = QtWidgets.QPushButton("Move To")
        self.focuser_move_button.clicked.connect(self.move_focuser_to)
        self.focuser_rel_move_input = QtWidgets.QLineEdit("100")
        self.focuser_move_up_button = QtWidgets.QPushButton("Up")
        self.focuser_move_up_button.clicked.connect(self.move_focuser_up)
        self.focuser_move_down_button = QtWidgets.QPushButton("Down")
        self.focuser_move_down_button.clicked.connect(self.move_focuser_down)
        self.focuser_home_button = QtWidgets.QPushButton("Home")
        self.focuser_home_button.clicked.connect(self.home_focuser)
        
        focuser_layout.addWidget(self.focuser_pos_label)
        focuser_layout.addWidget(self.focuser_move_input)
        focuser_layout.addWidget(self.focuser_move_button)
        focuser_layout.addWidget(self.focuser_rel_move_input)
        focuser_layout.addWidget(self.focuser_move_up_button)
        focuser_layout.addWidget(self.focuser_move_down_button)
        focuser_layout.addWidget(self.focuser_home_button)
        controls_layout.addWidget(focuser_group)

        controls_layout.addStretch()
        self.imaging_layout.addLayout(controls_layout)

        # Right side: Image view
        self.image_view = pg.ImageView()
        self.imaging_layout.addWidget(self.image_view, 4)

    def create_guiding_tab(self):
        self.guiding_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.guiding_tab, "Guiding")
        self.guiding_layout = QtWidgets.QHBoxLayout(self.guiding_tab)

        # Guiding controls
        guiding_controls_layout = QtWidgets.QVBoxLayout()

        self.connect_guider_button = QtWidgets.QPushButton("Connect Guider")
        self.connect_guider_button.clicked.connect(self.connect_guider)
        guiding_controls_layout.addWidget(self.connect_guider_button)

        # Guider Exposure
        self.guider_exp_label = QtWidgets.QLabel("Guider Exposure (s):")
        self.guider_exp_input = QtWidgets.QLineEdit(str(self.guider_exposure))
        self.guider_exp_input.editingFinished.connect(self.set_guider_exposure)
        guiding_controls_layout.addWidget(self.guider_exp_label)
        guiding_controls_layout.addWidget(self.guider_exp_input)

        # Guider Gain
        self.guider_gain_label = QtWidgets.QLabel("Guider Gain:")
        self.guider_gain_input = QtWidgets.QLineEdit("300")
        self.guider_gain_input.editingFinished.connect(self.set_guider_gain)
        guiding_controls_layout.addWidget(self.guider_gain_label)
        guiding_controls_layout.addWidget(self.guider_gain_input)

        self.start_guide_button = QtWidgets.QPushButton("Start Guiding")
        self.start_guide_button.clicked.connect(self.start_guiding)
        guiding_controls_layout.addWidget(self.start_guide_button)

        self.stop_guide_button = QtWidgets.QPushButton("Stop Guiding")
        self.stop_guide_button.clicked.connect(self.stop_guiding)
        guiding_controls_layout.addWidget(self.stop_guide_button)

        self.calibrate_mount_button = QtWidgets.QPushButton("Calibrate Mount")
        self.calibrate_mount_button.clicked.connect(self.calibrate_mount)
        guiding_controls_layout.addWidget(self.calibrate_mount_button)

        guiding_controls_layout.addStretch()
        self.guiding_layout.addLayout(guiding_controls_layout)

        # Guider image view and graphs
        guider_display_layout = QtWidgets.QVBoxLayout()
        self.guider_image_view = pg.ImageView()
        guider_display_layout.addWidget(self.guider_image_view, 3) # Give more space to image view

        # Graphs for dx/dy
        graph_layout = QtWidgets.QHBoxLayout()
        self.plt_bufsize = 100
        self.x_data = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.dx_buffer = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)
        self.dy_buffer = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)

        self.guide_graph_dx = pg.PlotWidget(title="Guiding Error (dx)")
        self.guide_graph_dx.showGrid(x=True, y=True)
        self.guide_curve_dx = self.guide_graph_dx.plot(self.x_data, np.zeros(self.plt_bufsize, dtype=np.float64), pen=(255,0,0))
        graph_layout.addWidget(self.guide_graph_dx)

        self.guide_graph_dy = pg.PlotWidget(title="Guiding Error (dy)")
        self.guide_graph_dy.showGrid(x=True, y=True)
        self.guide_curve_dy = self.guide_graph_dy.plot(self.x_data, np.zeros(self.plt_bufsize, dtype=np.float64), pen=(255,0,0))
        graph_layout.addWidget(self.guide_graph_dy)
        
        guider_display_layout.addLayout(graph_layout, 1) # Give less space to graphs

        self.guiding_layout.addLayout(guider_display_layout, 4)


    def create_system_tab(self):
        self.system_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.system_tab, "System")
        self.system_layout = QtWidgets.QVBoxLayout(self.system_tab)

        # Telescope connection
        self.skyx_connect_button = QtWidgets.QPushButton("Connect to TheSkyX")
        self.skyx_connect_button.clicked.connect(self.connect_skyx)
        self.skyx_status_label = QtWidgets.QLabel("TheSkyX: Not Connected")
        self.system_layout.addWidget(self.skyx_connect_button)
        self.system_layout.addWidget(self.skyx_status_label)

        # Log display
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.system_layout.addWidget(self.log_text)

    def connect_camera(self):
        try:
            if self.camera:
                self.camera.close()
            self.camera = zwoasi_wrapper(temp=float(self.temp_input.text()), exp=float(self.exp_input.text()), gain=int(self.gain_input.text()), crop=None, cam_name="ASI2600MM", live=False)
            self.log(f"Connected to camera: {self.camera.name()}")
            self.start_image_timer()
        except Exception as e:
            self.log(f"Failed to connect to camera: {e}")

    def connect_guider(self):
        try:
            if self.guider_camera:
                self.guider_camera.close()
            self.guider_camera = zwoasi_wrapper(temp=-10, exp=float(self.guider_exp_input.text()), gain=int(self.guider_gain_input.text()), crop=None, cam_name="ASI220MM", live=True)
            self.log(f"Connected to guider camera: {self.guider_camera.name()}")
            self.guider = guider(self.sky, self.guider_camera)
            self.start_guider_timer()
        except Exception as e:
            self.log(f"Failed to connect to guider camera: {e}")

    def connect_focuser(self):
        try:
            self.focuser = focuser()
            self.log("Connected to FLI Focuser.")
        except Exception as e:
            self.log(f"Failed to connect to focuser: {e}")

    def connect_skyx(self):
        try:
            self.sky = sky6RASCOMTele()
            self.sky.Connect()
            self.skyx_status_label.setText("TheSkyX: Connected")
            self.log("Connected to TheSkyX.")
        except Exception as e:
            self.skyx_status_label.setText("TheSkyX: Connection Failed")
            self.log(f"Failed to connect to TheSkyX: {e}")

    def start_image_timer(self):
        self.image_timer = QtCore.QTimer()
        self.image_timer.timeout.connect(self.update_image)
        self.image_timer.start(100)

    def start_guider_timer(self):
        self.guider_timer = QtCore.QTimer()
        self.guider_timer.timeout.connect(self.update_guider)
        self.guider_timer.start(50)

    def update_image(self):
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                self.image_view.setImage(np.rot90(frame), autoLevels=True, autoRange=False)

    def update_guider(self):
        if self.guider_camera and self.guider:
            frame = self.guider_camera.get_frame()
            if frame is not None:
                self.guider_image_view.setImage(np.rot90(frame), autoLevels=True, autoRange=False)
                
                max_y, max_x, val = self.finder.find_high_value_element(frame[32:-32, 32:-32])
                cy, cx, cv = compute_centroid_improved(frame, max_y + 32, max_x + 32)
                
                dx, dy = self.guider.pos_handler(cx, cy)
                if dx is not None and dy is not None:
                    self.update_guide_plots(dx, dy)

    def update_guide_plots(self, dx, dy):
        self.dx_buffer.append(dx)
        self.dy_buffer.append(dy)
        self.guide_curve_dx.setData(self.x_data, np.array(self.dx_buffer))
        self.guide_curve_dy.setData(self.x_data, np.array(self.dy_buffer))

    def update_status(self):
        if self.camera:
            cam_temp = self.camera.GetTemperature()
            cooler_power = self.camera.GetCoolerPower()
            if cam_temp is not None:
                self.cam_temp_label.setText(f"Camera Temp: {cam_temp:.1f}°C")
            if cooler_power is not None:
                self.cooler_power_label.setText(f"Cooler Power: {cooler_power}%")
        
        if self.focuser:
            pos = self.focuser.get_abs_pos()
            self.focuser_pos_label.setText(f"Position: {pos}")

    def set_exposure(self):
        if self.camera:
            try:
                exp = float(self.exp_input.text())
                self.log(f"Setting exposure to {exp}s.")
                self.camera.SetExposure(exp * 1000)
            except ValueError:
                self.log("Invalid exposure value.")

    def set_gain(self):
        if self.camera:
            try:
                gain = int(self.gain_input.text())
                self.log(f"Setting gain to {gain}.")
                self.camera.SetGain(gain)
            except ValueError:
                self.log("Invalid gain value.")

    def set_temperature(self):
        if self.camera:
            try:
                temp = float(self.temp_input.text())
                self.camera.SetTemperature(temp)
                self.log(f"Set target temperature to {temp}°C")
            except ValueError:
                self.log("Invalid temperature value.")

    def set_guider_exposure(self):
        if self.guider_camera:
            try:
                exp = float(self.guider_exp_input.text())
                self.log(f"Setting guider exposure to {exp}s.")
                self.guider_camera.SetExposure(exp * 1000)
            except ValueError:
                self.log("Invalid guider exposure value.")

    def set_guider_gain(self):
        if self.guider_camera:
            try:
                gain = int(self.guider_gain_input.text())
                self.log(f"Setting guider gain to {gain}.")
                self.guider_camera.SetGain(gain)
            except ValueError:
                self.log("Invalid guider gain value.")

    def move_focuser_to(self):
        if self.focuser:
            try:
                pos = int(self.focuser_move_input.text())
                self.focuser.move_to(pos)
                self.log(f"Moving focuser to {pos}")
            except ValueError:
                self.log("Invalid focuser position.")

    def move_focuser_up(self):
        if self.focuser:
            try:
                val = int(self.focuser_rel_move_input.text())
                self.focuser.move_focus(val)
                self.log(f"Moving focuser up by {val}")
            except ValueError:
                self.log("Invalid focuser move value.")

    def move_focuser_down(self):
        if self.focuser:
            try:
                val = int(self.focuser_rel_move_input.text())
                self.focuser.move_focus(-val)
                self.log(f"Moving focuser down by {val}")
            except ValueError:
                self.log("Invalid focuser move value.")

    def home_focuser(self):
        if self.focuser:
            self.focuser.home()
            self.log("Homing focuser.")

    def start_guiding(self):
        if self.guider:
            self.guider.guide()
            self.log("Guiding started.")

    def stop_guiding(self):
        if self.guider:
            self.guider.stop_guide()
            self.log("Guiding stopped.")

    def calibrate_mount(self):
        if self.guider:
            self.guider.calibrate_mount()
            self.log("Mount calibration started.")

    def log(self, message):
        self.log_text.append(f"{time.strftime('%H:%M:%S')} - {message}")

    def close_tab(self, index):
        widget = self.tabs.widget(index)
        if widget:
            widget.deleteLater()
        self.tabs.removeTab(index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Astronomy Control')
    parser.add_argument('--guider-exposure', type=float, default=0.1, help='Guider exposure time in seconds')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    main_win = MainApp(guider_exposure=args.guider_exposure)
    main_win.show()
    sys.exit(app.exec_())
