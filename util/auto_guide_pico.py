import zmq
import numpy as np
import time
from datetime import datetime
import cv2
import astropy

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
from PyQt5 import QtWidgets

from PyQt5.QtGui  import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMenu, QMenuBar, QAction

import os


from zwo_cam_interface import *

from astropy.io import fits

from util import *
import datetime
import random
import collections
import math
from scipy import ndimage
import mover
import pico
import skyx
from mount_corrector import MountCorrector

import logging


# Setup for the main application log
log_main = logging.getLogger(__name__)
log_main.setLevel(logging.INFO)
file_handler_main = logging.FileHandler('centroid.log')
stream_handler_main = logging.StreamHandler()
formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_main.setFormatter(formatter_main)
stream_handler_main.setFormatter(formatter_main)
log_main.addHandler(file_handler_main)
log_main.addHandler(stream_handler_main)

# Setup for the new problem-specific log file
log_aoscale = logging.getLogger('aoscale')
log_aoscale.setLevel(logging.INFO)
file_handler_aoscale = logging.FileHandler('aoscale.log', mode='w') # 'w' to overwrite on each run
formatter_aoscale = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
file_handler_aoscale.setFormatter(formatter_aoscale)
log_aoscale.addHandler(file_handler_aoscale)

log_aoscale.info("AOSCALE LOGGING STARTED")


ipc = IPC()

ipc.set_val("bump", [0,0])


pico_device = pico.ao()

class pico_AO:
    def __init__(self):
        # I might need to initialize the pico connection here
        pass

    def bump(self, dx, dy):
        # This method now expects dx and dy to be in the final physical units.
        pico_dx = int(dx)
        pico_dy = int(dy)
        
        log_main.debug(f"Bumping pico AO: dx={pico_dx}, dy={pico_dy}")
        logging.getLogger('aoscale').info(f"PICO BUMP: Physical Units=({pico_dx}, {pico_dy})")
        pico_device.move_relative(pico_dx, pico_dy)

#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------
from guider_hist import guider


from cv2 import medianBlur

class HighValueFinder:
    def __init__(self, search_box_size=72, blur_size=3):
        self.hint_x = None
        self.hint_y = None
        self.reference_value = None
        self.search_box_size = search_box_size
        self.blur_size = blur_size

    def find_high_value_element(self, array):
        array = array.astype('float32')
        filtered_array = medianBlur(array, self.blur_size)

        if self.hint_x is not None and self.hint_y is not None and self.reference_value is not None:
            x_start = max(0, self.hint_x - self.search_box_size // 2)
            x_end = min(array.shape[1], self.hint_x + self.search_box_size // 2)
            y_start = max(0, self.hint_y - self.search_box_size // 2)
            y_end = min(array.shape[0], self.hint_y + self.search_box_size // 2)
            
            search_area = filtered_array[y_start:y_end, x_start:x_end]
            
            local_max = np.max(search_area)
            if local_max < 0.4 * self.reference_value:
                return self._full_array_scan(filtered_array)
            
            local_rows, local_cols = np.where(search_area == local_max)
            
            col = local_cols[0] + x_start
            row = local_rows[0] + y_start
        else:
            col, row, val = self._full_array_scan(filtered_array)
        
        self.hint_x, self.hint_y = col, row
        self.reference_value = filtered_array[row, col]
        return col, row, filtered_array[row, col]

    def _full_array_scan(self, array):
        rows, cols = np.where(array == np.max(array))
        self.reference_value = array[rows[0], cols[0]]
        return cols[0], rows[0], array[rows[0], cols[0]]

    def reset(self):
        self.hint_x = None
        self.hint_y = None
        self.reference_value = None

def find_high_value_element(array, size=3):
    array = array.astype('float32')
    filtered_array = medianBlur(array, size)
    rows, cols = np.where(filtered_array == np.max(filtered_array))
    return cols[0], rows[0], filtered_array[rows[0], cols[0]]



class fake_cam:
    def __init__(self, temp, exp, gain, crop):
        
        log_main.info("init cam")
        self.frame = np.random.randint(0,4096, (512,512), dtype=np.uint16)
        self.stars_frame = self.stars(self.frame, 4, gain=2)

    def stars(self, image, number, max_counts=3000, gain=1):
        from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image
        flux_range = [max_counts/5, max_counts]
        y_max, x_max = image.shape
        xmean_range = [0.1 * x_max, 0.9 * x_max]
        ymean_range = [0.1 * y_max, 0.9 * y_max]
        xstddev_range = [3, 3]
        ystddev_range = [3, 3]
        params = dict([('amplitude', flux_range),
                      ('x_mean', xmean_range),
                      ('y_mean', ymean_range),
                      ('x_stddev', xstddev_range),
                      ('y_stddev', ystddev_range),
                      ('theta', [0, 2*np.pi])])
        sources = make_random_gaussians_table(number, params, seed=12345)
        star_im = make_gaussian_sources_image(image.shape, sources)
        return star_im         

    def get_frame(self):        
        self.frame = np.random.randint(0,512, (512,512), dtype=np.uint16)
        return self.frame + self.stars_frame.astype(np.uint16)
        
    def start(self):
        self.running = 1
    def close(self):
        self.running = 0
    def size_x(self):
        return 512
    def size_y(self):
        return 512

class FrameWindow(QtWidgets.QMainWindow):
    def __init__(self, ui_controller, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.ui_controller = ui_controller
        self.quit = 0
        self._createMenuBar()

    def on_auto_level(self):
        self.ui_controller.auto_level = True
        log_main.info("AUTO LEVEL")

    def _createMenuBar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        action_menu = menu_bar.addMenu('Action')
        new_action = QAction('Auto-Level', self)
        new_action.setShortcut('Ctrl+A')
        new_action.triggered.connect(self.on_auto_level)
        action_menu.addAction(new_action)
        menu_bar.addMenu(action_menu)

    def closeEvent(self, event):
        self.quit = 1
        log_main.info("quit")
        QtWidgets.QMainWindow.closeEvent(self, event)


class UI:
    def click(self, event):
        event.accept()      
        self.pos = event.pos()
        log_main.info("click %d %d", int(self.pos.x()),int(self.pos.y()))

    def convert_nparray_to_QPixmap(self,img):
        w,h = img.shape
        qimg = QImage(img.data, h, w, QImage.Format_Grayscale16) 
        qpixmap = QPixmap(qimg)
        return qpixmap

    def __init__(self,  args, sx, sy, guider, mount_corrector):
        self.sx = sx
        self.sy = sy
        self.t0 = time.perf_counter()
        self.idx = 0
        self.fits = fits
        self.capture_state = 0
        self.update_state = 1
        self.guider = guider
        self.mount_corrector = mount_corrector
        self.auto_level = False
        
        self.rms = 0
        self.pos = QPoint(256,256)
        self.array = np.random.randint(0,65000, (sx,sy), dtype=np.uint16)

        self.win = FrameWindow(self)
        self.EDGE = 32
        self.win.resize(800,900)
        
        self.imv = pg.ImageView()
        self.imv.setImage(self.array)
        self.imv.getImageItem().setAutoDownsample(active=True)
        self.win.setCentralWidget(self.imv)

        self.statusBar = QtWidgets.QStatusBar()

        temp_widget = QtWidgets.QWidget(self.win)
        temp_widget.setLayout(QtWidgets.QHBoxLayout())
        temp_widget.setFixedSize(1024, 256)
        self.zoom_view = QtWidgets.QLabel(self.win)
        temp_widget.layout().addWidget(self.zoom_view)

        self.combined_mover = mover.CombinedWidget()
        self.combined_mover.setFixedSize(290, 230)
        self.mover = self.combined_mover.mover
        temp_widget.layout().addWidget(self.combined_mover)
        self.statusBar.addPermanentWidget(temp_widget, 1)

        # --- Plots ---
        plot_layout = QtWidgets.QWidget()
        plot_layout.setLayout(QtWidgets.QVBoxLayout())
        
        self.plt_dx = pg.plot(title='dx')
        self.plt_bufsize = 100
        self.ccx = 0
        self.ccy = 0
        self.x1 = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.y1 = np.zeros(self.plt_bufsize, dtype=np.float64)
        self.databufferx = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)
        self.plt_dx.showGrid(x=True, y=True)
        self.plt_dx.setLabel('left', 'pos_x', 'pixels')
        self.plt_dx.setLabel('bottom', 'frame', 'f')
        self.curvex = self.plt_dx.plot(self.x1, self.y1, pen=(255,0,0))
        plot_layout.layout().addWidget(self.plt_dx)

        self.plt_dy = pg.plot(title='dy')
        self.x2 = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.y2 = np.zeros(self.plt_bufsize, dtype=np.float64)
        self.databuffery = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)
        self.plt_dy.showGrid(x=True, y=True)
        self.plt_dy.setLabel('left', 'pos_y', 'pixels')
        self.plt_dy.setLabel('bottom', 'frame', 'f')
        self.curvey = self.plt_dy.plot(self.x2, self.y2, pen=(255,0,0))
        plot_layout.layout().addWidget(self.plt_dy)

        self.pico_plot = pg.plot(title='Pico Offsets')
        self.pico_plot.setAspectLocked(True)
        self.pico_plot.showGrid(x=True, y=True)
        self.pico_plot.setLabel('left', 'Pico Y')
        self.pico_plot.setLabel('bottom', 'Pico X')
        self.pico_plot_data = self.pico_plot.plot(pen=None, symbol='o', symbolSize=5)
        self.pico_offset_buffer = collections.deque(maxlen=200)
        plot_layout.layout().addWidget(self.pico_plot)

        temp_widget.layout().addWidget(plot_layout)
        # --- End Plots ---

        rightlayout = QtWidgets.QWidget(self.win)
        rightlayout.setLayout(QtWidgets.QVBoxLayout())
        rightlayout.setFixedSize(564, 228)
        
        self.calibrate_ao_button =  QtWidgets.QPushButton("Calibrate AO")
        rightlayout.layout().addWidget(self.calibrate_ao_button)
        self.calibrate_mount_button = QtWidgets.QPushButton("Calibrate Mount")
        rightlayout.layout().addWidget(self.calibrate_mount_button)
        self.recenter_button = QtWidgets.QPushButton("Recenter Mount")
        rightlayout.layout().addWidget(self.recenter_button)

        self.guide_button =  QtWidgets.QPushButton("Guide")
        rightlayout.layout().addWidget(self.guide_button)
        self.bump_button =  QtWidgets.QPushButton("bump")
        rightlayout.layout().addWidget(self.bump_button)

        self.update_button =  QtWidgets.QPushButton("slow_update")
        rightlayout.layout().addWidget(self.update_button)
        self.txt1 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt1)
        self.txt1.setText("status_text 1")

        self.txt2 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt2)
        self.txt2.setText("status_text 2")

        self.txt3 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt3)
        self.txt3.setText("status_text 3")

        self.txt4 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt4)
        self.txt4.setText("status_text 4")

        self.statusBar.addPermanentWidget(rightlayout)
        self.win.setStatusBar(self.statusBar)
        
        self.win.setWindowTitle('qhycam guide')
        self.imv.getImageItem().mouseClickEvent = self.click
        self.cnt = 0

        self.calibrate_ao_button.clicked.connect(self.calibrate_ao_button_click)
        self.calibrate_mount_button.clicked.connect(self.calibrate_mount_button_click)
        self.recenter_button.clicked.connect(self.recenter_mount_button_click)
        self.update_button.clicked.connect(self.Update_buttonClick)
        self.guide_button.clicked.connect(self.Guide_buttonClick)
        self.bump_button.clicked.connect(self.rand_move)
  
        self.win.show()
        self.last_recenter_check_time = time.time()
    
    def rand_move(self):
        self.guider.bump(0.5, 0.5)

    def Update_buttonClick(self):
        if (self.update_state == 1):
            self.update_button.setText("fast_update")
            self.update_state = 0
        else:
            self.update_button.setText("slow_update")
            self.update_state = 1

    def calibrate_ao_button_click(self):
        self.guider.calibrate_ao()
        log_main.info("Calibrate AO")

    def calibrate_mount_button_click(self):
        self.mount_corrector.calibrate_mount()
        log_main.info("Calibrate Mount")

    def recenter_mount_button_click(self):
        log_main.info("Manual mount recenter triggered.")
        self.recenter_mount()

    def Guide_buttonClick(self):
        if self.guider.is_guiding:
            self.guider.stop_guide()
            pico_device.zero()
            log_main.info("Stop Guide and Center Pico")
            self.guide_button.setText("Guide")
        else:
            self.guider.start_guide()
            log_main.info("Start Guide")
            self.guide_button.setText("Stop Guide")

    def updateplot(self, x, y):
        self.databufferx.append(x)
        self.y1[:] = self.databufferx
        self.curvex.setData(self.x1, self.y1)

        self.databuffery.append(y)
        self.y2[:] = self.databuffery
        self.curvey.setData(self.x2, self.y2)
        
    def update_pico_plot(self, pico_x, pico_y):
        self.pico_offset_buffer.append({'x': pico_x, 'y': pico_y})
        x_vals = [item['x'] for item in self.pico_offset_buffer]
        y_vals = [item['y'] for item in self.pico_offset_buffer]
        self.pico_plot_data.setData(x_vals, y_vals)

    def clip(self, pos):
        if (pos.x() < self.EDGE): pos.setX(self.EDGE)
        if (pos.y() < self.EDGE): pos.setY(self.EDGE)
        if (pos.x() > (self.sx-self.EDGE)): pos.setX(self.sx-self.EDGE)
        if (pos.y() > (self.sy-self.EDGE)): pos.setY(self.sy-self.EDGE)
        return pos

    def update_status(self):
        rms_guide_x, rms_guide_y = self.guider.get_guide_rms()
        self.txt1.setText("min=" + "{:04f}".format(self.min) + " max=" + "{:04f}".format(self.max) + " frame=" + str(self.cnt) + " RMS=" + "{:.1f} ".format(self.rms))
        self.updateplot(self.cx, self.cy)

        pico_x, pico_y = pico_device.get_ao()
        self.txt2.setText(f"Pico Pos: X={pico_x}, Y={pico_y}")
        self.update_pico_plot(pico_x, pico_y)

        if (self.cnt % 30 == 0):
            self.temp = 0
            self.txt3.setText("Temp = " + str(self.temp) + " fps=" + "{:.2f}".format(self.fps))

    def update(self):
        self.imv.setImage(np.flip(np.rot90((self.array)), axis=0), autoRange=False, autoLevels=False, autoHistogramRange=False)
        
        if (self.auto_level):
            vmin = np.percentile(self.array, 3)
            vmax = np.percentile(self.array,93)
            self.imv.setLevels(vmin, vmax)
            self.auto_level = False

        self.txt4.setText("X="  + "{:.2f}".format(self.ccx) + " Y="  + "{:.2f}".format(self.ccy) )
        
        pos = self.clip(self.pos)
        sub = self.array[int(pos.y())-self.EDGE:int(pos.y())+self.EDGE, int(pos.x())-self.EDGE:int(pos.x())+self.EDGE].copy()

        self.min = np.min(sub)
        self.max = np.max(sub)
        self.rms = np.std(self.array)
        self.update_status()

        sub = sub - self.min
        max_val = self.max - self.min
        sub =  sub * (65535.0/((max_val+1)))
        sub = sub.astype(np.uint16)
        sub = cv2.resize(sub, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        pixmap = self.convert_nparray_to_QPixmap(sub)
        self.zoom_view.setPixmap(pixmap)

    def ipc_check(self):
        bump = ipc.get_val("bump")
        if bump is None:
            log_main.warning("IPC check failed. Is the state server running?")
            return
        if (bump[0] == 0.0 and bump[1] == 0):
            return
        
        log_main.info("IPC bump received. Triggering mount recenter.")
        self.recenter_mount()
        
        # Reset the IPC trigger
        ipc.set_val("bump", [0,0])

    def recenter_mount(self):
        if not self.guider.is_guiding:
            log_main.warning("Cannot recenter mount when not guiding.")
            return
        
        pico_x, pico_y = pico_device.get_ao()
        if self.mount_corrector.correct_mount_drift(pico_x, pico_y, self.guider):
            #pico_device.zero()
            self.pico_offset_buffer.clear()
            log_main.info("Pico centered after mount correction.")
        self.last_mount_correction_time = time.time()

    def process_image_frame(self, camera, finder):
        """Gets a frame and finds the star centroid."""
        result = camera.get_frame()
        if result is None:
            return False

        self.array = result
        max_y, max_x, val = finder.find_high_value_element(self.array[32:-32, 32:-32])
        self.cy, self.cx, cv = compute_centroid_improved(self.array, max_y + 32, max_x + 32)
        log_main.info("calc centroid = %f, %f, %f", self.cx, self.cy, val)
        return True

    def handle_guiding_and_calibration(self):
        """Routes the star position to the correct handler based on state."""
        if self.guider.ao_cal_state_count > 0:
            self.guider.pos_handler(self.cx, self.cy)
        elif self.mount_corrector.mount_cal_state_count > 0:
            self.mount_corrector.handle_calibrate_mount(self.cx, self.cy)
        elif self.guider.is_guiding:
            self.guider.pos_handler(self.cx, self.cy)

    def handle_periodic_recenter_check(self):
        """Periodically checks if the AO has drifted too far and triggers a mount recenter."""
        if self.guider.is_guiding and (time.time() - self.last_recenter_check_time > 60):
            pico_x, pico_y = pico_device.get_ao()
            log_main.info(f"Periodic check: Pico offset = ({pico_x}, {pico_y})")

            if abs(pico_x) > 150 or abs(pico_y) > 150:
                log_main.info(f"Pico offset > 150, triggering mount recenter.")
                self.recenter_mount()
            
            self.last_recenter_check_time = time.time()

    def mainloop(self, args, camera):
        finder = HighValueFinder()
        while(self.win.quit == 0):
            time.sleep(0.01)
            app.processEvents()

            if not self.process_image_frame(camera, finder):
                continue
            
            self.ipc_check()
            self.handle_guiding_and_calibration()
            self.handle_periodic_recenter_check()
            self.handle_periodic_rate_adjustment()

            self.idx += 1
            self.t1 = time.perf_counter()
            self.fps = 1.0 / ((self.t1 - self.t0) / self.idx) if self.idx > 0 else 0
            
            need_update = (self.update_state == 1) or (self.update_state == 0 and self.cnt % 10 == 0)
            if (need_update):
                self.update()

            self.cnt += 1

    def handle_periodic_rate_adjustment(self):
        """Periodically adjusts the mount's tracking rate to counter systematic drift."""
        if self.guider.is_guiding and (time.time() - self.mount_corrector.last_rate_adjustment_time > 300): # 5 minutes
            self.mount_corrector.adjust_tracking_rate()
            self.mount_corrector.last_rate_adjustment_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", type=float, default = 0.1, help="exposure in seconds (default 0.1)")
    parser.add_argument("-gain", "--gain", type=int, default = 300, help="camera gain (default 100)")
    parser.add_argument("-crop", "--crop", type=float, default = 1.0, help="crop ratio")
    parser.add_argument("-cam", "--cam", type=str, default = "220", help="cam name")
    args = parser.parse_args()

    skyx_controller = skyx.sky6RASCOMTele()
    mount_corrector = MountCorrector(skyx_controller)
    pico_dev = pico_AO()

    if (args.cam == -1):
        camera = fake_cam(100, args.exp, args.gain, args.crop)
    else:
        camera = zwoasi_wrapper(100, args.exp, args.gain, args.crop, args.cam, False)

    ao_guider = guider(pico_dev, camera)
    ui = UI(args, camera.size_x(), camera.size_y(), ao_guider, mount_corrector)

    camera.start()

    try:
        ui.mainloop(args, camera)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ao_guider.close()
        camera.close()
        pico_device.zero()
        log_main.info("Pico centered.")
        print("Program terminated.")
