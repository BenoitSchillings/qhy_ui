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
import qhyccd
from astropy.io import fits

from util import *
import datetime
import random
from ser import SerWriter
import skyx
import collections
import math
from scipy import ndimage
import mover

import logging as log

log.basicConfig(level=log.INFO)

sky = skyx.sky6RASCOMTele()

ipc = IPC()

ipc.set_val("bump", [0,0])


#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------
from guider import *


def rand_move():
    guider.bump()


class fake_cam:
    def __init__(self, temp, exp, gain, crop):
        
        log.info("init cam")
        self.frame = np.random.randint(0,4096, (512,512), dtype=np.uint16)
        self.stars_frame = self.stars(self.frame, 4, gain=2)

    def stars(self, image, number, max_counts=3000, gain=1):
        """
        Add some stars to the image.
        """
        from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image
        # Most of the code below is a direct copy/paste from
        # https://photutils.readthedocs.io/en/stable/_modules/photutils/datasets/make.html#make_100gaussians_image
        
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

        sources = make_random_gaussians_table(number, params,
                                              seed=12345)
        
        star_im = make_gaussian_sources_image(image.shape, sources)
        
        return star_im         

    

    def get_frame(self):        
        self.frame = np.random.randint(0,512, (512,512), dtype=np.uint16)
        

        return self.frame + ndimage.shift(self.stars_frame.astype(np.uint16), (guider.cheat_move_x, guider.cheat_move_y))
        
    def start(self):
        self.running = 1
        
    def close(self):
        self.running = 0

    def size_x(self):
        return 512
    
    def size_y(self):
        return 512



from qhy_cam_interface import *


class FrameWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.quit = 0
        self._createMenuBar()

    def on_auto_level(self):
        global ui

        ui.auto_level = True
        log.info("AUTO LEVEL")

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
        log.info("quit")
        QtWidgets.QMainWindow.closeEvent(self, event)


class UI:
    def click(self, event):
        event.accept()      
        self.pos = event.pos()
        log.info("click %d %d", int(self.pos.x()),int(self.pos.y()))

    def convert_nparray_to_QPixmap(self,img):
        w,h = img.shape

        qimg = QImage(img.data, h, w, QImage.Format_Grayscale16) 
        qpixmap = QPixmap(qimg)

        return qpixmap

        


    def __init__(self,  args, sx, sy, guider):
        self.sx = sx
        self.sy = sy
        self.t0 = time.perf_counter()
        self.idx = 0
        self.fits = fits
        self.capture_state = 0
        self.update_state = 1
        self.guider = guider
        self.auto_level = False
        
      	
        self.rms = 0
        self.pos = QPoint(256,256)
        self.array = np.random.randint(0,65000, (sx,sy), dtype=np.uint16)

        
        self.win = FrameWindow()
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
        self.mover = mover.Mover()
        self.mover.setFixedSize(200,200)

        temp_widget.layout().addWidget(self.mover)


        self.plt = pg.plot(title='dx')
        self.plt_bufsize = 100
        self.ccx = 0
        self.ccy = 0
        self.x1 = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.y1 = np.zeros(self.plt_bufsize, dtype=np.float64)
        self.databufferx = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)
        temp_widget.layout().addWidget(self.plt)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'pos_x', 'pixels')
        self.plt.setLabel('bottom', 'frame', 'f')
        self.curvex = self.plt.plot(self.x1, self.y1, pen=(255,0,0))
        self.statusBar.addPermanentWidget(temp_widget, 1)


        self.plt = pg.plot(title='dy')
        self.plt_bufsize = 100
        self.x2 = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.y2 = np.zeros(self.plt_bufsize, dtype=np.float64)
        self.databuffery = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)
        temp_widget.layout().addWidget(self.plt)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'pos_y', 'pixels')
        self.plt.setLabel('bottom', 'frame', 'f')
        self.curvey = self.plt.plot(self.x2, self.y2, pen=(255,0,0))
        self.statusBar.addPermanentWidget(temp_widget, 1)


        rightlayout = QtWidgets.QWidget(self.win)
        rightlayout.setLayout(QtWidgets.QVBoxLayout())
        rightlayout.setFixedSize(564, 228)
        
        

        self.calibrate_button_mount =  QtWidgets.QPushButton("Calibrate_mount")
        rightlayout.layout().addWidget(self.calibrate_button_mount)
        self.calibrate_button_ao =  QtWidgets.QPushButton("Calibrate_ao")
        rightlayout.layout().addWidget(self.calibrate_button_ao)

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
        
      
        self. win.setWindowTitle('qhycam guide')
        self.imv.getImageItem().mouseClickEvent = self.click
        self.cnt = 0

        self.calibrate_button_mount.clicked.connect(self.Calibrate_mount_buttonClick)
        self.calibrate_button_ao.clicked.connect(self.Calibrate_ao_buttonClick)
        self.update_button.clicked.connect(self.Update_buttonClick)
        self.guide_button.clicked.connect(self.Guide_buttonClick)
        self.bump_button.clicked.connect(self.Bump_buttonClick)
  
        self.win.show()
    


    def Update_buttonClick(self):
        #print("button")

        if (self.update_state == 1):
            self.update_button.setText("fast_update")
            self.update_state = 0
        else:
            self.update_button.setText("slow_update")
            self.update_state = 1

    def Bump_buttonClick(self):
        #print("bump")
        rand_move()



    def Calibrate_ao_buttonClick(self):
        self.guider.calibrate_ao()
        log.info("Calibrate_ao")


    def Calibrate_mount_buttonClick(self):
        self.guider.calibrate_mount()
        log.info("Calibrate_mount")

    def Guide_buttonClick(self):
        self.guider.guide()
        log.info("Guide")

    def updateplot(self, x, y):
        self.databufferx.append(x)
        self.y1[:] = self.databufferx
        self.curvex.setData(self.x1, self.y1)

        self.databuffery.append(y)
        self.y2[:] = self.databuffery
        self.curvey.setData(self.x2, self.y2)
        
       


    def clip(self, pos):
        if (pos.x() < self.EDGE):
            pos.setX(self.EDGE)
        if (pos.y() < self.EDGE):
            pos.setY(self.EDGE)

        if (pos.x() > (self.sx-self.EDGE)):
            pos.setX(self.sx-self.EDGE)
        if (pos.y() > (self.sy-self.EDGE)):
            pos.setY(self.sy-self.EDGE)

        return pos

    def update_status(self):
        self.txt1.setText("FWHM= " + "min=" + "{:04d}".format(self.min) + " max=" + "{:04d}".format(self.max) + " frame=" + str(self.cnt) + " RMS=" + "{:.1f} ".format(self.rms))
        self.updateplot(self.ccx, self.ccy)

        if (self.cnt % 30 == 0):
            if not (sky is None):
                p0 = sky.GetRaDec()
                
                self.txt2.setText("RA = " + p0[0][0:8] + " DEC=" + p0[1][0:8])

            self.temp = 0 #camera.qc.GetTemperature()
            self.txt3.setText("Temp = " + str(self.temp) + " fps=" + "{:.2f}".format(self.fps))




    def update(self):
        self.imv.setImage(np.flip(np.rot90((self.array)), axis=0), autoRange=False, autoLevels=False, autoHistogramRange=False) #, pos=[-1300,0],scale=[2,2])
        
        if (self.auto_level):
            vmin = np.percentile(self.array, 3)
            vmax = np.percentile(self.array,93)
            self.imv.setLevels(vmin, vmax)
            self.auto_level = False


        self.txt4.setText("X="  + "{:.2f}".format(self.ccx) + " Y="  + "{:.2f}".format(self.ccy) + " gx=" + "{:.2f}".format(self.guider.gain_x) + " gy=" + "{:.2f}".format(self.guider.gain_y))

        
        pos = self.clip(self.pos)
       

        sub = self.array[int(pos.y())-self.EDGE:int(pos.y())+self.EDGE, int(pos.x())-self.EDGE:int(pos.x())+self.EDGE].copy()

        self.min = np.min(sub)
        self.max = np.max(sub)



        self.rms = np.std(self.array)

        self.update_status()

        sub = sub - self.min
        max = self.max - self.min
        sub =  sub * (65535.0/((max+1)))
        sub = sub.astype(np.uint16)
        sub = cv2.resize(sub, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        pixmap = self.convert_nparray_to_QPixmap(sub)
        self.zoom_view.setPixmap(pixmap)


    def ipc_check(self):
        bump = ipc.get_val("bump")

        if (bump[0] == 0.0 and bump[1] == 0):
            return
        print("GOT BUMP")
        #self.guider.offset(bump[0], bump[1])
        self.guider.reset_ao()
        ipc.set_val("bump", [0,0])

    def mainloop(self, args, camera):
        global cheat_move_y
        global cheat_move_x

        mean_old = 0.0

        while(self.win.quit == 0):
            time.sleep(0.01)
           
            if (self.mover.moving()):
                rx, ry = self.mover.rate()
                #sky.rate(rx * 4.0, ry * 4.0)
                print("move at " + str(rx) + " " + str(ry))
           
            
            app.processEvents()

            
            result = camera.get_frame()
            #print(result)

            if (result is not None):
                self.array = result

                max_y, max_x = find_high_value_element(self.array[32:-32, 32:-32])
                #log.info("max value = %d %d", max_x, max_y)
                self.cy, self.cx, cv = compute_centroid(self.array, max_y + 32, max_x + 32)
                #log.info("calc centroid = %f %f", self.cx, self.cy)
                #self.cx = 0
                #self.cy = 0
                self.ipc_check()


                self.guider.pos_handler(self.cx, self.cy)
                ddx = 0
                ddy = 0
                
                self.ccx = ddx
                self.ccy = ddy

                self.idx = self.idx + 1
                self.t1 = time.perf_counter()
                #rand_move()
                self.fps = 1.0 / ((self.t1-self.t0)/self.idx)
                
                need_update = False
                if (self.update_state == 1):
                    need_update = True
                if (self.update_state == 0 and self.cnt % 10 == 0):
                    need_update = True

                if (need_update):
                    self.update()

                self.cnt = self.cnt + 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", type=float, default = 0.1, help="exposure in seconds (default 0.1)")
    parser.add_argument("-gain", "--gain", type=int, default = 100, help="camera gain (default 100)")
    parser.add_argument("-guide", "--guide", type=int, default = 0, help="frame per guide cycle (0 to disable)")
    
    parser.add_argument("-crop", "--crop", type=float, default = 1.0, help="crop ratio")
    parser.add_argument("-auto", "--auto", type=int, default = 0, help="start guiding automatically")
    parser.add_argument("-cam", "--cam", type=str, default = 0, help="cam name")
    args = parser.parse_args()

    try:
        sky.Connect()
    except:
        sky = None
        print("NO SKY")


    if (args.cam == -1):
        camera = fake_cam(-10, args.exp, args.gain, args.crop)
    else:
        camera = qhy_cam(-5, args.exp, args.gain, args.crop, args.cam, False)

    guider = guider(sky, camera)

    ui = UI(args, camera.size_x(), camera.size_y(), guider)
    
    camera.start()

    ui.mainloop(args, camera)


