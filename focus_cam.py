import zmq
import numpy as np
import time
from datetime import datetime
import cv2
import astropy

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMenu, QMenuBar, QAction

from PyQt5.QtGui  import *
from PyQt5.QtCore import *
import os

from astropy.io import fits

from util import *
import datetime
import random
from ser import SerWriter
import skyx
import collections
import math

import mover
import logging as log

from fli_focuser import *


log.basicConfig(level=log.INFO)


sky = skyx.sky6RASCOMTele()
ipc = IPC()


#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------


from qhy_cam_interface import *



class FrameWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.quit = 0
        self._createMenuBar()
        self.setWindowTitle(camera.name())

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
        print("quit")
        QtWidgets.QMainWindow.closeEvent(self, event)


class UI:
    def click(self, event):
        event.accept()      
        self.pos = event.pos()
        print ("click", int(self.pos.x()),int(self.pos.y()))

    def convert_nparray_to_QPixmap(self,img):
        w,h = img.shape

        qimg = QImage(img.data, h, w, QImage.Format_Grayscale16) 
        qpixmap = QPixmap(qimg)

        return qpixmap

        


    def __init__(self,  args, sx, sy, count, auto):
        self.sx = sx
        self.sy = sy
        
        self.idx = 0
        self.fits = fits
        self.update_state = 1
        self.auto = auto
        self.auto_level = False
        self.foc = focuser()
        self.foc_p0 = self.foc.get_abs_pos()
        self.focus_state = -1
     	
        self.rms = 0
        self.pos = QPoint(256,256)
        self.array = np.random.randint(0,65000, (sx,sy), dtype=np.uint16)
        self.frame_per_file = count
        
        self.win = FrameWindow()
        self.EDGE = 64
        
        self.win.resize(1500,1000)
        
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
        self.plt = pg.plot(title='Dynamic Plotting with PyQtGraph')
        self.plt_bufsize = 100
        self.x = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.y = np.zeros(self.plt_bufsize, dtype=np.float64)
        self.databuffer = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)

        temp_widget.layout().addWidget(self.plt)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'fwhm', 'pixels')
        self.plt.setLabel('bottom', 'frame', 'f')
        self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
        
        self.statusBar.addPermanentWidget(temp_widget, 1)


        rightlayout = QtWidgets.QWidget(self.win)
        rightlayout.setLayout(QtWidgets.QVBoxLayout())
        rightlayout.setFixedSize(464, 158)
        

        self.focus_button =  QtWidgets.QPushButton("Start Focus")
        rightlayout.layout().addWidget(self.focus_button)


        self.txt1 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt1)
        self.txt1.setText("status_text 1")


        self.txt2 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt2)
        self.txt2.setText("status_text 2")

        self.txt3 = QtWidgets.QLabel(self.win)
        rightlayout.layout().addWidget(self.txt3)
        self.txt3.setText("status_text 3")



        self.statusBar.addPermanentWidget(rightlayout)

        self.win.setStatusBar(self.statusBar)
        
      
        
        self.imv.getImageItem().mouseClickEvent = self.click
        self.cnt = 0

        self.focus_button.clicked.connect(self.focus_buttonClick)
 
        import sys
  
        self.win.show()
    



    def focus_buttonClick(self):
        self.focus_state = 0


    def updateplot(self, fwhm):
        self.databuffer.append(fwhm)
        self.y[:] = self.databuffer
        self.curve.setData(self.x, self.y)
        #self.app.processEvents()


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
        self.txt1.setText("FWHM= " + "{:.2f}  ".format(self.fwhm) + "min=" + "{:04d}".format(self.min) + " max=" + "{:04d}".format(self.max) + " frame=" + str(self.cnt) + " RMS=" + "{:.1f} ".format(self.rms))
        self.updateplot(self.fwhm)

        if (self.cnt % 5 == 2):
            if not (sky is None):
                p0 = sky.GetRaDec()
                try:
                    self.txt2.setText("RA = " + p0[0][0:8] + " DEC=" + p0[1][0:8])
                except:
                    print("erro")
            self.temp = camera.qc.GetTemperature()
            self.txt3.setText("Temp = " + str(self.temp) + " fps=" + "{:.2f}".format(self.fps))

    def add_lines(self, img):
         return img 


    def find_max_position(self, arr):
        """
        Find the position (row, column) of the maximum element in a 2D array.
        
        Args:
            arr (numpy.ndarray): 2D array.
            
        Returns:
            tuple: Position (row, column) of the maximum element.
        """
        # Get the flattened index of the maximum element
        max_idx = np.argmax(arr)
        
        # Convert the flattened index to row and column indices
        row, col = np.unravel_index(max_idx, arr.shape)
        
        return row, col


    def sharpness(self, img):
        row, col = self.find_max_position(img)

        hdf = compute_hfd(img[row-15:row+15,col-15:col+15])
        return hdf




    def handle_focus_frame(self, data):
        print("focus state is ", self.focus_state)

        if (self.focus_state > 370):
            min = find_best_focus(self.samples)
            print("min = ", find_best_focus(self.samples))
            self.focus_state = -1
            self.foc.move_to(int(self.foc_p0 - 280 + 20 * min))
            print("best focus = ", self.foc_p0 - 280 + 20 * min)
            return

        if (self.focus_state == 0):
            print("init set focus to ", self.foc_p0 - 280)
            self.p0 = self.foc_p0 - 280
            self.foc.move_to(self.foc_p0 - 280)
            self.focus_state = self.focus_state + 1
            self.samples =  np.array([], dtype=np.float32)  # You can use np.float32 if you need less precision

            self.focus_acc = 0
            return
        if (self.focus_state < 15):
            print("wait")
            self.focus_state = self.focus_state + 1
            return

        if (self.focus_state % 14 <= 3 and self.focus_state % 14 != 0):
            print("wait")
            self.focus_state = self.focus_state + 1
            return

        if (self.focus_state % 14 > 3):
            print("acc")
            self.focus_acc = self.focus_acc + self.sharpness(data)
            self.focus_state = self.focus_state + 1
            return

        if (self.focus_state % 14 == 0):
            self.samples = np.append(self.samples, self.focus_acc)
            print(self.samples)
            self.focus_acc = 0
            
            self.p0 = self.p0 + 20
            print("move focus up by 20 to", self.p0)
            self.foc.move_to(self.p0)
            self.focus_state = self.focus_state + 1
            return




        
    def update(self):
        def possible_star(array):
            max = np.max(array)
            min = np.min(array)
            std = np.std(array)

            return ((max - min) > (std*10))

        shape = self.array.shape
       
        #self.array[shape[0]//2-32:shape[0]//2+32, shape[1]//2-32:shape[1]//2+32] *= 2
 
        self.imv.setImage(np.flip(np.rot90((self.array)), axis=0), autoRange=False, autoLevels=False, autoHistogramRange=False) #, pos=[-1300,0],scale=[2,2])
 
        if (self.auto_level):
            vmin = np.percentile(self.array, 3)
            vmax = np.percentile(self.array,96)
            self.imv.setLevels(vmin, vmax)
            self.auto_level = False



# Find the index of the maximum value
        max_index = np.argmax(self.array)

# Convert the flattened index into a 2D index
        max_index_2d = np.unravel_index(max_index, self.array.shape)
        #print(max_index_2d)
        #self.pos.setX(max_index_2d[1])

        #self.pos.setY(max_index_2d[0])
        pos = self.clip(self.pos)
       

        sub = self.array[int(pos.y())-self.EDGE:int(pos.y())+self.EDGE, int(pos.x())-self.EDGE:int(pos.x())+self.EDGE].copy()

        self.min = np.min(sub)
        self.max = np.max(sub)

        if possible_star(sub):
            self.fwhm = fit_gauss_circular(sub)
        else:
            self.fwhm = 5.0

        self.rms = np.std(self.array)

        self.update_status()

        sub = sub - self.min
        max = self.max - self.min
        sub =  sub * (65535.0/((max+1)))
        sub = sub.astype(np.uint16)
        sub = cv2.resize(sub, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        pixmap = self.convert_nparray_to_QPixmap(sub)
        self.zoom_view.setPixmap(pixmap)



    def mainloop(self, args, camera):
        self.t0 = time.perf_counter()
        while(self.win.quit == 0):
            time.sleep(0.003)
            if (self.mover.moving()):
                rx, ry = self.mover.rate()
                #sky.rate(ry * 4.0, rx * 4.0)
                print("move at " + str(rx) + " " + str(ry))
            
            
            app.processEvents()
            result = camera.get_frame()
           

            if (result is not None):
                self.array = result

                self.cnt = self.cnt + 1


                if self.focus_state >= 0:
                    self.handle_focus_frame(self.array)


                self.idx = self.idx + 1
                self.t1 = time.perf_counter()

                self.fps = 1.0 / ((self.t1-self.t0)/self.idx)
                #print(self.fps)
                need_update = False
                if (self.update_state == 1):
                    need_update = True
                if (self.update_state == 0 and self.cnt % 10 == 0):
                    need_update = True

                if (need_update):
                    self.update()

                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", type=float, default = 0.1, help="exposure in seconds (default 0.1)")
    parser.add_argument("-gain", "--gain", type=int, default = 100, help="camera gain (default 100)")
    parser.add_argument("-bin", "--bin", type=int, default = 1, help="camera binning (default 1-6)")
    parser.add_argument("-count", "--count", type=int, default = 100, help="number of frames to capture")
    parser.add_argument("-crop", "--crop", type=float, default = 1.0, help="crop ratio")
    parser.add_argument("-auto", "--auto", type=int, default = 0, help="auto start focus")
    parser.add_argument("-cam", "--cam", type=str, default = "", help="cam name")
    args = parser.parse_args()

    try:
        sky.Connect()
    except:
        sky = None

    print("SKY ", sky)
    if not (sky is None):
        sky.bump(120,0)


    ipc.set_val("bump", [1.1,1.1])

    camera = qhy_cam(-22, args.exp, args.gain, args.crop, args.cam, False)

    ui = UI(args, camera.size_x(), camera.size_y(), args.count, args.auto)
    
    camera.start()

    ui.mainloop(args, camera)


