import zmq
import numpy as np
import time
from datetime import datetime
import cv2
import astropy
from util import *

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMenu, QMenuBar, QAction
from pyqtgraph import ImageItem

from PyQt5.QtGui  import *
from PyQt5.QtCore import *
import os

from astropy.io import fits


import datetime
import random

import collections
import math


import logging as log

log.basicConfig(level=log.INFO)



#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------

class CrosshairImageItem(ImageItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, *args, **kwargs):
        super().paint(painter, *args, **kwargs)

        # Draw crosshair
        painter.setPen(pg.mkPen(color='r'))
        painter.drawLine(self.boundingRect().center().x(), 0, self.boundingRect().center().x(), self.height())
        painter.drawLine(0, self.boundingRect().center().y(), self.width(), self.boundingRect().center().y())


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
        print("quit")
        QtWidgets.QMainWindow.closeEvent(self, event)


class UI:
    def click(self, event):
        event.accept()    
        self.pos = event.pos()
        print ("click", int(self.pos.x()), int(self.pos.y()))

    def convert_nparray_to_QPixmap(self, img):
        w, h = img.shape
        qimg = QImage(img.data, h, w, QImage.Format_Grayscale16) 
        qpixmap = QPixmap(qimg)
        return qpixmap

    def __init__(self, args, sx, sy, data):
        self.sx = sx
        self.sy = sy
        self.auto_level = False
        
        self.pos = QPoint(256,256)

        self.file_index = 0
        self.filenames = args.filenames 

        if len(self.filenames) == 0:
            raise ValueError("No image files provided!")

        self.array = self.load_image(self.filenames[self.file_index])

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

        rightlayout = QtWidgets.QWidget(self.win)
        rightlayout.setLayout(QtWidgets.QVBoxLayout())
        rightlayout.setFixedSize(464, 158)
         
        self.filename = QtWidgets.QLineEdit(self.filenames[self.file_index])  # Show initial filename
        rightlayout.layout().addWidget(self.filename)

        # UI Elements for navigation
        self.prev_button = QtWidgets.QPushButton("Previous", self.win)
        self.prev_button.clicked.connect(self.prev_image)
        rightlayout.layout().addWidget(self.prev_button)

        self.next_button = QtWidgets.QPushButton("Next", self.win)
        self.next_button.clicked.connect(self.next_image)
        rightlayout.layout().addWidget(self.next_button)


        self.statusBar.addPermanentWidget(rightlayout)
        self.win.setStatusBar(self.statusBar)

        self.imv.getImageItem().mouseClickEvent = self.click
        self.win.show()

    def load_image(self, filename):
        if filename.endswith('.fits'):
            data = fits.getdata(filename, ext=0)
        else:  
            data = cv2.imread(filename)
        return data

    def prev_image(self):
        self.file_index = (self.file_index - 1) % len(self.filenames)
        self.load_and_update()

    def next_image(self):
        self.file_index = (self.file_index + 1) % len(self.filenames)
        self.load_and_update()

    def load_and_update(self):
        self.array = self.load_image(self.filenames[self.file_index])
        self.update()
        self.filename.setText(self.filenames[self.file_index])  


    def clip(self, pos):
        if (pos.x() < self.EDGE):
            pos.setX(self.EDGE)
        if (pos.y() < self.EDGE):
            pos.setY(self.EDGE)

        if (pos.x() > (self.sx-self.EDGE)):
            pos.setX(self.sx-self.EDGE)
        if (pos.y() > (self.sy-self.EDGE)):
            pos.setY(self.sy-self.EDGE)

    def update(self):

        shape = self.array.shape
       
 
        self.imv.setImage(np.flip(np.rot90((self.array)), axis=0), autoRange=False, autoLevels=False, autoHistogramRange=False) #, pos=[-1300,0],scale=[2,2])
 
        if (self.auto_level):
            vmin = np.percentile(self.array, 3)
            vmax = np.percentile(self.array,96)
            self.imv.setLevels(vmin, vmax)
            self.auto_level = False

        pos = self.clip(self.pos)
       
        print(self.pos)
        sub = self.array[int(pos.y())-self.EDGE:int(pos.y())+self.EDGE, int(pos.x())-self.EDGE:int(pos.x())+self.EDGE].copy()

        self.min = np.min(sub)
        self.max = np.max(sub)


        self.rms = np.std(self.array)

        #print(fit_gauss_circular(sub))
       
        sub = sub - self.min
        max = self.max - self.min
        sub =  sub * (65535.0/((max+1)))
        sub = sub.astype(np.uint16)
        #sub = cv2.resize(sub, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #pixmap = self.convert_nparray_to_QPixmap(sub)
        #self.zoom_view.setPixmap(pixmap)



    def mainloop(self, args):

        while(self.win.quit == 0):
            time.sleep(0.02)
            app.processEvents()
            self.update()

                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", type=str, help="List of image filenames")  # filenames as arguments
    args = parser.parse_args()

    # Assuming you have data loading logic, get data based on filenames
    ui = UI(args, 1024, 1024, None)  # Adjust dimensions as needed
    ui.mainloop(args)


