import zmq
import numpy as np
import time
from datetime import datetime
import cv2
import astropy
from astropy.io import fits
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
from PyQt5 import QtWidgets

from PyQt5.QtGui  import *
from PyQt5.QtCore import *
import os
import qhyccd
import pywt

from util import *
import datetime
import random
from ser import SerWriter
import skyx
import collections
import math
4
import mover

sky = skyx.sky6RASCOMTele()

#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------

def denoise_image(image, wavelet="db1", thresholding="soft"):
    # Perform a 2D wavelet transform on the image
    coeffs = pywt.wavedec2(image, wavelet)
    print(len(coeffs))
    denoised_coeffs = pywt.threshold(coeffs, value = 20)
     # Reconstruct the image from the thresholded coefficients
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    
    return denoised_image


class qhy_cam:
    def __init__(self, temp, exp, gain, sx, sy):
        self.qc = qhyccd.qhyccd()
        self.dt = exp
    
        self.qc.SetBit(16)
        self.qc.SetUSB(1)
        self.qc.SetTemperature(temp)
        
        self.qc.SetROI(0,0,sx,sy)
        self.qc.SetExposure(self.dt*1000)
        print(self.qc.GetTemperature())
        self.qc.SetGain(gain)
        print(self.qc.pixelw)
        

 
    def get_frame(self):        
        self.frame = self.qc.GetLiveFrame()
        

        return self.frame
        
    def start(self):
        self.running = 1

        self.qc.BeginLive()
        
    def close(self):
        self.running = 0
        self.qc.StopLive()


#-------------------------------------app = QApplication(sys.argv)-------------------


class FrameWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.quit = 0


    def closeEvent(self, event):
        self.quit = 1
        print("quit")
        QtWidgets.QMainWindow.closeEvent(self, event)

#--------------------------------------------------------

class UI:
    def click(self, event):
        event.accept()      
        self.pos = event.pos()
        print (int(self.pos.x()),int(self.pos.y()))

    def convert_nparray_to_QPixmap(self,img):
        w,h = img.shape

        qimg = QImage(img.data, h, w, QImage.Format_Grayscale16) 
        qpixmap = QPixmap(qimg)

        return qpixmap



    def __init__(self,  args, sx, sy):
        self.sx = sx
        self.sy = sy
        self.capture_state = 0
        self.update_state = 1
        self.rms = 0
        self.pos = QPoint(256,256)
        self.array = np.random.randint(0,65000, (sx,sy), dtype=np.uint16)
        
        self.win = FrameWindow()
        self.EDGE = 16
        
        self.win.resize(1500,1100)
        
        self.imv = pg.ImageView()
        self.imv.setImage(self.array)
        
      
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
        self.plt_bufsize = 200
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
        rightlayout.setFixedSize(564, 228)
        
        self.filename = QtWidgets.QLineEdit(args.filename)
        rightlayout.layout().addWidget(self.filename)

        

        self.capture_button =  QtWidgets.QPushButton("Start Capture")
        rightlayout.layout().addWidget(self.capture_button)

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



        self.statusBar.addPermanentWidget(rightlayout)

        self.win.setStatusBar(self.statusBar)
        
      
        self. win.setWindowTitle('qhycam')
        self.imv.getImageItem().mouseClickEvent = self.click
        self.cnt = 0

        self.capture_button.clicked.connect(self.Capture_buttonClick)
        self.update_button.clicked.connect(self.Update_buttonClick)
        
        self.win.show()
    


    def Update_buttonClick(self):
        print("button")

        if (self.update_state == 1):
            self.update_button.setText("fast_update")
            self.update_state = 0
        else:
            self.update_button.setText("slow_update")
            self.update_state = 1


    def toggle_capture(self):
        if (self.capture_state == 0):
            
            self.capture_button.setText("Stop Capture")
            vnow = time.time_ns()
            self.capture_filename = self.filename.text() + str(vnow) + ".ser"
            self.capture_file = SerWriter(self.capture_filename)
            self.capture_file.set_sizes(self.sx, self.sy, 2)
            self.capture_state = 1
            self.cnt = 0
        else:
            self.capture_state = 0
            self.capture_button.setText("Start Capture")
            self.capture_file.close()

    def Capture_buttonClick(self):
        self.toggle_capture()


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

        #print(pos)
        return pos

    def update(self):
        #self.array = denoise_image(self.array.copy())
        #print(self.array)
        self.imv.setImage(np.flip(np.rot90(self.array), axis=0), autoRange=False, autoLevels=False, autoHistogramRange=False) #, pos=[-1300,0],scale=[2,2])
      
        pos = self.clip(self.pos)
       

        sub = self.array[int(pos.y())-self.EDGE:int(pos.y())+self.EDGE, int(pos.x())-self.EDGE:int(pos.x())+self.EDGE].copy()

        min = np.min(sub)
        max = np.max(sub)
        fwhm = fit_gauss_circular(sub)

        self.rms = np.std(self.array)

        self.txt1.setText("FWHM= " + "{:.2f}  ".format(fwhm) + "min=" + str(min) + " max=" + str(max) + " frame=" + str(self.cnt) + " RMS=" + "{:.1f} ".format(self.rms))
        self.updateplot(fwhm)

        if (self.cnt % 30 == 0):
            if not (sky is None):
                p0 = sky.GetRaDec()
                
                self.txt2.setText("RA = " + p0[0][0:8] + " DEC=" + p0[1][0:8])

            temp = camera.qc.GetTemperature()
            self.txt3.setText("Temp = " + str(temp))


        sub = sub - min
        max = max - min
        sub =  sub * (65535.0/((max+1)))
        sub = sub.astype(np.uint16)
        sub = cv2.resize(sub, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        pixmap = self.convert_nparray_to_QPixmap(sub)
        self.zoom_view.setPixmap(pixmap)



    def mainloop(self, args, camera):

 
        while(self.win.quit == 0):
            time.sleep(0.008)
            if (self.mover.moving()):
                rx, ry = self.mover.rate()
                print("move at " + str(rx) + " " + str(ry))
            
            self.statusBar.showMessage(str(self.cnt), 2000)
            app.processEvents()
            self.array = camera.get_frame()
            if (self.capture_state == 1):
                self.capture_file.add_image(self.array)
                if (self.cnt > 3000):
                    self.toggle_capture()
                    self.toggle_capture()

            need_update = False
            if (self.update_state == 1):
                need_update = True
            if (self.update_state == 0 and self.cnt % 30 == 0):
                need_update = True
            #if (self.cnt % 30 == 15):
                #print(camera.vcam.temp)

            if (need_update):
                self.update()
             #if (cnt % 10 == 0):
            #    imv.ui.histogram.setImageItem(pg.ImageItem(array))
            self.cnt = self.cnt + 1

        if (self.capture_state == 1):
            self.capture_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default = 'emccd_capture_', help="generic file name")
    parser.add_argument("-exp", type=float, default = 0.1, help="exposure in seconds (default 0.1)")
    parser.add_argument("-gain", "--gain", type=int, default = 100, help="camera gain (default 100)")
    parser.add_argument("-bin", "--bin", type=int, default = 1, help="camera binning (default 1-6)")
    parser.add_argument("-guide", "--guide", type=int, default = 0, help="frame per guide cycle (0 to disable)")
    parser.add_argument("-count", "--count", type=int, default = 100, help="number of frames to capture")
    args = parser.parse_args()

    try:
        sky.Connect()
    except:
        sky = None

    #if not (sky is None):
        #sky.bump(120,0)



    camera = qhy_cam(-12, args.exp, args.gain, 1800, 800)
    ui = UI(args, 1800, 800)
    
    camera.start()

    ui.mainloop(args, camera)


