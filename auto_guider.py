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

ipc.set_val("test", [1,2])

print(ipc.get_val("test"))

#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------
import pickle

cheat_move_x = 0
cheat_move_y = 0


def fbump(dx, dy):
    global cheat_move_x
    global cheat_move_y

    cheat_move_x += dx / 50.0
    cheat_move_y += dy / 50.0


def rand_move():
    fbump(random.uniform(-277, 270), random.uniform(-270, 270))

class LastNValues:
    def __init__(self, n):
        self.n = n
        self.values = []

    def add_value(self, x):
        if len(self.values) == self.n:
            # If the array is already full, remove the first element
            self.values.pop(0)
        self.values.append(x)

    def same_sign(self):
        if not self.values or len(self.values) < self.n:
            # If the array is empty or not full yet, return False
            return False
        return (self.values[0] > 0 and self.values[-1] > 0) or (self.values[0] < 0 and self.values[-1] < 0)

class guider:
    def __init__(self, mount, camera):
        log.info("init")
        self.reset()
        self.mount = mount
        self.camera = camera
        self.guide_inited = 0
        self.filter = KalmanFilter(0.1)
        self.gain_x = 110.0
        self.gain_y = 110.0
        N = 4
        self.last_x = LastNValues(N)
        self.last_y = LastNValues(N)
        self.load_state("guide.data")


    def start_calibrate(self):
        log("calibrate")
        self.cal_state = 20

    def stop_calibrate(self):
        self.cal_state = 0

    def start_guide(self):
        self.is_guiding = 1

    def stop_guide(self):
        self.is_guiding = 0

    def save_state(self, filename):
        settings = {}

        settings['mount_dx1'] = self.mount_dx1
        settings['mount_dx2'] = self.mount_dx2
        settings['mount_dy1'] = self.mount_dy1
        settings['mount_dy2'] = self.mount_dy2

        settings['gain_x'] = self.gain_x
        settings['gain_y'] = self.gain_y

        with open(filename, "wb") as f:
            pickle.dump(settings, f)

    def load_state(self, filename):
        """Load the state of the object from a file.

        Arguments:
        filename -- the name of the file to load the state from
        """
        try:
            with open(filename, "rb") as f:
                settings = pickle.load(f)
                self.mount_dx1 = settings['mount_dx1']
                self.mount_dx2 = settings['mount_dx2']
                self.mount_dy1 = settings['mount_dy1']
                self.mount_dy2 = settings['mount_dy2']

                self.gain_x = settings['gain_x']
                self.gain_y = settings['gain_y']
               
        except Exception as e:
            log.critical("An error occurred while loading the state:", e)
            self.reset()

    def reset(self):
        """Reset the object's state to its default values."""
        self.cal_state = 0
        self.is_guiding = 0
        
        self.mount_dx1 = 0
        self.mount_dy1 = 0
        self.mount_dx2 = 0
        self.mount_dy2 = 0
        self.guide_state = 0
        self.cal_state = 0



    def new_pos(self, x, y):
        log.info("new pos %d %d", x, y)

    def set_pos(self, x, y):
        log.info("set pos %d %d", x, y)

    def calibrate(self):
        self.cal_state = 20
        self.guide_state = 0


    def guide(self):
        self.guide_state = 1

    def handle_calibrate(self, x, y):
        if (self.cal_state == 20):
            self.pos_x0 = x
            self.pos_y0 = y
            fbump(-300, 0)
            log.info("Move Left")

        if (self.cal_state == 15):
            self.pos_x1 = x
            self.pos_y1 = y
            fbump(300, 0)
            log.info("Move Right")


        if (self.cal_state == 10):
            self.pos_x2 = x
            self.pos_y2 = y
            fbump(0, -300)
            log.info("Move Up")


        if (self.cal_state == 5):
            self.pos_x3 = x
            self.pos_y3 = y
            fbump(0, 300)
            log.info("Move Down")


        if (self.cal_state == 1):
            self.calc_calibration()

        self.cal_state = self.cal_state - 1
        if (self.cal_state < 0):
            self.cal_state = 0

    def calc_calibration(self):
        log.info("calc cal")
        self.mount_dx1 = self.pos_x1 - self.pos_x0       
        self.mount_dy1 = self.pos_y1 - self.pos_y0

        self.mount_dx2 = self.pos_x3 - self.pos_x2      
        self.mount_dy2 = self.pos_y3 - self.pos_y2

        self.save_state("guide.data")


    def calibrate_state(self):
        return cal_state

    def distance(self, x, y):
        return np.sqrt(x*x+y*y)


    def offset(self, dx, dy):
        self.center_x = self.center_x + dx
        self.center_y = self.center_y + dy

        log.info("new guide position %f %f", self.center_x, self_center_y)

    def handle_guide(self, x, y):
        if (self.guide_inited == 0):
            self.center_x = x
            self.center_y = y
            self.guide_inited = 1
        else:
            dx = x - self.center_x
            dy = y - self.center_y

           
            self.dis = self.distance(dx,dy)

            if (self.dis > 20.0):
                return
            self.filter.update(GPoint(dx,dy))
            val = self.filter.value()

            log.info("e0", self.filter.value())

            self.last_x.add_value(dx)
            self.last_y.add_value(dy)

            if (self.last_x.same_sign()):
                self.gain_x = self.gain_x + 0.1
            else:
                self.gain_x = self.gain_x - 0.1
                
            if (self.last_y.same_sign()):
                self.gain_y = self.gain_y + 0.1
            else:
                self.gain_y = self.gain_y - 0.1
                             

            tx = self.error_to_tx(self.gain_x * dx, self.gain_y * dy)
            ty = self.error_to_ty(self.gain_x *dx, self.gain_y * dy)

            log.info("ERROR %f %f %f %f", dx, dy, tx, ty)
            fbump(tx, ty)
            #self.mount(bump, tx, ty)

        log.info("get guide point %f %f", x, y)

    def pos_handler(self, x, y):
        log.info("handler %f %f", x, y)
        if self.cal_state != 0:
            self.handle_calibrate(x, y)
        if self.guide_state != 0:
            self.handle_guide(x, y)

            
    def error_to_tx(self, mx, my):
        num = (self.mount_dy2 * mx) - (self.mount_dx2 * my)
        den = (self.mount_dx1 * self.mount_dy2) - (self.mount_dx2 * self.mount_dy1)

        return num / den

    def error_to_ty(self, mx, my):
        num = (self.mount_dy1 * mx) - (self.mount_dx1 * my)
        den = (self.mount_dx2 * self.mount_dy1) - (self.mount_dx1 * self.mount_dy2)

        return num / den



class fake_cam:
    def __init__(self, temp, exp, gain, crop):
        
        log.info("init cam")
        self.frame = np.random.randint(0,4096, (512,512), dtype=np.uint16)
        self.stars_frame = self.stars(self.frame, 4, gain=2)

    def stars(self, image, number, max_counts=3000, gain=1):
        global cheat_move_y
        global cheat_move_x
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
        

        return self.frame + ndimage.shift(self.stars_frame.astype(np.uint16), (cheat_move_x, cheat_move_y))
        
    def start(self):
        self.running = 1
        
    def close(self):
        self.running = 0

    def size_x(self):
        return 512
    
    def size_y(self):
        return 512



class qhy_cam:
    def __init__(self, temp, exp, gain, crop):
        self.qc = qhyccd.qhyccd()
        self.dt = exp
        self.qc.GetSize()
        self.qc.SetBit(16)
        self.qc.SetUSB(11)
        self.qc.SetOffset(144)
        self.qc.SetTemperature(temp)
        self.sizex = int(self.qc.image_size_x * crop)
        self.sizey = int(self.qc.image_size_y * crop)

        ddx = self.qc.image_size_x - self.sizex
        ddy = self.qc.image_size_y  -self.sizey
        ddx = ddx // 2
        ddy = ddy // 2

        self.qc.SetROI(ddx,ddy,ddx + self.sizex,ddy + self.sizey)
        self.qc.SetExposure(self.dt*1000)
       
        self.qc.SetGain(gain)
        
   
    def get_frame(self):        
        self.frame = self.qc.GetLiveFrame()
        #self.qc.GetStatus()
        return self.frame
        
    def start(self):
        self.running = 1

        self.qc.BeginLive()
        
    def close(self):
        self.running = 0
        self.qc.StopLive()

    def size_x(self):
        return self.sizex
    
    def size_y(self):
        return self.sizey


class FrameWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.quit = 0


    def closeEvent(self, event):
        self.quit = 148
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
        
      	
        self.rms = 0
        self.pos = QPoint(256,256)
        self.array = np.random.randint(0,65000, (sx,sy), dtype=np.uint16)

        
        self.win = FrameWindow()
        self.EDGE = 16
        
        self.win.resize(800,900)
        
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
        self.plt = pg.plot(title='dx')
        self.plt_bufsize = 200
        self.x = np.linspace(-self.plt_bufsize, 0.0, self.plt_bufsize)
        self.y = np.zeros(self.plt_bufsize, dtype=np.float64)
        self.databuffer = collections.deque([0.0]*self.plt_bufsize, self.plt_bufsize)

        temp_widget.layout().addWidget(self.plt)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'pos_x', 'pixels')
        self.plt.setLabel('bottom', 'frame', 'f')
        self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
        
        self.statusBar.addPermanentWidget(temp_widget, 1)



        rightlayout = QtWidgets.QWidget(self.win)
        rightlayout.setLayout(QtWidgets.QVBoxLayout())
        rightlayout.setFixedSize(564, 228)
        
        

        self.calibrate_button =  QtWidgets.QPushButton("Calibrate")
        rightlayout.layout().addWidget(self.calibrate_button)
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

        self.calibrate_button.clicked.connect(self.Calibrate_buttonClick)
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



    def Calibrate_buttonClick(self):
        self.guider.calibrate()
        log.info("Calibrate")

    def Guide_buttonClick(self):
        self.guider.guide()
        log.info("Guide")

    def updateplot(self, x):
        self.databuffer.append(x)
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
        self.txt1.setText("FWHM= " + "min=" + "{:04d}".format(self.min) + " max=" + "{:04d}".format(self.max) + " frame=" + str(self.cnt) + " RMS=" + "{:.1f} ".format(self.rms))
        self.updateplot(self.cx)

        if (self.cnt % 30 == 0):
            if not (sky is None):
                p0 = sky.GetRaDec()
                
                self.txt2.setText("RA = " + p0[0][0:8] + " DEC=" + p0[1][0:8])

            self.temp = 0 #camera.qc.GetTemperature()
            self.txt3.setText("Temp = " + str(self.temp) + " fps=" + "{:.2f}".format(self.fps))




    def update(self):
        self.imv.setImage(np.flip(np.rot90((self.array)), axis=0), autoRange=False, autoLevels=False, autoHistogramRange=False) #, pos=[-1300,0],scale=[2,2])
        self.imv.setLevels(0,3384)
        self.txt4.setText("X="  + "{:.2f}".format(self.cx) + " Y="  + "{:.2f}".format(self.cy) + " gx=" + "{:.2f}".format(self.guider.gain_x) + " gy=" + "{:.2f}".format(self.guider.gain_y))

        
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

        if (bump[0] != 0.0 and bump[1] != 0):
            return

        self.guider.offset(bump[0], bump[1])
        ipc.set_val("bump", [0,0])

    def mainloop(self, args, camera):
        global cheat_move_y
        global cheat_move_x

        mean_old = 0.0

        while(self.win.quit == 0):
            time.sleep(0.002)
            if (self.mover.moving()):
                rx, ry = self.mover.rate()
                cheat_move_x = cheat_move_x + rx
                cheat_move_y = cheat_move_y + ry
               
                log.info("move at %f %f", rx, ry)
            
            
            app.processEvents()
            self.array = camera.get_frame()
            mean_new = np.mean(self.array)

            if (mean_new != mean_old):
                mean_old = mean_new

                max_y, max_x = find_high_value_element(self.array[16:-16, 16:-16])
                log.info("max value = %d %d", max_x, max_y)
                self.cy, self.cx, cv = compute_centroid(self.array, max_y + 16, max_x + 16)
                log.info("calc centroid = %f %f", self.cx, self.cy)

                self.ipc_check()


                self.guider.pos_handler(self.cx, self.cy)
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
    args = parser.parse_args()

    try:
        sky.Connect()
    except:
        sky = None

    #if not (sky is None):
        #sky.bump(120,0)



    camera = fake_cam(-10, args.exp, args.gain, args.crop)
    guider = guider(sky, camera)

    ui = UI(args, camera.size_x(), camera.size_y(), guider)
    
    camera.start()

    ui.mainloop(args, camera)


