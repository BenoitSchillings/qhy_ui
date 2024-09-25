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

log.basicConfig(level=log.INFO)


sky = skyx.sky6RASCOMTele()
ipc = IPC()


#--------------------------------------------------------
app = QtWidgets.QApplication([])

#--------------------------------------------------------
import argparse
#--------------------------------------------------------


from qhy_cam_interface import *




if __name__ == "__main__":

    
    camera = qhy_cam(30, 1, 1, 1.0, "268", False)


    camera.start()


    while(True):

        print(camera.qc.GetTemperature())
        time.sleep(5)