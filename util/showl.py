import sys
import numpy as np
import cv2
from astropy.io import fits
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer

import pyqtgraph as pg
import argparse


class ImageNavigator(QtWidgets.QMainWindow):
    def __init__(self, images):
        super().__init__()
        self.images = [np.rot90(image) for image in images]  # Rotate each image by 90 degrees
        self.currentIndex = 0
        self.histLevels = None  # Store histogram levels
        self.prevTransform = None  # Initialize variable to store previous transformation

        self.initUI()
        
    def initUI(self):
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.centralWidget.setLayout(self.layout)
        
        self.imageView = pg.ImageView()
        self.layout.addWidget(self.imageView)
        
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.buttonLayout)
        
        self.prevButton = QtWidgets.QPushButton("Previous")
        self.prevButton.clicked.connect(self.prevImage)
        self.buttonLayout.addWidget(self.prevButton)
        
        self.nextButton = QtWidgets.QPushButton("Next")
        self.nextButton.clicked.connect(self.nextImage)
        self.buttonLayout.addWidget(self.nextButton)
        
        self.updateImage()
        self.resize(1600, 1200)  # Set the window size to 1600x1200

        
        self.toggleFullScreenAction = QtWidgets.QAction("Toggle Full-Screen", self)
        self.toggleFullScreenAction.setShortcut("f")
        self.toggleFullScreenAction.triggered.connect(self.toggleFullScreen)
        self.addAction(self.toggleFullScreenAction)

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


    def prevImage(self):
        if self.currentIndex > 0:
            self.saveHistogramSettings()
            self.saveViewTransform()
            self.currentIndex -= 1
            self.updateImage()
        
    def nextImage(self):
        if self.currentIndex < len(self.images) - 1:
            self.saveHistogramSettings()
            self.saveViewTransform()
            self.currentIndex += 1
            self.updateImage()
        
    def saveHistogramSettings(self):
        # Save the current histogram settings
        self.histLevels = self.imageView.getHistogramWidget().getLevels()
        
    def saveViewTransform(self):
        # Save the current view transformation matrix
        self.prevTransform = self.imageView.view.getViewBox().transform()

    def updateImage(self):
        self.imageView.setImage(self.images[self.currentIndex], autoHistogramRange=False)
        if self.histLevels:
            self.imageView.setLevels(*self.histLevels)  # Apply saved histogram settings
        self.restoreViewTransform()  # Restore the view state after setting the image
        #QTimer.singleShot(50, self.restoreViewTransform)


    def restoreViewTransform(self):
        # Restore the saved view transformation if it exists
        if self.prevTransform:
            print("yes")
            self.imageView.view.getViewBox().setTransform(self.prevTransform)

def loadFITSFiles(fileList):
    images = []
    for file in fileList:
        data = fits.getdata(file, ext=0)
        images.append(data)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FITS Viewer with Navigation")
    parser.add_argument("files", nargs='+', help="List of FITS files to view")
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    
    images = loadFITSFiles(args.files)
    navigator = ImageNavigator(images)
    navigator.show()
    
    sys.exit(app.exec_())
