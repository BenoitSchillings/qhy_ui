import sys
import os
import numpy as np
import cv2
from astropy.io import fits
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer
from util import *

import pyqtgraph as pg
import argparse

def compute_hfd(image):
    """
    Compute the half flux diameter (HFD) of a star image in a 2D array.
    
    Args:
        image (numpy.ndarray): 2D array containing the star image.
        
    Returns:
        float: The half flux diameter (HFD) of the star image.
    """
    # Find the centroid (center of mass) of the star image
    total_flux = np.sum(image)
    y, x = np.indices(image.shape)
    y_centroid = np.sum(y * image) / total_flux
    x_centroid = np.sum(x * image) / total_flux
    
    # Sort the pixel values in descending order
    sorted_pixels = np.sort(image.ravel())[::-1]
    
    # Calculate the cumulative sum of the sorted pixel values
    cumsum = np.cumsum(sorted_pixels)
    
    # Find the radius at which the cumulative sum reaches half of the total flux
    half_flux = total_flux / 2
    idx = np.searchsorted(cumsum, half_flux, side='right')
    radius = np.sqrt((idx - 1) / np.pi)
    
    # The HFD is twice the radius
    return 2 * radius

class ImageNavigator(QtWidgets.QMainWindow):
    def __init__(self, images, filenames):
        super().__init__()
        self.images = [np.rot90(image) for image in images]  # Rotate each image by 90 degrees
        self.filenames = filenames
        self.currentIndex = 0
        self.histLevels = None  # Store histogram levels
        self.prevTransform = None  # Initialize variable to store previous transformation

        self.initUI()
        
    def click(self, event):
        event.accept()      
        self.pos = event.pos()
        print("click", int(self.pos.x()), int(self.pos.y()))
        x = int(self.pos.x())
        y = int(self.pos.y())
        data = self.images[self.currentIndex][x-10:x+10, y-10:y+10]
        data = data - np.min(data)
        print("max = ", np.max(data))
        print("hfd = ", compute_hfd(data))

        fwhm = fit_gauss_circular(data)
        print("fwhm = ", fwhm)

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

        # Add existing actions
        self.toggleFullScreenAction = QtWidgets.QAction("Toggle Full-Screen", self)
        self.toggleFullScreenAction.setShortcut("f")
        self.toggleFullScreenAction.triggered.connect(self.toggleFullScreen)
        self.addAction(self.toggleFullScreenAction)

        self.deleteFileAction = QtWidgets.QAction("Delete File", self)
        self.deleteFileAction.setShortcut("d")
        self.deleteFileAction.triggered.connect(self.deleteFile)
        self.addAction(self.deleteFileAction)

        self.showFilenameAction = QtWidgets.QAction("Show Filename", self)
        self.showFilenameAction.setShortcut("n")
        self.showFilenameAction.triggered.connect(self.showFilename)
        self.addAction(self.showFilenameAction)

        # Add new auto-scale action
        self.autoScaleAction = QtWidgets.QAction("Auto Scale", self)
        self.autoScaleAction.setShortcut("Ctrl+A")
        self.autoScaleAction.triggered.connect(self.autoScale)
        self.addAction(self.autoScaleAction)

        self.imageView.getImageItem().mouseClickEvent = self.click

    def autoScale(self):
        """Set image levels to 3% and 70% percentiles of the data."""
        current_image = self.images[self.currentIndex]
        min_level = np.percentile(current_image, 2)
        max_level = np.percentile(current_image, 97)
        self.imageView.setLevels(min_level, max_level)
        self.histLevels = (min_level, max_level)

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

    def restoreViewTransform(self):
        # Restore the saved view transformation if it exists
        if self.prevTransform:
            self.imageView.view.getViewBox().setTransform(self.prevTransform)

    def deleteFile(self):
        if self.currentIndex >= 0 and self.currentIndex < len(self.filenames):
            filename = self.filenames[self.currentIndex]
            try:
                os.remove(filename)
                self.images.pop(self.currentIndex)
                self.filenames.pop(self.currentIndex)
                if self.currentIndex >= len(self.images):
                    self.currentIndex = len(self.images) - 1
                self.updateImage()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to delete file: {e}')

    def showFilename(self):
        if self.currentIndex >= 0 and self.currentIndex < len(self.filenames):
            filename = self.filenames[self.currentIndex]
            print(f"Current file: {filename}")

def loadFITSFiles(fileList):
    images = []
    filenames = []
    for file in fileList:
        data = fits.getdata(file, ext=0)
        images.append(data)
        filenames.append(file)
    return images, filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FITS Viewer with Navigation")
    parser.add_argument("files", nargs='+', help="List of FITS files to view")
    
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    images, filenames = loadFITSFiles(args.files)
    navigator = ImageNavigator(images, filenames)
    navigator.show()

    sys.exit(app.exec_())