import sys
import os
import numpy as np
import cv2
from astropy.io import fits
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer
from util import * # Assuming util.py with fit_gauss_circular exists

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
    if total_flux == 0:
        return 0 # Avoid division by zero
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
    def __init__(self, filenames):
        super().__init__()
        # Store file paths instead of image data to save memory
        self.filenames = filenames
        self.currentImage = None # This will hold the data for the current image
        self.currentIndex = 0
        self.histLevels = None  # Store histogram levels
        self.prevTransform = None  # Initialize variable to store previous transformation

        self.initUI()
        
    def click(self, event):
        event.accept()   
        if self.currentImage is None:
            return
            
        self.pos = event.pos()
        x = int(self.pos.x())
        y = int(self.pos.y())
        
        # Ensure coordinates are within image bounds
        if 0 <= x < self.currentImage.shape[0] and 0 <= y < self.currentImage.shape[1]:
            # Extract a 20x20 box around the click
            data = self.currentImage[max(0, x-10):x+10, max(0, y-10):y+10]
            if data.size == 0:
                return

            data = data - np.min(data)
            print("--- Star Analysis ---")
            print(f"Clicked at: ({x}, {y})")
            print(f"Max value in box: {np.max(data)}")
            print(f"HFD = {compute_hfd(data):.2f}")
            fwhm = fit_gauss_circular(data)
            print(f"FWHM = {fwhm:.2f}")
            print("---------------------")

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
        
        self.resize(1600, 1200) # Set a default window size

        # --- Setup Actions and Shortcuts ---
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

        # New auto-scale action with 'A' shortcut
        self.autoScale90Action = QtWidgets.QAction("Auto Scale to 90%", self)
        self.autoScale90Action.setShortcut("A")
        self.autoScale90Action.triggered.connect(self.setAutoScale90)
        self.addAction(self.autoScale90Action)

        self.imageView.getImageItem().mouseClickEvent = self.click
        
        # Load the first image
        self.updateImage()

    def setAutoScale90(self):
        """Set image levels to background (median) and 90th percentile."""
        if self.currentImage is None:
            return
            
        flat_data = self.currentImage.flatten()
        background = np.median(flat_data)
        max_level = np.percentile(flat_data, 90)
        
        # Fallback for low-contrast images
        if background >= max_level:
            max_level = np.percentile(flat_data, 99)

        self.imageView.setLevels(background, max_level)
        self.histLevels = (background, max_level)

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
        # Use self.filenames to check length
        if self.currentIndex < len(self.filenames) - 1:
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
        """Loads and displays the image at the current index."""
        if not self.filenames or self.currentIndex < 0:
            self.imageView.clear()
            self.currentImage = None
            self.setWindowTitle("FITS Viewer")
            return

        filename = self.filenames[self.currentIndex]
        try:
            # Load one FITS file at a time
            with fits.open(filename) as hdul:
                data = hdul[0].data
                if data is None:
                    raise ValueError("No data in primary HDU.")
                # Keep original rotation for consistency
                self.currentImage = np.rot90(data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load FITS file {filename}:\n{e}')
            self.imageView.clear()
            self.currentImage = None
            self.setWindowTitle(f"Error loading {os.path.basename(filename)}")
            return

        # Update window title with filename and index
        title = f"{os.path.basename(filename)} ({self.currentIndex + 1}/{len(self.filenames)})"
        self.setWindowTitle(title)
        
        self.imageView.setImage(self.currentImage, autoHistogramRange=False)
        
        if self.histLevels:
            self.imageView.setLevels(*self.histLevels)
        else:
            # Set a default level for the first image loaded
            self.setAutoScale90()

        self.restoreViewTransform()

    def restoreViewTransform(self):
        if self.prevTransform:
            self.imageView.view.getViewBox().setTransform(self.prevTransform)

    def deleteFile(self):
        if not self.filenames or self.currentIndex < 0:
            return

        filename = self.filenames[self.currentIndex]
        
        if True:
            try:
                os.remove(filename)
                print(f"Deleted file: {filename}")
                # Remove from list, no need to touch self.images anymore
                self.filenames.pop(self.currentIndex)
                
                if not self.filenames:
                    self.currentIndex = -1
                    self.updateImage() # Clear the view
                else:
                    if self.currentIndex >= len(self.filenames):
                        self.currentIndex = len(self.filenames) - 1
                    self.updateImage() # Load the next/previous image
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to delete file: {e}')

    def showFilename(self):
        if self.currentIndex >= 0 and self.currentIndex < len(self.filenames):
            filename = self.filenames[self.currentIndex]
            print(f"Current file: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FITS Viewer with on-demand loading.")
    parser.add_argument("files", nargs='+', help="List of FITS files to view")
    
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    
    if not args.files:
        print("Error: No FITS files specified.", file=sys.stderr)
        sys.exit(1)
        
    # Pass the list of filenames directly to the navigator
    navigator = ImageNavigator(args.files)
    navigator.show()

    sys.exit(app.exec_())
