import sys
import os
import numpy as np
from astropy.io import fits
from PyQt5 import QtWidgets
from util import fit_gauss_circular
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


class FlatFieldViewer(QtWidgets.QMainWindow):
    def __init__(self, calibrated_image, output_filename):
        super().__init__()
        self.currentImage = calibrated_image
        self.output_filename = output_filename
        self.histLevels = None

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

        self.resize(1600, 1200)

        # --- Setup Actions and Shortcuts ---
        self.toggleFullScreenAction = QtWidgets.QAction("Toggle Full-Screen", self)
        self.toggleFullScreenAction.setShortcut("f")
        self.toggleFullScreenAction.triggered.connect(self.toggleFullScreen)
        self.addAction(self.toggleFullScreenAction)

        self.showFilenameAction = QtWidgets.QAction("Show Filename", self)
        self.showFilenameAction.setShortcut("n")
        self.showFilenameAction.triggered.connect(self.showFilename)
        self.addAction(self.showFilenameAction)

        self.autoScale90Action = QtWidgets.QAction("Auto Scale to 90%", self)
        self.autoScale90Action.setShortcut("A")
        self.autoScale90Action.triggered.connect(self.setAutoScale90)
        self.addAction(self.autoScale90Action)

        self.saveAction = QtWidgets.QAction("Save Output", self)
        self.saveAction.setShortcut("s")
        self.saveAction.triggered.connect(self.saveOutput)
        self.addAction(self.saveAction)

        self.imageView.getImageItem().mouseClickEvent = self.click

        # Display the calibrated image
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

    def updateImage(self):
        """Display the calibrated image."""
        if self.currentImage is None:
            self.imageView.clear()
            self.setWindowTitle("Flat Field Calibration")
            return

        self.setWindowTitle(f"Flat Field Calibrated: {self.output_filename}")

        self.imageView.setImage(self.currentImage, autoHistogramRange=False)
        self.setAutoScale90()

    def saveOutput(self):
        """Save the calibrated image to a FITS file."""
        if self.currentImage is None:
            return

        try:
            # Rotate back to original orientation before saving
            data_to_save = np.rot90(self.currentImage, k=-1)
            hdu = fits.PrimaryHDU(data_to_save)
            hdu.writeto(self.output_filename, overwrite=True)
            print(f"Saved calibrated image to: {self.output_filename}")
            QtWidgets.QMessageBox.information(self, 'Success', f'Saved to {self.output_filename}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to save file:\n{e}')

    def showFilename(self):
        print(f"Output file: {self.output_filename}")


def load_fits(filename):
    """Load a FITS file and return the data."""
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError("No data in primary HDU.")
            return data.astype(np.float64)
    except Exception as e:
        print(f"Error loading {filename}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flat field calibration viewer.")
    parser.add_argument("image", help="Input image FITS file")
    parser.add_argument("flat_field", help="Flat field FITS file")
    parser.add_argument("flat_bias", help="Flat field bias FITS file")
    parser.add_argument("dark", help="Dark frame FITS file")
    parser.add_argument("-o", "--output", default="output.fits",
                        help="Output filename (default: output.fits)")

    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    image = load_fits(args.image)

    print(f"Loading flat field: {args.flat_field}")
    flat_field = load_fits(args.flat_field)

    print(f"Loading flat bias: {args.flat_bias}")
    flat_bias = load_fits(args.flat_bias)

    print(f"Loading dark: {args.dark}")
    dark = load_fits(args.dark)

    # Perform calibration: (image - dark) / (flat_field - flat_bias)
    print("Performing flat field calibration...")
    flat_corrected = flat_field - flat_bias

    # Avoid division by zero
    flat_corrected[flat_corrected == 0] = 1.0

    calibrated = (image - dark) / flat_corrected

    # Rotate for display (same as s1.py)
    calibrated_rotated = np.rot90(calibrated)

    print("Calibration complete. Launching viewer...")
    print(f"Press 's' to save to {args.output}")

    app = QtWidgets.QApplication(sys.argv)
    viewer = FlatFieldViewer(calibrated_rotated, args.output)
    viewer.show()

    sys.exit(app.exec_())
