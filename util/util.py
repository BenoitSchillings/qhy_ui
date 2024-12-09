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

import numpy as np

from scipy.optimize import leastsq


from scipy.optimize import curve_fit

def find_minimum_parabola(data):
    """
    Fit a parabola to the given data and find the position of the minimum.
    
    Args:
        data (numpy.ndarray): 1D array of float values representing the data.
        
    Returns:
        float: The position of the minimum of the fitted parabola.
    """
    # Ensure the input data is a 1D numpy array
    data = np.asarray(data).ravel()
    
    # Define the parabola function
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Get the indices of the data points
    x = np.arange(len(data))
    
    # Fit the parabola using non-linear least squares
    try:
        popt, pcov = curve_fit(parabola, x, data, p0=[1, 1, 1])
    except RuntimeError:
        # If the fit fails, try again with a different initial guess
        try:
            popt, pcov = curve_fit(parabola, x, data, p0=[-1, -1, -1])
        except RuntimeError:
            # If the fit still fails, return None
            return None
    
    # Calculate the minimum position
    a, b, c = popt
    if a == 0:
        # If the parabola is a straight line, return None
        return None
    else:
        min_position = -b / (2 * a)
        return min_position

        
def compute_hfd(image):
    """
    Compute the half flux diameter (HFD) of a star image in a 2D array.
    
    Args:
        image (numpy.ndarray): 2D array containing the star image.
        
    Returns:
        float: The half flux diameter (HFD) of the star image.
    """
    # Find the centroid (center of mass) of the star image
    image = image - np.min(image)
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
    hfd = 2 * radius
    
    return hfd
    
def extract_centered_subarray(array, subarray_size):
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2-dimensional")
    
    if subarray_size % 2 == 0:
        raise ValueError("Subarray size must be odd")
    
    # Find the coordinates of the maximum value
    max_coord = np.unravel_index(np.argmax(array), array.shape)

    # Calculate the center of the subarray
    center_x, center_y = max_coord
    
    # Calculate the half-width of the subarray
    half_width = subarray_size // 2

    # Calculate the boundaries of the subarray
    start_x = max(0, center_x - half_width)
    end_x = min(array.shape[0], center_x + half_width + 1)
    start_y = max(0, center_y - half_width)
    end_y = min(array.shape[1], center_y + half_width + 1)

    # Extract the subarray
    subarray = array[start_x:end_x, start_y:end_y]

    # If the subarray is smaller than the desired size due to edge cases, pad it
    if subarray.shape != (subarray_size, subarray_size):
        padded = np.zeros((subarray_size, subarray_size))
        x_offset = half_width - center_x if center_x < half_width else 0
        y_offset = half_width - center_y if center_y < half_width else 0
        padded[x_offset:x_offset+subarray.shape[0], 
               y_offset:y_offset+subarray.shape[1]] = subarray
        return padded
    
    return subarray


    
def fit_gauss_circular(data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D circular gaussian PSF.
    ---------------------
    Inputs

    * data (2D Numpy array) = small subimage
    ---------------------
    Output (list) = list with 6 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm is the gaussian PSF full width half maximum (fit result) in pixels
    ---------------------
    """
    
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height < 200:
        return 0
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    
    #---------------------------------------------------------------------------------
    sig = fwhm / (2.*np.sqrt(2.*np.log(2.)))
    width = 0.5/np.square(sig)
    
    p0 = floor, height, mean_x, mean_y, width

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def gauss(floor, height, mean_x, mean_y, width):        
        return lambda x,y: floor + height*np.exp(-np.abs(width)*((x-mean_x)**2+(y-mean_y)**2))

    def err(p,data):
        return np.ravel(gauss(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=300)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2]
    mean_y = p[3]

    sig = np.sqrt(0.5/np.abs(p[4]))
    fwhm = sig * (2.*np.sqrt(2.*np.log(2.)))    
    
    output = [maxi, floor, height, mean_x, mean_y, fwhm]
    return fwhm


def fit_gauss_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical gaussian PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 8 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    ---------------------
    """

    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor
    
    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm
    sig_1 = fwhm_1 / (2.*np.sqrt(2.*np.log(2.)))
    sig_2 = fwhm_2 / (2.*np.sqrt(2.*np.log(2.)))    

    angle = 0.

    p0 = floor, height, mean_x, mean_y, sig_1, sig_2, angle

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def gauss(floor, height, mean_x, mean_y, sig_1, sig_2, angle):
    
        A = (np.cos(angle)/sig_1)**2. + (np.sin(angle)/sig_2)**2.
        B = (np.sin(angle)/sig_1)**2. + (np.cos(angle)/sig_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./(sig_1**2.)-1./(sig_2**2.))

        #do not forget factor 0.5 in exp(-0.5*r**2./sig**2.)    
        return lambda x,y: floor + height*np.exp(-0.5*(A*((x-mean_x)**2)+B*((y-mean_y)**2)+C*(x-mean_x)*(y-mean_y)))

    def err(p,data):
        return np.ravel(gauss(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    
    #angle gives the direction of the p[4]=sig_1 axis, starting from x (vertical) axis, clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):

        fwhm_large = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))    
        angle = np.arctan(np.tan(p[6]))
            
    else:    #then sig_1 is the smallest : we want angle to point to sig_y, the largest
    
        fwhm_large = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))    
        angle = np.arctan(np.tan(p[6]+np.pi/2.))
    
    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]
    return output

def fit_moffat_circular(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D circular moffat PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 7 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm, beta]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm is the gaussian PSF full width half maximum (fit result) in pixels
    - beta is the "beta" parameter of the moffat function
    ---------------------
    """
    
    #---------------------------------------------------------------------------------
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))

    beta = 4
    
    p0 = floor, height, mean_x, mean_y, fwhm, beta

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def moffat(floor, height, mean_x, mean_y, fwhm, beta):
        alpha = 0.5*fwhm/np.sqrt(2.**(1./beta)-1.)    
        return lambda x,y: floor + height/((1.+(((x-mean_x)**2+(y-mean_y)**2)/alpha**2.))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    fwhm = np.abs(p[4])
    beta = p[5]
    
    output = [maxi, floor, height, mean_x, mean_y, fwhm, beta]
    return output

def fit_moffat_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical moffat PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 9 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle, beta]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results), 
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    - beta is the "beta" parameter of the moffat function    
    ---------------------
    """
    
    #---------------------------------------------------------------------------------
    #find starting values
    maxi = data.max()
    floor = np.ma.median(data.flatten())
    height = maxi - floor
    if height==0.0:                #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(data.flatten())
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm

    angle = 0.
    beta = 4
    
    p0 = floor, height, mean_x, mean_y, fwhm_1, fwhm_2, angle, beta

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def moffat(floor, height, mean_x, mean_y, fwhm_1, fwhm_2, angle, beta):
        
        alpha_1 = 0.5*fwhm_1/np.sqrt(2.**(1./beta)-1.)
        alpha_2 = 0.5*fwhm_2/np.sqrt(2.**(1./beta)-1.)
    
        A = (np.cos(angle)/alpha_1)**2. + (np.sin(angle)/alpha_2)**2.
        B = (np.sin(angle)/alpha_1)**2. + (np.cos(angle)/alpha_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./alpha_1**2. - 1./alpha_2**2.)
        
        return lambda x,y: floor + height/((1.+ A*((x-mean_x)**2) + B*((y-mean_y)**2) + C*(x-mean_x)*(y-mean_y))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(data), maxfev=1000)
    p = p[0]
    
    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]
    beta = p[7]
    
    #angle gives the direction of the p[4]=fwhm_1 axis, starting from x (vertical) axis, clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):

        fwhm_large = np.abs(p[4])
        fwhm_small = np.abs(p[5])
        angle = np.arctan(np.tan(p[6]))
            
    else:    #then fwhm_1 is the smallest : we want angle to point to sig_y, the largest
    
        fwhm_large = np.abs(p[5])
        fwhm_small = np.abs(p[4])
        angle = np.arctan(np.tan(p[6]+np.pi/2.))

    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle, beta]
    return output



from cv2 import medianBlur

import numpy as np
from cv2 import medianBlur

class HighValueFinder:
    def __init__(self, search_box_size=32, blur_size=3):
        self.hint_x = None
        self.hint_y = None
        self.reference_value = None
        self.search_box_size = search_box_size
        self.blur_size = blur_size

    def find_high_value_element(self, array):
        array = array.astype('float32')
        filtered_array = medianBlur(array, self.blur_size)

        if self.hint_x is not None and self.hint_y is not None and self.reference_value is not None:
            # Define the search box boundaries
            x_start = max(0, self.hint_x - self.search_box_size // 2)
            x_end = min(array.shape[1], self.hint_x + self.search_box_size // 2)
            y_start = max(0, self.hint_y - self.search_box_size // 2)
            y_end = min(array.shape[0], self.hint_y + self.search_box_size // 2)
            
            # Extract the search box
            search_area = filtered_array[y_start:y_end, x_start:x_end]
            
            # Find the maximum value within the search box
            local_max = np.max(search_area)
            
            # If the local max is less than half the reference value, do a full scan
            if local_max < 0.5 * self.reference_value:
                return self._full_array_scan(filtered_array)
            
            local_rows, local_cols = np.where(search_area == local_max)
            
            # Translate local coordinates back to global coordinates
            col = local_cols[0] + x_start
            row = local_rows[0] + y_start
        else:
            # If no hint is available, do a full array scan
            col, row, val = self._full_array_scan(filtered_array)
        
        # Update hint and reference value for next call
        self.hint_x, self.hint_y = col, row
        self.reference_value = filtered_array[row, col]
        
        return col, row, filtered_array[row, col]

    def _full_array_scan(self, array):
        rows, cols = np.where(array == np.max(array))
        return cols[0], rows[0], array[rows[0], cols[0]]

    def reset(self):
        self.hint_x = None
        self.hint_y = None
        self.reference_value = None

def find_high_value_element(array, size=3):
    array = array.astype('float32')
    filtered_array = medianBlur(array, size)
    rows, cols = np.where(filtered_array == np.max(filtered_array))
    return cols[0], rows[0], filtered_array[rows[0], cols[0]]


def find_high_value_element(array, size = 3):
  array = array.astype('float32')
  #print(array.dtype)
  filtered_array = medianBlur(array,3)


  #filtered_array = array

  # Find the indices of the maximum value
  row, col = np.where(filtered_array == np.max(filtered_array))


  return col[0], row[0]

  
def compute_centroid(array, x, y):


  # Find the indices and values of the star pixels in the array

  N = 32
  array = array[y - N: y + N, x - N:x + N]

  array = array - (np.min(array) + 1.5 * np.std(array))

  rows, cols = np.where(array > 0.0)
  values = array[rows, cols]
  #print(values.shape)

  # Compute the centroid using a weighted average
  centroid_row = np.sum(rows * values) / np.sum(values)
  centroid_col = np.sum(cols * values) / np.sum(values)
  centroid_value = np.max(values)

  return centroid_col + x - N, centroid_row + y - N, centroid_value


from scipy import ndimage
from skimage.filters import threshold_otsu

def compute_centroid_improved(array, x, y, initial_size=32, final_size=16, iterations=3):
    def iterate_centroid(sub_array, size):
        # Use Otsu's method for adaptive thresholding
        thresh = threshold_otsu(sub_array)
        binary = sub_array > thresh
        
        # Use center of mass for initial estimate
        cy, cx = ndimage.center_of_mass(binary)
        
        # Refine using weighted centroid
        rows, cols = np.where(binary)
        values = sub_array[rows, cols]
        centroid_row = np.sum(rows * values) / np.sum(values)
        centroid_col = np.sum(cols * values) / np.sum(values)
        
        return centroid_col, centroid_row

    current_x, current_y = x, y
    current_size = initial_size

    for _ in range(iterations):
        half_size = current_size // 2
        sub_array = array[int(current_y - half_size):int(current_y + half_size),
                          int(current_x - half_size):int(current_x + half_size)]
        
        dx, dy = iterate_centroid(sub_array, current_size)
        
        current_x += dx - half_size
        current_y += dy - half_size
        
        current_size = max(final_size, current_size // 2)

    # Final refined centroid
    final_half_size = final_size // 2
    final_array = array[int(current_y - final_half_size):int(current_y + final_half_size),
                        int(current_x - final_half_size):int(current_x + final_half_size)]
    
    final_dx, final_dy = iterate_centroid(final_array, final_size)
    
    final_x = current_x + final_dx - final_half_size
    final_y = current_y + final_dy - final_half_size
    
    centroid_value = np.max(final_array)
    
    return final_x, final_y, centroid_value


from scipy.optimize import minimize


def find_optimal_scaling(array1, array2):
    def minimize_std(K):
        # Calculate the difference array
        diff = array1 - array2 * K

        # Calculate the standard deviation of the difference array
        std = np.std(diff)
        #print(std)
        # Return the standard deviation as the optimization target
        return std

    # Initialize the optimization parameters
    x0 = [1.0]  # Initial value for K
    bounds = [(0.0, None)]  # Bounds for K (must be non-negative)

    # Perform the optimization
    result = minimize(minimize_std, x0, bounds=bounds)

    # Extract the optimized value for K
    K_opt = result.x[0]

    return K_opt




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

        self.imageView.getImageItem().mouseClickEvent = self.click

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
            #reply = QtWidgets.QMessageBox.question(self, 'Delete File', f'Are you sure you want to delete {filename}?',
            #                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            #if reply == QtWidgets.QMessageBox.Yes:
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
        data = np.fliplr(fits.getdata(file, ext=0))
        downscaled_data = data[::2, ::2]
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




import zmq

class IPC:
    def __init__(self, type = zmq.REQ):
        self.context = zmq.Context()
        self.socket = self.context.socket(type)

        if (type == zmq.REQ):
            self.socket.connect("tcp://localhost:5555")
        else:
            self.socket.bind("tcp://*:5555")


    def get(self):
        count = self.socket.poll(timeout=30)
        if (count != 0):
            obj = self.socket.recv_pyobj()
            return obj
        else:
            return None

    def send(self, msg):
        self.socket.send_pyobj(msg)

    def close(self):
        # Close the socket
        self.socket.close()

        # Terminate the context
        self.context.term()

    def set_val(self, name, val):
        ob = [name, val]
        self.send(ob)
        res = self.get()

    def get_val(self, name):
        ob = [name, -1]
        self.send(ob)
        res = self.get()
        return res


