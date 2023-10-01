import numpy as np
import time
import cv2
import astropy
import sys
from astropy.io import fits
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import median_filter

#--------------------------------------------------------

def crop_center(img,crop):
    y,x = img.shape
    startx = x//2 - crop//2
    starty = y//2 - crop//2    
    return img[starty:starty+crop, startx:startx+crop]

#--------------------------------------------------------

def crop_a(img,crop):
    y,x = img.shape
    startx = x//4 - crop//2
    starty = y//4 - crop//2    
    return img[starty:starty+crop, startx:startx+crop]


def crop(a):
	return(a[20:, 20:])

	
#--------------------------------------------------------

def sharpness(img):
	v1 = np.sum(np.diff(img, axis=0)**2)
	v2 = np.sum(np.diff(img, axis=1)**2)

	return(v1+v2)

def bin(a):
	return(a[0:None:2, 0:None:2] + a[1:None:2, 0:None:2] + a[0:None:2, 1:None:2] + a[1:None:2, 1:None:2])


from astropy.io import fits

from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u


def sigma_reject(arr, sigma_threshold=1.5):
    """
    Perform sigma-rejection and averaging on a 3D numpy array.
    
    Parameters:
    arr (numpy.ndarray): The input 3D numpy array of shape (count, size_x, size_y)
    sigma_threshold (float): The number of standard deviations for rejection criterion.
    
    Returns:
    numpy.ndarray: A 2D numpy array of shape (size_x, size_y)
    """
    
    # Step 1: Calculate the mean and standard deviation for each pixel across the `count` axis
    mean_values = np.mean(arr, axis=0)
    std_values = np.std(arr, axis=0)
    
    # Step 2: Create a mask where we set True for the elements we want to keep
    mask = (arr >= mean_values - sigma_threshold * std_values) & (arr <= mean_values + sigma_threshold * std_values)
    
    # Step 3: Compute the average, but for each pixel only include values where the mask is True
    sum_values = np.sum(arr * mask, axis=0)
    count_values = np.sum(mask, axis=0)
    
    # Avoid division by zero by setting zero-count pixels to 1
    count_values[count_values == 0] = 1
    
    avg_values = sum_values / count_values
    
    return avg_values

from scipy.optimize import minimize



def find_optimal_scaling(array1, dk):
    def minimize_std(K):
        # Calculate the difference array
        diff = array1 - dk * K

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

def best_dark(image, dark):
	k = find_optimal_scaling(image, dark)
	print("k is ", str(k))
	d0 = dark * k 

	delta = np.percentile(dark, 3) - np.percentile(d0, 3)

	d0 = d0 + delta

	image = image - d0

	return image


import astroalign as aa

def match(image1, image2):
	p, (pos_img, pos_img_rot) = aa.find_transform(image2, image1)

	print("Rotation: {:.2f} degrees".format(p.rotation * 180.0 / np.pi))
	print("\nScale factor: {:.2f}".format(p.scale))
	print("\nTranslation: (x, y) = ({:.2f}, {:.2f})".format(*p.translation))
	print("\nTranformation matrix:\n{}".format(p.params))

	for (x1, y1), (x2, y2) in zip(pos_img, pos_img_rot):
	    print("({:.2f}, {:.2f}) is source --> ({:.2f}, {:.2f}) in target"
	          .format(x1, y1, x2, y2))
	
	img_aligned, footprint = aa.apply_transform(p, image2, image1)


	return img_aligned


files = args = sys.argv[1:]

print(files)

def fn(idx):
	return files[idx]


frame = 0
img0 = crop(fits.getdata(fn(0), ext=0))
sum = np.empty_like(img0, dtype=float)

flat = crop(fits.getdata("flat.fits", ext=0).astype(np.float))
bias = crop(fits.getdata("bias.fits", ext=0).astype(np.float))
dark = crop(fits.getdata("dark.fits", ext=0).astype(np.float))

flat = flat - bias
flat = np.clip(flat, 1000, 128000)


print("bias", bias.mean(), bias.min(), bias.max(), bias[0][0])
print("flat", flat.mean(), flat.min(), flat.max(), flat[0][0])
print("dark", dark.mean(), dark.min(), dark.max(), dark[0][0])

flat = flat / np.mean(flat)


img0 = best_dark(img0, dark)


img0 = img0 / flat
ref_level = np.percentile(img0, 3)

array3d = (np.empty_like(img0))
array3d = np.expand_dims(array3d, axis=0)


for frame in range(0, len(files)):
	img = crop(fits.getdata(fn(frame), ext=0).astype(float))
	ref_level1 = np.percentile(img0, 3)
	img = img + (ref_level - ref_level1)
	img = best_dark(img, dark)

	img = img / flat

	try:
		shifted = match(img0, img)
		sum +=  shifted
		#array3d = np.concatenate([array3d, np.expand_dims(shifted, axis=0)])
	except:
		print("ERROR")
#sum1 = sigma_reject(array3d)

hdr = fits.header.Header()
fits.writeto("result" + str(time.time()) + ".fits", sum.astype(np.float32), hdr, overwrite=True)

