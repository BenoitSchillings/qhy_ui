import numpy as np
import time
import cv2
import astropy
import sys
from astropy.io import fits
import image_registration
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


#--------------------------------------------------------

def sharpness(img):
	v1 = np.sum(np.diff(img, axis=0)**2)
	v2 = np.sum(np.diff(img, axis=1)**2)

	return(v1+v2)

def bin(a):
	return(a[0:None:2, 0:None:2] + a[1:None:2, 0:None:2] + a[0:None:2, 1:None:2] + a[1:None:2, 1:None:2])


from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u

def find_stars(image_data):
    bkg_sigma = mad_std(image_data)    # estimate background noise
    daofind = DAOStarFinder(fwhm=7.0, threshold=8.*bkg_sigma)  
    sources = daofind(image_data - np.median(image_data))
    print(sources)
    return sources

from sklearn.linear_model import RANSACRegressor

def get_shift(star_list1, star_list2):
    # We're going to estimate the translation from image 2 to image 1.
    # So the "independent variable" is the star positions in image 2
    # and the "dependent variable" is the star positions in image 1.
    X = np.array([star_list2['xcentroid'], star_list2['ycentroid']]).T
    y = np.array([star_list1['xcentroid'], star_list1['ycentroid']]).T

    # Estimate the translation using RANSAC
    ransac = RANSACRegressor(residual_threshold=5.0)  # 5 pixel tolerance
    ransac.fit(X, y)
    translation = ransac.estimator_.intercept_

    return translation

#--------------------------------------------------------
import random

files = args = sys.argv[1:]
#random.shuffle(files)
print(files)

def fn(idx):
	return files[idx]


frame = 0
img0 = fits.getdata(fn(0), ext=0)
sum = np.empty_like(img0, dtype=float)

flat = fits.getdata("flat.fits", ext=0).astype(np.float)
bias = fits.getdata("bias.fits", ext=0).astype(np.float)
dark = fits.getdata("dark.fits", ext=0).astype(np.float)

flat = flat - bias
flat = np.clip(flat, 1000, 128000)
#flat = np.abs(flat) + 0.1

print("bias", bias.mean(), bias.min(), bias.max(), bias[0][0])
print("flat", flat.mean(), flat.min(), flat.max(), flat[0][0])
print("dark", dark.mean(), dark.min(), dark.max(), dark[0][0])
#dark = dark - 90
flat = flat / np.mean(flat)
#dark = dark - bias

img0 = img0 - (1.0*dark)
img0 = img0 / flat
ref_level = np.percentile(img0, 20)
list_1 = find_stars(img0)

img_prop = [('idx', int), ('dx', float), ('dy', float), ('delta', float), ('name', str)]
images_prop = np.array([], dtype=img_prop) 


for frame in range(0, len(files)):
	img = fits.getdata(fn(frame), ext=0).astype(np.float)
	
	img = img -  (1.0*dark)
	#print("top ", img[0][0])
	img = img / flat
	ref_level1 = np.percentile(img, 20)
	list_2 = find_stars(img)
	print(get_shift(list_1, list_2))


	print("mean ", img.mean())
	yoff,xoff = image_registration.cross_correlation_shifts(crop_center(img, 3048), crop_center(img0, 3048))
	#yoff1,xoff1 = image_registration.cross_correlation_shifts(crop_a(img, 2048), crop_a(img0, 2048))
	print(yoff,xoff)
	
	if (np.abs(yoff) < 2100.0 and np.abs(xoff) < 2100.0):
		shifted = np.roll(np.roll(img,int(round(yoff)),1),int(round(xoff)),0)
		#shifted = image_registration.fft_tools.shift.shift2d(img, yoff, xoff)
		#delta = sharpness(crop_center(shifted, 1024))
		delta = -np.mean(shifted[1000:-1000,1400:-1400] - img0[1000:-1000,1400:-1400])
		print(delta)
		
		element = [(frame, xoff, yoff, -delta, fn(frame))]
		images_prop = np.append(images_prop, np.array(element, dtype=img_prop))

		#frame = frame + 1

		sum +=  shifted


hdr = fits.header.Header()
fits.writeto("result" + str(time.time()) + ".fits", sum.astype(np.float32), hdr, overwrite=True)

#stack 90 % of frames
sum = np.empty_like(img0, dtype=float)
images_prop = np.sort(images_prop, order='delta')
print(images_prop)
