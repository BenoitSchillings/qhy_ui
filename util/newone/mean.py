import numpy as np
import time
import sys
from astropy.io import fits



files = args = sys.argv[1:]

def fn(idx):
	return files[idx]


frame = 0
img0 = fits.getdata(fn(0), ext=0) * 0.0

count = len(files)
print(count)
for idx in range(count):
	print(fn(idx))
	img0 = img0 + (fits.getdata(fn(idx), ext=0))

img0 = img0 / (count * 1.0)
#img0 = img0 / np.max(img0)
hdr = fits.header.Header()
fits.writeto("mean.fits", (img0).astype(np.float32), hdr, overwrite=True)
