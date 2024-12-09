import numpy as np
from pprint import pprint
from util import *
import astropy.io.fits as fits

class SerWriter(object):

    def __init__(self, fname):
        self._fid = open(fname, 'wb')

        

    def write_header(self):
        self.write_at(26, np.int32(self.xsize))
        self.write_at(30, np.int32(self.ysize))
        self.write_at(34, np.int32(self.depth * 8))
        print("final count is ", self.count)
        self.write_at(38, np.int32(self.count))

    def set_sizes(self, xsize, ysize, depth):

        self.xsize = xsize
        self.ysize = ysize
        self.depth = depth
        self.image_size = (self.xsize * self.ysize)
        self.count = 0
        self.write_header()

        
    def write_at(self, pos, value):
        self._fid.seek(pos)
        value.tofile(self._fid)
        


    def count(self):
        return self.count

    def add_image(self, img):
        if (self.depth == 2):
            self.write_at(178 + self.count * self.image_size * self.depth, img)
        else:
            self.img.write_at(178 + self.count * self.image_size, img)

        self.count = self.count + 1 
            
        

    def close(self):
        self.write_header()
        self._fid.close()



class Ser(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self.get_sizes()

    def get_sizes(self):
        self.xsize = np.int32(self.read_at(26, 1, np.int32)[0])
        self.ysize = np.int32(self.read_at(30, 1, np.int32)[0])
        self.depth = np.int32(self.read_at(34, 1, np.int32)[0]) // 8
        self.count = np.int32(self.read_at(38, 1, np.int32)[0])
        pprint(vars(self))
        self.image_size = self.xsize * self.ysize

    def read_at(self, pos, size, ntype):
        #print(pos)
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)
        
        

    def swap16(x):
        return uint16.from_bytes(x.to_bytes(2, byteorder='little'), byteorder='big', signed=False)

    def count():
        return self.count

    def load_img(self, image_number):
        if (image_number >= self.count or image_number < 0):
            return 0
            
        if (self.depth == 2):
            img = self.read_at((178) + image_number * self.image_size * self.depth, self.image_size, np.uint16)
        else:
            img = self.read_at((178) + image_number * self.image_size, self.image_size, np.uint8)  
            img = img.astype(np.uint16)
            img = img * 255 
            
        out = img.reshape((self.ysize, self.xsize)).astype(np.uint16)
        
        
        #out = out[0:1024,0:1024]
        return out

    def close(self):
        self._fid.close()


import cv2

from scipy.ndimage import gaussian_filter

def find_brightest_pixel(image: np.ndarray) -> tuple[int, int]:
    """Returns (y, x) coordinates of brightest pixel after Gaussian filtering"""
    filtered = gaussian_filter(image.astype(float), sigma=2)

    return np.unravel_index(np.argmax(filtered), filtered.shape)

def find_max_position(array_2d):
    # Convert to numpy array if it isn't already
    array_2d = np.array(array_2d)
    
    # Find the index of maximum value
    index = np.argmax(array_2d)
    
    # Convert flat index to 2D coordinates
    row, col = np.unravel_index(index, array_2d.shape)
    
    return row, col

def fwhm(array):
# Find the index of the maximum value
        filtered = gaussian_filter(array.astype(float), sigma=2)

        y, x = find_max_position(filtered)
        EDGE = 20

        sub = array[int(y)-EDGE:int(y)+EDGE, int(x)-EDGE:int(x)+EDGE].copy()



        fwhm = fit_gauss_circular(extract_centered_subarray(sub, 21))
        print(fwhm)
        return fwhm


if __name__ == "__main__":
    import sys
    
    
    
    fid = Ser(sys.argv[-1])


    sum = fid.load_img(0) * 1.0
    ref_y, ref_x = find_brightest_pixel(sum)
    cnt = 1.0
    print(fid.count)
    for i in range(111, fid.count - 1):
        print(i)
        img = fid.load_img(i)
        fwh = fwhm(img)
        print("max = ", np.max(img), " fwhm = ", fwh)
        if (fwh < 7.0):


            y, x = find_brightest_pixel(img)
            print(x,y)
            if (x > 900):
                dy = ref_y - y
                dx = ref_x - x
                shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
                

                #fid1.add_image(img)
                sum = sum + shifted
                v = sum / cnt
                cnt = cnt + 1
                print(cnt)
                v = v / np.max(v)
                if (cnt % 20 == 0):
                    fits.writeto("stack.fits", 10000.0*v.astype(np.float32), overwrite=True)
            cv2.imshow("image",450.0*(v - np.min(v)))
            cv2.waitKey(1)
        
        


