import numpy as np
from astropy.io import fits
import argparse
import os
from numba import jit

class Ser:
    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self.get_sizes()
        self._fid.close()
        self.mmap = np.memmap(fname, dtype=np.uint8, mode='r')
    
    def get_sizes(self):
        self.xsize = np.int32(self.read_at(26, 1, np.int32)[0])
        self.ysize = np.int32(self.read_at(30, 1, np.int32)[0])
        self.depth = np.int32(self.read_at(34, 1, np.int32)[0]) // 8
        self.count = np.int32(self.read_at(38, 1, np.int32)[0])
        self.image_size = self.xsize * self.ysize
    
    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)
    
    def load_img(self, image_number):
        if image_number >= self.count or image_number < 0:
            return None
        
        start = 178 + image_number * self.image_size * self.depth
        end = start + self.image_size * self.depth
        
        if self.depth == 2:
            img = self.mmap[start:end].view(np.uint16)
        else:
            img = self.mmap[start:end].astype(np.uint16) * 255
        
        return img.reshape((self.ysize, self.xsize))

@jit(nopython=True)
def update_lowest_values_fast(current_lowest, new_frame):
    for i in range(current_lowest.shape[0]):
        if new_frame[i] < current_lowest[i, -1]:
            current_lowest[i, -1] = new_frame[i]
            # Simple insertion sort
            j = current_lowest.shape[1] - 2
            while j >= 0 and current_lowest[i, j] > current_lowest[i, j+1]:
                current_lowest[i, j], current_lowest[i, j+1] = current_lowest[i, j+1], current_lowest[i, j]
                j -= 1
    return current_lowest

def compute_low_value_average(ser_file, n_lowest=500):
    total_pixels = ser_file.ysize * ser_file.xsize
    lowest_values = np.full((total_pixels, n_lowest), np.iinfo(np.uint16).max, dtype=np.uint16)
    
    chunk_size = 500  # Process 100 frames at a time
    for start_frame in range(0, ser_file.count, chunk_size):
        end_frame = min(start_frame + chunk_size, ser_file.count)
        chunk = np.stack([ser_file.load_img(i).ravel() for i in range(start_frame, end_frame)])
        
        for frame in chunk:
            lowest_values = update_lowest_values_fast(lowest_values, frame)
        
        print(f"Processed frames {start_frame} to {end_frame-1}")
    
    # Compute the average of the n_lowest values for each pixel
    result = np.mean(lowest_values, axis=1).reshape((ser_file.ysize, ser_file.xsize)).astype(np.float32)
    
    return result

def ser_to_fits(input_file, output_file):
    ser_file = Ser(input_file)
    print(f"File info: {ser_file.xsize}x{ser_file.ysize}, {ser_file.count} frames, {ser_file.depth} bytes per pixel")
    
    low_value_avg = compute_low_value_average(ser_file)
    
    # Create a new FITS file
    hdu = fits.PrimaryHDU(data=low_value_avg)
    
    # Add some metadata to the FITS header
    hdu.header['NFRAMES'] = ser_file.count
    hdu.header['COMMENT'] = 'Created from SER file using average of 100 lowest values per pixel'
    
    # Write the FITS file
    hdu.writeto(output_file, overwrite=True)

def main():
    parser = argparse.ArgumentParser(description='Convert SER file to FITS file using average of 100 lowest values per pixel.')
    parser.add_argument('-input', required=True, help='Input .ser file')
    parser.add_argument('-output', required=True, help='Output .fits file')
    
    args = parser.parse_args()
    
    ser_to_fits(args.input, args.output)
    print(f"Converted {args.input} to {args.output}")

if __name__ == "__main__":
    main()