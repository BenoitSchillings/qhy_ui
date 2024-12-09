import numpy as np
import sys
from astropy.io import fits
from typing import List
import gc

def calculate_chunk_size(total_files: int, image_width: int, available_memory_gb: float = 3.0) -> int:
    """Calculate optimal chunk size based on available memory."""
    available_memory = available_memory_gb * 0.8 * 1024**3
    bytes_per_row = total_files * image_width * 4  # 4 bytes per float32
    max_rows = int(available_memory / bytes_per_row)
    return max(10, max_rows)

def calculate_bottom_10_average(files: List[str], output_file: str) -> None:
    """Calculate average of 10 lowest pixel values at each position."""
    
    # Get image dimensions
    with fits.open(files[0]) as hdul:
        shape = hdul[0].data.shape
    
    # Calculate optimal chunk size
    chunk_size = calculate_chunk_size(len(files), shape[1])
    print(f"Processing in chunks of {chunk_size} rows")
    
    # Create output file
    empty_data = np.zeros(shape, dtype=np.float32)
    hdr = fits.Header()
    fits.writeto(output_file, empty_data, hdr, overwrite=True)
    del empty_data
    gc.collect()
    
    with fits.open(output_file, mode='update') as output_hdul:
        # Process chunks
        for start_row in range(0, shape[0], chunk_size):
            end_row = min(start_row + chunk_size, shape[0])
            current_chunk_size = end_row - start_row
            
            print(f"Processing rows {start_row}-{end_row} ({end_row/shape[0]*100:.1f}% complete)")
            
            # Allocate chunk buffer
            chunk_buffer = np.empty((len(files), current_chunk_size, shape[1]), dtype=np.float32)
            
            # Read chunk from all files
            for i, file in enumerate(files):
                with fits.open(file) as hdul:
                    chunk_buffer[i] = hdul[0].data[start_row:end_row]
            
            # For each pixel position, sort values and take average of 10 lowest
            # Reshape to 2D array where each column contains all values for one pixel
            reshaped = chunk_buffer.reshape(len(files), -1)
            # Sort along axis 0 (across files)
            sorted_values = np.sort(reshaped, axis=0)
            # Take average of 10 lowest values
            bottom_10_avg = np.mean(sorted_values[:10], axis=0)
            # Reshape back to original chunk shape
            bottom_10_avg = bottom_10_avg.reshape(current_chunk_size, shape[1])
            
            # Store results
            output_hdul[0].data[start_row:end_row] = bottom_10_avg
            
            # Clean up chunk buffer
            del chunk_buffer, reshaped, sorted_values, bottom_10_avg
            gc.collect()
            
            # Flush periodically
            if start_row % (chunk_size * 10) == 0:
                output_hdul.flush()

def main():
    files = sys.argv[1:]
    if not files:
        print("Please provide FITS files as arguments")
        return
    
    print(f"Processing {len(files)} files...")
    calculate_bottom_10_average(files, "bottom_10_avg.fits")
    print("Done!")

if __name__ == "__main__":
    main()