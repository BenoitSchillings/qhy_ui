import numpy as np
import sys
from astropy.io import fits
from typing import List
import gc

def calculate_chunk_size(total_files: int, image_width: int, available_memory_gb: float = 3.0) -> int:
    """Calculate optimal chunk size based on available memory."""
    # Convert GB to bytes and use 80% of available memory to be safe
    available_memory = available_memory_gb * 0.8 * 1024**3
    
    # Memory needed per row (all files)
    bytes_per_row = total_files * image_width * 4  # 4 bytes per float32
    
    # Calculate number of rows we can process at once
    max_rows = int(available_memory / bytes_per_row)
    
    # Return a reasonable chunk size (minimum 10 rows)
    return max(10, max_rows)

def calculate_median_fits(files: List[str], output_file: str) -> None:
    """Calculate median image using chunk processing."""
    
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
            
            # Calculate median for the entire chunk at once
            output_hdul[0].data[start_row:end_row] = np.median(chunk_buffer, axis=0)
            
            # Clean up chunk buffer
            del chunk_buffer
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
    calculate_median_fits(files, "median.fits")
    print("Done!")

if __name__ == "__main__":
    main()