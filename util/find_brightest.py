import numpy as np
from astropy.io import fits
import sys

def find_brightest_pixels(filename="mean.fits", num_pixels=10):
    """
    Opens a FITS file, finds the coordinates of the N brightest pixels,
    and prints them to the console.

    Args:
        filename (str): The name of the FITS file to open.
        num_pixels (int): The number of brightest pixels to find.
    """
    try:
        with fits.open(filename) as hdul:
            # Assume the primary HDU contains the image data
            data = hdul[0].data

            if data is None:
                print(f"Error: No data found in the primary HDU of {filename}", file=sys.stderr)
                return

            # Flatten the 2D array to a 1D array to easily find the N largest values
            flat_data = data.flatten()

            # Get the indices of the N largest values in the flattened array.
            # argsort() sorts in ascending order, so we take the last N indices.
            brightest_indices_flat = np.argsort(flat_data)[-num_pixels:]

            # Convert the flat indices back to 2D coordinates
            brightest_coords_2d = np.unravel_index(brightest_indices_flat, data.shape)

            print(f"Coordinates of the {num_pixels} brightest pixels in {filename} (x, y):")
            
            # Reverse the sorted list to print from brightest to least bright
            # and pair the y (row) and x (col) coordinates together.
            coords = zip(brightest_coords_2d[1][::-1], brightest_coords_2d[0][::-1])

            for i, (x, y) in enumerate(coords):
                value = data[y, x] # Get the actual pixel value for display
                print(f"{i+1:2d}: ({x:4d}, {y:4d}) - Value: {value}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    find_brightest_pixels()
