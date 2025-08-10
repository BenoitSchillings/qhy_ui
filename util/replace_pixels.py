import numpy as np
from astropy.io import fits
import sys

def replace_pixels_with_mean(data_array, coordinates):
    """
    Replaces specified pixels in an array with the mean of their 8 neighbors.

    This function is useful for removing cosmic rays or hot pixels. It operates
    on a copy of the array to ensure that the calculation for each pixel is
    based on the original, unmodified neighbors.

    Args:
        data_array (np.ndarray): The 2D NumPy array representing the image.
        coordinates (list): A list of (x, y) tuples for the pixels to be replaced.

    Returns:
        np.ndarray: A new array with the specified pixels corrected.
    """
    # Work on a copy to avoid modifying the original array in place
    corrected_array = np.copy(data_array)
    height, width = data_array.shape

    for x, y in coordinates:
        # Ensure the coordinate is within the image bounds
        if not (0 <= y < height and 0 <= x < width):
            print(f"Warning: Coordinate ({x}, {y}) is outside the image bounds. Skipping.", file=sys.stderr)
            continue

        neighbor_values = []
        # Iterate through the 3x3 neighborhood
        for j in range(max(0, y - 1), min(height, y + 2)):
            for i in range(max(0, x - 1), min(width, x + 2)):
                # Skip the center pixel itself
                if i == x and j == y:
                    continue
                neighbor_values.append(data_array[j, i])

        # Calculate the mean of the neighbors and replace the pixel in the copied array
        if neighbor_values:
            mean_value = np.mean(neighbor_values)
            corrected_array[y, x] = mean_value
            # print(f"Pixel ({x}, {y}) with value {data_array[y, x]:.2f} replaced by mean {mean_value:.2f}")

    return corrected_array

if __name__ == "__main__":
    # The list of (x, y) coordinates you provided
    hot_pixel_coords = [
        (1992, 195),
        (2846, 2808),
        (1724, 1429),
        (1139, 347),
        (3007, 1885),
        (4430, 3357),
        (1699, 1491),
        (4183, 819),
        (6135, 265),
        (102, 3337)
    ]

    # --- Example Usage ---
    # This example demonstrates how you would use the function.
    # It first creates a dummy FITS file, then corrects it.
    
    print("--- Creating a dummy FITS file for demonstration ---")
    # Create a 400x400 array of random noise
    dummy_data = np.random.rand(400, 400) * 100
    # Add a few obvious "hot pixels" to be corrected
    dummy_coords = [(50, 50), (150, 200), (300, 100)]
    for x, y in dummy_coords:
        dummy_data[y, x] = 1000.0 # Make them stand out
    
    hdu = fits.PrimaryHDU(dummy_data)
    hdul = fits.HDUList([hdu])
    dummy_filename = "dummy_image.fits"
    hdul.writeto(dummy_filename, overwrite=True)
    print(f"Saved '{dummy_filename}' with hot pixels.")

    print("\n--- Running Correction ---")
    try:
        with fits.open(dummy_filename) as hdul:
            original_data = hdul[0].data
            
            print("Original values at coordinates:")
            for x, y in dummy_coords:
                print(f"  ({x}, {y}): {original_data[y, x]:.2f}")

            # Run the correction function
            corrected_data = replace_pixels_with_mean(original_data, dummy_coords)
            
            print("\nCorrected values at coordinates:")
            for x, y in dummy_coords:
                print(f"  ({x}, {y}): {corrected_data[y, x]:.2f}")

            # Save the corrected image to a new file
            corrected_filename = "corrected_image.fits"
            fits.writeto(corrected_filename, corrected_data, overwrite=True)
            print(f"\nSuccessfully saved the corrected image to '{corrected_filename}'")

    except Exception as e:
        print(f"An error occurred during the example: {e}", file=sys.stderr)

    # To use this with your actual 'mean.fits' file and your list of coordinates,
    # you would do something like this:
    #
    # with fits.open('mean.fits') as hdul:
    #     image_data = hdul[0].data
    #     corrected_image = replace_pixels_with_mean(image_data, hot_pixel_coords)
    #     fits.writeto('mean_corrected.fits', corrected_image, overwrite=True)
