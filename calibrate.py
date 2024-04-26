import argparse
import astropy.io.fits as fits
import numpy as np


from scipy.optimize import minimize


def find_optimal_scaling(array1, array2):
    def minimize_std(K):
        # Calculate the difference array
        diff = array1 - array2 * K

        # Calculate the standard deviation of the difference array
        std = np.std(diff)
        #print(K, std)
        # Return the standard deviation as the optimization target
        return std

    # Initialize the optimization parameters
    x0 = [1.0]  # Initial value for K
    bounds = [(0, None)]  # Bounds for K (must be non-negative)

    # Perform the optimization
    result = minimize(minimize_std, x0, bounds=bounds, tol = 0.0000001)

    # Extract the optimized value for K
    K_opt = result.x[0]

    return K_opt



def find_optimal_dark_coefficient(image_data, dark_data):
    return find_optimal_scaling(image_data, dark_data)

def crop(im):
    print(im.shape)
    im = (im[30:, :])
    print(im.shape)
    return im

def calibrate_image(image_file, dark_file, flat_file, bias_file):
    """
    Calibrates a single FITS image using an optimized dark frame coefficient.

    Args:
        image_file: The path to the FITS image file.
        dark_file: The path to the FITS dark frame file.
        flat_file: The path to the FITS flat field file.
        bias_file: The path to the FITS bias frame file.

    Returns:
        The calibrated image data.
    """

    # Load the dark, flat field, and bias frames
    dark_data = crop(fits.getdata(dark_file)) * 1.0
    flat_data = crop(fits.getdata(flat_file)) * 1.0
    bias_data = crop(fits.getdata(bias_file)) * 1.0

    flat_data = flat_data - bias_data
    flat_data = np.clip(flat_data, 100, 1e9)
    print(np.min(flat_data))
    print(flat_data)
    flat_data = flat_data / np.mean(flat_data)
    

    dark_data = dark_data - bias_data



    # Load the image data
    image_data = crop(fits.getdata(image_file)) * 1.0
    print("v1", np.mean(image_data))
    image_data = image_data - bias_data
    print("v2", np.mean(image_data))

    # Find the optimal dark coefficient
    optimal_coefficient = find_optimal_dark_coefficient(image_data, dark_data)
    print("optimal = ", optimal_coefficient)

    # Calibrate the image using the optimal coefficient
    
    calibrated_data = image_data - (optimal_coefficient * dark_data)
    print("v3", np.mean(calibrated_data))
    calibrated_data /= flat_data
    print("v4", np.mean(calibrated_data))
    print(calibrated_data)

    return calibrated_data


def calibrate_images(image_files, dark_file, flat_file, bias_file):
    """
    Calibrates a list of FITS images using optimized dark frame coefficients.

    Args:
        image_files: A list of FITS image file paths.
        dark_file: The path to the FITS dark frame file.
        flat_file: The path to the FITS flat field file.
        bias_file: The path to the FITS bias frame file.

    Returns:
        A list of calibrated image files.
    """

    # Create a list to store the calibrated image files
    calibrated_files = []

    # Iterate over each image
    for image_file in image_files:
        print(f"Calibrating {image_file}")

        # Calibrate the image and get the calibrated data
        calibrated_data = calibrate_image(image_file, dark_file, flat_file, bias_file)

        # Save the calibrated mage to a new file
        output_file = image_file.replace(".fits", "_calibrated.fits")
        fits.writeto(output_file, calibrated_data.astype(np.float32), overwrite=True)

        # Append the calibrated file to the list
        calibrated_files.append(output_file)

    return calibrated_files

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments for the image files, dark frame file, flat field file, and bias file
    parser.add_argument("image_files", nargs="+", help="Image files to calibrate")
    parser.add_argument("-d", "--dark", required=True, help="Dark frame file")
    parser.add_argument("-f", "--flat", required=True, help="Flat field file")
    parser.add_argument("-b", "--bias", required=True, help="Bias frame file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Calibrate the images
    calibrated_files = calibrate_images(args.image_files, args.dark, args.flat, args.bias)

    # Print the list of calibrated image files
    print(calibrated_files)
