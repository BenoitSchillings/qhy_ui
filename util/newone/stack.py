import argparse
import astropy.io.fits as fits
import numpy as np

import astroalign as aa

import numpy as np
from astropy.io import fits

import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, MedianBackground

def estimate_background_noise(data, method='2d', sigma=3.0, maxiters=10, box_size=(50, 50)):
    """
    Estimate the background noise of a star field image.

    Parameters:
    - data (numpy.ndarray): The input image data array.
    - method (str): The method for noise estimation ('simple' for sigma_clipped_stats, '2d' for Background2D).
    - sigma (float): The sigma for sigma clipping.
    - maxiters (int): The maximum number of iterations for sigma clipping.
    - box_size (tuple): The box size for Background2D analysis.

    Returns:
    - float: The estimated background median.
    - float: The estimated background noise (standard deviation).
    """
    if method == 'simple':
        # Use sigma-clipped statistics to estimate background noise
        mean, median, std = sigma_clipped_stats(data, sigma=sigma, maxiters=maxiters)
        return median, std

    elif method == '2d':
        # Use Background2D for a detailed background estimation
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, box_size, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        return bkg.background_median, bkg.background_rms_median

    else:
        raise ValueError("Unsupported method specified. Use 'simple' or '2d'.")

def crop_image(data):
    """
    Crop the image to the center 2/3 of its original size.

    Parameters:
    - data (numpy.ndarray): The input image data array.

    Returns:
    - numpy.ndarray: Cropped image data.
    """
    h, w = data.shape
    new_h, new_w = h * 2 // 3, w * 2 // 3
    start_h, start_w = h // 6, w // 6
    return data[start_h:start_h + new_h, start_w:start_w + new_w]

def bin_image_2x2(data):
    """
    Perform 2x2 binning on an image represented by a numpy array, cropping if necessary.

    Parameters:
    - data (numpy.ndarray): The input image data array.

    Returns:
    - numpy.ndarray: The binned image data.
    """
    # Ensure dimensions are even by cropping if necessary
    if data.shape[0] % 2 != 0:
        data = data[:-1, :]
    if data.shape[1] % 2 != 0:
        data = data[:, :-1]

    # Reshape the array to prepare for binning
    # This reshapes the array so every four elements in a 2x2 block are in one row
    reshaped_data = data.reshape(data.shape[0] // 2, 2, data.shape[1] // 2, 2)
    
    # Sum across the axes for the 2x2 blocks
    # Summing or averaging can be done here. Example uses sum:
    binned_data = reshaped_data.sum(axis=(1, 3))

    return binned_data

def compute_sharpness(data):
    """
    Process the image to find sum of squares of differences between adjacent pixels 
    that are more than twice the background noise.

    Parameters:
    - data (numpy.ndarray): The input image data array.

    Returns:
    - float: The square root of the sum of squared differences.
    """
    # Crop the image
    cropped_data = crop_image(bin_image_2x2(data))

    # Estimate background noise
    _, background_noise = estimate_background_noise(cropped_data, method='simple', sigma=3.0, maxiters=10)

    # Initialize the sum of squares
    sum_of_squares = 0

    # Iterate over the array excluding the last row and column
    # for i in range(cropped_data.shape[0] - 1):
    #     for j in range(cropped_data.shape[1] - 1):
    #         # Differences between adjacent pixels
    #         diff_right = np.abs(cropped_data[i, j] - cropped_data[i, j + 1])
    #         diff_down = np.abs(cropped_data[i, j] - cropped_data[i + 1, j])

    #         # Check if differences are more than twice the background noise
    #         if diff_right > 2 * background_noise:
    #             sum_of_squares += diff_right ** 2
    #         if diff_down > 2 * background_noise:
    #             sum_of_squares += diff_down ** 2

    diff_right = np.abs(cropped_data[:, :-1] - cropped_data[:, 1:])
    diff_down = np.abs(cropped_data[:-1, :] - cropped_data[1:, :])

    # Mask where differences are greater than twice the background noise
    mask_right = diff_right > 2 * background_noise
    mask_down = diff_down > 2 * background_noise

    # Calculate squared differences where the condition is met
    squared_diff_right = np.where(mask_right, diff_right ** 2, 0)
    squared_diff_down = np.where(mask_down, diff_down ** 2, 0)

    # Sum all squared differences
    sum_of_squares = np.sum(squared_diff_right) + np.sum(squared_diff_down)
    # Return the square root of the sum of squares
    return np.sqrt(sum_of_squares)



def align(image_data, reference):

    
    transf, (source_list, target_list) = aa.find_transform(image_data, reference)
    #print(transf)

    registered_image, mask = aa.apply_transform(transf, image_data, reference)
    #print(registered_image)
    #registered_image

    return registered_image


def robust_mean_std(array, n_sigma=6):
    """
    Computes the robust mean and standard deviation of an array along the first axis, rejecting outliers.

    Args:
        array: A NumPy array of shape (N, A, B).
        n_sigma: The number of standard deviations to use for outlier rejection.

    Returns:
        A tuple containing the robust mean and standard deviation arrays of shape (A, B).
    """

    # Compute the mean and standard deviation along the first axis
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    # Create a mask for outliers
    outlier_mask = np.abs(array - mean) > (n_sigma * 112 * std)
    true_count = np.count_nonzero(outlier_mask)
    false_count = np.count_nonzero(~outlier_mask)

    print("true ", true_count, " false ", false_count)
    # Set outlier values to NaN
    array[outlier_mask] = np.nan

    # Compute the robust mean and standard deviation (ignoring NaN values)
    robust_mean = np.nanmean(array, axis=0)
    robust_std = np.nanstd(array, axis=0)

    return robust_mean, robust_std

def stack_images(image_files):

    sum = fits.getdata(image_files[0]) * 0.0
    sum = np.expand_dims(sum, axis=0)
    sum = sum.astype(np.float32)
    ref = fits.getdata(image_files[0]) * 1.0

    count = 0

    ref_level = np.percentile(ref[300:-300,300:-300], 10)

    N = 0
    for image_file in image_files:
        print(f"stacking {image_file}")
        image_data = fits.getdata(image_file) * 1.0
        cur = align(image_data, ref).astype(np.float32)

        r1 = np.percentile(cur[300:-300,300:-300], 10)
        cur = cur + -r1 + ref_level
       
        # Calibrate the image and get the calibrated data
        
        print("sharpness = ", compute_sharpness(cur))
        sum = np.concatenate((sum, np.expand_dims(cur, axis=0)), axis=0)
        print(sum.shape)

        count = count + 1.0

        if (count == 20  or  image_file == image_files[-1]):
            mean, std = robust_mean_std(sum, 2)
            fits.writeto("stack" + str(N) + ".fits", (mean * count).astype(np.float32), overwrite=True)
            N = N + 1
            sum = fits.getdata(image_files[0]) * 0.0
            sum = np.expand_dims(sum, axis=0)
            sum = sum.astype(np.float32)
            ref = fits.getdata(image_files[0]) * 1.0          
            count = 0
        



if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add arguments for the image files, dark frame file, flat field file, and bias file
    parser.add_argument("image_files", nargs="+", help="Image files to calibrate")

    # Parse the command line arguments
    args = parser.parse_args()

    # stack the images
    stack_images(args.image_files)

  
