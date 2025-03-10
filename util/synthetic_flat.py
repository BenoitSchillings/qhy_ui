#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astronomy Image Stacking Tool

This script processes multiple FITS images to create an optimized stack with
flat field correction and dark frame subtraction.

Usage:
    python stack.py -o output.fits data*.fits
"""

import os
import sys
import glob
import argparse
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from scipy.spatial import KDTree
from scipy.optimize import minimize


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stack astronomy images with flat field correction.')
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('-o', '--output', required=True, help='Output FITS file')
    parser.add_argument('-d', '--dark', default='dark.fits', help='Dark frame FITS file')
    parser.add_argument('--no-dark', action='store_true', help='Skip dark frame subtraction')
    parser.add_argument('--no-flat', action='store_true', help='Skip flat field correction')
    parser.add_argument('--max-iter', type=int, default=20, help='Maximum iterations for flat field estimation')
    parser.add_argument('--saturation', type=float, default=0.9, help='Saturation threshold (0-1)')
    parser.add_argument('--high-value', type=float, default=98, help='High value percentile threshold')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def load_fits_image(filename):
    """Load a FITS image file."""
    try:
        with fits.open(filename) as hdul:
            # Get the first image HDU
            for hdu in hdul:
                if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                    if hdu.data is not None:
                        header = hdu.header
                        data = hdu.data.astype(np.float32)
                        return data, header
            
            raise ValueError(f"No valid image data found in {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None, None


def find_stars(image, fwhm=3.0, threshold=5.0):
    """
    Find stars in an image using DAOStarFinder.
    
    Parameters:
    image: 2D numpy array
    fwhm: Full width at half maximum of the PSF
    threshold: Detection threshold in sigma
    
    Returns:
    x, y: Arrays of star coordinates
    """
    # Calculate image statistics with sigma clipping to ignore stars
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    # Create a DAOStarFinder object
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    
    # Find stars
    sources = daofind(image - median)
    
    if sources is None or len(sources) == 0:
        return np.array([]), np.array([])
    
    # Return x and y coordinates
    return sources['xcentroid'], sources['ycentroid']


def match_star_patterns(ref_stars, target_stars, max_distance=50.0):
    """
    Match star patterns between two images to find the transformation.
    
    Parameters:
    ref_stars: Tuple of (x, y) arrays for reference stars
    target_stars: Tuple of (x, y) arrays for target stars
    max_distance: Maximum distance for considering a star pair
    
    Returns:
    transformation: Dictionary with dx, dy, angle (in radians)
    success: Boolean indicating if matching was successful
    """
    ref_x, ref_y = ref_stars
    target_x, target_y = target_stars
    
    # Check if we have enough stars
    if len(ref_x) < 3 or len(target_x) < 3:
        return {'dx': 0, 'dy': 0, 'angle': 0}, False
    
    # Combine x and y coordinates
    ref_coords = np.column_stack((ref_x, ref_y))
    target_coords = np.column_stack((target_x, target_y))
    
    # Function to calculate transformation error
    def transformation_error(params):
        dx, dy, angle = params
        
        # Create rotation matrix
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Apply rotation and shift to target coordinates
        transformed = np.dot(target_coords, rotation.T) + np.array([dx, dy])
        
        # Build KD-Tree for efficient nearest neighbor search
        tree = KDTree(ref_coords)
        
        # Find nearest neighbors
        distances, _ = tree.query(transformed, k=1)
        
        # Calculate mean squared distance
        return np.mean(np.square(distances))
    
    # Initial guess (shift only)
    initial_params = [0, 0, 0]
    
    # Optimize transformation parameters
    result = minimize(
        transformation_error,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': 100}
    )
    
    # Check if optimization succeeded
    if not result.success:
        return {'dx': 0, 'dy': 0, 'angle': 0}, False
    
    # Get optimized parameters
    dx, dy, angle = result.x
    
    # Calculate final error
    final_error = transformation_error(result.x)
    
    # Check if error is acceptable
    success = final_error < max_distance
    
    return {'dx': dx, 'dy': dy, 'angle': angle}, success


def apply_transformation(image, transformation):
    """
    Apply transformation (shift and rotation) to an image.
    
    Parameters:
    image: 2D numpy array
    transformation: Dictionary with dx, dy, angle
    
    Returns:
    transformed_image: 2D numpy array
    """
    from scipy.ndimage import rotate, shift
    
    # Extract transformation parameters
    dx = transformation['dx']
    dy = transformation['dy']
    angle_rad = transformation['angle']
    angle_deg = angle_rad * 180 / np.pi
    
    # First rotate the image
    if abs(angle_deg) > 0.01:  # Only rotate if angle is significant
        rotated = rotate(image, angle_deg, reshape=False, order=3)
    else:
        rotated = image
    
    # Then shift the image
    if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only shift if displacement is significant
        transformed = shift(rotated, (dy, dx), order=3)
    else:
        transformed = rotated
    
    return transformed


def fill_holes_in_flat_field(flat_field):
    """
    Fill holes in the flat field by extrapolating from neighboring areas.
    
    Parameters:
    flat_field: 2D numpy array - the flat field with holes
    
    Returns:
    None - modifies flat_field in-place
    """
    # Identify problematic pixels
    mask = ~np.isfinite(flat_field) | (flat_field <= 0)
    
    if not np.any(mask):
        return  # No holes to fill
    
    # Create a copy of the original flat field
    valid_flat = np.copy(flat_field)
    valid_flat[mask] = np.nan
    
    # Get coordinates of all pixels
    y_coords, x_coords = np.indices(flat_field.shape)
    
    # Create a mask of valid pixels
    valid_mask = ~mask
    
    # For each hole pixel, find the nearest valid pixels and use their weighted average
    for i in range(flat_field.shape[0]):
        for j in range(flat_field.shape[1]):
            if mask[i, j]:
                # Calculate Euclidean distance to each valid pixel
                distances = np.sqrt((y_coords - i)**2 + (x_coords - j)**2)
                
                # Only consider valid pixels
                valid_distances = distances[valid_mask]
                valid_values = valid_flat[valid_mask]
                
                # Find nearest neighbors (up to 16)
                if len(valid_distances) > 0:
                    # Sort by distance
                    sorted_indices = np.argsort(valid_distances)
                    
                    # Take up to 16 nearest neighbors
                    k = min(16, len(sorted_indices))
                    nearest_indices = sorted_indices[:k]
                    
                    # Get distances and values for these neighbors
                    nearest_distances = valid_distances[nearest_indices]
                    nearest_values = valid_values[nearest_indices]
                    
                    # Skip if any of the nearest values are NaN
                    if np.any(np.isnan(nearest_values)):
                        continue
                    
                    # Weight by inverse distance
                    weights = 1.0 / (nearest_distances + 1e-10)
                    
                    # Normalize weights
                    weights = weights / np.sum(weights)
                    
                    # Weighted average
                    flat_field[i, j] = np.sum(nearest_values * weights)
    
    # For any remaining holes, try a more aggressive approach
    mask = ~np.isfinite(flat_field) | (flat_field <= 0)
    if np.any(mask):
        # For remaining holes, use a Gaussian-weighted interpolation
        # First, dilate the mask to ensure we're getting values from valid regions
        dilated_mask = binary_dilation(mask, iterations=3)
        boundary_mask = dilated_mask & ~mask
        
        # Use the values at the boundary for a distance-based interpolation
        if np.any(boundary_mask):
            # Get the distance transform
            dist = distance_transform_edt(mask)
            
            # Build a weighted sum based on distance
            weighted_sum = np.zeros_like(flat_field)
            weight_sum = np.zeros_like(flat_field)
            
            for i in range(flat_field.shape[0]):
                for j in range(flat_field.shape[1]):
                    if boundary_mask[i, j]:
                        value = flat_field[i, j]
                        if np.isfinite(value) and value > 0:
                            # Add weight to all hole pixels based on distance
                            for ii in range(flat_field.shape[0]):
                                for jj in range(flat_field.shape[1]):
                                    if mask[ii, jj]:
                                        # Calculate distance
                                        d = np.sqrt((ii-i)**2 + (jj-j)**2)
                                        # Weight is inverse square of distance
                                        if d > 0:
                                            w = 1.0 / (d**2)
                                            weighted_sum[ii, jj] += value * w
                                            weight_sum[ii, jj] += w
            
            # Apply the weighted values
            valid_weights = weight_sum > 0
            if np.any(valid_weights & mask):
                flat_field[valid_weights & mask] = weighted_sum[valid_weights & mask] / weight_sum[valid_weights & mask]
    
    # Final pass - any remaining holes get the median of valid values
    mask = ~np.isfinite(flat_field) | (flat_field <= 0)
    if np.any(mask):
        valid_values = flat_field[~mask]
        if len(valid_values) > 0:
            median_value = np.median(valid_values)
            flat_field[mask] = median_value


def estimate_flat_field(images, transformations, max_iter=20, saturation_threshold=0.9, high_value_percentile=98):
    """
    Estimate flat field from a set of images with known transformations.
    
    Parameters:
    images: List of 2D numpy arrays
    transformations: List of transformation dictionaries
    max_iter: Maximum number of iterations
    saturation_threshold: Threshold for saturation (0-1)
    high_value_percentile: Percentile threshold for high values
    
    Returns:
    flat_field: 2D numpy array
    """
    if not images or len(images) == 0:
        raise ValueError("No images provided for flat field estimation")
    
    h, w = images[0].shape
    n = len(images)
    
    # Find the global saturation level
    max_possible_value = np.max([np.max(img) for img in images])
    saturation_level = saturation_threshold * max_possible_value
    
    # Calculate high value threshold using the specified percentile across all images
    all_values = np.concatenate([img.flatten() for img in images])
    high_value_threshold = np.percentile(all_values, high_value_percentile)
    
    # Initialize flat field
    flat = np.ones((h, w))
    
    # Iteratively refine the flat field
    for iteration in range(max_iter):
        # Create a set of corrected images
        corrected_images = [img / flat for img in images]
        
        # Create a reference image by averaging the registered corrected images
        reference = np.zeros((h, w))
        counts = np.zeros((h, w))
        
        for i, img in enumerate(corrected_images):
            # Apply inverse transformation to register the image
            registered_img = apply_transformation(img, {
                'dx': -transformations[i]['dx'],
                'dy': -transformations[i]['dy'],
                'angle': -transformations[i]['angle']
            })
            
            # Create mask for valid pixels
            valid_mask = np.ones((h, w), dtype=bool)
            
            # Exclude regions affected by the transformation (near edges)
            edge_margin = 10  # pixels to exclude from edges
            valid_mask[:edge_margin, :] = False
            valid_mask[-edge_margin:, :] = False
            valid_mask[:, :edge_margin] = False
            valid_mask[:, -edge_margin:] = False
            
            # Exclude saturated pixels and very high values
            sat_mask = images[i] < saturation_level
            high_val_mask = images[i] < high_value_threshold
            
            # Combine all masks
            combined_mask = valid_mask & sat_mask & high_val_mask
            
            reference[combined_mask] += registered_img[combined_mask]
            counts[combined_mask] += 1
        
        # Avoid division by zero
        valid_pixels = counts > 0
        reference[valid_pixels] /= counts[valid_pixels]
        
        # Update flat field
        new_flat = np.zeros_like(flat)
        new_counts = np.zeros_like(flat)
        
        for i, img in enumerate(images):
            # Apply transformation to reference to match original image orientation
            transformed_ref = apply_transformation(reference, transformations[i])
            
            # Create mask for valid pixels
            valid_mask = np.ones((h, w), dtype=bool)
            
            # Exclude regions affected by the transformation (near edges)
            edge_margin = 10  # pixels to exclude from edges
            valid_mask[:edge_margin, :] = False
            valid_mask[-edge_margin:, :] = False
            valid_mask[:, :edge_margin] = False
            valid_mask[:, -edge_margin:] = False
            
            # Exclude saturated pixels and very high values
            sat_mask = img < saturation_level
            high_val_mask = img < high_value_threshold
            
            # Only consider pixels with significant signal (above noise floor)
            signal_mask = img > np.median(img) * 0.1
            
            # Combine all masks
            combined_mask = valid_mask & sat_mask & high_val_mask & signal_mask
            
            # Skip regions where transformed reference is too close to zero
            ref_valid = transformed_ref > np.median(transformed_ref) * 0.05
            combined_mask = combined_mask & ref_valid
            
            # Update flat field where mask is True
            ratio = np.zeros_like(img)
            ratio[combined_mask] = img[combined_mask] / transformed_ref[combined_mask]
            new_flat += ratio
            new_counts += combined_mask.astype(int)
        
        # Handle regions with sufficient data
        valid_updates = new_counts > 0
        flat[valid_updates] = new_flat[valid_updates] / new_counts[valid_updates]
        
        # Normalize flat field to have median 1
        if np.sum(valid_updates) > 0:  # Only normalize if we have valid pixels
            flat = flat / np.median(flat[valid_updates])
    
    # Fill holes in the flat field
    fill_holes_in_flat_field(flat)
    
    return flat


def stack_images(images, transformations, flat_field=None):
    """
    Stack images after applying transformations and flat field correction.
    
    Parameters:
    images: List of 2D numpy arrays
    transformations: List of transformation dictionaries
    flat_field: 2D numpy array or None (if no flat field correction)
    
    Returns:
    stacked_image: 2D numpy array
    """
    if not images or len(images) == 0:
        raise ValueError("No images provided for stacking")
    
    h, w = images[0].shape
    
    # Initialize stack with zeros
    stack_sum = np.zeros((h, w))
    counts = np.zeros((h, w))
    
    # For each image
    for i, img in enumerate(images):
        # Apply flat field correction if provided
        if flat_field is not None:
            corrected = img / flat_field
        else:
            corrected = img
        
        # Apply inverse transformation to register the image
        registered = apply_transformation(corrected, {
            'dx': -transformations[i]['dx'],
            'dy': -transformations[i]['dy'],
            'angle': -transformations[i]['angle']
        })
        
        # Create mask for valid pixels
        valid_mask = np.ones((h, w), dtype=bool)
        
        # Exclude regions affected by the transformation (near edges)
        edge_margin = 10  # pixels to exclude from edges
        valid_mask[:edge_margin, :] = False
        valid_mask[-edge_margin:, :] = False
        valid_mask[:, :edge_margin] = False
        valid_mask[:, -edge_margin:] = False
        
        # Add to stack
        stack_sum[valid_mask] += registered[valid_mask]
        counts[valid_mask] += 1
    
    # Avoid division by zero
    valid_pixels = counts > 0
    stacked = np.zeros_like(stack_sum)
    stacked[valid_pixels] = stack_sum[valid_pixels] / counts[valid_pixels]
    
    return stacked


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if input files exist
    if not args.files:
        print("Error: No input files specified")
        sys.exit(1)
    
    # Load dark frame if specified
    dark_frame = None
    if not args.no_dark:
        if os.path.exists(args.dark):
            dark_frame, _ = load_fits_image(args.dark)
            if dark_frame is None:
                print(f"Warning: Could not load dark frame {args.dark}, proceeding without dark subtraction")
        else:
            print(f"Warning: Dark frame {args.dark} not found, proceeding without dark subtraction")
    
    # Load all input images
    images = []
    headers = []
    for filename in args.files:
        if args.verbose:
            print(f"Loading {filename}...")
        
        image_data, image_header = load_fits_image(filename)
        if image_data is not None:
            # Subtract dark frame if available
            if dark_frame is not None:
                if dark_frame.shape == image_data.shape:
                    image_data = image_data - dark_frame
                else:
                    print(f"Warning: Dark frame shape {dark_frame.shape} does not match image shape {image_data.shape}")
            
            images.append(image_data)
            headers.append(image_header)
    
    if not images:
        print("Error: No valid images loaded")
        sys.exit(1)
    
    if args.verbose:
        print(f"Loaded {len(images)} images")
    
    # Use the first image as reference
    reference_image = images[0]
    
    # Find stars in the reference image
    if args.verbose:
        print("Finding stars in reference image...")
    
    ref_stars = find_stars(reference_image)
    
    if len(ref_stars[0]) == 0:
        print("Warning: No stars found in reference image")
    elif args.verbose:
        print(f"Found {len(ref_stars[0])} stars in reference image")
    
    # Find transformations for all images
    transformations = []
    valid_images = []
    valid_headers = []
    
    for i, image in enumerate(images):
        if i == 0:
            # Reference image has identity transformation
            transformations.append({'dx': 0, 'dy': 0, 'angle': 0})
            valid_images.append(image)
            valid_headers.append(headers[i])
            continue
        
        if args.verbose:
            print(f"Aligning image {i+1}/{len(images)}...")
        
        # Find stars in the current image
        stars = find_stars(image)
        
        if len(stars[0]) == 0:
            print(f"Warning: No stars found in image {i+1}, skipping")
            continue
        
        # Match star patterns to find transformation
        transformation, success = match_star_patterns(ref_stars, stars)
        
        if not success:
            print(f"Warning: Failed to align image {i+1}, skipping")
            continue
        
        if args.verbose:
            print(f"  Transformation: dx={transformation['dx']:.2f}, dy={transformation['dy']:.2f}, angle={transformation['angle']*180/np.pi:.2f}°")
        
        transformations.append(transformation)
        valid_images.append(image)
        valid_headers.append(headers[i])
    
    if len(valid_images) < 2:
        print("Error: Not enough valid images for stacking")
        sys.exit(1)
    
    if args.verbose:
        print(f"Successfully aligned {len(valid_images)} out of {len(images)} images")
    
    # Estimate flat field if not disabled
    flat_field = None
    if not args.no_flat:
        if args.verbose:
            print("Estimating flat field...")
        
        flat_field = estimate_flat_field(
            valid_images,
            transformations,
            max_iter=args.max_iter,
            saturation_threshold=args.saturation,
            high_value_percentile=args.high_value
        )
    
    # Stack images
    if args.verbose:
        print("Stacking images...")
    
    stacked_image = stack_images(valid_images, transformations, flat_field)
    
    # Save the stacked image
    if args.verbose:
        print(f"Saving output to {args.output}...")
    
    # Use header from the first valid image
    hdu = fits.PrimaryHDU(data=stacked_image, header=valid_headers[0])
    
    # Add some processing history
    hdu.header['HISTORY'] = f'Created by stack.py from {len(valid_images)} images'
    hdu.header['HISTORY'] = f'Dark frame: {"No" if args.no_dark else args.dark}'
    hdu.header['HISTORY'] = f'Flat field correction: {"No" if args.no_flat else "Yes"}'
    
    # Write to file
    hdu.writeto(args.output, overwrite=True)
    
    if args.verbose:
        print("Done!")


if __name__ == "__main__":
    main()
