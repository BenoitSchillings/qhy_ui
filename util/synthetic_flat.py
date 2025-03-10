def iterative_flat_field_astronomy(images, offsets, max_iter=20, saturation_threshold=0.9, high_value_percentile=98):
    import numpy as np
    from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
    
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
            dx, dy = offsets[i]
            shifted = np.roll(img, (-dx, -dy), axis=(1, 0))
            
            # Create mask for valid pixels after shift
            valid_mask = np.ones((h, w), dtype=bool)
            if dx > 0:
                valid_mask[:, -dx:] = False
            elif dx < 0:
                valid_mask[:, :-dx] = False
            if dy > 0:
                valid_mask[-dy:, :] = False
            elif dy < 0:
                valid_mask[:-dy, :] = False
            
            # Exclude saturated pixels and very high values
            sat_mask = images[i] < saturation_level
            high_val_mask = images[i] < high_value_threshold
            
            # Combine all masks
            combined_mask = valid_mask & sat_mask & high_val_mask
            
            reference[combined_mask] += shifted[combined_mask]
            counts[combined_mask] += 1
        
        # Avoid division by zero
        valid_pixels = counts > 0
        reference[valid_pixels] /= counts[valid_pixels]
        
        # Update flat field
        new_flat = np.zeros_like(flat)
        new_counts = np.zeros_like(flat)
        
        for i, img in enumerate(images):
            dx, dy = offsets[i]
            shifted_ref = np.roll(reference, (dx, dy), axis=(1, 0))
            
            # Create mask for valid pixels after shift
            valid_mask = np.ones((h, w), dtype=bool)
            if dx > 0:
                valid_mask[:, :dx] = False
            elif dx < 0:
                valid_mask[:, dx:] = False
            if dy > 0:
                valid_mask[:dy, :] = False
            elif dy < 0:
                valid_mask[dy:, :] = False
            
            # Exclude saturated pixels and very high values
            sat_mask = img < saturation_level
            high_val_mask = img < high_value_threshold
            
            # Only consider pixels with significant signal (above noise floor)
            signal_mask = img > np.median(img) * 0.1
            
            # Combine all masks
            combined_mask = valid_mask & sat_mask & high_val_mask & signal_mask
            
            # Skip regions where shifted reference is too close to zero
            ref_valid = shifted_ref > np.median(shifted_ref) * 0.05
            combined_mask = combined_mask & ref_valid
            
            # Update flat field where mask is True
            ratio = np.zeros_like(img)
            ratio[combined_mask] = img[combined_mask] / shifted_ref[combined_mask]
            new_flat += ratio
            new_counts += combined_mask.astype(int)
        
        # Handle regions with sufficient data
        valid_updates = new_counts > 0
        flat[valid_updates] = new_flat[valid_updates] / new_counts[valid_updates]
        
        # Normalize flat field to have median 1
        if np.sum(valid_updates) > 0:  # Only normalize if we have valid pixels
            flat = flat / np.median(flat[valid_updates])
    
    # Fill holes in the flat field via interpolation from neighbors
    fill_holes_in_flat_field(flat)
    
    return flat

def fill_holes_in_flat_field(flat_field, max_hole_size=100):
    """
    Fill holes in the flat field by extrapolating from neighboring areas.
    
    Parameters:
    flat_field: 2D numpy array - the flat field with holes
    max_hole_size: int - maximum size of holes to fill (in pixels)
    
    Returns:
    None - modifies flat_field in-place
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt
    
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
                    max_dist = np.max(nearest_distances)
                    if max_dist > 0:
                        # Add small epsilon to avoid division by zero
                        weights = 1.0 / (nearest_distances + 1e-10)
                        
                        # Normalize weights
                        weights = weights / np.sum(weights)
                        
                        # Weighted average
                        flat_field[i, j] = np.sum(nearest_values * weights)
                    else:
                        # If distance is zero (shouldn't happen), use simple average
                        flat_field[i, j] = np.mean(nearest_values)
    
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
