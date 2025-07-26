from ser import *
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sep

def find_brightest_stars(image, num_stars=4, min_separation=50):
    """
    Find positions of brightest stars using SEP (Python implementation of SExtractor).
    
    Args:
        image: 2D numpy array containing the image data
        num_stars: Number of stars to detect
        min_separation: Minimum separation between stars in pixels
    """
    # SEP requires background-subtracted data in native byte order
    data = image.astype(np.float64).copy(order='C')
    
    # Measure the background
    bkg = sep.Background(data)
    
    # Subtract the background
    data_sub = data - bkg.back()
    
    # Extract sources
    objects = sep.extract(data_sub, 3.0, err=bkg.globalrms, minarea=5)
    
    if len(objects) == 0:
        print("No stars found in image")
        return []
    
    # Create list of stars with their properties
    stars = [(obj['y'], obj['x'], obj['peak']) for obj in objects]
    
    # Sort by peak intensity
    stars.sort(key=lambda x: x[2], reverse=True)
    
    # Filter stars by minimum separation
    filtered_stars = []
    for star in stars:
        if not filtered_stars:  # Always keep brightest star
            filtered_stars.append(star)
            continue
            
        # Check distance to all already-kept stars
        is_far_enough = True
        for kept_star in filtered_stars:
            dist = np.sqrt((star[0] - kept_star[0])**2 + 
                         (star[1] - kept_star[1])**2)
            if dist < min_separation:
                is_far_enough = False
                break
                
        if is_far_enough:
            filtered_stars.append(star)
            
        if len(filtered_stars) >= num_stars:
            break
    
    return filtered_stars

def display_image_with_stars(image, star_positions, fig, ax, im, scale_factor=4):
    """
    Update the display with new image and star positions.
    """
    # Scale down the image
    h, w = image.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    scaled_img = np.zeros((new_h, new_w))
    
    for i in range(new_h):
        for j in range(new_w):
            scaled_img[i, j] = np.mean(image[i*scale_factor:(i+1)*scale_factor, 
                                           j*scale_factor:(j+1)*scale_factor])
    
    # Calculate reasonable vmin/vmax for better contrast
    mean = np.mean(scaled_img)
    std = np.std(scaled_img)
    vmin = mean - 2*std
    vmax = mean + 5*std
    
    # Update image data
    im.set_data(scaled_img)
    im.set_clim(vmin=vmin, vmax=vmax)
    
    # Clear previous circles and text
    for artist in ax.patches + ax.texts:
        artist.remove()
    
    # Add circles for each star
    circle_radius = 10
    #print("\nDisplaying stars at (scaled coordinates):")
    for y, x, intensity in star_positions:
        # Scale star positions
        scaled_x = x / scale_factor
        scaled_y = y / scale_factor
        #print(f"Star at ({scaled_x:.1f}, {scaled_y:.1f}) with intensity {intensity:.1f}")
        
        # Add circle
        circle = Circle((scaled_x, scaled_y), circle_radius, 
                       fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
        
        # Add intensity and FWHM labels
        ax.text(scaled_x + circle_radius + 1, scaled_y, 
                f'I={intensity:.0f}', 
                color='lime', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.7))
    
    # Ensure the display shows the full image
    ax.set_xlim(0, new_w)
    ax.set_ylim(new_h, 0)  # Inverted y-axis for image coordinates
    
    plt.draw()
    plt.pause(0.01)

def measure_stars(image):
    """
    Measure positions of 3 brightest stars, sort by X position, and print in a single line.
    """
    image = image.astype(float)
    star_measurements = find_brightest_stars(image, num_stars=3, min_separation=50)
    
    # Sort by X position (index 1 in the tuples)
    star_measurements.sort(key=lambda x: x[1])
    
    # Print in a single line format: x1,y1 x2,y2 x3,y3
    positions = ' '.join([f"x={x:.1f}  y={y:.1f}," for y, x, _ in star_measurements])
    print(positions)
    
    return star_measurements
# Main script
plt.ion()  # Turn on interactive plotting

# Create figure and axes once
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title('Detected Stars (SEP/SExtractor)')


fid = Ser(sys.argv[2])
N = fid.count



for idx in range(N):
    image = fid.load_img(idx) * 1.0

    if (idx == 0):
        mean = image
    else:
        mean = mean + image

    dark = mean / (N*1.0)


# Create initial image object
first_image = Ser(sys.argv[1]).load_img(0)
h, w = first_image.shape
scaled_h, scaled_w = h//4, w//4
im = ax.imshow(np.zeros((scaled_h, scaled_w)), cmap='gray')

fid = Ser(sys.argv[1])
N = fid.count
print(f"\nProcessing {N} frames from {sys.argv[1]}")
print("=" * 50)

for idx in range(N):
    image = fid.load_img(idx) #- dark
    #if (idx % 100 == 0):
        #print(f"\nProcessing frame {idx}")
    
    # Measure star positions
    star_measurements = measure_stars(image)
    
    # Update display with new image and star positions
    display_image_with_stars(image, star_measurements, fig, ax, im, scale_factor=4)
    
    # Save the processed image with header containing star positions
    fn = sys.argv[1] + str(idx) + ".fits"
    hdr = fits.header.Header()
    
    for i, (y, x, intensity) in enumerate(star_measurements, 1):
        hdr[f'STAR{i}Y'] = (y, f'Y position of star {i}')
        hdr[f'STAR{i}X'] = (x, f'X position of star {i}')
        hdr[f'STAR{i}I'] = (intensity, f'Peak intensity of star {i}')
    
    fits.writeto(fn, image, hdr, overwrite=True)

fid.close()
plt.ioff()