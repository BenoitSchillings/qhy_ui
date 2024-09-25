import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage

def load_fits(filename):
    with fits.open(filename) as hdul:
        return hdul[0].data

def save_psf(psf, filename='psf.fits'):
    hdu = fits.PrimaryHDU(psf)
    hdu.writeto(filename, overwrite=True)
    print(f"PSF saved as {filename}")

def normalize_psf(psf):
    # Sort the pixel values
    sorted_pixels = np.sort(psf.flatten())
    # Find the threshold value (90th percentile)
    threshold = sorted_pixels[int(0.9 * len(sorted_pixels))]
    # Create a normalized PSF
    normalized_psf = np.where(psf > threshold, psf, 0)
    return normalized_psf

def calculate_centroid(data):
    total = np.sum(data)
    if total == 0:
        return data.shape[0] / 2, data.shape[1] / 2  # Return center if sum is zero
    Y, X = np.indices(data.shape)
    cx = np.sum(X * data) / total
    cy = np.sum(Y * data) / total
    return cy, cx

def extract_psf(data, x, y, size=32):
    half = size // 2
    
    # Extract a larger area for normalization and centroid calculation
    extract_size = size * 2
    y_start = max(0, int(y - extract_size//2))
    y_end = min(data.shape[0], int(y + extract_size//2))
    x_start = max(0, int(x - extract_size//2))
    x_end = min(data.shape[1], int(x + extract_size//2))
    
    larger_psf = data[y_start:y_end, x_start:x_end]
    
    # Normalize the larger PSF
    normalized_psf = normalize_psf(larger_psf)
    
    # Calculate centroid of the normalized PSF
    cy, cx = calculate_centroid(normalized_psf)
    
    # Adjust extraction based on centroid
    cy += y_start
    cx += x_start
    
    # Extract centered PSF
    y_start = max(0, int(cy - half))
    y_end = min(data.shape[0], int(cy + half))
    x_start = max(0, int(cx - half))
    x_end = min(data.shape[1], int(cx + half))
    
    psf = data[y_start:y_end, x_start:x_end]
    
    # Pad the PSF if it's smaller than the desired size
    if psf.shape[0] < size or psf.shape[1] < size:
        psf_padded = np.zeros((size, size))
        y_offset = half - (cy - y_start)
        x_offset = half - (cx - x_start)
        psf_padded[int(y_offset):int(y_offset)+psf.shape[0], int(x_offset):int(x_offset)+psf.shape[1]] = psf
        psf = psf_padded
    
    return psf, (cx, cy)

def on_click(event):
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        psf, (cx, cy) = extract_psf(data, x, y)
        save_psf(psf)
        
        # Update the plot to show the selected region
        if hasattr(on_click, 'rect'):
            on_click.rect.remove()
        on_click.rect = Rectangle((cx-16, cy-16), 32, 32, fill=False, color='r')
        ax.add_patch(on_click.rect)
        fig.canvas.draw()

        # Display the extracted PSF
        fig_psf, ax_psf = plt.subplots()
        im_psf = ax_psf.imshow(psf, cmap='viridis', origin='lower')
        ax_psf.set_title('Extracted PSF')
        plt.colorbar(im_psf)
        plt.show()

# Load the FITS file
filename = input("Enter the path to your FITS file: ")
data = load_fits(filename)

# Display the image
fig, ax = plt.subplots()
vmin, vmax = np.percentile(data, [1, 99])  # Scale to 1st and 99th percentiles
im = ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar(im)

# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)

plt.title("Click to extract PSF")
plt.show()