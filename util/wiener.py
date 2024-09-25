import numpy as np
from astropy.io import fits
import argparse

def load_fits(filename):
    with fits.open(filename) as hdul:
        return hdul[0].data

def save_fits(data, filename):
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename, overwrite=True)
    print(f"Deconvolved image saved as {filename}")


from numpy.fft import fft2, ifft2
from skimage import color, data, restoration


def wiener_deconvolution(img, kernel, K):
    return(restoration.wiener(img, kernel, K, clip=False))

def normalize_psf(psf):
    psf = psf.astype(np.float64)
    psf_min = np.min(psf)
    psf_max = np.max(psf)
    print(psf_min, psf_max)

    normalized_psf = (psf - psf_min) 
    psf_min = np.min(normalized_psf)
    psf_max = np.max(normalized_psf)
    print("after norm", psf_min, psf_max)
    normalized_psf = normalized_psf / np.max(normalized_psf)
    print(normalized_psf)
    return normalized_psf


def main():
    parser = argparse.ArgumentParser(description='Perform Wiener deconvolution on a FITS image.')
    parser.add_argument('image', help='Input image filename (FITS)')
    parser.add_argument('psf', help='PSF filename (FITS)')
    parser.add_argument('output', help='Output filename for deconvolved image (FITS)')
    parser.add_argument('coefficient', type=float, help='Coefficient for Wiener deconvolution')
    
    args = parser.parse_args()
    
    # Load the image and PSF
    image = load_fits(args.image)
    psf = load_fits(args.psf)
    psf = normalize_psf(psf)
    print(f"Image shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}")
    print(f"PSF shape: {psf.shape}, min: {np.min(psf)}, max: {np.max(psf)}")
    
    print("Original PSF values:")
    np.set_printoptions(threshold=np.inf, precision=6)
    #print(psf)
    np.set_printoptions()  # Reset print options
    
    # Ensure PSF is smaller than or equal to the image
    if psf.shape[0] > image.shape[0] or psf.shape[1] > image.shape[1]:
        raise ValueError("PSF dimensions must be smaller than or equal to image dimensions")
    
    # Pad PSF to match image size if necessary
    if psf.shape != image.shape:
        psf_padded = np.zeros_like(image)
        y_offset = (image.shape[0] - psf.shape[0]) // 2
        x_offset = (image.shape[1] - psf.shape[1]) // 2
        psf_padded[y_offset:y_offset+psf.shape[0], x_offset:x_offset+psf.shape[1]] = psf
        psf = psf_padded
    
    # Perform Wiener deconvolution
    deconvolved = wiener_deconvolution(image, psf, args.coefficient)
    
    print(f"Deconvolved image min: {np.min(deconvolved)}, max: {np.max(deconvolved)}")
    
    # Save the deconvolved image
    save_fits(deconvolved, args.output)

if __name__ == "__main__":
    main()