from astropy.io import fits
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources
from astropy.stats import sigma_clipped_stats
from photutils.centroids import centroid_2dg
from scipy.optimize import curve_fit
import sys
from pathlib import Path

def gaussian_2d(xy, amplitude, xc, yc, sigma_x, sigma_y, offset):
    x, y = xy
    exp_term = -((x-xc)**2/(2*sigma_x**2) + (y-yc)**2/(2*sigma_y**2))
    return (offset + amplitude * np.exp(exp_term)).ravel()

def measure_star_fwhm(data, x, y, box_size=21):
    try:
        x, y = int(x), int(y)
        half_box = box_size // 2
        cutout = data[y-half_box:y+half_box+1, x-half_box:x+half_box+1]
        
        y_coords, x_coords = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
        coords = (x_coords, y_coords)
        
        max_val = np.max(cutout)
        min_val = np.min(cutout)
        p0 = [max_val - min_val, half_box, half_box, 2.0, 2.0, min_val]
        
        popt, _ = curve_fit(gaussian_2d, coords, cutout.ravel(), p0=p0)
        fwhm_x = 2.355 * abs(popt[3])
        fwhm_y = 2.355 * abs(popt[4])
        return np.mean([fwhm_x, fwhm_y])
    except Exception:
        return None

def analyze_image_fwhm(fits_path, min_peak=20000, max_peak=40000):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    
    mean, median, std = sigma_clipped_stats(data)
    daofind = DAOStarFinder(fwhm=6.0, threshold=5.*std)
    sources = daofind(data - median)
    
    if sources is None:
        return None
    
    peak_values = sources['peak']
    good_sources = sources[(peak_values > min_peak) & (peak_values < max_peak)]
    
    fwhms = []
    for star in good_sources:
        fwhm = measure_star_fwhm(data, star['xcentroid'], star['ycentroid'])
        if fwhm is not None:
            fwhms.append(fwhm)
    
    if not fwhms:
        return None
        
    fwhms = np.array(fwhms)
    stats = {
        'mean_fwhm': np.mean(fwhms),
        'median_fwhm': np.median(fwhms),
        'std_fwhm': np.std(fwhms),
        'n_stars': len(fwhms)
    }
    
    return stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py path/to/fits/*.fits")
        sys.exit(1)
    
    fits_files = sys.argv[1:]
    
    print(f"{'Filename':40} {'Mean FWHM':>10} {'Median FWHM':>12} {'Std FWHM':>10} {'N Stars':>8}")
    print("-" * 80)
    
    for fits_path in fits_files:
        path = Path(fits_path)
        stats = analyze_image_fwhm(fits_path)
        if stats:
            print(f"{path.name:40} {stats['mean_fwhm']:10.2f} {stats['median_fwhm']:12.2f} "
                  f"{stats['std_fwhm']:10.2f} {stats['n_stars']:8d}")
            
            # Remove any existing .good.fits or .bad.fits extension
            base_name = path.stem.replace('.good', '').replace('.bad', '')
            new_ext = '.good.fits' if stats['median_fwhm'] <= 8 else '.bad.fits'
            new_path = path.with_name(base_name + new_ext)
            #path.rename(new_path)
            
            quality = "good" if stats['median_fwhm'] <= 8 else "bad"
            print(f"Renamed to {new_path.name} ({quality} FWHM: {stats['median_fwhm']:.2f})")
        else:
            print(f"{path.name:40} No valid measurements")

if __name__ == "__main__":
    main()