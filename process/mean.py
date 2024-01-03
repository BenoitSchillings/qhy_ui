#compute a map of the standard deviation of an array of pixels


from ser import *
import sys
from astropy.io import fits

fid = Ser(sys.argv[1])

mean = fid.load_img(0) * 1.0
N = fid.count
print(N)
print("count ", fid.count)
for idx in range(1, N):
    sum = fid.load_img(idx)
    mean = mean + sum
    print(idx, np.mean(sum), sum)
    if (idx % 100 == 0):
        print(idx)
fid.close()

mean = mean / N
hdr = fits.header.Header()
fits.writeto(sys.argv[2], (mean).astype(np.float32), hdr, overwrite=True)




