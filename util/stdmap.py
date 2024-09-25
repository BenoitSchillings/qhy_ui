#compute a map of the standard deviation of an array of pixels


from ser import *
import sys
from astropy.io import fits

fid = Ser(sys.argv[-1])

mean = fid.load_img(0) * 0.0
K = 50
N = 19

for slice in range(N):
    sum = fid.load_img(slice*K)
    sum = np.expand_dims(sum, axis=0)   
    print(sum.shape)
    images = []

    for i in range(slice*K, slice*K+K):
        print(i)
        img = fid.load_img(i)

        images.append(img)
        del img

    sum = np.stack(images, axis=0)
    print(sum.shape)
    del(images)
    stdev = np.std(sum, axis=0)
    print(stdev.shape)

    mean = mean + stdev

fid.close()

mean = mean / N
print(np.percentile(mean, 80))
print(np.percentile(mean, 97))
cval = np.percentile(mean, 99)

print("mean noise std is " + str(np.mean(mean)) + " clip =" + str(cval))
mean = np.clip(mean, cval, cval + 1)
mean = mean - cval

print(mean)
hdr = fits.header.Header()
fits.writeto("mask.fits", (mean).astype(np.float32), hdr, overwrite=True)
    