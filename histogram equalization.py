from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *
from collections import Counter, OrderedDict

im = cv2.imread("/Users/virajj/Downloads/starfish.jpg", 0)
res = numpy.empty(im.shape, dtype=uint8)
hist,bins = numpy.histogram(im.flatten(),256,[0,256]) #hist gives number of pixels for every grayscale
cdf = hist.cumsum() # get cumulative sum of hist array
im_size = im.shape[0]*im.shape[1]

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        res[i][j] = (max(cdf[im[i][j]], 1) - 1)*255/im_size # histogram equalization

cv2.imshow("Original Image", im)
cv2.waitKey(0)
cv2.imshow("Resulting Image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
pyplot.hist(im.ravel(), 256, [0, 256])
pyplot.show()
pyplot.hist(res.ravel(), 256, [0, 256])
pyplot.show()
