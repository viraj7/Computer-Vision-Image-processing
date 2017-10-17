from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *

im = cv2.imread("/Users/virajj/Downloads/starfish.jpg", 0)
res1 = numpy.empty(im.shape, dtype=uint8)
res2 = numpy.empty(im.shape, dtype=uint8)
res3 = numpy.empty(im.shape, dtype=uint8)
res4 = numpy.empty(im.shape, dtype=uint8)
c = [1, 10, 100, 1000]
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        res1[i][j] = c[0]*math.log10(1+im[i][j])
        res2[i][j] = c[1]*math.log10(1+im[i][j])
        res3[i][j] = c[2]*math.log10(1+im[i][j])
        res4[i][j] = c[3]*math.log10(1+im[i][j])

cv2.imshow("Original Image", im)
cv2.waitKey(0)
cv2.imshow("Resulting Image, c = 1", res1)
cv2.waitKey(0)
cv2.imshow("Resulting Image, c = 10", res2)
cv2.waitKey(0)
cv2.imshow("Resulting Image, c = 100", res3)
cv2.waitKey(0)
cv2.imshow("Resulting Image, c = 1000", res4)
cv2.waitKey(0)
cv2.destroyAllWindows()
