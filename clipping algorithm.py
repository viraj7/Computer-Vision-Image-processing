from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *

a = 50
b = 150
beta = 2

im = cv2.imread("/Users/virajj/Downloads/starfish.jpg", 0)

res = numpy.empty(im.shape, dtype=uint8)

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if im[i][j] >= 0 and im[i][j] < a:
            res[i][j] = 0
        elif im[i][j] >= a and im[i][j] < b:
            res[i][j] = beta*(im[i][j] - a)
        elif im[i][j] >= b and im[i][j] < 255:
            res[i][j] = beta*(b - a)


cv2.imshow("Original Image", im)
cv2.waitKey(0)
cv2.imshow("Resulting Image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()            
