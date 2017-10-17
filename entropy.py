from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *

def im_threshold(im):
    res = numpy.empty(im.shape, dtype=uint8)
    hist, bins = numpy.histogram(im.flatten(),256,[0,256])
    H = 0   # H -> entropy
    T = 0   # T -> threshold
    N = im.shape[0]*im.shape[1]   # N -> Total number of pixels
    P = hist/N  # Array of Probability of each grayscale
    #print "prob", len(hist)
    pc = P.cumsum()
    print N
    for t in range(0, 255):
        a = P[:t]/sum(P[:t])    # P(i)/P(i1)+P(i2)+...+P(iT-1) where 0 < i < T
        b = P[t:]/sum(P[t:])    # P(i)/P(iT)+..+P(i255) where T < i < 255
        ha = sum(a)*numpy.log(sum(a))  # H(A)
        hb = sum(b)*numpy.log(sum(b))  # H(B)
        #ha = sum(map(lambda x: x*numpy.log(x), a))
        #hb = sum(map(lambda x: x*numpy.log(x), b))
        ht = -ha-hb  # H(T) = H(A) + H(B)
        if ht > H:  # choosing T such that H(T) is maximum
            H = ht
            T = t

    #print T
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] < T: # if intensity below T set to 0
                res[i][j] = 0
            elif im[i][j] >= T:   # if intensity greater than or equal to T, set to 255
                res[i][j] = 255


    cv2.imshow("Original Image", im)
    cv2.waitKey(0)
    cv2.imshow("Resulting Image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

im_threshold(cv2.imread("/Users/virajj/Downloads/reindeer.jpg", 0))
im_threshold(cv2.imread("/Users/virajj/Downloads/horse.jpg", 0))
im_threshold(cv2.imread("/Users/virajj/Downloads/starfish.jpg", 0))
