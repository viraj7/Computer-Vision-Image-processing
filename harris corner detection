from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *
from scipy.ndimage.filters import gaussian_filter, laplace

def gen_gauss1d_k(ar, sig):# Returns a list of 1-D Gaussian filter
    return list(map(lambda x: math.e**(-(x**2/(2*sig**2)))/(math.sqrt(2*math.pi)*sig), ar))

def harris(im):
    '''
    gauss1d_k = numpy.array(gen_gauss1d_k([-1, 0, 1], 1))
    res1 = numpy.zeros(im.shape, dtype=numpy.uint8)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if j >= (len(gauss1d_k)//2) and j <= (im.shape[1] - len(gauss1d_k)//2 - 1):#Skipping the border pixels
                res1[i][j] = int(sum(im[i][j-len(gauss1d_k)//2:j+1+len(gauss1d_k)//2]*gauss1d_k))
            else: # copying the border pixels of the original image as it is
                res1[i][j] = im[i][j]
    '''

    Gim = gaussian_filter(im, 1.5)
    print Gim
    derx = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) #derivative filter in X direction
    dery = derx.transpose() # derivative filter on y component
    Ix = cv2.filter2D(im, -1, derx) # Derivative of X component of Image
    Iy = cv2.filter2D(im, -1, dery)
    Ixy = cv2.filter2D(im, -1, dery)
    Lx = laplace(Ix)
    Ly = laplace(Iy)
    Lxy = laplace(Ixy)
    alpha = 0.04
    R = numpy.empty(im.shape)
    res=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            h2 = numpy.matrix([[Lx[i, j]*Lx[i, j], Lxy[i, j]], [Lxy[i, j], Ly[i, j]*Ly[i, j]]])
            R[i, j] = numpy.linalg.det(h2) - (alpha*numpy.trace(h2))


    th = numpy.max(R*0.99)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if R[i, j] > th:
                res[i, j] = [0,0,255]

    cv2.imshow("Gim", Gim)
    cv2.waitKey(0)
    cv2.imshow("Ix", Ix)
    cv2.waitKey(0)
    cv2.imshow("Iy", Iy)
    cv2.waitKey(0)
    cv2.imshow("Lx", Lx)
    cv2.waitKey(0)
    cv2.imshow("Ly", Ly)
    cv2.waitKey(0)
    cv2.imshow("Lxy", Lxy)
    cv2.waitKey(0)
    cv2.imshow("res", res)
    cv2.waitKey(0)

harris(cv2.imread("/Users/virajj/Downloads/input1.png", 0))
