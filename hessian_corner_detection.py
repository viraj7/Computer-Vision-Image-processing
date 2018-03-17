from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *
from scipy.ndimage.filters import gaussian_filter

def hess_corner_det(im, sig, th):
    im = gaussian_filter(im, sig)
    derx = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) #derivative filter in X direction
    dery = derx.transpose() # derivative filter on y component
    Ix = cv2.filter2D(im, -1, derx) # Derivative of X component of Image
    Iy = cv2.filter2D(im, -1, dery) # Derivative of Y component of Image
    Ixx = cv2.filter2D(Ix, -1, derx) # 2nd order Derivative of X component of Image
    Iyy = cv2.filter2D(Iy, -1, dery) # 2nd order Derivative of Y component of Image
    Ixy = cv2.filter2D(Ix, -1, dery) # Derivative w.r.t Y on Ix

    #for i in range(1, im.shape[0]-1):
    #    for j in range(1, im.shape[1]-1):
    #        Ix[i][j] = (sum(im[i-1][j-1:j+2]*derx[0][0:3]) + sum(im[i][j-1:j+2]*derx[1][0:3]) + sum(im[i+1][j-1:j+2]*derx[2][0:3]))/3
    #        Iy[i][j] = (sum(im[i-1][j-1:j+2]*dery[0][0:3]) + sum(im[i][j-1:j+2]*dery[1][0:3]) + sum(im[i+1][j-1:j+2]*dery[2][0:3]))/3

    eigenvalues = numpy.empty(im.shape)
    res=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) # resultant image for plotting colourd pixels
    for i in range(im.shape[0]): # for every row(Y)
        for j in range(im.shape[1]): # for every column(X)
            l1, l2 = numpy.linalg.eigvals([[Ixx[i][j], Ixy[i][j]], [Ixy[i][j], Iyy[i][j]]]) #get eigenvalues of Hessian matrix for each pixel
            if fabs(l1) > th and fabs(l2) > th:
                res[i, j] = [0,0,255]

                '''
    for i in range(im.shape[0]): # for every row(Y)
        for j in range(im.shape[1]): # for every column(X)
            if eigenvalues[i, j][0] > 0.5*numpy.max(eigenvalues) and eigenvalues[i, j][1] > 0.5*numpy.max(eigenvalues):
                res[i, j] = [0,0,255] # for high eigenvalues, change the pixel color to Red
                '''


    cv2.imshow("or", im)
    cv2.waitKey(0)
    cv2.imshow("Ix", Ix)
    cv2.waitKey(0)
    cv2.imshow("Iy", Iy)
    cv2.waitKey(0)
    cv2.imshow("Ixx", Ixx)
    cv2.waitKey(0)
    cv2.imshow("Iyy", Iyy)
    cv2.waitKey(0)
    cv2.imshow("Ixy", Ixy)
    cv2.waitKey(0)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hess_corner_det(cv2.imread("/Users/virajj/Downloads/input1.png", 0), 1.5, 200)
hess_corner_det(cv2.imread("/Users/virajj/Downloads/input2.png", 0), 2, 210)
hess_corner_det(cv2.imread("/Users/virajj/Downloads/input3.png", 0), 1.5, 150)
