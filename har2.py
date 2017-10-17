from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *
from scipy.ndimage.filters import gaussian_filter, laplace, gaussian_laplace

def l_o_g(x, y, sigma):
    # Formatted this way for readability
    nom = ( (y**2)+(x**2)-2*(sigma**2) )
    denom = ( (2*math.pi*(sigma**6) ))
    expo = math.exp( -((x**2)+(y**2))/(2*(sigma**2)) )
    return nom*expo/denom


# Create the laplacian of the gaussian, given a sigma
# Note the recommended size is 7 according to this website http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
# Experimentally, I've found 6 to be much more reliable for images with clear edges and 4 to be better for images with a lot of little edges
def create_log(sigma, size = 7):
    w = math.ceil(float(size)*float(sigma))
    print w
    # If the dimension is an even number, make it uneven
    if(w%2 == 0):
        w = w + 1
    # Now make the mask
    l_o_g_mask = []
    w_range = w//2
    print "Going from " + str(-w_range) + " to " + str(w_range)
    for i in range(-w_range, w_range):
        for j in range(-w_range, w_range):
            l_o_g_mask.append(l_o_g(i,j,sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w, w)
    return l_o_g_mask


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

    #Gim = gaussian_filter(im, 1.5)
    #print Gim
    #log = create_log(2, 7)
    derx = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) #derivative filter in X direction
    dery = derx.transpose() # derivative filter on y component
    Ix = cv2.filter2D(im, -1, derx) # Derivative of X component of Image
    Iy = cv2.filter2D(im, -1, dery)
    Ixy = cv2.filter2D(im, -1, dery)
    Lx = gaussian_laplace(Ix, 1.5)#cv2.filter2D(Ix, -1, log)
    Ly = gaussian_laplace(Iy, 1.5)#cv2.filter2D(Iy, -1, log)
    Lxy = gaussian_laplace(Ixy, 1.5)#cv2.filter2D(Ixy, -1, log)
    alpha = 0.05
    R = numpy.empty(im.shape)
    res=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            h2 = numpy.matrix([[Lx[i, j]*Lx[i, j], Lxy[i, j]], [Lxy[i, j], Ly[i, j]*Ly[i, j]]])
            R[i, j] = numpy.linalg.det(h2) - (alpha*numpy.trace(h2))


    th = (numpy.max(numpy.absolute(R)))*0.99
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if fabs(R[i, j]) > th:
                res[i, j] = [0,0,255]

    cv2.imshow("im", im)
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
#harris(cv2.imread("/Users/virajj/Downloads/input2.png", 0))
#harris(cv2.imread("/Users/virajj/Downloads/input3.png", 0))
