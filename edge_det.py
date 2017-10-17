from __future__ import division
import cv2, scipy, numpy, math
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

def gen_gauss1d_k(ar, sig):# Returns a list of 1-D Gaussian filter
    return list(map(lambda x: math.e**(-(x**2/(2*sig**2)))/(math.sqrt(2*math.pi)*sig), ar))


def edge_det(im, sig, ar):
    gauss1d_k = numpy.array(gen_gauss1d_k(ar, sig))

    #Applying Gaussian smoothing on X component of the image
    cv2.imshow("Original image", im)
    cv2.waitKey(0)
    res1 = numpy.zeros(im.shape, dtype=numpy.uint8)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if j >= (len(gauss1d_k)//2) and j <= (im.shape[1] - len(gauss1d_k)//2 - 1):#Skipping the border pixels
                res1[i][j] = int(sum(im[i][j-len(gauss1d_k)//2:j+1+len(gauss1d_k)//2]*gauss1d_k))
            else: # copying the border pixels of the original image as it is
                res1[i][j] = im[i][j]
    cv2.imshow("Gaussian on X for sig=%d" %sig, res1)
    cv2.waitKey(0)

    #Applying Gaussian smoothing on Y component of the image
    cim = im.transpose()
    res2 = numpy.zeros(cim.shape, dtype=numpy.uint8)
    for i in range(cim.shape[0]):
        for j in range(cim.shape[1]):
            if j >= (len(gauss1d_k)//2) and j <= (cim.shape[1] - len(gauss1d_k)//2 - 1):
                res2[i][j] = int(sum(cim[i][j-len(gauss1d_k)//2:j+1+len(gauss1d_k)//2]*gauss1d_k))
            else:
                res2[i][j] = cim[i][j]
    res2 = res2.transpose()
    cv2.imshow("Gaussian on Y", res2)
    cv2.waitKey(0)


    #Apply derivative of Gaussian on result of X component of the image convolved with gaussian

    gauss1d_k_grad = [ gauss1d_k[i] - gauss1d_k[i-1] if i > 0 else 0 for i in range(len(gauss1d_k))]
    #gauss1d_k_grad = numpy.gradient(gauss1d_k)
    res3 = numpy.zeros(res1.shape, dtype=numpy.uint8)
    for i in range(res1.shape[0]):
        for j in range(res1.shape[1]):
            if j >= (len(gauss1d_k_grad)//2) and j <= (res1.shape[1] - len(gauss1d_k_grad)//2 - 1):#Skipping the border pixels
                res3[i][j] = int(sum(res1[i][j-len(gauss1d_k_grad)//2:j+1+len(gauss1d_k_grad)//2]*gauss1d_k_grad))
            else: # copying the border pixels of the original image as it is
                res3[i][j] = res1[i][j]
    cv2.imshow("1st der Gauss on X", res3)
    cv2.waitKey(0)

    #Apply derivative of Gaussian on Y component of the image
    res2 = res2.transpose()
    res4 = numpy.zeros(res2.shape, dtype=numpy.uint8)
    for i in range(res2.shape[0]):
        for j in range(res2.shape[1]):
            if j >= (len(gauss1d_k_grad)//2) and j <= (res2.shape[1] - len(gauss1d_k_grad)//2 - 1):#Skipping the border pixels
                res4[i][j] = int(sum(res2[i][j-len(gauss1d_k_grad)//2:j+1+len(gauss1d_k_grad)//2]*gauss1d_k_grad))
            else: # copying the border pixels of the original image as it is
                res4[i][j] = res2[i][j]
    res4 = res4.transpose()
    cv2.imshow("1st der Gauss on Y", res4)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    res5 = numpy.zeros(res3.shape, dtype=numpy.uint8)
    orientation = numpy.empty(res3.shape)
    for i in range(res3.shape[0]):
        for j in range(res3.shape[1]):
            res5[i][j] = int(math.sqrt(res3[i][j]**2 + res4[i][j]**2)) #calculating magnitude
            orientation[i][j] = math.degrees(math.atan2(res4[i][j], res3[i][j])) #Getting orientation for every pixel
    #res5 = numpy.uint8(numpy.sqrt(numpy.add(numpy.power(res3, 2), numpy.power(res4, 2))))
    #res5 = numpy.array(res5, dtype=numpy.uint8)
    cv2.imshow("Magnitude", res5)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #non maximum suppression
    #print numpy.amax(orientation), numpy.amin(orientation)
    #print set(orientation.flatten())
    res6 = numpy.zeros(res5.shape, dtype=numpy.uint8)
    for i in range(1, res5.shape[0]-1):
        for j in range(1, res5.shape[1]-1):
            if orientation[i][j] < 30: # check for pixels on right and left
                if res5[i][j] <= res5[i][j+1] or res5[i][j] <= res5[i][j-1]: # compare with pixels on left and right
                    res6[i][j] = 0
                else:
                    res6[i][j] = res5[i][j]
            elif orientation[i][j] > 30 and orientation[i][j] <= 60:
                if res5[i][j] <= res5[i+1][j-1] or res5[i][j] <= res5[i-1][j+1]: # check for pixels in diagonal direction
                    res6[i][j] = 0
                else:
                    res6[i][j] = res5[i][j]
            elif orientation[i][j] > 60:
                if res5[i][j] <= res5[i-1][j] or res5[i][j] <= res5[i+1][j]: # compare with pixels on up and down
                    res6[i][j] = 0
                else:
                    res6[i][j] = res5[i][j]
    #        if res5[i][j]

    cv2.imshow("Non max suppression", res6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #hysterisis thresholding

    res7 = numpy.zeros(res5.shape, dtype=numpy.uint8)
    high, low = 0.7*numpy.amax(res6), 0.3*numpy.amin(res6) # setting high and low for thresholding
    for i in range(1, res6.shape[0]-1):
        for j in range(1, res6.shape[1]-1):
            if res6[i][j] < low:
                res7[i][j] = 0 # if pixel is below low
            elif res6[i][j] > high:
                res7[i][j] = res6[i][j] # if pixel is above high highlight the pixel as edge
            else: # if the pixel is between low and high, compare it with neighbooring pixels
                    if res6[i][j+1] >= high or res6[i][j-1] >= high or res6[i-1][j-1] >= high or res6[i+1][j+1] >= high or res6[i+1][j-1] >= high or res6[i-1][j+1] >= high or res6[i-1][j] >= high or res6[i+1][j] >= high:
                        res7[i][j] = res6[i][j] # if one of the neighbouring pixel is an edge, pixel is considered as edge
                    else: # if none of the neighboring pixels are edge pixels
                        res7[i][j] = 0 # set pixel value to zero
    cv2.imshow("Hysterisis threshold", res7)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


edge_det(cv2.imread("/Users/virajj/programs/cv/reindeer.jpg", 0), 1, [-1, 0, 1])
edge_det(cv2.imread("/Users/virajj/programs/cv/reindeer.jpg", 0), 1.5, [-2, -1, 0, 1, 2])
edge_det(cv2.imread("/Users/virajj/programs/cv/reindeer.jpg", 0), 2, [-3, -2, -1, 0, 1, 2, 3])

edge_det(cv2.imread("/Users/virajj/programs/cv/pyramid.jpg", 0), 1, [-1, 0, 1])
edge_det(cv2.imread("/Users/virajj/programs/cv/pyramid.jpg", 0), 1.5, [-2, -1, 0, 1, 2])
edge_det(cv2.imread("/Users/virajj/programs/cv/pyramid.jpg", 0), 2, [-3, -2, -1, 0, 1, 2, 3])

edge_det(cv2.imread("/Users/virajj/programs/cv/starfish.jpg", 0), 1, [-1, 0, 1])
edge_det(cv2.imread("/Users/virajj/programs/cv/starfish.jpg", 0), 1.5, [-2, -1, 0, 1, 2])
edge_det(cv2.imread("/Users/virajj/programs/cv/starfish.jpg", 0), 2, [-3, -2, -1, 0, 1, 2, 3])

# For sigma = 1 we get best results.
# As the sigma increases, edge detection deteriorates
