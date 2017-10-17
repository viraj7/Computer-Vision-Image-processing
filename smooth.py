from __future__ import division
import cv2, scipy, numpy, math
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
im = cv2.imread("/Users/virajj/Downloads/reindeer.jpg", 0)

def gen_gauss1d_k(ar, sig):# Returns a list of 1-D Gaussian filter
    return list(map(lambda x: math.e**(-(x**2/(2*sig**2)))/(math.sqrt(2*math.pi)*sig), ar))

gauss1d_k = numpy.array(gen_gauss1d_k([-4, -3, -2, -1, 0, 1, 2, 3, 4], 3))

#Applying Gaussian smoothing on X component of the image
res1 = numpy.zeros(im.shape, dtype=numpy.uint8)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if j >= (len(gauss1d_k)//2) and j <= (im.shape[1] - len(gauss1d_k)//2 - 1):#Skipping the border pixels
            res1[i][j] = int(sum(im[i][j-len(gauss1d_k)//2:j+1+len(gauss1d_k)//2]*gauss1d_k))
        else: # copying the border pixels of the original image as it is
            res1[i][j] = im[i][j]
cv2.imshow("img1", res1)
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

cv2.imshow("img2", res2)
cv2.waitKey(0)


#Apply derivative of Gaussian on result of X component of the image convolved with gaussian

gauss1d_k_grad = numpy.gradient(gauss1d_k)
res3 = numpy.zeros(res1.shape, dtype=numpy.uint8)
for i in range(res1.shape[0]):
    for j in range(res1.shape[1]):
        if j >= (len(gauss1d_k_grad)//2) and j <= (res1.shape[1] - len(gauss1d_k_grad)//2 - 1):#Skipping the border pixels
            res3[i][j] = int(sum(res1[i][j-len(gauss1d_k_grad)//2:j+1+len(gauss1d_k_grad)//2]*gauss1d_k_grad))
        else: # copying the border pixels of the original image as it is
            res3[i][j] = res1[i][j]
cv2.imshow("img3", res3)
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
cv2.imshow("img4", res4)
cv2.waitKey(0)
#cv2.destroyAllWindows()

res5 = numpy.zeros(res3.shape, dtype=numpy.uint8)
orientation = numpy.empty(res3.shape)
for i in range(res3.shape[0]):
    for j in range(res3.shape[1]):
        res5[i][j] = int(math.sqrt(res3[i][j]**2 + res4[i][j]**2))
        orientation[i][j] = math.degrees(math.atan2(res4[i][j], res3[i][j]))

#res5 = numpy.uint8(numpy.sqrt(numpy.add(numpy.power(res3, 2), numpy.power(res4, 2))))
#res5 = numpy.array(res5, dtype=numpy.uint8)
cv2.imshow("img5", res5)
cv2.waitKey(0)
#cv2.destroyAllWindows()

#non maximum suppression
print numpy.amax(orientation), numpy.amin(orientation)
res6 = numpy.zeros(res5.shape, dtype=numpy.uint8)
for i in range(1, res5.shape[0]-1):
    for j in range(1, res5.shape[1]-1):
        if orientation[i][j] < 30:
            if res5[i][j] <= res5[i][j+1] or res5[i][j] <= res5[i][j-1]:
                res6[i][j] = 0
            else:
                res6[i][j] = res5[i][j]
        elif orientation[i][j] > 30 and orientation[i][j] < 60:
            if res5[i][j] <= res5[i+1][j-1] or res5[i][j] <= res5[i-1][j+1]:
                res6[i][j] = 0
            else:
                res6[i][j] = res5[i][j]
        else:
            if res5[i][j] <= res5[i-1][j] or res5[i][j] <= res5[i+1][j]:
                res6[i][j] = 0
            else:
                res6[i][j] = res5[i][j]
#        if res5[i][j]

cv2.imshow("img6", res6)
cv2.waitKey(0)
cv2.destroyAllWindows()
