import cv2 as cv
import time
import numpy as np
#read the image
g_kernel = cv.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv.CV_32F)

img = cv.imread('lion.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
filtered_img = cv.filter2D(img, cv.CV_8UC3, g_kernel)

#display on screen
cv.imshow('image', img)
cv.imshow('filtered image', filtered_img)

#save the image
#cv.imwrite('lion_grey.png', img)
#cv.imwrite('batmangrey.png', img)

#close the program
#h, w = g_kernel.shape[:2]
#g_kernel = cv.resize(g_kernel, (3*w, 3*h), interpolation=cv.INTER_CUBIC)
#cv.imshow('gabor kernel (resized)', g_kernel)
cv.waitKey(0)
cv.destroyAllWindows()





