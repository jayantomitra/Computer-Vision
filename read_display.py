from __future__ import division
import cv2 as cv
import time
#import numpy

#read the image
img = cv.imread('coder.jpg', 0)

#do the processing

#display on screen

screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

cv.namedWindow('coder.jpg', cv.WINDOW_NORMAL)
cv.resizeWindow('coder.jp', window_width, window_height)
cv.imshow('images', img)
time.sleep(10)
#save the image

#cv.imwrite('batmangrey.png', img)

#close the program
cv.destroyAllWindows()

