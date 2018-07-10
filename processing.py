import cv2 as cv
import time
import numpy as np

#read the image
img = cv.imread('lion.jpg', 1)

#do the processing

blur = cv.GaussianBlur(img,(5,5),0)
blur2 = cv.bilateralFilter(img,9,75,75)

#display on screen
cv.imshow('image', blur)
cv.imshow('image2', blur2)

#save the image
#cv.imwrite('lion_grey.png', img)
#cv.imwrite('batmangrey.png', img)

#close the program
cv.waitKey(300)
time.sleep(10)
cv.destroyAllWindows()

