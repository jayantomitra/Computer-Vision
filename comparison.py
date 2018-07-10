import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import scipy
from imageio import imread
from scipy.spatial import distance
#comparison
image1 = imread('vishu_processed_palmprint_img1.png')
image2 = imread('processed_palmprint_img2.png')

img1 = np.reshape(image1,(1,921600))
img2 = np.reshape(image2,(1,599592))
print img1
print img2
matching = distance.euclidean(img1, img2)
print matching
if matching <= 0.9:
    print "final output => palmprints dont match"
else:
    print "final output => palmprints match"


#close the program
cv.waitKey(0)
cv.destroyAllWindows()