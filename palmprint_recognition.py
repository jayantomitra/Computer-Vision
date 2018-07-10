import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.spatial import distance


#read the image
g_kernel = cv.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv.CV_32F)

img1 = cv.imread('vishu1.jpeg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
filtered_img = cv.filter2D(img1, cv.CV_8UC3, g_kernel)

img2 = cv.imread('palmprint2.jpeg')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

filtered_img2 = cv.filter2D(img2, cv.CV_8UC3, g_kernel)


'''dft flow for first image of palm'''
dft1 = cv.dft(np.float32(img1),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift1 = np.fft.fftshift(dft1)

magnitude_spectrum1 = 20*np.log(cv.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))

plt.subplot(121),plt.imshow(filtered_img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

'''dft flow for second image of palm'''
dft2 = cv.dft(np.float32(img2),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift2 = np.fft.fftshift(dft2)

magnitude_spectrum2 = 20*np.log(cv.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

plt.subplot(121),plt.imshow(filtered_img2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#display on screen
cv.imshow('image', img1)
cv.waitKey(1)
cv.destroyAllWindows()
cv.imshow('filtered image', filtered_img)
cv.waitKey(1)
cv.destroyAllWindows()
cv.imshow('image2', img2)
cv.waitKey(1)
cv.destroyAllWindows()
cv.imshow('filtered image 2', filtered_img2)
cv.waitKey(1)
cv.destroyAllWindows()

#save the image
cv.imwrite("vishu_processed_palmprint_img1.png", magnitude_spectrum1)
cv.imwrite("vishu_processed_palmprint_img2.png", magnitude_spectrum2)

#close the program
cv.waitKey(0)
cv.destroyAllWindows()


#comparison
image1 = cv.imread('processed_palmprint_img1.png')
image2 = cv.imread('processed_palmprint_img2.png')
print " next is matching"
