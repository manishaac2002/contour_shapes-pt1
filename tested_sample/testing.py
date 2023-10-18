import cv2
from numpy import array    
img = cv2.imread('images/-_2003880-_URO-_20442_20200519_Kidney_0002.JPG')
img2 = array( 200  * (img[:,:,2] > img[:,:, 1]), dtype='uint8')

edges = cv2.Canny(img2, 70, 50)
cv2.imshow('edges.png', edges)
cv2.waitKey(0)