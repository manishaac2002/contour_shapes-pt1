#converted into bitwise

import cv2
import matplotlib.pyplot as plt

# Read the image
image2 = cv2.imread('images/-_2003880-_URO-_20442_20200519_Kidney_0002.JPG') 

# Display the image
plt.figure(figsize=[10,10])
plt.imshow(image2[:,:,::-1]);plt.title("Original Image");plt.axis("off")

image2_copy = image2.copy()
 
# Convert the image to gray-scale
gray = cv2.cvtColor(image2_copy, cv2.COLOR_BGR2GRAY)
 
# Display the result
plt.imshow(gray, cmap="gray");plt.title("Gray-scale Image");plt.axis("off")

gray_inverted = cv2.bitwise_not(gray)

contours, hierarchy = cv2.findContours(gray_inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# draw all the contours found
image2_copy2 = cv2.drawContours(gray_inverted, contours, -1, (0, 0, 255), 2)

# Plot both of the resuts for comparison
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image2_copy[::,:-1]);plt.title("Without Thresholding");plt.axis('off')
plt.subplot(122);plt.imshow(image2_copy2[::,:-1]);plt.title("With Threshloding");plt.axis('off');
plt.show()
