# contours 6 pre-processing image 
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
 
# Display the result
plt.figure(figsize=[10,10])
plt.imshow(gray_inverted ,cmap="gray");plt.title("Bitwise Inverted");plt.axis("off")

# Find the contours from the inverted gray-scale image
contours, hierarchy = cv2.findContours(gray_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# draw all contours
cv2.drawContours(image2_copy, contours, -1, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(image2_copy[:,:,::-1]);plt.title("Contours Detected");plt.axis("off")
# plt.show()

# Create a binary thresholded image
_, binary = cv2.threshold(gray_inverted, 250, 1, cv2.THRESH_BINARY)

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(binary, cmap="gray");plt.title("Binary Image");plt.axis("off")
# plt.show()

# Make a copy of the source image.
image2_copy2 = image2.copy()

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# draw all the contours found
image2_copy2 = cv2.drawContours(image2_copy2, contours, -1, (0, 0, 255), 2)

# Plot both of the resuts for comparison
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image2_copy[:,:,::-1]);plt.title("Without Thresholding");plt.axis('off')
plt.subplot(122);plt.imshow(image2_copy2[:,:,::-1]);plt.title("With Threshloding");plt.axis('off')
plt.show()



