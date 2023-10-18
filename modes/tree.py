# import cv2
# import matplotlib.pyplot as plt
# # Read the image in color mode for drawing purposes.
# image1 = cv2.imread('images/-_2003880-_URO-_20442_20200519_Kidney_0002.JPG')  

# src_copy = image1.copy()
 
# # Convert to gray-scale
# imageGray = cv2.cvtColor(src_copy,cv2.COLOR_BGR2GRAY)
 
# # Find all contours in the image while maintaining a hierarchy
# contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# # Draw all the contours.
# contour_image = cv2.drawContours(src_copy, contours, -1, (0,255,0), 3);
 
# # Print the number of Contours returned
# print("Number of Contours Returned: {}".format(len(contours)))
 
# # Display the results.
# plt.figure(figsize=[10,10])
# plt.imshow(contour_image);plt.axis("off");plt.title('Retrieval Mode: RETR_TREE');
# plt.show()

# using 6 pre-processing
import cv2
import matplotlib.pyplot as plt

# Read the image
image1 = cv2.imread('images/-_2003880-_URO-_20442_20200519_Kidney_0002.JPG') 

# Display the image
plt.figure(figsize=[10,10])
plt.imshow(image1[:,:,::-1]);plt.title("Original Image");plt.axis("off")

image1_copy = image1.copy()
 
# Convert the image to gray-scale
gray = cv2.cvtColor(image1_copy, cv2.COLOR_BGR2GRAY)
 
# Display the result
plt.imshow(gray, cmap="gray");plt.title("Gray-scale Image");plt.axis("off")

gray_inverted = cv2.bitwise_not(gray)
 
# Display the result
plt.figure(figsize=[10,10])
plt.imshow(gray_inverted ,cmap="gray");plt.title("Bitwise Inverted");plt.axis("off")

# Find the contours from the inverted gray-scale image
contours, hierarchy = cv2.findContours(gray_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# draw all contours
cv2.drawContours(image1_copy, contours, -1, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(image1_copy[:,:,::-1]);plt.title("Contours Detected");plt.axis("off")
# plt.show()

# Create a binary thresholded image
_, binary = cv2.threshold(gray_inverted, 250, 1, cv2.THRESH_BINARY)

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(binary, cmap="gray");plt.title("Binary Image");plt.axis("off")
# plt.show()

# src_copy = image1.copy()
 
# # Convert to gray-scale
# imageGray = cv2.cvtColor(src_copy,cv2.COLOR_BGR2GRAY)
 
# Find all contours in the image while maintaining a hierarchy
contours, hierarchy = cv2.findContours(gray_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# Draw all the contours.
contour_image = cv2.drawContours(gray_inverted, contours, -1, (0,255,0), 3)
 
# Print the number of Contours returned
print("Number of Contours Returned: {}".format(len(contours)))
 
# Display the results.
plt.figure(figsize=[10,10])
plt.imshow(contour_image);plt.axis("off");plt.title('Retrieval Mode: RETR_TREE')
plt.show()
