import cv2
import matplotlib.pyplot as plt 
# Read the image
image4 = cv2.imread('sword.image.jpg') 

# Create a copy 
image4_copy = image4.copy()

# Convert to gray-scale
imageGray = cv2.cvtColor(image4_copy,cv2.COLOR_BGR2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(imageGray, 220, 255, cv2.THRESH_BINARY_INV)

# Detect and draw external contour
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Select a contour
contour = contours[0]

# Draw the selected contour
cv2.drawContours(image4_copy, contour, -1, (0,255,0), 3)

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(image4_copy[:,:,::-1]);plt.title("Sword Contour");plt.axis("off");
plt.show()