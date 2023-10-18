# contours using edge detection
import cv2
import matplotlib.pyplot as plt

# Read the image
image3 = cv2.imread('images/-_2003880-_URO-_20442_20200519_Kidney_0002.JPG') 
# plt.imshow(image3)
# plt.show()

# Display the image
# plt.figure(figsize=[10,10])

# Blur the image to remove noise
blurred_image = cv2.GaussianBlur(image3,(5,5),0)

# Apply canny edge detection
edges = cv2.Canny(blurred_image, 100, 160)

# Display the resultant binary image of edges
# plt.figure(figsize=[10,10])
# plt.imshow(edges,cmap='Greys_r');plt.title("Edges Image");plt.axis("off")

# Detect the contour using the edges
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours
image3_copy = image3.copy()
cv2.drawContours(image3_copy, contours, -1, (0, 255, 0), 2)

# Display the drawn contours
# plt.figure(figsize=[10,10])

image3_copy2 = image3.copy()

# Remove noise from the image
blurred = cv2.GaussianBlur(image3_copy2,(3,3),0)

# Convert the image to gray-scale
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# Perform adaptive thresholding 
binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 5)

# Detect and Draw contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image3_copy2, contours, -1, (0, 255, 0), 2)

# Plotting both results for comparison
# plt.figure(figsize=[14,10])
plt.subplot(211);plt.imshow(image3_copy2[:,:,::-1]);plt.title("Using Adaptive Thresholding");plt.axis('off')
plt.subplot(212);plt.imshow(image3_copy[:,:,::-1]);plt.title("Using Edge Detection");plt.axis('off')
plt.show()