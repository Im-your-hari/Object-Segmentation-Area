'''
import cv2
import numpy as np

# Load the segmented color image
segmented_image = cv2.imread('videoplayback (1).avi')

# Convert the segmented image to grayscale
gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to create a binary mask
_, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Draw a rectangle around the segmented portion
    cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the segmented image with contours
cv2.imshow('Segmented Image with Contours', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


'''
import cv2

# Read the image
image = cv2.imread("IMG_1366.JPG")

#image = image.resize(400,400)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Find contours
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the area of each contour
for contour in contours:
    area = cv2.contourArea(contour)
    print(f"Contour area: {area}")

# Display the image with contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Image with Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
import cv2
import numpy as np

# Load the segmented image
image = cv2.imread('test1.jpeg', cv2.IMREAD_GRAYSCALE)

# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Create a mask image for visualization
mask = np.zeros_like(image)

# Draw contours on the mask image
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Display the mask image
#cv2.imshow('Contours', mask)
cv2.imshow('Contours', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Access contour pixels
for contour in contours:
    #print(contour)
    for pixel in contour:
        x, y = pixel[0]
        print(f"Pixel at ({x}, {y}) is part of the contour.")

'''

'''
import cv2
import numpy as np

# rgb 229, 0, 178 # the purple carpet in RGB (sampled with IrfanView)
# b,g,r = 178, 0, 229 # cv2 uses BGR

#239,132,140

#class_color = [178, 0, 229]
class_color = [140,132,239]
multiclassImage = cv2.imread("test2.jpeg")
cv2.imshow("MULTI", multiclassImage)
filteredImage = multiclassImage.copy()
low = np.array(class_color);

mask = cv2.inRange(filteredImage, low, low)
filteredImage[mask == 0] = [0, 0, 0]
filteredImage[mask != 0] = [255,255,255]
cv2.imshow("FILTER", filteredImage)
# numberPixelsFancier = len(cv2.findNonZero(filteredImage[...,0]))
# That also works and returns 14861 - without conversion, taking one color channel
bwImage = cv2.cvtColor(filteredImage, cv2.COLOR_BGR2GRAY)  
cv2.imshow("BW", bwImage)
numberPixels = len(cv2.findNonZero(bwImage))
print(numberPixels)
cv2.waitKey(0)
'''
import cv2
import numpy as np

image = cv2.imread('test3.jpeg')
cv2.imshow('Image', image)
# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow("Hsv",hsv_image)

# Define range of red color in HSV
##lower_blue = np.array([160,50,50])
#lower_blue = np.array([0, 100, 20])
#Better Result kittiya range :: lower[160,50,50]-upper[180,255,255]
##upper_blue = np.array([180,255,255])
#upper_blue = np.array([10, 255, 255])

#Dark Region____________________________________
lower_red_dark = np.array([0,100,100])
upper_red_dark = np.array([20,255,255])
#_______________________________________________
#Light Region___________________________________
lower_blue_light = np.array([160,100,100])
upper_blue_light = np.array([180,255,255])
#_______________________________________________

mask_dark = cv2.inRange(hsv_image, lower_red_dark, upper_red_dark)

mask_light = cv2.inRange(hsv_image, lower_blue_light, upper_blue_light)

#cv2.imshow("mask",mask)
# Bitwise-AND mask and original image
segmented_image_dark = cv2.bitwise_and(image, image, mask=mask_dark)
segmented_image_light = cv2.bitwise_and(image, image, mask=mask_light)


# Display the result
cv2.imshow('Segmented Image dark', segmented_image_dark)
cv2.imshow('Segmented Image light', segmented_image_light)

pixel_count_dark = cv2.countNonZero(mask_dark)
pixel_count_light = cv2.countNonZero(mask_light)


print(pixel_count_dark)
print(pixel_count_light)

print("Total Area = {}pixels".format(pixel_count_dark+pixel_count_light))

cv2.waitKey(0)
cv2.destroyAllWindows()

