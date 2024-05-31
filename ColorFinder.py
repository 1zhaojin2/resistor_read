import cv2
import numpy as np
import os

# Create directory to save segmented bands
output_dir = 'segmented_bands'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the image
image = cv2.imread('median_resistor_0.jpg')

# Convert image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the background color range
lower_background = np.array([30, 0, 100])
upper_background = np.array([180, 255, 255])

# Create a mask for the background
background_mask = cv2.inRange(hsv_image, lower_background, upper_background)

# Invert the mask to isolate the bands
bands_mask = cv2.bitwise_not(background_mask)

# Apply the mask to the image to isolate the bands
isolated_bands = cv2.bitwise_and(image, image, mask=bands_mask)

# Find contours of the bands
contours, _ = cv2.findContours(bands_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left to right
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Extract individual bands, crop, and save them
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    band = image[y:y+h, x:x+w]

    # Crop out the left and right 3 pixels
    if w > 6:  # Ensure the width is greater than 6 pixels
        band = band[:, 3:w-3]

    # Save the band as an image file
    output_path = os.path.join(output_dir, f'band_{idx}.png')
    cv2.imwrite(output_path, band)
    print(f'Saved: {output_path}')

    # Display the band (optional)
    cv2.imshow(f'Band - {idx}', band)

cv2.waitKey(0)
cv2.destroyAllWindows()
