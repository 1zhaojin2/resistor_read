"""
Image Cropping and Resizing:
    Once the Haar Cascade has identified the resistor's location, crop this area from the original image to focus solely
     on the resistor.
    Resize the cropped image to a standard size to normalize the processing regardless of the original size. This helps
     in consistent color band segmentation.
Image Preprocessing:
    Convert the cropped image to a grayscale image to help in thresholding and edge detection, which can further assist
     in isolating the color bands from the body of the resistor.
    Apply filters, such as a Gaussian Blur, to smooth out the image, reducing noise and enhancing the detection of the
     edges of the color bands.
Segmentation of Color Bands:
    Use edge detection techniques (like the Sobel filter or Canny edge detector) to find vertical boundaries of color
     bands. This will help in segmenting the image into distinct color bands.
    You might also consider using morphological operations like dilation and erosion to clarify the separation between
     bands.
Color Recognition:
    For each segmented color band, extract the dominant color. This can be done by converting the color space to HSV and
     calculating the histogram to find the most frequent hue values.
    Map these dominant hue values to the closest standard resistor color codes.
Avoiding Background/Body Color:
    To ensure the machine does not count the background or body color of the resistor, you can set thresholds based on
     the color distribution. Typically, the body of the resistor (often tan or blue) can be distinct from the vibrant
     colors used for the bands.
    Analyze only those segments that fall within the expected size range of a color band compared to the overall length
     of the resistor, excluding any large areas that extend beyond typical band dimensions.
Validation and Calibration:
    Validate the system with known resistors under various lighting conditions to calibrate your color detection
     algorithms.
    Adjust thresholds and algorithms based on this testing to improve accuracy and robustness.
"""

import cv2
import numpy as np
import math

band_areas = []

COLOUR_BOUNDS = [
    [(0, 0, 0)     , (179, 255, 40)   , "BLACK"  , 0 , (255,255,0)],
    [(0, 68, 48)   , (117, 255, 97)  , "BROWN"  , 1 , (0,255,102)], #adjusted
    [(0, 187, 125)  , (39, 255, 255)  , "RED"    , 2 , (128,0,128)],
    [(6, 197, 87)  , (20, 255, 255)  , "ORANGE" , 3 , (0,128,255)], #adjusted
    [(22, 103, 164), (35, 255, 255)  , "YELLOW" , 4 , (0,255,255)], #adjusted
    [(30, 76, 89) , (87, 255, 184)   , "GREEN"  , 5 , (0,255,0)],
    [(113, 40, 82)   , (125, 255, 255)  , "BLUE"   , 6 , (255,0,0)], #adjusted
    [(130, 40, 100), (140, 250, 220) , "PURPLE" , 7 , (255,0,127)], #adjusted
    [(0, 0, 50)    , (179, 50, 80)   , "GRAY"   , 8 , (128,128,128)],
    [(0, 0, 0)    , (0, 0, 0)  , "WHITE"  , 9 , (255,255,255)], #ignoring white for now lol
    [(19, 120, 80), (23, 255, 255), "GOLD", 10, (0,215,255)],  # Approximate for Gold
    [(0, 0, 0), (0, 0, 50), "SILVER", 11, (192,192,192)]  # Very light gray, almost white
    # reminder to self - add white color
]


def load_and_detect_resistors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('cascade/haarcascade_resistors_0.xml')
    resistors = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)  # Adjusted parameters

    # Filter detections by size (example sizes, adjust accordingly)
    min_width, min_height = 100, 50  # Minimum acceptable dimensions for a resistor
    filtered_resistors = [r for r in resistors if r[2] >= min_width and r[3] >= min_height]

    return img, filtered_resistors



def crop_resistor(img, x, y, w, h):
    # Calculate the vertical middle point of the resistor
    mid_y = y + h // 2

    # Adjust the x coordinate to start 50 pixels more to the right
    # and reduce the width by 100 pixels (50 from each side)
    start_x = x + 110
    end_x = x + w - 110

    # Ensure the new x coordinates are not out of image bounds
    start_x = max(start_x, 0)
    end_x = min(end_x, img.shape[1])

    # Define the crop region for y coordinate
    start_y = max(mid_y , 0)
    end_y = min(mid_y + 20, img.shape[0])

    # Crop the image around the middle of the resistor
    return img[start_y:end_y, start_x:end_x]

def compute_vertical_averages(cropped_img):
    # Compute the average of each column (axis=0) for all color channels
    vertical_averages = np.mean(cropped_img, axis=0)

    # Normalize the averages to span the full range of colors
    vertical_averages = cv2.normalize(vertical_averages, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    vertical_averages = np.array(vertical_averages, dtype=np.uint8)

    # Create an image to display the averages
    # The height is 20 pixels, and we replicate the averages array 20 times vertically
    average_img = np.tile(vertical_averages, (20, 1, 1))

    return average_img


# Include this in your main code where appropriate to test the flood fill
# Draw the results on your cropped_img to see how well the flood fill is working


def preprocess_image(cropped_img):
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_img)

    highlight_threshold = 180

    v[v > highlight_threshold] = highlight_threshold

    processed_hsv_img = cv2.merge([h, s, v])

    highlight_removed_img = cv2.cvtColor(processed_hsv_img, cv2.COLOR_HSV2BGR)

    return highlight_removed_img



def display_images(images, titles):
    for img, title in zip(images, titles):
        cv2.imshow(title, img)


def main(image_path):
    img, resistors = load_and_detect_resistors(image_path)
    if img is None or resistors is None:
        return

    # Draw bounding boxes around detected resistors
    for (x, y, w, h) in resistors:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw red rectangles

    processed_images = []
    for (x, y, w, h) in resistors:
        cropped_img = crop_resistor(img, x, y, w, h)
        preprocessed_img = preprocess_image(cropped_img)

        processed_images.append((cropped_img, 'Cropped Resistor'))
        processed_images.append((preprocessed_img, 'Preprocessed Image with Edges'))
        average_img = compute_vertical_averages(preprocessed_img)
        processed_images.append((average_img, 'Vertical Average Profile'))

    # Display all cropped, preprocessed images, and vertical averages
    display_images(*zip(*processed_images))


# Run the main function with the specified image
main('pic4.jpg')
cv2.waitKey()
cv2.destroyAllWindows()

