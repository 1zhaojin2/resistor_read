import cv2
import numpy as np
import os

def setup_debug_interface():
    cv2.namedWindow("frame")
    cv2.createTrackbar("lh", "frame", 0, 179, lambda x: None)
    cv2.createTrackbar("uh", "frame", 0, 179, lambda x: None)
    cv2.createTrackbar("ls", "frame", 0, 255, lambda x: None)
    cv2.createTrackbar("us", "frame", 0, 255, lambda x: None)
    cv2.createTrackbar("lv", "frame", 0, 255, lambda x: None)
    cv2.createTrackbar("uv", "frame", 0, 255, lambda x: None)

def findResistors(liveimg, cascade_path):
    rectCascade = cv2.CascadeClassifier(cascade_path)
    gliveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
    ressFind = rectCascade.detectMultiScale(gliveimg, 1.1, 25)
    resClose = []
    for (x, y, w, h) in ressFind:
        roi_color = liveimg[y:y+h, x:x+w]
        temp_img = np.copy(roi_color)
        temp_resized_img = cv2.resize(temp_img, (400, 200))
        if findBands(temp_resized_img, True):  # Simplified check for bands
            resClose.append((roi_color, (x, y, w, h)))
    return resClose

def findBands(roi_color, DEBUG):
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([22, 103, 164]), np.array([35, 255, 255]))
    result = cv2.bitwise_and(roi_color, roi_color, mask=mask)
    if DEBUG:
        cv2.imshow("Band Detection", result)
        cv2.waitKey(0)
    return mask.any()  # Returns True if any band is detected

def test_color_masks(image_path, cascade_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    detected_resistors = findResistors(img, cascade_path)
    for roi_color, (x, y, w, h) in detected_resistors:
        cv2.imshow(f"Resistor Detected at ({x}, {y})", roi_color)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# Usage
test_color_masks("pic3.jpg", os.getcwd() + "/cascade/haarcascade_resistors_0.xml")
