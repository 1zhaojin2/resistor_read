import cv2
import numpy as np
import os

objCascade = cv2.CascadeClassifier('cascade/haarcascade_resistors_0.xml')

def setup_debug_interface():
    cv2.namedWindow("frame")
    cv2.createTrackbar("lh", "frame", 0, 179, lambda x: None)
    cv2.createTrackbar("uh", "frame", 0, 179, lambda x: None)
    cv2.createTrackbar("ls", "frame", 0, 255, lambda x: None)
    cv2.createTrackbar("us", "frame", 0, 255, lambda x: None)
    cv2.createTrackbar("lv", "frame", 0, 255, lambda x: None)
    cv2.createTrackbar("uv", "frame", 0, 255, lambda x: None)

def findResistors(liveimg, cascade_path):
    global objCascade
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
    mask = cv2.inRange(hsv, np.array([3, 102, 108]), np.array([16, 187, 153]))
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
    if not detected_resistors:
        print("No resistors found.")
        return
    for roi_color, (x, y, w, h) in detected_resistors:
        cv2.imshow(f"Resistor Detected at ({x}, {y})", roi_color)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# Usage
test_color_masks("0.jpg", "cascade/haarcascade_resistors_0.xml")
