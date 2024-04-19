import cv2
import numpy as np
import os
import imutils

DEBUG = True
COLOUR_BOUNDS = [
    # Adjusted BLACK and BROWN, tune these based on further observations
    [(0, 0, 0)     , (179, 255, 40)   , "BLACK"  , 0 , (255,255,0)],  # Tightened value range for BLACK
    [(5, 140, 16)   , (12, 255, 151)  , "BROWN"  , 1 , (0,255,102)], # Expanded hue and value for BROWN
    [(0, 187, 125)  , (39, 255, 255)  , "RED"    , 2 , (128,0,128)], # Adjusted to include desired RED
    [(10, 70, 70)  , (25, 255, 60)  , "ORANGE" , 3 , (0,128,255)],
    [(30, 170, 100), (40, 250, 255)  , "YELLOW" , 4 , (0,255,255)],
    [(30, 76, 89) , (87, 255, 184)   , "GREEN"  , 5 , (0,255,0)],
    [(65, 0, 85)   , (115, 30, 147)  , "BLUE"   , 6 , (255,0,0)],
    [(120, 40, 100), (140, 250, 220) , "PURPLE" , 7 , (255,0,127)],
    [(0, 0, 50)    , (179, 50, 80)   , "GRAY"   , 8 , (128,128,128)],
    [(0, 0, 90)    , (179, 15, 250)  , "WHITE"  , 9 , (255,255,255)],
]


RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def empty(x):
    pass

def init(DEBUG):
    if (DEBUG):
        cv2.namedWindow("frame")
        cv2.createTrackbar("lh", "frame",0,179, empty)
        cv2.createTrackbar("uh", "frame",0,179, empty)
        cv2.createTrackbar("ls", "frame",0,255, empty)
        cv2.createTrackbar("us", "frame",0,255, empty)
        cv2.createTrackbar("lv", "frame",0,255, empty)
        cv2.createTrackbar("uv", "frame",0,255, empty)
    tPath = os.getcwd()
    rectCascade = cv2.CascadeClassifier(tPath + "/cascade/haarcascade_resistors_0.xml")
    return rectCascade

def printResult(color_code_positions, liveimg, resPos):

    color_names = {0: "BLACK", 1: "BROWN", 2: "RED", 3: "ORANGE", 4: "YELLOW",
                   5: "GREEN", 6: "BLUE", 7: "PURPLE", 8: "GRAY", 9: "WHITE"}
    
    # Sort the dictionary by x-coordinate (key) and extract the codes
    sorted_positions = sorted(color_code_positions.items())
    color_codes = [pos[1][0] for pos in sorted_positions]  # List of color codes in order of appearance
    detected_colors = [f"{color_names[code]} at {pos[1][1]}" for code, pos in zip(color_codes, sorted_positions)]
    detected_colors_str = "; ".join(detected_colors)
    print(f"Detected Colors: {detected_colors_str}")

    # Base value and multiplier extraction based on the number of bands
    if len(color_codes) >= 3:
        if len(color_codes) == 3:
            base_value = int(f"{color_codes[0]}{color_codes[1]}")
            multiplier = 10 ** color_codes[2]
        elif len(color_codes) == 4 or len(color_codes) == 5 or len(color_codes) == 6:
            base_value = int(f"{color_codes[0]}{color_codes[1]}{color_codes[2]}"[:3])  # Ensures only the first three are digits
            multiplier = 10 ** color_codes[3]

        final_resistance = base_value * multiplier
        display_message = f"{final_resistance} OHMS"
        print(f"Resistor Value: {final_resistance} Ohms")
    else:
        display_message = "Insufficient data to calculate resistance."

    # Display results
    x, y, w, h = resPos
    cv2.rectangle(liveimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(liveimg, display_message, (x + w + 10, y), FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Optionally display all detected colors as well
    text_y = y + 20
    for color_text in detected_colors:
        cv2.putText(liveimg, color_text, (x, text_y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        text_y += 20

    if DEBUG:
        # Show the final image with annotations in debug mode
        cv2.imshow("Final Result", liveimg)




def validContour(cnt):
    # Check if the contour area is at least 150 square pixels
    if cv2.contourArea(cnt) < 10:
        return False
    else:
        x, y, w, h = cv2.boundingRect(cnt)
    return True

def findResistors(liveimg, rectCascade):
    gliveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
    resClose = []
    ressFind = rectCascade.detectMultiScale(gliveimg, 1.1, 25)
    for (x, y, w, h) in ressFind:
        roi_gray = gliveimg[y:y+h, x:x+w]
        roi_color = liveimg[y:y+h, x:x+w]
        secondPass = rectCascade.detectMultiScale(roi_gray, 1.01, 5)
        if (len(secondPass) != 0):
            resClose.append((np.copy(roi_color), (x, y, w, h)))
    return resClose

def findBands(resistorInfo, DEBUG):
    resImg = cv2.resize(resistorInfo[0], (400, 200))
    midY = resImg.shape[0] // 2
    cv2.line(resImg, (0, midY), (resImg.shape[1], midY), (0, 0, 255), 2)  # Draw a middle red line

    # Adjust ROI to below the middle line for scanning bands
    roi = resImg[midY + 1: midY + 20, :]
    pre_bil = cv2.bilateralFilter(roi, 5, 80, 80)
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 59, 5)
    thresh = cv2.bitwise_not(thresh)

    color_code_positions = {}
    last_accepted_position = -30  # Initialize with a value that allows the first detection

    for clr in COLOUR_BOUNDS:
        mask = cv2.inRange(hsv, clr[0], clr[1])
        if clr[2] == "RED":  # Combining the 2 RED ranges in hsv
            redMask2 = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
            mask = cv2.bitwise_or(redMask2, mask, mask)

        # Apply the threshold mask to get only the relevant color parts
        mask = cv2.bitwise_and(mask, thresh, mask=mask)

        if DEBUG:
            window_name = f"Mask for {clr[2]}"  # Unique window name for each color
            cv2.imshow(window_name, mask)  # Display each mask in its own window

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if validContour(contour):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"]) + midY  # Adjust y-coordinate to global image context
                    if abs(cX - last_accepted_position) >= 30:
                        color_code_positions[cX] = (clr[3], (cX, cY))  # Store color code and coordinates
                        last_accepted_position = cX  # Update last accepted position

    cv2.imshow('Contour Display', pre_bil)  # Optional: Display the image with drawn contours
    return color_code_positions





# MAIN
rectCascade = init(DEBUG)

cliveimg = cv2.imread("pic1.jpg")
cliveimg = cv2.resize(cliveimg, (0,0), fx=0.2, fy=0.2) 

resClose = findResistors(cliveimg, rectCascade)
for res in resClose:
    color_code_positions = findBands(res, DEBUG)
    printResult(color_code_positions, cliveimg, res[1])
cv2.imshow("Frame", cliveimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
