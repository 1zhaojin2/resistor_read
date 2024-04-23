import cv2
import numpy as np
import os
import imutils



band_areas = []  # List to hold area of each detected band
DEBUG = True
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

tolerance_codes = {
    10: "5%",  # Gold
    11: "10%"  # Silver
}

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

    color_names = {
    0: "BLACK",
    1: "BROWN",
    2: "RED",
    3: "ORANGE",
    4: "YELLOW",
    5: "GREEN",
    6: "BLUE",
    7: "PURPLE",
    8: "GRAY",
    9: "WHITE",
    10: "GOLD",
    11: "SILVER"  # Make sure these new entries are present
    }

    temperature_list = {
    0: "250 ppm/K",
    1: "100 ppm/K",
    2: "50 ppm/K",
    3: "15 ppm/K",
    4: "25 ppm/K",
    5: "20 ppm/K",
    6: "10 ppm/K",
    7: "5 ppm/K",
    8: "1 ppm/K"
    }

    
    # Sort the dictionary by x-coordinate (key) and extract the codes
    sorted_positions = sorted(color_code_positions.items())
    color_codes = [pos[1][0] for pos in sorted_positions]  # List of color codes in order of appearance
    detected_colors = [f"{color_names[code]} at {pos[1][1]}" for code, pos in zip(color_codes, sorted_positions)]
    detected_colors_str = "; ".join(detected_colors)
    print(f"Detected Colors: {detected_colors_str}")

    # Base value and multiplier extraction based on the number of bands
    if len(color_codes) >= 3:
        if len(color_codes) == 3:
            base_value = int(str(color_codes[0]) + str(color_codes[1]))
            multiplier = 10 ** color_codes[2]
        elif len(color_codes) == 4:
            base_value = int(str(color_codes[0]) + str(color_codes[1]))
            multiplier = 10 ** color_codes[2]
            if tolerance_codes.get(color_codes[3]):
                tolerance = tolerance_codes[color_codes[3]]
            print(f"Tolerance: {tolerance}")
        elif len(color_codes) == 5:
            base_value = int(str(color_codes[0]) + str(color_codes[1]) + str(color_codes[2]))
            multiplier = 10 ** color_codes[3]
            if tolerance_codes.get(color_codes[4]):
                tolerance = tolerance_codes[color_codes[4]]
            print(f"Tolerance: {tolerance}")
        elif len(color_codes) == 6:
            base_value = int(str(color_codes[0]) + str(color_codes[1]) + str(color_codes[2]))
            multiplier = 10 ** color_codes[3]
            if tolerance_codes.get(color_codes[4]):
                tolerance = tolerance_codes[color_codes[4]]
            temperature = color_codes[5]
            temperature = temperature_list[temperature]
            print(f"Tolerance: {tolerance}")
            print(f"Temperature: {temperature}")

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
    if cv2.contourArea(cnt) < 150:
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
            # Check for bands within this roi before adding
            temp_img = np.copy(roi_color)
            temp_resized_img = cv2.resize(temp_img, (400, 200))
            color_bands = findBands((temp_resized_img, None), DEBUG)
            if color_bands:  # Only append if there are detected color bands
                resClose.append((roi_color, (x, y, w, h)))
    return resClose


def findBands(resistorInfo, DEBUG):
    resImg = cv2.resize(resistorInfo[0], (400, 200))
    midY = resImg.shape[0] // 2
    cv2.line(resImg, (0, midY), (resImg.shape[1], midY), (0, 0, 255), 2)  # Visual reference

    # Pre-processing for mask detection
    roi = resImg[midY + 1: midY + 20, :]
    pre_bil = cv2.bilateralFilter(roi, 5, 80, 80)
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 59, 5)
    thresh = cv2.bitwise_not(thresh)

    color_code_positions = {}
    last_accepted_position = -30

    for clr in COLOUR_BOUNDS:
        mask = cv2.inRange(hsv, clr[0], clr[1])
        
        if clr[2] == "RED":  # Handling red color specifics
            redMask2 = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
            mask = cv2.bitwise_or(redMask2, mask, mask)

        if DEBUG:
            window_name = f"Mask for {clr[2]}"  # Unique window name for each color
            cv2.imshow(window_name, mask)  # Display each mask in its own window

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if validContour(contour):
                area = cv2.contourArea(contour)
                band_areas.append((clr[2], area))
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"]) + midY  # Adjust y-coordinate to global image context
                    if abs(cX - last_accepted_position) >= 30:
                        color_code_positions[cX] = (clr[3], (cX, cY))  # Store color code and coordinates
                        last_accepted_position = cX  # Update last accepted position

    if DEBUG:
        area_image = np.zeros((500, 300, 3), dtype=np.uint8)
        y0, dy = 20, 20
        for i, (color, area) in enumerate(band_areas):
            cv2.putText(area_image, f"{color}: {area:.2f} px^2", (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Band Areas", area_image)

    cv2.imshow('Contour Display', pre_bil)  # Optional: Display the image with drawn contours
    return color_code_positions




# MAIN
rectCascade = init(DEBUG)

cliveimg = cv2.imread("pic1.jpg")
if cliveimg is None:
    print("Image not found. Please check the file path.")
else:
    cliveimg = cv2.resize(cliveimg, (0,0), fx=0.2, fy=0.2) 

    resClose = findResistors(cliveimg, rectCascade)
    for res in resClose:
        color_code_positions = findBands(res, DEBUG)
        if color_code_positions:  # Check again to ensure there are detected bands
            printResult(color_code_positions, cliveimg, res[1])

    cv2.imshow("Frame", cliveimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
