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
    cv2.line(resImg, (0, midY), (resImg.shape[1], midY), (0, 0, 255), 2)  # Visual reference line

    # Extract the line at midY for color analysis
    line_data = resImg[midY, :, :]
    hsv_line = cv2.cvtColor(line_data.reshape(1, line_data.shape[0], 3), cv2.COLOR_BGR2HSV)

    color_sequence = []
    current_color = None
    start_position = 0

    # Identify colors along the line
    for i, pixel in enumerate(hsv_line[0]):
        color = identifyColor(pixel)  # Match HSV values to predefined color bounds
        if color != current_color:
            if current_color is not None:
                length = i - start_position
                color_sequence.append((current_color, start_position, length))
            current_color = color
            start_position = i

    # Add the last color segment
    color_sequence.append((current_color, start_position, len(hsv_line[0]) - start_position))

    # Filter out the background/base color by frequency
    color_counts = {}
    for color, _, length in color_sequence:
        if color in color_counts:
            color_counts[color] += length
        else:
            color_counts[color] = length

    base_color = max(color_counts, key=color_counts.get)
    filtered_colors = [(color, start, length) for color, start, length in color_sequence if color != base_color]

    # Show results in DEBUG mode
    if DEBUG:
        print("Color sequence detected (color, start, length):", filtered_colors)
        print("Base color detected as most frequent:", base_color)

        debug_img = np.copy(resImg)
        for color, start, length in filtered_colors:
            cv2.rectangle(debug_img, (start, midY-10), (start+length, midY+10), (255, 255, 255), 1)
            cv2.putText(debug_img, color, (start, midY-15), FONT, 0.4, (255, 255, 255), 1)

        cv2.imshow('Color Detection Along Line', debug_img)

    return filtered_colors

def identifyColor(hsv_pixel):
    # Define a function to map an HSV pixel to a color name based on your COLOUR_BOUNDS
    hue, sat, val = hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]
    for bounds in COLOUR_BOUNDS:
        if bounds[0][0] <= hue <= bounds[1][0] and bounds[0][1] <= sat <= bounds[1][1] and bounds[0][2] <= val <= bounds[1][2]:
            return bounds[2]
    return 'Unknown'

# This function should be part of your resistor analysis pipeline, replacing or augmenting the current findBands logic.



# MAIN
rectCascade = init(DEBUG)

cliveimg = cv2.imread("pic3.jpg")
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
