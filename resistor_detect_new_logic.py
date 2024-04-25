import cv2
import numpy as np

# Constants for red color bounds
RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)

# Define color bounds
COLOUR_BOUNDS = [
    [(0, 0, 0), (179, 255, 40), "BLACK", 0, (255, 255, 0)],
    [(0, 19, 55), (179, 190, 78), "BROWN", 1, (0, 255, 102)],
    [(0, 187, 125), (39, 255, 255), "RED", 2, (128, 0, 128)],
    [(6, 197, 87), (20, 255, 255), "ORANGE", 3, (0, 128, 255)],
    [(22, 103, 164), (35, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(30, 76, 89), (87, 255, 184), "GREEN", 5, (0, 255, 0)],
    [(113, 40, 82), (125, 255, 255), "BLUE", 6, (255, 0, 0)],
    [(130, 40, 100), (140, 250, 220), "PURPLE", 7, (255, 0, 127)],
    [(0, 0, 50), (179, 50, 80), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 0), (0, 0, 0), "WHITE", 9, (255, 255, 255)],  # Ignored for now
    [(19, 120, 80), (23, 255, 255), "GOLD", 10, (0, 215, 255)],
    [(0, 0, 0), (0, 0, 50), "SILVER", 11, (192, 192, 192)]
]
tolerance_codes = {
    10: "5%",  # Gold
    11: "10%"  # Silver
}

def load_and_detect_resistors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('cascade/haarcascade_resistors_0.xml')
    resistors = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)

    filtered_resistors = [r for r in resistors if r[2] >= 100 and r[3] >= 50]  # Filter by size
    return img, filtered_resistors

def crop_resistor(img, x, y, w, h):
    start_x = x + 110
    end_x = x + w - 110
    start_y = max(y + h // 2, 0)
    end_y = min(start_y + 20, img.shape[0])
    return img[start_y:end_y, start_x:end_x]

def compute_vertical_medians(cropped_img):
    median_values = np.zeros((1, cropped_img.shape[1], 3), dtype=np.uint8)
    for i in range(cropped_img.shape[1]):
        column = cropped_img[:, i, :]
        median_values[0, i, 0] = np.median(column[:, 0])
        median_values[0, i, 1] = np.median(column[:, 1])
        median_values[0, i, 2] = np.median(column[:, 2])
    median_img = np.tile(median_values, (20, 1, 1))
    return median_img

def preprocess_image(cropped_img):
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    h_filtered = cv2.bilateralFilter(h, 9, 75, 75)
    s_filtered = cv2.bilateralFilter(s, 9, 75, 75)
    v_filtered = cv2.bilateralFilter(v, 9, 75, 75)
    v[v > 180] = 180  # Threshold highlights
    filtered_hsv = cv2.merge([h_filtered, s_filtered, v_filtered])
    return cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)

def validContour(cnt):
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Check if the width of the bounding rectangle is at least 10 pixels
    if w < 10:
        return False
    
    return True

def printResult(bands, img, resPos, DEBUG):
    # Definitions for color and temperature properties
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
        11: "SILVER"
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

    color_codes = []
    for x, (color_name, _) in bands:
        # Assuming 'color_names' is a reverse dictionary from the color name to its index
        color_codes.append(COLOUR_BOUNDS[color_name][3])  # Get the index/code from color name

    # Definitions for display and calculations
    display_message = "Insufficient data to calculate resistance."
    detected_colors = [f"{name} at {x}" for x, (name, _) in bands]

    if len(color_codes) >= 3:
        try:
            if len(color_codes) == 3:
                base_value = int(str(color_codes[0]) + str(color_codes[1]))
                multiplier = 10 ** color_codes[2]
            elif len(color_codes) == 4:
                base_value = int(str(color_codes[0]) + str(color_codes[1]))
                multiplier = 10 ** color_codes[2]
                tolerance = tolerance_codes.get(color_codes[3], "Unknown")
            elif len(color_codes) == 5:
                base_value = int(str(color_codes[0]) + str(color_codes[1]) + str(color_codes[2]))
                multiplier = 10 ** color_codes[3]
                tolerance = tolerance_codes.get(color_codes[4], "Unknown")
            elif len(color_codes) == 6:
                base_value = int(str(color_codes[0]) + str(color_codes[1]) + str(color_codes[2]))
                multiplier = 10 ** color_codes[3]
                tolerance = tolerance_codes.get(color_codes[4], "Unknown")
                temperature = temperature_list.get(color_codes[5], "Unknown")

            final_resistance = base_value * multiplier
            display_message = f"{final_resistance} OHMS"
            print(f"Resistor Value: {final_resistance} Ohms")
            if tolerance:
                print(f"Tolerance: {tolerance}")
            if temperature:
                print(f"Temperature: {temperature}")
        except Exception as e:
            print(f"Error calculating resistance: {e}")

    # Display results on image
    x, y, w, h = resPos
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, display_message, (x + w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Optionally display all detected colors
    text_y = y + 20
    for color_text in detected_colors:
        cv2.putText(img, color_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        text_y += 20

    if DEBUG:
        cv2.imshow("Final Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
def findBands(median_img, DEBUG=False):
    resized_img = cv2.resize(median_img, (400, 200))  # Resize image for processing
    hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    bands = {}
    last_pos = -30  # Initialize last accepted position

    # Iterate over the defined color bounds to create masks and find contours
    for bounds in COLOUR_BOUNDS:
        lower_bound, upper_bound, color_name, _, _ = bounds
        mask = cv2.inRange(hsv, tuple(lower_bound), tuple(upper_bound))

        if color_name == "RED":  # Special handling for red if needed
            red_mask = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
            mask = cv2.bitwise_or(mask, red_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            if validContour(contour):
                if DEBUG:
                    cv2.imshow('Contour', cv2.drawContours(resized_img, [contour], -1, (0, 255, 0), 2))
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # Store band info only if it is sufficiently far from the last accepted position
                    if abs(cx - last_pos) >= 30:
                        bands[cx] = (color_name, (cx, cy))
                        last_pos = cx  # Update last accepted position

    if DEBUG:
        cv2.imshow('Processed Bands', resized_img)  # Optional: Display the image with drawn contours
        cv2.waitKey(0)

    return bands


def display_images(images, titles):
    for img, title in zip(images, titles):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(image_path):
    img, resistors = load_and_detect_resistors(image_path)
    if img is None or resistors is None:
        print("No resistors detected.")
        return

    for x, y, w, h in resistors:
        cropped_img = crop_resistor(img, x, y, w, h)
        preprocessed_img = preprocess_image(cropped_img)
        median_img = compute_vertical_medians(preprocessed_img)
        bands = findBands(median_img, DEBUG=True)
        print("Detected bands:", bands)

        # Check the structure of color_code_positions, excluding 'last_pos'
        for key, value in bands.items():
            if key == 'last_pos':
                continue
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                print(f"Error: Value for {key} is not a list or tuple with at least two elements.")
                return

        printResult(bands, img, (x, y, w, h), DEBUG=True)
        display_images([cropped_img, preprocessed_img, median_img], ['Cropped', 'Preprocessed', 'Median'])

main('pic4.jpg')


