import cv2
import numpy as np
from time import sleep
from inference_sdk import InferenceHTTPClient
import pickle

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TR7fHlxhfPCLx6wzniIC"
)

model_path = 'segmented_bands/color_knn_model.pkl'
with open(model_path, 'rb') as f:
    knn_model = pickle.load(f)

# Constants for red color bounds
RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)

lower_background = np.array([30, 0, 100])
upper_background = np.array([160, 255, 255])

# Define color bounds
COLOUR_BOUNDS = [
    [(0, 0, 0), (179, 255, 75), "BLACK", 0, (255, 255, 0)],
    [(0, 0, 0), (92, 255, 100), "BROWN", 1, (0, 255, 102)],
    [(148, 95, 99), (177, 158, 125), "RED", 2, (128, 0, 128)],
    [(5, 101, 0), (70, 195, 149), "ORANGE", 3, (0, 128, 255)],
    [(22, 103, 164), (35, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(30, 76, 89), (87, 255, 184), "GREEN", 5, (0, 255, 0)],
    [(113, 40, 82), (125, 255, 255), "BLUE", 6, (255, 0, 0)],
    [(130, 40, 100), (140, 250, 220), "PURPLE", 7, (255, 0, 127)],
    [(0, 0, 0), (0, 0, 0), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 0), (0, 0, 0), "WHITE", 9, (255, 255, 255)],  # Ignored for now
    [(0, 0, 0), (0, 0, 0), "GOLD", 10, (0, 215, 255)],
    [(0, 0, 0), (0, 0, 50), "SILVER", 11, (192, 192, 192)]
]

tolerance_codes = {
    1: "±1%",  # Brown
    2: "±2%",  # Red
    3: "±0.05",  # Orange
    4: "±0.02",  # Yellow
    5: "±0.5%",  # Green
    6: "±0.25%",  # Blue
    7: "±0.1%",  # Purple
    8: "±0.01",  # Gray

    10: "5%",  # Gold
    11: "10%"  # Silver
}

def load_and_detect_resistors(image_path):
    # Perform inference with the Roboflow model
    result = CLIENT.infer(image_path, model_id="detect-r/1")
    predictions = result['predictions']
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return None, None
    
    resistors = []
    for prediction in predictions:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])
        resistors.append((x, y, w, h))
    
    return img, resistors

def crop_resistor(img, x, y, w, h):
    start_x = x + 50
    end_x = x + w - 50
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

def validContour(cnt):
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Check if the width of the bounding rectangle is at least 10 pixels
    if w < 5:
        return False
    
    return True

def printResult(bands, img, resPos, DEBUG):
    
    results = []

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

    # Dictionary to convert color names to their respective codes
    color_code_dict = {name: code for (_, _, name, code, _) in COLOUR_BOUNDS}

    # Sort the bands by their x-coordinate and convert to a list
    sorted_bands = sorted(bands.items(), key=lambda item: item[0])

    print("Sorted bands:", sorted_bands)

    resistance = ""
    tolerance = ""
    temperature = ""

    if not sorted_bands:
        print("Error: No bands detected.")
        return results

    try:
    # Convert color names to resistor codes
        band_codes = [color_code_dict[band[1][0]] for band in sorted_bands]

        # Determine the base value and multiplier based on the number of bands
        if len(band_codes) >= 3:
            if len(band_codes) == 3:
                base_value = int(f"{band_codes[0]}{band_codes[1]}")
                multiplier = 10 ** band_codes[2]
            elif len(band_codes) == 4:
                base_value = int(f"{band_codes[0]}{band_codes[1]}")
                multiplier = 10 ** band_codes[2]
                tolerance = tolerance_codes.get(band_codes[3], "Unknown tolerance")
            elif len(band_codes) == 5:
                base_value = int(f"{band_codes[0]}{band_codes[1]}{band_codes[2]}")
                multiplier = 10 ** band_codes[3]
                tolerance = tolerance_codes.get(band_codes[4], "Unknown tolerance")
            elif len(band_codes) == 6:
                base_value = int(f"{band_codes[0]}{band_codes[1]}{band_codes[2]}")
                multiplier = 10 ** band_codes[3]
                tolerance = tolerance_codes.get(band_codes[4], "Unknown tolerance")
                temperature = temperature_list.get(band_codes[5], "Unknown temperature coefficient")

            final_resistance = base_value * multiplier
            resistance = f"{final_resistance} Ohms"  # Formatting resistance with units

    except Exception as e:
        print(f"Error processing bands: {e}")
        return results


    # Draw the resistor and the text
    cv2.rectangle(img, (resPos[0], resPos[1]), (resPos[0] + resPos[2], resPos[1] + resPos[3]), (0, 255, 0), 2)
    cv2.putText(img, f"Resistance: {resistance}", (resPos[0], resPos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f"Tolerance: {tolerance}", (resPos[0], resPos[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f"Temperature: {temperature}", (resPos[0], resPos[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    results.append(resistance)
    results.append(tolerance)
    results.append(temperature)

    if DEBUG:
        print(f"Resistance: {resistance}")
        print(f"Tolerance: {tolerance}")
        print(f"Temperature: {temperature}")

    return results

def preprocess_image(cropped_img):
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    h_filtered = cv2.bilateralFilter(h, 9, 75, 75)
    s_filtered = cv2.bilateralFilter(s, 9, 75, 75)
    v_filtered = cv2.bilateralFilter(v, 9, 75, 75)
    v[v > 180] = 180  # Threshold highlights
    filtered_hsv = cv2.merge([h_filtered, s_filtered, v_filtered])
    return cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)


# Function to extract features (average color)
def extract_features(image):
    if image is None:
        raise ValueError("Invalid image provided")
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color

# Predict the color of a new image
def predict_color(image):
    avg_color = extract_features(image)
    avg_color = avg_color.reshape(1, -1)  # Reshape to match the model input
    prediction = knn_model.predict(avg_color)
    return prediction[0]
    
def findBands(median_img, DEBUG=True):
    hsv_image = cv2.cvtColor(median_img, cv2.COLOR_BGR2HSV)
    background_mask = cv2.inRange(hsv_image, lower_background, upper_background)
    bands_mask = cv2.bitwise_not(background_mask)
    isolated_bands = cv2.bitwise_and(median_img, median_img, mask=bands_mask)
    contours, _ = cv2.findContours(bands_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if validContour(cnt)]
    band_colors = {}
    for i, cnt in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        band = median_img[y:y+h, x:x+w]
        if w > 6:
            band = band[:, 3:w-3]
        avg_color = np.mean(band, axis=(0, 1))
        color_name = predict_color(band)
        band_colors[x] = (color_name, avg_color)
        if DEBUG:
            print(f"Band {i + 1}: {color_name}")
    return band_colors


def display_images(images, titles):
    for img, title in zip(images, titles):
        cv2.imwrite(f"{title}.jpg", img)

def main(image_path):
    img, resistors = load_and_detect_resistors(image_path)
    if img is None or resistors is None:
        print("No resistors detected.")
        return

    for i, (x, y, w, h) in enumerate(resistors):
        cropped_img = crop_resistor(img, x, y, w, h)
        cv2.imwrite(f"cropped_resistor_{i}.jpg", cropped_img)
        preprocessed_img = preprocess_image(cropped_img)
        cv2.imwrite(f"preprocessed_resistor_{i}.jpg", preprocessed_img)
        median_img = compute_vertical_medians(preprocessed_img)
        cv2.imwrite(f"median_resistor_{i}.jpg", median_img)
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

main('pic1.jpg')
