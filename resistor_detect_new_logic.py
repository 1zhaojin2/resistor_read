import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from picamera2 import Picamera2, Preview
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TR7fHlxhfPCLx6wzniIC"
)

# Constants for red color bounds
RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)

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

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT) # Solenoid
GPIO.setup(18, GPIO.OUT)

def useSolenoid():
    GPIO.output(25, 1)
    sleep(0.5)
    GPIO.output(25, 0)

def rotate_servo():
    try:
        pwm = GPIO.PWM(18, 50)
        pwm.start(0) 
        # Set duty cycle for counterclockwise rotation
        pwm.ChangeDutyCycle(5.7)  # Adjust this value if needed for your servo
        sleep(0.5)
    finally:
        pwm.stop()

def destroy():
    GPIO.cleanup()

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (854, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()

def take_picture():
    picam2.capture_file("pic1.jpg")
    print("Picture taken")

button_pin = 23
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

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
    
def findBands(median_img, DEBUG=True):
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

        # Save each color mask as an image
        if DEBUG:
            cv2.imwrite(f"mask_{color_name}.jpg", mask)
            
        # Process each contour
        for contour in contours:
            if validContour(contour):
                if DEBUG:
                    cv2.imwrite('contour.jpg', cv2.drawContours(resized_img.copy(), [contour], -1, (0, 255, 0), 2))
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # Store band info only if it is sufficiently far from the last accepted position
                    if abs(cx - last_pos) >= 30:
                        bands[cx] = (color_name, (cx, cy))
                        last_pos = cx  # Update last accepted position

    if DEBUG:
        cv2.imwrite('processed_bands.jpg', resized_img)  # Optional: Save the image with drawn contours

    return bands

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
        results = printResult(bands, img, (x, y, w, h), True)

        if results:
            resistance_value = int(results[0].split()[0])
            if resistance_value < 1000:
                rotate_servo()
                useSolenoid()
            elif resistance_value < 10000:
                rotate_servo()
                rotate_servo()
                useSolenoid()
            elif resistance_value < 100000:
                rotate_servo()
                rotate_servo()
                rotate_servo()
                useSolenoid()
            elif resistance_value < 1000000:
                rotate_servo()
                rotate_servo()
                rotate_servo()
                rotate_servo()
                useSolenoid()
            elif resistance_value < 10000000:
                rotate_servo()
                rotate_servo()
                rotate_servo()
                rotate_servo()
                rotate_servo()
                useSolenoid()
    
    cv2.imwrite('final_result.jpg', img)

    return

try:
    while(True):
        # check if button is pushed
        if GPIO.input(button_pin) == GPIO.HIGH:
            print("Button pushed")
            take_picture()
            main('pic1.jpg')
except KeyboardInterrupt:
    destroy()
