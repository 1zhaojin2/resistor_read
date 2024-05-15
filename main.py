import cv2
import numpy as np
from resistor_detect_new_logic import load_and_detect_resistors, crop_resistor, preprocess_image, compute_vertical_medians, findBands, printResult
from camera import take_picture
from motors import setup, rotate_servo, destroy
import RPi.GPIO as GPIO

button = 17

setup()
GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def main(image_path):
    results = []
    DEBUG = True
    img, resistors = load_and_detect_resistors(image_path)
    if img is None or resistors is None:
        print("No resistors detected.")
        return

    for x, y, w, h in resistors:
        cropped_img = crop_resistor(img, x, y, w, h)
        preprocessed_img = preprocess_image(cropped_img)
        median_img = compute_vertical_medians(preprocessed_img)
        bands = findBands(median_img, DEBUG=True)
        # print("Detected bands:", bands)

        # Check the structure of color_code_positions, excluding 'last_pos'
        for key, value in bands.items():
            if key == 'last_pos':
                continue
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                print(f"Error: Value for {key} is not a list or tuple with at least two elements.")
                return

        results = printResult(bands, img, (x, y, w, h), DEBUG)
    
    
    # if the resistance is between 0 and 1000 ohms, rotate the servo once
    if results[0] < 1000:
        rotate_servo()
    elif results[0] < 10000:
        rotate_servo()
        rotate_servo()
    elif results[0] < 100000:
        rotate_servo()
        rotate_servo()
        rotate_servo()
    elif results[0] < 1000000:
        rotate_servo()
        rotate_servo()
        rotate_servo()
        rotate_servo()
    elif results[0] < 10000000:
        rotate_servo()
        rotate_servo()
        rotate_servo()
        rotate_servo()
        rotate_servo()





    if DEBUG:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main('pic4.jpg')

while(True):
    # check if button is pushed
    while GPIO.input(button) == GPIO.HIGH:
        pass
    
    rotate_servo()
    take_picture()
    main('pic.jpg')






