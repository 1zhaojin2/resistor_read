import cv2
import numpy as np
from resistor_detect_new_logic import load_and_detect_resistors, crop_resistor, preprocess_image, compute_vertical_medians, findBands, printResult
import RPi.GPIO as GPIO
from time import sleep

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
    
    
rotate_servo()
useSolenoid()
