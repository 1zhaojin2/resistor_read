import RPi.GPIO as GPIO 
import time

GPIO.setmode(GPIO.BCM) 
    
# give servo pin i put 18 as a placeholder 
def setup():
    global pwm
    servo_pin = 18 
    GPIO.setup(servo_pin, GPIO.OUT)
    GPIO.output(servo_pin, GPIO.LOW)  # Set ServoPin to low
    pwm = GPIO.PWM(servo_pin, 50) # 50 Hz
    # neutral position apparently 
    pwm.start(7.5)  
 
def rotate_servo(): 
    
    try: 
        duty_cycle = 7.5 + (60 / 18)
        pwm.ChangeDutyCycle(duty_cycle) 
        time.sleep(0.5) # rotate one section (1s turns it twice)
        
    finally: 
        pwm.stop() 
        GPIO.cleanup()

def destroy():
    pwm.stop()
    GPIO.cleanup()

if __name__ == '__main__':     #Program start from here
    setup()
    try:
        rotate_servo()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the program destroy() will be executed.
        destroy()