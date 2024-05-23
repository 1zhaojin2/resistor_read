from picamera2 import Picamera2, Preview
import time
import keyboard

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (854, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()

def take_picture():
    picam2.capture("pic.jpg")
    print("Picture taken")
    
keyboard.add_hotkey('space', take_picture)

