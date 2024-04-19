from picamera2 import Picamera2, Preview
import time
import keyboard

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (854, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
num = 0

print("start")
while True:
    num+=1
    keyboard.wait('esc')
    name = "pic" + str(num)
    picam2.capture_file(name+".jpg")
