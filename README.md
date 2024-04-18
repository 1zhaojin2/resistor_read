# resistor_read

This Python project takes in an image of a resistor, use harr cascade to locate the resistor, zooms in and use the color bands to calculate the resistance. Works for 3, 4, 5 and 6 band resistors. 

Warning: The HSV minimum and maximum values in resistor_detect.py can vary between different lighting conditions and cameras, please adjust them according to your use case.