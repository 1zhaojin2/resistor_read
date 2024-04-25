import cv2
import numpy as np

clr = [(22, 103, 164), (35, 255, 255)  , "YELLOW" , 4 , (0,255,255)] #adjusted

def nothing(x):
    pass

# Load an image
image_path = '0.jpg'  # Update this to the path of your image
img = cv2.imread(image_path)
img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
if img is None:
    print("Error: Image not found.")
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('image')

    # Create trackbars for color change
    cv2.createTrackbar('H Lower', 'image', 0, 179, nothing)
    cv2.createTrackbar('S Lower', 'image', 0, 255, nothing)
    cv2.createTrackbar('V Lower', 'image', 0, 255, nothing)
    cv2.createTrackbar('H Upper', 'image', 0, 179, nothing)
    cv2.createTrackbar('S Upper', 'image', 0, 255, nothing)
    cv2.createTrackbar('V Upper', 'image', 0, 255, nothing)

    while True:
        h_l = cv2.getTrackbarPos('H Lower', 'image')
        s_l = cv2.getTrackbarPos('S Lower', 'image')
        v_l = cv2.getTrackbarPos('V Lower', 'image')
        h_u = cv2.getTrackbarPos('H Upper', 'image')
        s_u = cv2.getTrackbarPos('S Upper', 'image')
        v_u = cv2.getTrackbarPos('V Upper', 'image')

        lower_bound = np.array([h_l, s_l, v_l])
        upper_bound = np.array([h_u, s_u, v_u])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Original', img)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)    
        cv2.imshow("HSV Image", hsv)


        if cv2.waitKey(1) & 0xFF == 27:  # Press 'esc' to close
            break

    cv2.destroyAllWindows()