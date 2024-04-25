import cv2
import numpy as np

def show_region_and_all_matched_areas(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Make a copy of the image to work on
        img_copy = image.copy()

        # Mask used for flood filling
        # Note: the size needs to be 2 pixels more than the image
        h, w = img_copy.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill settings
        floodflags = 8
        floodflags |= cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

        # Define the color range threshold for flood filling
        loDiff = (50, 50, 50, 50)
        upDiff = (50, 50, 50, 50)

        # Perform flood fill from point (x, y)
        cv2.floodFill(img_copy, mask, (x, y), 0, loDiff, upDiff, floodflags)

        # Extract the HSV values using the mask (subtract 1 from mask dimensions to match image size)
        highlighted_hsv = hsv_image[mask[1:-1, 1:-1] == 1]

        # Display the highlighted area in a separate window
        highlighted_area = np.zeros_like(image)
        highlighted_area[mask[1:-1, 1:-1] == 1] = image[mask[1:-1, 1:-1] == 1]
        cv2.imshow("Highlighted Area", highlighted_area)

        # Calculate the range of HSV values and display in another window
        if highlighted_hsv.size > 0:
            min_hsv = highlighted_hsv.min(axis=0)
            max_hsv = highlighted_hsv.max(axis=0)
            text = f"Area: {np.sum(mask == 1)} px²\nH range: {min_hsv[0]}-{max_hsv[0]}\nS range: {min_hsv[1]}-{max_hsv[1]}\nV range: {min_hsv[2]}-{max_hsv[2]}"
            
            # Create a mask for all areas matching the HSV range
            range_mask = cv2.inRange(hsv_image, min_hsv, max_hsv)

            # Create an image to display the matched areas
            matched_areas_img = cv2.bitwise_and(image, image, mask=range_mask)
            cv2.imshow("All Matched Areas", matched_areas_img)
        else:
            text = "Area: 0 px²\nH range: N/A\nS range: N/A\nV range: N/A"
            cv2.imshow("All Matched Areas", np.zeros_like(image))

        # Create a black image to display the text
        stats_img = np.zeros((200, 400, 3), dtype=np.uint8)
        y0, dy = 50, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(stats_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Statistics", stats_img)

# Load an image
image = cv2.imread('median_img.jpg')
# image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

# Check if image is loaded properly
if image is None:
    print("Error: Image not found.")
else:
    # Create a window to display the image
    cv2.namedWindow("Image Display")

    # Set the mouse callback function to track mouse movements
    cv2.setMouseCallback("Image Display", show_region_and_all_matched_areas)

    # Display the image
    cv2.imshow("Image Display", image)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
