import cv2
import numpy as np
import matplotlib.pyplot as plt

def flood_fill_analysis(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found.")
        return

    # Define the distance between seed points and flood fill parameters
    seed_distance = 10
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    flags = 4
    flags |= cv2.FLOODFILL_MASK_ONLY
    flags |= (255 << 8)

    # Create a mask for flood filling to get unique regions
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Color collection
    colors = []

    # Calculate the range for seed points around the center
    start_x = max(center_x - (center_x % seed_distance), seed_distance)
    start_y = max(center_y - (center_y % seed_distance), seed_distance)
    end_x = min(center_x + (center_x % seed_distance) + seed_distance, w - seed_distance)
    end_y = min(center_y + (center_y % seed_distance) + seed_distance, h - seed_distance)

    # Perform flood fill from each seed point
    for x in range(start_x, end_x, seed_distance):
        for y in range(start_y, end_y, seed_distance):
            # Reset mask each time
            temp_mask = np.zeros((h + 2, w + 2), np.uint8)
            # Flood fill
            num, im, temp_mask, rect = cv2.floodFill(image.copy(), temp_mask, (x, y), 0, (10, 10, 10), (10, 10, 10), flags)
            # Get color statistics
            if num > 0:  # if the area has been filled
                mask |= temp_mask[1:-1, 1:-1]
                colors.append(image[y, x].tolist())  # get the color at the seed

    # Show unique colors found
    unique_colors = np.unique(colors, axis=0)

    # Plot the unique colors
    plt.figure(figsize=(10, 2))
    for i, color in enumerate(unique_colors):
        plt.fill_between([i, i+1], 0, 1, color=np.array(color)/255)
    plt.xlim(0, len(unique_colors))
    plt.axis('off')
    plt.title('Unique Colors Found by Flood Fill')
    plt.show()

    return unique_colors

# Usage
unique_colors = flood_fill_analysis('median_img.jpg')
print("Unique colors:", unique_colors)
