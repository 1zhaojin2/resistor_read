import cv2

# Load the image
image_path = "pic1.jpg"
image = cv2.imread(image_path)

# Bounding box parameters from the inference result
x_center = 990.9375
y_center = 517.5
width = 646.875
height = 232.5
confidence = 0.8778327703475952
class_label = '3'

# Calculate the top-left and bottom-right coordinates of the bounding box
x1 = int(x_center - width / 2)
y1 = int(y_center - height / 2)
x2 = int(x_center + width / 2)
y2 = int(y_center + height / 2)

# Draw the bounding box on the image
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Put the class label and confidence score on the image
label = f'{class_label}: {confidence:.2f}'
cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with the bounding box
output_path = "pic1_with_bounding_box.jpg"
cv2.imwrite(output_path, image)

print(f"Image saved to {output_path}")
