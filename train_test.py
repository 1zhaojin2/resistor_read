import pickle
import numpy as np
import cv2
import os

# Load the trained model
model_path = 'segmented_bands/color_knn_model.pkl'
with open(model_path, 'rb') as f:
    knn_model = pickle.load(f)

# Function to extract features (average color)
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color

# Predict the color of a new image
def predict_color(image_path):
    avg_color = extract_features(image_path)
    avg_color = avg_color.reshape(1, -1)  # Reshape to match the model input
    prediction = knn_model.predict(avg_color)
    return prediction[0]

# Example usage: predict the color of a new image
new_image_path = 'segmented_bands/brown9.png'  # Update this path to your new image
predicted_color = predict_color(new_image_path)
print(f'The predicted color for the image {new_image_path} is: {predicted_color}')
