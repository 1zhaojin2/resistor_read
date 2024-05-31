import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# Ensure the correct path to the CSV file
csv_file_path = 'labels.csv'  # Update this if your file is in a different location

# Check if the file exists
if not os.path.isfile(csv_file_path):
    raise FileNotFoundError(f"No such file or directory: '{csv_file_path}'")

# Load the dataset
data = pd.read_csv(csv_file_path)

# Function to extract features (average color)
def extract_features(image_path):
    image = cv2.imread(image_path)
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color

# Extract features for all images
features = []
labels = []
for index, row in data.iterrows():
    image_path = row['image_path']
    color_name = row['color_name']
    avg_color = extract_features(image_path)
    features.append(avg_color)
    labels.append(color_name)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model for future use
import pickle
with open('color_knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Model saved as color_knn_model.pkl")
