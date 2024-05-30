from inference_sdk import InferenceHTTPClient

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TR7fHlxhfPCLx6wzniIC"
)

# Path to the image
image_path = "pic1.jpg"

# Send the image for inference using the file path
result = CLIENT.infer(image_path, model_id="detect-r/1")

# Print the result
print(result)
