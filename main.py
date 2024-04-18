import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_buffered_image(mat):
    if mat.ndim == 2:
        return mat
    elif mat.shape[2] == 3:
        return cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    return mat

def show_result(img):
    plt.imshow(img, cmap='gray')
    plt.title("Webcam Feed")
    plt.show()

def do_magic(image):
    small = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small = image[50:image.shape[0]-50, 50:image.shape[1]-50]
    mean = cv2.mean(small)[:3]

    nums = np.zeros(small.shape[1], dtype=int)
    avgy = np.zeros(small.shape[1], dtype=float)

    for x in range(small.shape[1]):
        for y in range(small.shape[0]):
            pixel = small[y, x]
            if all(pixel > np.array(mean) * 0.5):
                nums[x] += 1
                avgy[x] += y

    avgy[nums > 2] /= nums[nums > 2]

    # Example detection based on brightness, this would need to be adjusted
    # according to the actual detection logic needed.
    threshold_indices = nums > (np.mean(nums) + 15)
    minx, maxx = np.where(threshold_indices)[0][[0, -1]]
    line_color = (255, 0, 0)
    cv2.line(image, (minx, int(avgy[minx])), (maxx, int(avgy[maxx])), line_color, 4)

    return image

def main():
    print("Hello, OpenCV")
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera Error")
    else:
        print("Camera OK?")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, -1)
                frame = do_magic(frame)
                img = to_buffered_image(frame)
                show_result(img)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
