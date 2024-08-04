import cv2

for i in range(10):  # Check up to index 10
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Found webcam at index: {i}")
        cap.release()