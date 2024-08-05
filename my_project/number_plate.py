import cv2

# Initialize the cascade classifier with the path to the XML file
plate_cascade = cv2.CascadeClassifier('/home/llen/Documents/Project/pyhon/py-num-plate-detection/model/carplate.xml')

# Initialize video capture from the camera (or a file path)
cap = cv2.VideoCapture(0)  # 0 for default camera; replace with a file path if needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around detected plates
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Number Plate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
