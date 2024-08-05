import cv2
import pytesseract
import csv
import time
import re

# Specify the path to the Tesseract executable if not added to the system's PATH
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Path to the Haarcascade classifier file
harcascade_path = r"/home/llen/Documents/Project/pyhon/py-num-plate-detection/my_project/model/carplate.xml"

# Initialize VideoCapture for two cameras
cap_enter = cv2.VideoCapture(0)  # Camera for entry
cap_exit = cv2.VideoCapture("http://10.10.1.162:4747/video")   # Camera for exit

# Set the width and height of the captured video
cap_enter.set(3, 640)
cap_enter.set(4, 480)
cap_exit.set(3, 640)
cap_exit.set(4, 480)

# Minimum area for a detected plate to be considered valid
min_area = 500
plate_history = {}  # Dictionary to store plate history

# Create a CSV file to save the license plate data
output_file_path = r"/home/llen/Documents/Project/pyhon/py-num-plate-detection/my_project/license_plates.csv"
with open(output_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Plate Number', 'License Plate', 'Status']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize the Haarcascade classifier
    plate_cascade = cv2.CascadeClassifier(harcascade_path)

    try:
        # Start the main loop
        while True:
            # Read frames from both cameras
            success_enter, img_enter = cap_enter.read()
            success_exit, img_exit = cap_exit.read()

            if not success_enter or not success_exit:
                print("Error: Unable to capture video from one or both cameras.")
                break

            # Preprocess the entry camera image
            img_gray_enter = cv2.cvtColor(img_enter, cv2.COLOR_BGR2GRAY)
            _, img_thresh_enter = cv2.threshold(img_gray_enter, 128, 255, cv2.THRESH_BINARY)
            img_denoised_enter = cv2.fastNlMeansDenoising(img_thresh_enter, None, 30, 7, 21)

            # Detect plates in the entry camera image
            plates_enter = plate_cascade.detectMultiScale(img_denoised_enter, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in plates_enter:
                area = w * h
                if area > min_area:
                    img_roi = img_enter[y: y + h, x: x + w]
                    plate_text = pytesseract.image_to_string(img_roi, config='--psm 8').strip()
                    if plate_text and re.match(r'^[A-Z0-9]{1,7}$', plate_text):
                        plate_history[plate_text] = 'entered'
                        writer.writerow({'Plate Number': len(plate_history), 'License Plate': plate_text, 'Status': 'entered'})
                    # Debugging output
                    cv2.rectangle(img_enter, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_enter, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Preprocess the exit camera image
            img_gray_exit = cv2.cvtColor(img_exit, cv2.COLOR_BGR2GRAY)
            _, img_thresh_exit = cv2.threshold(img_gray_exit, 128, 255, cv2.THRESH_BINARY)
            img_denoised_exit = cv2.fastNlMeansDenoising(img_thresh_exit, None, 30, 7, 21)

            # Detect plates in the exit camera image
            plates_exit = plate_cascade.detectMultiScale(img_denoised_exit, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in plates_exit:
                area = w * h
                if area > min_area:
                    img_roi = img_exit[y: y + h, x: x + w]
                    plate_text = pytesseract.image_to_string(img_roi, config='--psm 8').strip()
                    if plate_text and re.match(r'^[A-Z0-9]{1,7}$', plate_text):
                        if plate_text in plate_history and plate_history[plate_text] == 'entered':
                            writer.writerow({'Plate Number': len(plate_history) + 1, 'License Plate': plate_text, 'Status': 'exited'})
                            del plate_history[plate_text]  # Remove from history after exiting
                    # Debugging output
                    cv2.rectangle(img_exit, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_exit, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the processed images
            cv2.imshow("Entry Camera", img_enter)
            cv2.imshow("Exit Camera", img_exit)

            # Check for user input to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break  # Exit the loop if 'q' or ESC is pressed

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources and close windows
        cap_enter.release()
        cap_exit.release()
        cv2.destroyAllWindows()

